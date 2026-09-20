"""
Autonomous landing on a moving rover.
Camera -> ArUco -> ray/plane 2D->3D -> Kalman filter (rover p,v,a)
       -> state machine SEARCH/TRACK/DESCEND/LANDED
       -> cascaded position/velocity PID (+ optional RL residual on acceleration)
       -> geometric attitude PD -> 4-rotor mixer -> forces [N]
Needs only numpy + cv2. Optional: policy_weights.py (exported PPO actor).
"""
import math
import numpy as np
import cv2

try:
    from policy_weights import WEIGHTS as _RL_WEIGHTS
except Exception:
    _RL_WEIGHTS = None

# ---------------- constants (match local_arena.py) ----------------
G, MASS, L_ARM = 9.81, 1.2, 0.25
I_BODY = np.array([0.0174, 0.0174, 0.0331])   # estimated from the arena geometry
DT = 1.0 / 240.0
F_MAX = 12.0                                  # per-rotor limit [N]

CAM_OFFSET_B = np.array([0.0, 0.0, -0.1])
CAM_FOV_DEG = 60.0                            # vertical FOV (pybullet convention)
MARKER_Z, MARKER_SIZE = 0.104, 0.8            # top of plate, plate edge [m]
TOUCH_Z = 0.144                               # drone-origin height when chassis rests on pad
Z_CUT = 0.152                                 # cut motors below this

# ---------------- tunables ----------------
Z_TRACK, Z_SEARCH, BLIND_Z = 1.5, 3.5, 1.15
V_DESC_MAX, V_DESC_MIN = 0.60, 0.30
KP_POS, KP_VEL, KI_VEL = 2.0, 5.0, 1.0        # cascaded: position -> velocity -> accel
KP_Z, KP_VZ = 2.0, 6.0
KR, KW = 250.0, 28.0                          # attitude PD (rad/s^2 per rad, per rad/s)
TAU_MAX = 1.5
POLICY_DECIM = 8                              # RL runs at 30 Hz, held between updates
RES_SCALE = np.array([1.5, 1.5, 1.5])         # max residual accel [m/s^2]
OBS_DIM = 16

SEARCH, TRACK, DESCEND, LANDED = 0, 1, 2, 3


def quat_to_R(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# ---------------- camera model ----------------
class Camera:
    """Body frame: camera looks along -z_b, image-up = +x_b, image-right = -y_b."""

    def __init__(self, w=320, h=240, fov_deg=CAM_FOV_DEG):
        self.w, self.h = w, h
        self.f = (h / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        self.cx, self.cy = w / 2.0, h / 2.0

    def pose(self, pos, R):
        cam = np.asarray(pos) + R @ CAM_OFFSET_B
        return cam, -R[:, 1], R[:, 0], -R[:, 2]          # pos, right, up, forward

    def project(self, P, pos, R):
        cam, right, up, fwd = self.pose(pos, R)
        d = np.asarray(P) - cam
        zc = d @ fwd
        if zc < 0.1:
            return None
        return self.cx + self.f * (d @ right) / zc, self.cy - self.f * (d @ up) / zc

    def backproject(self, u, v, pos, R, plane_z):
        """Pixel -> world point on horizontal plane z=plane_z (uses full attitude)."""
        cam, right, up, fwd = self.pose(pos, R)
        ray = right * ((u - self.cx) / self.f) - up * ((v - self.cy) / self.f) + fwd
        if ray[2] > -1e-6:
            return None
        t = (plane_z - cam[2]) / ray[2]
        return cam + t * ray if t > 0 else None


# ---------------- ArUco tracker ----------------
_DICTS = ["DICT_6X6_250", "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_1000",
          "DICT_5X5_250", "DICT_4X4_250", "DICT_7X7_250", "DICT_ARUCO_ORIGINAL"]
_FLIPS = [None, 1, 0, -1]       # mirror variants (texture may be mirrored on the box top)


class ArucoTracker:
    def __init__(self, cam):
        self.cam = cam
        self.cands = [(d, f) for d in _DICTS if hasattr(cv2.aruco, d) for f in _FLIPS]
        self.locked, self.marker_id, self.rr = None, None, 0
        self._det = {}

    def _detector(self, dname):
        if dname in self._det:
            return self._det[dname]
        dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dname))
        if hasattr(cv2.aruco, "ArucoDetector"):                    # OpenCV >= 4.7
            prm = cv2.aruco.DetectorParameters()
            try: prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            except Exception: pass
            det = cv2.aruco.ArucoDetector(dic, prm)
            fn = lambda g: det.detectMarkers(g)[:2]
        else:                                                      # older API
            prm = cv2.aruco.DetectorParameters_create()
            try: prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            except Exception: pass
            fn = lambda g: cv2.aruco.detectMarkers(g, dic, parameters=prm)[:2]
        self._det[dname] = fn
        return fn

    def _try(self, gray, cand, pos, R):
        dname, flip = cand
        g = gray if flip is None else cv2.flip(gray, flip)
        corners, ids = self._detector(dname)(g)
        if ids is None or len(ids) == 0:
            return None
        H, W = gray.shape[:2]
        for c, mid in zip(corners, ids.flatten()):
            if self.marker_id is not None and mid != self.marker_id:
                continue
            pts = c.reshape(4, 2).astype(float)
            if flip in (1, -1): pts[:, 0] = (W - 1) - pts[:, 0]
            if flip in (0, -1): pts[:, 1] = (H - 1) - pts[:, 1]
            world = [self.cam.backproject(u, v, pos, R, MARKER_Z) for u, v in pts]
            if any(P is None for P in world):
                continue
            world = np.array(world)
            edge = np.linalg.norm(world - np.roll(world, -1, axis=0), axis=1).mean()
            if abs(edge - MARKER_SIZE) > 0.25 * MARKER_SIZE:      # geometry sanity gate
                continue
            ctr = world.mean(axis=0)
            e0 = world[1] - world[0]
            return dict(center_w=ctr[:2], yaw=math.atan2(e0[1], e0[0]),
                        pix=pts.mean(0), id=int(mid), size=float(edge))
        return None

    def detect(self, frame, pos, R):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.locked is None:                       # one candidate per frame until first hit
            cand = self.cands[self.rr % len(self.cands)]
            self.rr += 1
            res = self._try(gray, cand, pos, R)
            if res is not None:
                self.locked, self.marker_id = cand, res["id"]
                print(f"[VISION] locked dict={cand[0]} flip={cand[1]} id={res['id']}")
            return res
        return self._try(gray, self.locked, pos, R)


# ---------------- rover Kalman filter (p, v, a per axis) ----------------
class RoverKF:
    def __init__(self, sigma=0.02, jerk_std=0.6, tau_a=3.0):
        self.sm2, self.tau = sigma ** 2, tau_a
        d = DT
        self.F = np.array([[1, d, 0.5 * d * d], [0, 1, d], [0, 0, math.exp(-d / tau_a)]])
        self.Q = jerk_std ** 2 * np.array([[d**5 / 20, d**4 / 8, d**3 / 6],
                                           [d**4 / 8, d**3 / 3, d**2 / 2],
                                           [d**3 / 6, d**2 / 2, d]])
        self.reset()

    def reset(self):
        self.x = np.zeros((2, 3))        # rows: x,y ; cols: p,v,a
        self.P = np.diag([1.0, 4.0, 4.0])
        self.init, self.t_since = False, 0.0

    def predict(self):
        self.x = self.x @ self.F.T
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.t_since += DT

    def update(self, z_xy):
        if not self.init:
            self.x[:, 0], self.x[:, 1:] = z_xy, 0.0
            self.P = np.diag([self.sm2, 4.0, 4.0])
            self.init, self.t_since = True, 0.0
            return
        S = self.P[0, 0] + self.sm2
        K = self.P[:, 0] / S
        self.x += np.outer(z_xy - self.x[:, 0], K)
        self.P = self.P - np.outer(K, self.P[0, :])
        self.t_since = 0.0


class NumpyPolicy:
    def __init__(self, w):
        self.w = [np.array(w[k], dtype=float) for k in ("w0", "b0", "w1", "b1", "w2", "b2")]

    def __call__(self, obs):
        w0, b0, w1, b1, w2, b2 = self.w
        h = np.tanh(w0 @ obs + b0)
        h = np.tanh(w1 @ h + b1)
        return np.clip(w2 @ h + b2, -1.0, 1.0)


# ---------------- flight controller ----------------
class FlightController:
    def __init__(self, use_rl=True):
        self.mass, self.gravity = MASS, G
        self.cam = Camera()
        self.tracker = ArucoTracker(self.cam)
        self.kf = RoverKF()
        self.policy = NumpyPolicy(_RL_WEIGHTS) if (use_rl and _RL_WEIGHTS is not None) else None
        self.res = np.zeros(3)
        self.last_obs = None
        self.state = SEARCH
        self.k = 0
        self.R_prev = None
        self.int_v = np.zeros(2)
        self.vz_ref = 0.0
        self.vz_prev = 0.0
        self.align_t = 0.0
        self.search_anchor, self.search_t = None, 0.0
        self.last_det = None
        self.info = {}

    # ---- arena entry point ----
    def calculate_forces(self, camera_frame, drone_pos, drone_ori, drone_vel):
        pos = np.asarray(drone_pos, dtype=float)
        vel = np.asarray(drone_vel, dtype=float)
        R = quat_to_R(drone_ori)
        meas = None
        if camera_frame is not None and self.state != LANDED:
            h, w = camera_frame.shape[:2]
            if (w, h) != (self.cam.w, self.cam.h):
                self.cam = Camera(w, h)
                self.tracker.cam = self.cam
            det = self.tracker.detect(camera_frame, pos, R)
            if det is not None:
                self.last_det, meas = det, det["center_w"]
        return self.control(meas, pos, R, vel)

    # ---- everything except pixel processing (shared with the training env) ----
    def control(self, meas_xy, pos, R, vel):
        self.k += 1
        z = pos[2]

        # angular velocity (body) by finite-differencing attitude (not provided by arena)
        if self.R_prev is None:
            W = np.zeros(3)
        else:
            M = self.R_prev.T @ R
            W = vee(0.5 * (M - M.T)) / DT
        self.R_prev = R.copy()

        if self.state == LANDED:
            return 0.0, 0.0, 0.0, 0.0

        # ---- [B] state estimation ----
        self.kf.predict()
        if meas_xy is not None:
            if self.kf.init and self.kf.t_since > 3.0:
                self.kf.reset()
            if (not self.kf.init) or np.linalg.norm(meas_xy - self.kf.x[:, 0]) < 1.5:
                self.kf.update(meas_xy)
        have = self.kf.init and self.kf.t_since < 1.0
        age = self.kf.t_since if self.kf.init else 3.0
        if self.kf.init:
            p_r, v_r, a_r = self.kf.x[:, 0].copy(), self.kf.x[:, 1].copy(), self.kf.x[:, 2].copy()
        else:
            p_r, v_r, a_r = pos[:2].copy(), np.zeros(2), np.zeros(2)
        e_p, e_v = p_r - pos[:2], v_r - vel[:2]
        e_xy = float(np.linalg.norm(e_p))

        # ---- landing state machine ----
        if self.state == SEARCH:
            if have:
                self.state, self.search_anchor = TRACK, None
        elif self.state == TRACK:
            if not have:
                self._enter_search(p_r if self.kf.init else pos[:2])
            else:
                aligned = (e_xy < 0.12 and np.linalg.norm(e_v) < 0.35
                           and age < 0.1 and abs(z - Z_TRACK) < 0.2)
                self.align_t = self.align_t + DT if aligned else 0.0
                if self.align_t > 0.4:
                    self.state = DESCEND
        elif self.state == DESCEND:
            if z <= Z_CUT or self._impact(z, vel[2]):
                self.state = LANDED                         # emergency motor cutoff
                return 0.0, 0.0, 0.0, 0.0
            if (z > 0.7 and e_xy > 0.45 and age < 0.3) or (z > BLIND_Z and age > 0.6):
                self.state, self.align_t = TRACK, 0.0       # abort, re-acquire

        # ---- [C] references ----
        if self.state == SEARCH:
            ref_p, ref_v, ref_a = self._search_ref(pos)
            vz_cmd, tilt_lim = float(np.clip(1.0 * (Z_SEARCH - z), -0.5, 1.0)), 0.35
        else:
            ref_p, ref_v, ref_a = p_r, v_r, a_r
            tilt_lim = 0.5 if z > 1.0 else 0.25
            if self.state == TRACK:
                vz_cmd = float(np.clip(KP_Z * (Z_TRACK - z), -1.0, 1.0))
            else:
                vz_cmd = -self._desc_speed(z, e_xy)

        # cascaded PID: position -> velocity -> acceleration (+ rover accel feed-forward)
        v_cmd = ref_v + KP_POS * (ref_p - pos[:2])
        rel = v_cmd - ref_v
        n = np.linalg.norm(rel)
        if n > 3.0:
            v_cmd = ref_v + rel * (3.0 / n)
        ev = v_cmd - vel[:2]
        if self.state in (TRACK, DESCEND) and e_xy < 0.6:
            self.int_v = np.clip(self.int_v + ev * DT, -0.3, 0.3)
        else:
            self.int_v[:] = 0.0
        a_xy = ref_a + KP_VEL * ev + KI_VEL * self.int_v

        self.vz_ref += float(np.clip(vz_cmd - self.vz_ref, -2.0 * DT, 2.0 * DT))
        a_z = KP_VZ * (self.vz_ref - vel[2])

        # ---- RL residual on commanded acceleration ----
        if self.state in (TRACK, DESCEND):
            if self.policy is not None and self.k % POLICY_DECIM == 0 and self.last_obs is not None:
                self.res = self.policy(self.last_obs)
            a_xy = a_xy + RES_SCALE[:2] * self.res[:2]
            a_z += RES_SCALE[2] * self.res[2]

        # ---- desired thrust vector -> desired attitude (geometric control) ----
        F = self.mass * np.array([a_xy[0], a_xy[1], G + a_z])
        F[2] = max(F[2], 0.3 * self.mass * G)
        fh, fh_max = np.linalg.norm(F[:2]), F[2] * math.tan(tilt_lim)
        if fh > fh_max:
            F[:2] *= fh_max / fh
        b3d = F / np.linalg.norm(F)
        b1c = np.array([R[0, 0], R[1, 0], 0.0])             # yaw is unactuated: hold current heading
        nb = np.linalg.norm(b1c)
        b1c = b1c / nb if nb > 1e-3 else np.array([1.0, 0.0, 0.0])
        b2d = np.cross(b3d, b1c)
        b2d /= np.linalg.norm(b2d)
        b1d = np.cross(b2d, b3d)
        Rd = np.column_stack((b1d, b2d, b3d))
        eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        tau = I_BODY * (-KR * eR - KW * W) + np.cross(W, I_BODY * W)
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        T = float(np.clip(F @ R[:, 2], 0.0, 4 * F_MAX))

        # ---- motor mixer (rotors: f1 (+L,+L) f2 (-L,+L) f3 (+L,-L) f4 (-L,-L)) ----
        A, B = tau[0] / L_ARM, -tau[1] / L_ARM
        f = np.clip(np.array([T + A + B, T + A - B, T - A + B, T - A - B]) / 4.0, 0.0, F_MAX)

        self.last_obs = self._build_obs(pos, R, vel, e_p, e_v, a_r, age)
        self.info = dict(state=self.state, e_xy=e_xy, age=age)
        if self.k % 120 == 0:
            print(f"t={self.k*DT:5.1f}s state={self.state} e_xy={e_xy:.2f} age={age:.2f} z={z:.2f}")
        return float(f[0]), float(f[1]), float(f[2]), float(f[3])

    # ---- helpers ----
    def _desc_speed(self, z, e_xy):
        h = z - TOUCH_Z
        v = float(np.clip(V_DESC_MIN + 0.55 * (h - 0.15), V_DESC_MIN, V_DESC_MAX))
        lo = 0.3 if z < BLIND_Z else 0.0                     # once blind, keep committing
        return v * float(np.clip(1.0 - (e_xy - 0.08) / 0.30, lo, 1.0))

    def _impact(self, z, vz):
        hit = z < 0.30 and self.vz_prev < -0.12 and (vz - self.vz_prev) > 0.15
        self.vz_prev = vz
        return hit

    def _enter_search(self, xy):
        self.state, self.search_anchor, self.search_t = SEARCH, np.array(xy, float), 0.0
        self.int_v[:] = 0.0

    def _search_ref(self, pos):
        if self.search_anchor is None:
            self.search_anchor, self.search_t = pos[:2].copy(), 0.0
        self.search_t += DT
        ts = max(0.0, self.search_t - 1.5)
        r, ph = min(0.25 * ts, 6.0), 0.6 * ts
        return (self.search_anchor + r * np.array([math.cos(ph), math.sin(ph)]),
                np.zeros(2), np.zeros(2))

    def _build_obs(self, pos, R, vel, e_p, e_v, a_r, age):
        b3 = R[:, 2]
        o = np.array([e_p[0], e_p[1], e_v[0] / 2, e_v[1] / 2, a_r[0] / 2, a_r[1] / 2,
                      (pos[2] - TOUCH_Z) / 2.0, vel[2], 3.0 * b3[0], 3.0 * b3[1],
                      min(age, 3.0) / 3.0, 1.0 if age < 0.5 * DT else 0.0,
                      1.0 if self.state == DESCEND else 0.0,
                      self.res[0], self.res[1], self.res[2]], dtype=np.float32)
        return np.clip(o, -5.0, 5.0)