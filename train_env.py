import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from controller import (FlightController, quat_to_R, Camera, CAM_OFFSET_B, MARKER_Z,
                        DT, OBS_DIM, DESCEND, L_ARM)


def spawn_rover(xy):
    bv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05])
    bc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05])
    mv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.002])
    mc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.002])
    return p.createMultiBody(
        baseMass=10.0, baseCollisionShapeIndex=bc, baseVisualShapeIndex=bv,
        basePosition=[xy[0], xy[1], 0.05], linkMasses=[0.01],
        linkCollisionShapeIndices=[mc], linkVisualShapeIndices=[mv],
        linkPositions=[[0, 0, 0.052]], linkOrientations=[[0, 0, 0, 1]],
        linkInertialFramePositions=[[0, 0, 0]], linkInertialFrameOrientations=[[0, 0, 0, 1]],
        linkParentIndices=[0], linkJointTypes=[p.JOINT_FIXED], linkJointAxis=[[0, 0, 1]])


def spawn_drone(pos):
    cv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.04])
    cc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.04])
    rv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.12, length=0.01)
    rc = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.12, height=0.01)
    L, H = 0.25, 0.04
    return p.createMultiBody(
        baseMass=1.0, baseCollisionShapeIndex=cc, baseVisualShapeIndex=cv, basePosition=list(pos),
        linkMasses=[0.05] * 4, linkCollisionShapeIndices=[rc] * 4, linkVisualShapeIndices=[rv] * 4,
        linkPositions=[[L, L, H], [-L, L, H], [L, -L, H], [-L, -L, H]],
        linkOrientations=[[0, 0, 0, 1]] * 4, linkInertialFramePositions=[[0, 0, 0]] * 4,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4, linkParentIndices=[0] * 4,
        linkJointTypes=[p.JOINT_FIXED] * 4, linkJointAxis=[[0, 0, 1]] * 4)


def arena_traj():   # exact trajectory used in local_arena.py
    return dict(ax=(1.2, 0.5), wx=(0.4, 1.1), px=(0, 0), ay=(1.2, 0.5), wy=(0.3, 0.9), py=(0, 0))


def random_traj(rng, s):   # same family (2 sinusoids/axis), randomised
    return dict(ax=(rng.uniform(0.8, 1.5) * s, rng.uniform(0.0, 0.8) * s),
                wx=(rng.uniform(0.25, 0.6), rng.uniform(0.7, 1.4)), px=tuple(rng.uniform(0, 2 * np.pi, 2)),
                ay=(rng.uniform(0.8, 1.5) * s, rng.uniform(0.0, 0.8) * s),
                wy=(rng.uniform(0.2, 0.5), rng.uniform(0.6, 1.2)), py=tuple(rng.uniform(0, 2 * np.pi, 2)))


def traj_vel(tr, t):
    vx = tr["ax"][0] * math.cos(tr["wx"][0] * t + tr["px"][0]) + tr["ax"][1] * math.cos(tr["wx"][1] * t + tr["px"][1])
    vy = tr["ay"][0] * math.sin(tr["wy"][0] * t + tr["py"][0]) + tr["ay"][1] * math.sin(tr["wy"][1] * t + tr["py"][1])
    return vx, vy


class LandingEnv(gym.Env):
    """One env per process (pybullet default client). Use SubprocVecEnv."""

    def __init__(self, controller_factory=None, decim=8, max_time=25.0, ramp_steps=150_000,
                 fixed_difficulty=None, arena=False, seed=None, gui=False):
        super().__init__()
        self.factory = controller_factory or (lambda: FlightController(use_rl=False))
        self.decim, self.max_time, self.ramp_steps = decim, max_time, ramp_steps
        self.fixed_difficulty, self.arena = fixed_difficulty, arena
        self.rng = np.random.default_rng(seed)
        self.steps_done = 0
        self.cam = Camera()
        self.observation_space = spaces.Box(-5.0, 5.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _difficulty(self):
        if self.fixed_difficulty is not None:
            return self.fixed_difficulty
        return min(1.0, 0.25 + 0.75 * self.steps_done / self.ramp_steps)   # curriculum

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        d = self._difficulty()
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT)
        p.loadURDF("plane.urdf")
        if self.arena:
            self.traj, rov0, off, z0, eul = arena_traj(), np.zeros(2), np.zeros(2), 3.0, [0, 0, 0]
        else:
            self.traj = random_traj(self.rng, 0.3 + 0.7 * d)
            rov0 = self.rng.uniform(-1, 1, 2)
            off = self.rng.uniform(-1, 1, 2) * (0.3 + 0.9 * d)
            z0 = self.rng.uniform(2.5, 3.5)
            eul = [self.rng.uniform(-.1, .1), self.rng.uniform(-.1, .1), self.rng.uniform(-.3, .3)]
        self.rover = spawn_rover(rov0)
        self.drone = spawn_drone([rov0[0] + off[0], rov0[1] + off[1], z0])
        p.resetBasePositionAndOrientation(self.drone, [rov0[0] + off[0], rov0[1] + off[1], z0],
                                          p.getQuaternionFromEuler(eul))
        self.t, self.z_prev, self.prev_action = 0.0, z0, np.zeros(3)
        self.ctrl = self.factory()
        self._advance()
        return self._obs(), {}

    def _obs(self):
        o = self.ctrl.last_obs
        return np.zeros(OBS_DIM, np.float32) if o is None else o.astype(np.float32)

    def _vision(self, pos, R):
        """Analytic stand-in for ArUco: all 4 corners must be inside the image."""
        if self.rng.random() < 0.02:
            return None
        rp, rq = p.getBasePositionAndOrientation(self.rover)
        Rr = quat_to_R(rq)
        for cx, cy in 0.4 * np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]):
            uv = self.cam.project(np.array(rp) + Rr @ np.array([cx, cy, 0.054]), pos, R)
            if uv is None or not (6 < uv[0] < self.cam.w - 6 and 6 < uv[1] < self.cam.h - 6):
                return None
        h = max(pos[2] + CAM_OFFSET_B[2] - MARKER_Z, 0.1)
        return np.array(rp[:2]) + self.rng.normal(0, 0.004 + 0.004 * h, 2)

    def _apply(self, f):
        L = L_ARM
        for fi, r in zip(f, ([L, L, 0], [-L, L, 0], [L, -L, 0], [-L, -L, 0])):
            p.applyExternalForce(self.drone, -1, [0, 0, fi], r, p.LINK_FRAME)

    def _score(self):   # mirrors CompetitionScorer
        pos, quat = p.getBasePositionAndOrientation(self.drone)
        vel, _ = p.getBaseVelocity(self.drone)
        rp, _ = p.getBasePositionAndOrientation(self.rover)
        e = math.hypot(pos[0] - rp[0], pos[1] - rp[1])
        if pos[2] < 0.08:
            return "ground"
        if pos[2] <= 0.16 and e < 0.5:
            if vel[2] < -0.65:
                return "fast"
            eu = p.getEulerFromQuaternion(quat)
            if max(abs(eu[0]), abs(eu[1])) > 0.52:
                return "tilt"
            if self.t > 2.0 and abs(vel[2]) < 0.05:
                return "success"
        return None

    def _advance(self):
        status = None
        for _ in range(self.decim):
            p.resetBaseVelocity(self.rover, [*traj_vel(self.traj, self.t), 0])
            pos, quat = p.getBasePositionAndOrientation(self.drone)
            vel, _ = p.getBaseVelocity(self.drone)
            R = quat_to_R(quat)
            f = self.ctrl.control(self._vision(pos, R), np.array(pos), R, np.array(vel))
            self._apply(f)
            p.stepSimulation()
            self.t += DT
            status = self._score()
            if status:
                break
        return status

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        if self.ctrl.policy is None:
            self.ctrl.res = action.astype(float)
        status = self._advance()
        self.steps_done += 1

        pos, quat = p.getBasePositionAndOrientation(self.drone)
        vel, _ = p.getBaseVelocity(self.drone)
        rp, _ = p.getBasePositionAndOrientation(self.rover)
        rv, _ = p.getBaseVelocity(self.rover)
        R = quat_to_R(quat)
        e = math.hypot(pos[0] - rp[0], pos[1] - rp[1])
        vrel = math.hypot(vel[0] - rv[0], vel[1] - rv[1])
        tilt = math.acos(float(np.clip(R[2, 2], -1, 1)))

        r = 0.5 * math.exp(-(e / 0.3) ** 2) + 0.25 * math.exp(-(vrel / 0.5) ** 2) - 0.5 * tilt
        r -= 0.02 * float(action @ action) + 0.05 * float(np.sum((action - self.prev_action) ** 2))
        if self.ctrl.state == DESCEND:
            r += 10.0 * (self.z_prev - pos[2]) * math.exp(-(e / 0.25) ** 2)   # descend only when aligned
            if e > 0.4:
                r -= 0.5
        terminated = status is not None
        if status == "success":
            r += 50.0 + 100.0 * math.exp(-(e / 0.2) ** 2)
        elif terminated:
            r -= 100.0
        truncated = (not terminated) and self.t >= self.max_time
        if truncated:
            r -= 20.0
        self.prev_action, self.z_prev = action.copy(), pos[2]
        info = dict(status=status, err=e, state=self.ctrl.state, t=self.t)
        return self._obs(), float(r), terminated, truncated, info

    def close(self):
        if p.isConnected():
            p.disconnect()