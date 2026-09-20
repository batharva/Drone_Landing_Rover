# local_arena.py - LOCAL TESTING ENVIRONMENT (not the official evaluator)
# Physics, rover kinematics, actuator model and scoring are identical to the original.
import argparse
import math
import os
import tempfile
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from controller import FlightController

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_NAMES = ["SEARCH", "TRACK", "DESCEND", "LANDED"]


class CompetitionScorer:
    def __init__(self):
        self.xy_errors = []
        self.sim_time = 0.0
        self.run_complete = False

        # You must design your landing logic to beat these thresholds!
        self.MAX_SAFE_Z_VELOCITY = -0.65  # m/s
        self.MAX_SAFE_TILT = 0.52         # radians (~30 degrees)
        self.ROVER_SURFACE_Z = 0.15       # meters

    def update(self, drone_pos, drone_vel, drone_ori, rover_pos, dt):
        if self.run_complete:
            return

        self.sim_time += dt
        error_dist = math.sqrt((drone_pos[0] - rover_pos[0]) ** 2 + (drone_pos[1] - rover_pos[1]) ** 2)
        self.xy_errors.append(error_dist)

        if drone_pos[2] < 0.08:
            self._trigger_failure("GROUND COLLISION: Missed the rover entirely.")
            return

        if drone_pos[2] <= self.ROVER_SURFACE_Z + 0.01 and error_dist < 0.5:
            if drone_vel[2] < self.MAX_SAFE_Z_VELOCITY:
                self._trigger_failure(f"CRASH: Descent velocity too high ({drone_vel[2]:.2f} m/s).")
                return

            euler = p.getEulerFromQuaternion(drone_ori)
            max_tilt = max(abs(euler[0]), abs(euler[1]))
            if max_tilt > self.MAX_SAFE_TILT:
                self._trigger_failure(f"CRASH: Excessive tilt ({math.degrees(max_tilt):.1f} deg). Propeller strike.")
                return

            if self.sim_time > 2.0 and abs(drone_vel[2]) < 0.05:
                self._trigger_success(error_dist)

    def _trigger_failure(self, reason):
        print(f"\n[LOCAL TEST FAILED] {reason}")
        self.run_complete = True

    def _trigger_success(self, final_error):
        print(f"\n[LOCAL TEST SUCCESS] Safe Touchdown Confirmed! (touchdown xy error {final_error:.3f} m)")
        self.run_complete = True


def setup_environment(show_panels=False):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    if not show_panels:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)   # hide side panels (cleaner video)
    p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=45, cameraPitch=-35,
                                 cameraTargetPosition=[0, 0, 0])


def _marker_mesh_path(half=0.4, z=0.002):
    """Flat quad with explicit UVs, sitting on top of the plate (link frame)."""
    path = os.path.join(tempfile.gettempdir(), "arena_marker_plate.obj")
    with open(path, "w") as f:
        f.write(f"v {-half} {-half} {z}\nv {half} {-half} {z}\nv {half} {half} {z}\nv {-half} {half} {z}\n")
        f.write("vt 0 1\nvt 1 1\nvt 1 0\nvt 0 0\n")
        f.write("vn 0 0 1\n")
        f.write("f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n")
    return path


def spawn_rover(box_texture=False):
    # Base rover chassis (1m x 1m x 0.1m)
    base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05], rgbaColor=[1, 1, 1, 1])
    base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05])

    # Marker plate (0.8m x 0.8m). Collision is the original box; visual is a mesh with explicit UVs
    # (or the original textured box when box_texture=True).
    if box_texture:
        marker_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.002], rgbaColor=[1, 1, 1, 1])
    else:
        marker_vis = p.createVisualShape(p.GEOM_MESH, fileName=_marker_mesh_path(), rgbaColor=[1, 1, 1, 1])
    marker_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.002])

    rover_id = p.createMultiBody(
        baseMass=10.0,
        baseCollisionShapeIndex=base_col,
        baseVisualShapeIndex=base_vis,
        basePosition=[0, 0, 0.05],
        linkMasses=[0.01],
        linkCollisionShapeIndices=[marker_col],
        linkVisualShapeIndices=[marker_vis],
        linkPositions=[[0, 0, 0.052]],
        linkOrientations=[[0, 0, 0, 1]],
        linkInertialFramePositions=[[0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1]],
        linkParentIndices=[0],
        linkJointTypes=[p.JOINT_FIXED],
        linkJointAxis=[[0, 0, 1]]
    )

    tex_path = os.path.join(HERE, "aruco_marker.png")
    tex_id = p.loadTexture(tex_path)
    if tex_id < 0:
        raise FileNotFoundError(f"Could not load texture: {tex_path}")
    p.changeVisualShape(rover_id, 0, textureUniqueId=tex_id, rgbaColor=[1, 1, 1, 1])
    return rover_id


def spawn_quadcopter():
    chassis_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.04], rgbaColor=[0.15, 0.15, 0.15, 1])
    chassis_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.04])
    rotor_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.12, length=0.01, rgbaColor=[0.4, 0.4, 0.4, 0.8])
    rotor_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.12, height=0.01)

    L = 0.25
    H = 0.04
    drone_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=chassis_col,
        baseVisualShapeIndex=chassis_vis,
        basePosition=[0, 0, 3.0],
        linkMasses=[0.05] * 4,
        linkCollisionShapeIndices=[rotor_col] * 4,
        linkVisualShapeIndices=[rotor_vis] * 4,
        linkPositions=[[L, L, H], [-L, L, H], [L, -L, H], [-L, -L, H]],
        linkOrientations=[[0, 0, 0, 1]] * 4,
        linkInertialFramePositions=[[0, 0, 0]] * 4,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
        linkParentIndices=[0, 0, 0, 0],
        linkJointTypes=[p.JOINT_FIXED] * 4,
        linkJointAxis=[[0, 0, 1]] * 4
    )
    return drone_id


def get_drone_camera_image(drone_id, width=320, height=240, fov=60):
    pos, ori = p.getBasePositionAndOrientation(drone_id)
    rot_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    camera_pos = np.array(pos) + rot_matrix.dot(np.array([0, 0, -0.1]))
    camera_vector = rot_matrix.dot(np.array([0, 0, -1]))
    up_vector = rot_matrix.dot(np.array([1, 0, 0]))
    view_matrix = p.computeViewMatrix(camera_pos, camera_pos + camera_vector, up_vector)
    projection_matrix = p.computeProjectionMatrixFOV(fov, width / height, 0.1, 20.0)
    _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, projection_matrix,
                                       renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return cv2.cvtColor(np.reshape(rgb, (height, width, 4)).astype(np.uint8), cv2.COLOR_RGBA2BGR)


def show_camera(frame, brain):
    view = frame.copy()                       # overlay on a copy; controller gets the raw frame
    s = getattr(brain, "state", None)
    if isinstance(s, int) and 0 <= s < len(STATE_NAMES):
        e = (getattr(brain, "info", None) or {}).get("e_xy", 0.0)
        cv2.putText(view, f"{STATE_NAMES[s]}  err={e:.2f}m", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Drone Downward Camera", view)
    cv2.waitKey(1)


def make_brain(use_rl):
    try:
        return FlightController(use_rl=use_rl)
    except TypeError:                          # starter-template controller has no use_rl argument
        return FlightController()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rl", action="store_true", help="ignore policy_weights.py (PID only)")
    ap.add_argument("--fast", action="store_true", help="do not sleep to real time")
    ap.add_argument("--box-texture", action="store_true", help="original box texture mapping")
    ap.add_argument("--panels", action="store_true", help="keep PyBullet side panels")
    ap.add_argument("--no-follow", action="store_true", help="fixed 3D camera")
    args = ap.parse_args()

    setup_environment(show_panels=args.panels)
    rover_id = spawn_rover(box_texture=args.box_texture)
    drone_id = spawn_quadcopter()

    scorer = CompetitionScorer()
    my_drone_brain = make_brain(use_rl=not args.no_rl)

    dt = 1.0 / 240.0
    t = 0.0
    step = 0
    L = 0.25

    print("[SYSTEM] Simulation active. Testing local control logic...")

    try:
        while p.isConnected():
            # Rover Kinematics
            rover_vx = 1.2 * math.cos(t * 0.4) + 0.5 * math.cos(t * 1.1)
            rover_vy = 1.2 * math.sin(t * 0.3) + 0.5 * math.sin(t * 0.9)
            p.resetBaseVelocity(rover_id, linearVelocity=[rover_vx, rover_vy, 0])

            # Sensors
            drone_pos, drone_ori = p.getBasePositionAndOrientation(drone_id)
            drone_vel, _ = p.getBaseVelocity(drone_id)
            rover_pos, _ = p.getBasePositionAndOrientation(rover_id)
            camera_frame = get_drone_camera_image(drone_id)
            show_camera(camera_frame, my_drone_brain)

            # Scorer Update
            scorer.update(drone_pos, drone_vel, drone_ori, rover_pos, dt)
            if scorer.run_complete:
                time.sleep(5)
                break

            # The arena asks the controller for the motor forces
            f1, f2, f3, f4 = my_drone_brain.calculate_forces(camera_frame, drone_pos, drone_ori, drone_vel)
            if not all(math.isfinite(f) for f in (f1, f2, f3, f4)):
                print(f"[SYSTEM] Controller returned non-finite forces at t={t:.2f}s: {(f1, f2, f3, f4)}")
                break

            # Apply Actuator Forces
            p.applyExternalForce(drone_id, -1, [0, 0, f1], [L, L, 0], p.LINK_FRAME)
            p.applyExternalForce(drone_id, -1, [0, 0, f2], [-L, L, 0], p.LINK_FRAME)
            p.applyExternalForce(drone_id, -1, [0, 0, f3], [L, -L, 0], p.LINK_FRAME)
            p.applyExternalForce(drone_id, -1, [0, 0, f4], [-L, -L, 0], p.LINK_FRAME)

            p.stepSimulation()
            if not args.fast:
                time.sleep(dt)
            t += dt
            step += 1

            if step % 120 == 0:                                   # telemetry every 0.5 s
                s = getattr(my_drone_brain, "state", None)
                name = STATE_NAMES[s] if isinstance(s, int) and 0 <= s < 4 else "-"
                err = math.hypot(drone_pos[0] - rover_pos[0], drone_pos[1] - rover_pos[1])
                print(f"t={t:5.1f}s {name:7s} drone=({drone_pos[0]:+.2f},{drone_pos[1]:+.2f},{drone_pos[2]:.2f}) "
                      f"rover=({rover_pos[0]:+.2f},{rover_pos[1]:+.2f}) xy_err={err:.2f} vz={drone_vel[2]:+.2f}")

            if not args.no_follow and step % 4 == 0:               # view only, no physics effect
                p.resetDebugVisualizerCamera(6, 45, -35, [rover_pos[0], rover_pos[1], 0.5])
    except p.error:
        print("[SYSTEM] Physics window closed by user. Exiting...")
    finally:
        cv2.destroyAllWindows()
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()