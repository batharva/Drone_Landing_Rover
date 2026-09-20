import numpy as np
import pybullet as p
import local_arena as A
from controller import Camera, ArucoTracker, quat_to_R

A.setup_environment()
rover, drone = A.spawn_rover(), A.spawn_quadcopter()      # rover stays at (0,0), static
tracker = ArucoTracker(Camera())
tests = [(0, 0, 3.0, 0, 0), (0.5, -0.3, 2.0, 0, 0), (-0.6, 0.4, 1.6, 0.1, -0.1), (0.2, 0.2, 1.3, 0, 0.15),
         (0, 0, 1.0, 0, 0)]
for dx, dy, z, roll, pitch in tests:
    p.resetBasePositionAndOrientation(drone, [dx, dy, z], p.getQuaternionFromEuler([roll, pitch, 0]))
    frame = A.get_drone_camera_image(drone)
    pos, q = p.getBasePositionAndOrientation(drone)
    R, det = quat_to_R(q), None
    for _ in range(80):                                    # cycles dictionaries/flips until lock
        det = tracker.detect(frame, np.array(pos), R)
        if det is not None:
            break
    rp, _ = p.getBasePositionAndOrientation(rover)
    if det is None:
        print(f"z={z}: NOT DETECTED")
    else:
        err = np.linalg.norm(det["center_w"] - np.array(rp[:2]))
        print(f"z={z} offset=({dx},{dy}) tilt=({roll},{pitch}): est-vs-true error {err*100:.1f} cm, "
              f"edge {det['size']:.2f} m (expect 0.80)")