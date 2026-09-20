# local_arena.py (YOUR LOCAL TESTING ENVIRONMENT - DO NOT EDIT)
import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import cv2

# This imports YOUR code from controller.py
from controller import FlightController

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
        error_dist = math.sqrt((drone_pos[0]-rover_pos[0])**2 + (drone_pos[1]-rover_pos[1])**2)
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
                self._trigger_failure(f"CRASH: Excessive tilt ({math.degrees(max_tilt):.1f}°). Propeller strike.")
                return
                
            if self.sim_time > 2.0 and abs(drone_vel[2]) < 0.05:
                self._trigger_success(error_dist)

    def _trigger_failure(self, reason):
        print(f"\n[LOCAL TEST FAILED] {reason}")
        self.run_complete = True

    def _trigger_success(self, final_error):
        print(f"\n[LOCAL TEST SUCCESS] Safe Touchdown Confirmed!")
        self.run_complete = True

def setup_environment():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

def spawn_rover():
    # Base rover chassis (1m x 1m x 0.1m)
    base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05], rgbaColor=[1, 1, 1, 1])
    base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05])
    
    # Flat top marker plate (0.8m x 0.8m, flush on top surface to prevent texture stretching)
    marker_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.002], rgbaColor=[1, 1, 1, 1])
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
    
    # Apply ArUco texture cleanly onto the dedicated top plate link (link index 0)
    tex_id = p.loadTexture("aruco_marker.png")
    p.changeVisualShape(rover_id, 0, textureUniqueId=tex_id)
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
        linkMasses=[0.05]*4, 
        linkCollisionShapeIndices=[rotor_col]*4, 
        linkVisualShapeIndices=[rotor_vis]*4,
        linkPositions=[[L, L, H], [-L, L, H], [L, -L, H], [-L, -L, H]], 
        linkOrientations=[[0, 0, 0, 1]]*4,
        linkInertialFramePositions=[[0, 0, 0]]*4, 
        linkInertialFrameOrientations=[[0, 0, 0, 1]]*4,
        linkParentIndices=[0, 0, 0, 0], 
        linkJointTypes=[p.JOINT_FIXED]*4, 
        linkJointAxis=[[0, 0, 1]]*4
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
    _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return cv2.cvtColor(np.reshape(rgb, (height, width, 4)).astype(np.uint8), cv2.COLOR_RGBA2BGR)

def main():
    setup_environment()
    rover_id = spawn_rover()
    drone_id = spawn_quadcopter()
    
    scorer = CompetitionScorer()
    
    # -----------------------------------------------------------------------
    # THIS INITIALIZES YOUR CUSTOM CODE FROM controller.py
    # -----------------------------------------------------------------------
    my_drone_brain = FlightController()
    
    dt = 1.0 / 240.0
    t = 0.0
    
    print("[SYSTEM] Simulation active. Testing local control logic...")
    
    while True:
        if not p.isConnected():
            print("[SYSTEM] Physics window closed by user. Exiting...")
            break
            
        # Rover Kinematics
        rover_vx = 1.2 * math.cos(t * 0.4) + 0.5 * math.cos(t * 1.1)
        rover_vy = 1.2 * math.sin(t * 0.3) + 0.5 * math.sin(t * 0.9)
        p.resetBaseVelocity(rover_id, linearVelocity=[rover_vx, rover_vy, 0])
        
        # Sensors
        drone_pos, drone_ori = p.getBasePositionAndOrientation(drone_id)
        drone_vel, _ = p.getBaseVelocity(drone_id)
        rover_pos, _ = p.getBasePositionAndOrientation(rover_id)
        camera_frame = get_drone_camera_image(drone_id)
        
        cv2.imshow("Drone Downward Camera", camera_frame)
        cv2.waitKey(1)

        # Scorer Update
        scorer.update(drone_pos, drone_vel, drone_ori, rover_pos, dt)
        if scorer.run_complete:
            time.sleep(5)
            break

        # -----------------------------------------------------------------------
        # THE ARENA ASKS YOUR CODE FOR THE MOTOR FORCES
        # -----------------------------------------------------------------------
        f1, f2, f3, f4 = my_drone_brain.calculate_forces(camera_frame, drone_pos, drone_ori, drone_vel)

        # Apply Actuator Forces
        L = 0.25
        p.applyExternalForce(drone_id, -1, [0, 0, f1], [ L,  L, 0], p.LINK_FRAME)
        p.applyExternalForce(drone_id, -1, [0, 0, f2], [-L,  L, 0], p.LINK_FRAME)
        p.applyExternalForce(drone_id, -1, [0, 0, f3], [ L, -L, 0], p.LINK_FRAME)
        p.applyExternalForce(drone_id, -1, [0, 0, f4], [-L, -L, 0], p.LINK_FRAME)
        
        p.stepSimulation()
        time.sleep(dt)
        t += dt

if __name__ == '__main__':
    main()