# Autonomous Drone Precision Landing on a Moving Rover

An autonomous control system designed to perform high-precision landing of a quadcopter onto a continuously moving rover pad in a PyBullet simulation. The architecture employs a **hybrid approach** combining computer vision, state estimation, cascaded PID baseline control, and a zero-dependency **Reinforcement Learning (PPO) residual controller**.

---

## Key Features & Architecture

- **Computer Vision Pipeline:** ArUco marker detection with auto-dictionary scanning and dynamic backprojection (2D camera frame to 3D world space).
- **State Estimation:** 3-axis Kalman Filter tracking rover position, velocity, and acceleration to maintain trajectory predictions even during camera occlusion/blind spots at lower altitudes.
- **Cascaded Control System:** Cascaded Position $\rightarrow$ Velocity $\rightarrow$ Acceleration PID controller paired with geometric attitude PD tracking and a 4-rotor force mixer.
- **Hybrid RL Residual:** A PPO policy trained in Gymnasium that provides fine acceleration corrections (residuals) on top of the PID baseline.
- **Zero-Dependency Inference:** The trained RL actor is exported to pure NumPy (`policy_weights.py`), eliminating the need for PyTorch or Stable-Baselines3 during runtime.

---

## Project Structure

```text
Intra/
├── local_arena.py          # PyBullet simulation environment (with telemetry overlay)
├── controller.py           # Core submission controller (Vision + KF + PID + RL)
├── policy_weights.py       # (Optional) Exported NumPy weights for RL actor network
├── train_env.py            # Headless Gymnasium environment for training
├── train.py                # SB3 PPO training script with curriculum learning
├── export_policy.py        # Exports PyTorch PPO checkpoint to pure NumPy dictionary
├── evaluate.py             # Evaluation & benchmarking script for PID vs. PID+RL
├── vision_check.py         # Visual geometry & ArUco detection diagnostic tool
├── aruco_marker.png        # ArUco target texture
└── requirements.txt        # Python dependency list
```

---

## Prerequisites & Installation

### Requirements

- Python 3.8+
- Windows / Linux / macOS

### Setup Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

> [!IMPORTANT]
> Install **only** `opencv-contrib-python`. Do not install `opencv-python` alongside it, as it creates namespace conflicts with `cv2.aruco`.

```bash
pip install numpy pybullet opencv-contrib-python gymnasium stable-baselines3 torch tensorboard
```

---

## How to Run

### 1. Vision & Geometry Check

Verify that ArUco marker detection, camera backprojection, and 3D pose calculations match the simulator physics:

```bash
python vision_check.py
```

### 2. Run PID Baseline Simulation

To test the baseline PID controller in the simulation environment without RL weights:

```bash
python local_arena.py --no-rl
```

### 3. Train the RL Policy

Train the PPO residual agent across multiple CPU cores:

```bash
python train.py --timesteps 3000000 --envs 8
```

- Monitor training progress via TensorBoard:
  ```bash
  tensorboard --logdir runs
  ```

### 4. Export Policy to NumPy

Export the trained PyTorch PPO checkpoint (`ppo_landing.zip`) into standalone NumPy weights (`policy_weights.py`):

```bash
python export_policy.py ppo_landing.zip
```

### 5. Evaluate Performance

Compare performance, landing success rate, touchdown error, and tracking RMS error across episodes:

- **Evaluate PID Baseline:**
  ```bash
  python evaluate.py --episodes 50
  ```
- **Evaluate Hybrid PID + RL Policy:**
  ```bash
  python evaluate.py --rl --episodes 50
  ```
- **Run Evaluation on Exact Arena Trajectory:**
  ```bash
  python evaluate.py --rl --arena
  ```

### 6. Run Full Arena Demo (PID + RL)

Once `policy_weights.py` is present in the working directory, running `local_arena.py` automatically loads and applies the RL residual controller:

```bash
python local_arena.py
```

---

## System Tuning Guide

If tuning the controller under modified vehicle parameters, refer to the guidance below:

| Symptom                                            | Action / Tuning Adjustment                                                                                           |
| :------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| **Attitude wobble or high-frequency buzzing**      | Lower attitude gains $K_R$ (e.g., 250 $\rightarrow$ 150) and $K_W$ (28 $\rightarrow$ 20).                            |
| **Sluggish tilt / response time**                  | Increase $K_R$ and $K_W$ together while maintaining $K_W \approx 1.8 \cdot \sqrt{K_R}$.                              |
| **Lags behind moving rover**                       | Raise position gain $K_{P,\text{POS}}$ (2 $\rightarrow$ 3) and velocity gain $K_{P,\text{VEL}}$ (5 $\rightarrow$ 7). |
| **Overshooting during tracking**                   | Lower position gain $K_{P,\text{POS}}$.                                                                              |
| **Premature descent before alignment**             | Tighten alignment conditions in `TRACK` mode (`e_xy < 0.08`).                                                        |
| **Touchdown error $> 0.3\text{ m}$**               | Raise maximum descent velocity `V_DESC_MAX` (up to 0.6 m/s) or increase `BLIND_Z` cutoff.                            |
| **Impact velocity near limit ($0.65\text{ m/s}$)** | Lower minimum descent speed `V_DESC_MIN` or motor cutoff height `Z_CUT`.                                             |

---

## Submission Deliverables Checklist

- [x] `controller.py` (Main controller file)
- [x] `policy_weights.py` (Exported NumPy RL weights, optional)
- [x] `README.md` (System documentation and setup guidelines)
- [x] Demo Video / Performance Logs
