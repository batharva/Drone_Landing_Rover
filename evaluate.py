import argparse
import numpy as np
from controller import FlightController
from train_env import LandingEnv

ap = argparse.ArgumentParser()
ap.add_argument("--rl", action="store_true", help="use policy_weights.py residual")
ap.add_argument("--arena", action="store_true", help="exact local_arena trajectory (1 deterministic run)")
ap.add_argument("--episodes", type=int, default=50)
a = ap.parse_args()

if a.rl and FlightController(use_rl=True).policy is None:
    raise SystemExit("policy_weights.py not found - run export_policy.py first")

env = LandingEnv(controller_factory=lambda: FlightController(use_rl=a.rl),
                 fixed_difficulty=1.0, arena=a.arena, seed=123)
n = 1 if a.arena else a.episodes
results, track_err = [], []
for i in range(n):
    env.reset()
    done = False
    while not done:
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        done = term or trunc
        if info["state"] in (1, 2):    # TRACK, DESCEND
            track_err.append(info["err"])
    results.append((info["status"], info["err"]))
    print(f"ep {i:3d}: {info['status']}  touchdown xy err {info['err']:.3f} m  t={info['t']:.1f}s")

ok = [r for r in results if r[0] == "success"]
print(f"\nmode={'PID+RL' if a.rl else 'PID baseline'}  success {len(ok)}/{n}")
if ok:
    print(f"touchdown err mean {np.mean([r[1] for r in ok]):.3f}  max {np.max([r[1] for r in ok]):.3f} m")
if track_err:
    print(f"tracking RMS xy error (TRACK+DESCEND): {np.sqrt(np.mean(np.square(track_err))):.3f} m")