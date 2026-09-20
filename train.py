import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from train_env import LandingEnv


def make_env(rank, seed, ramp):
    def _f():
        return Monitor(LandingEnv(seed=seed + rank, ramp_steps=ramp))
    return _f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ramp = int(0.4 * a.timesteps / a.envs)          # curriculum: easy -> full difficulty over first 40%
    venv = SubprocVecEnv([make_env(i, a.seed, ramp) for i in range(a.envs)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)   # reward-norm only

    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, learning_rate=3e-4, clip_range=0.2, ent_coef=0.0,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-1.0),
                tensorboard_log="runs", verbose=1, device="cpu", seed=a.seed)
    ckpt = CheckpointCallback(save_freq=max(100_000 // a.envs, 1), save_path="checkpoints",
                              name_prefix="ppo_landing")
    model.learn(total_timesteps=a.timesteps, callback=ckpt, progress_bar=False)
    model.save("ppo_landing")