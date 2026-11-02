import argparse
import os
from pathlib import Path

from model import build_policy
import numpy as np
import torch
import gymnasium as gym
import gym_pusht



RUNS_DIR = Path(__file__).resolve().parent / "runs"   # same folder the trainer writes to
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Where the render window opens (x,y). Without this the window system chose the second
# monitor on this machine. Override from the shell: SDL_VIDEO_WINDOW_POS=2000,100 python ...
os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "200,150")

def load_model_and_data(run_name: str = "MSEPolicy"):

    checkpoint = torch.load(RUNS_DIR / run_name / "final.pt", map_location=DEVICE)
    cfg = checkpoint["config"]
    STATE_DIM = 5
    ACTION_DIM = 2
    # policy type and widths are saved by the trainer (older MSE checkpoints: fall back to config)
    policy_type = checkpoint.get("policy_type", "mse")
    hidden_dims = checkpoint.get("hidden_dims", cfg.get("hidden_dims"))
    model = build_policy(policy_type, STATE_DIM, ACTION_DIM, cfg["chunk_size"], hidden_dims)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(DEVICE)

    # Saved as tensors; the env speaks numpy, so keep the stats as numpy arrays.
    normalizer = {k: v.cpu().numpy() for k, v in checkpoint["normalizer"].items()}

    return model, normalizer, cfg


def predict_chunk(policy, normalizer, observation, env, num_steps: int = 10) -> np.ndarray:
    """One policy query: raw observation (5,) -> chunk of env-ready actions (chunk, 2)."""
    # obs (5,) float64 pixels -> normalize -> float32 tensor on device
    obv_normalized = (observation - normalizer["state_mean"]) / normalizer["state_std"]
    obv_tensor = torch.from_numpy(obv_normalized).float().to(DEVICE)
    # (5,) -> (chunk, 2) normalized actions
    actions = policy.sample_actions(obv_tensor, num_steps=num_steps).cpu().numpy()
    # back to pixel coordinates, inside the env's action range, float32
    actions = actions * normalizer["action_std"] + normalizer["action_mean"]
    return np.clip(actions, env.action_space.low, env.action_space.high).astype(np.float32)


def rollout(policy, normalizer, env, seed: int, num_steps: int = 10) -> tuple[float, bool]:
    """Run exactly one episode. Returns (max reward reached, whether it succeeded)."""
    observation, info = env.reset(seed=seed)
    max_reward = 0.0
    success = False
    done = False

    while not done:
        # Receding horizon: execute the whole chunk, then query the policy again.
        for action in predict_chunk(policy, normalizer, observation, env, num_steps):
            observation, reward, terminated, truncated, info = env.step(action)
            if env.render_mode == "human":
                env.render()
            max_reward = max(max_reward, float(reward))
            success = success or bool(info["is_success"])
            done = terminated or truncated
            if done:
                break

    return max_reward, success


def evaluate(policy, normalizer, num_episodes: int = 100, num_render_episodes: int = 0,
             num_steps: int = 10) -> dict:
    """Run num_episodes seeded episodes (seeds 0..num_episodes-1) and aggregate.

    The first num_render_episodes episodes are shown in a window (real time);
    the remaining ones run headless. num_steps is the Euler step count for the
    flow policy (ignored by the MSE policy).
    """
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    render_env = None
    if num_render_episodes > 0:
        render_env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="human")

    max_rewards, successes = [], []
    for seed in range(num_episodes):
        ep_env = render_env if seed < num_render_episodes else env
        max_reward, success = rollout(policy, normalizer, ep_env, seed, num_steps)
        max_rewards.append(max_reward)
        successes.append(success)
        if render_env is not None and seed == num_render_episodes - 1:
            render_env.close()   # close the window as soon as the last shown episode ends
            render_env = None
    env.close()
    if render_env is not None:
        render_env.close()

    return {
        "mean_max_reward": float(np.mean(max_rewards)),
        "std_max_reward": float(np.std(max_rewards)),
        "success_rate": float(np.mean(successes)),
        "max_rewards": max_rewards,
    }


def main():

    parser = argparse.ArgumentParser(description="Evaluate a trained Push-T policy.")
    parser.add_argument("--run_name", default="MSEPolicy", help="folder under runs/")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render_episodes", type=int, default=5, help="first N episodes shown in a window")
    args = parser.parse_args()

    policy, normalizer, cfg = load_model_and_data(args.run_name)
    results = evaluate(policy, normalizer, num_episodes=args.episodes,
                       num_render_episodes=args.render_episodes,
                       num_steps=cfg["flow_num_steps"])
    print(f"run: {args.run_name}")
    print(f"episodes: {len(results['max_rewards'])}")
    print(f"mean max reward: {results['mean_max_reward']:.3f} +/- {results['std_max_reward']:.3f}")
    print(f"success rate:    {results['success_rate']:.2%}")

if __name__ == "__main__":
    main()
