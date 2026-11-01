from model import MSEPolicy
import numpy as np
import torch
import gymnasium as gym
import gym_pusht



CHECKPOINT_PATH = "/home/parikshat/MyWorkings/learning/imitation_learning/pushT/runs/MSEPolicy/final.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_data(CHECKPOINT_PATH= CHECKPOINT_PATH):

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    cfg = checkpoint["config"]
    STATE_DIM = 5
    ACTION_DIM = 2
    model = MSEPolicy(STATE_DIM, cfg["chunk_size"] * ACTION_DIM, cfg["hidden_dims"])
    model.load_state_dict(checkpoint["model"])
    model.eval().to(DEVICE)

    # Saved as tensors; the env speaks numpy, so keep the stats as numpy arrays.
    normalizer = {k: v.cpu().numpy() for k, v in checkpoint["normalizer"].items()}

    return model, normalizer, cfg


def predict_chunk(policy, normalizer, observation, env) -> np.ndarray:
    """One policy query: raw observation (5,) -> chunk of env-ready actions (chunk, 2)."""
    # obs (5,) float64 pixels -> normalize -> float32 tensor on device
    obv_normalized = (observation - normalizer["state_mean"]) / normalizer["state_std"]
    obv_tensor = torch.from_numpy(obv_normalized).float().to(DEVICE)
    # (5,) -> (chunk, 2) normalized actions
    actions = policy.sample_actions(obv_tensor).cpu().numpy()
    # back to pixel coordinates, inside the env's action range, float32
    actions = actions * normalizer["action_std"] + normalizer["action_mean"]
    return np.clip(actions, env.action_space.low, env.action_space.high).astype(np.float32)


def rollout(policy, normalizer, env, seed: int) -> tuple[float, bool]:
    """Run exactly one episode. Returns (max reward reached, whether it succeeded)."""
    observation, info = env.reset(seed=seed)
    max_reward = 0.0
    success = False
    done = False

    while not done:
        # Receding horizon: execute the whole chunk, then query the policy again.
        for action in predict_chunk(policy, normalizer, observation, env):
            observation, reward, terminated, truncated, info = env.step(action)
            if env.render_mode == "human":
                env.render()
            max_reward = max(max_reward, float(reward))
            success = success or bool(info["is_success"])
            done = terminated or truncated
            if done:
                break

    return max_reward, success


def evaluate(policy, normalizer, num_episodes: int = 100, render: bool = False) -> dict:
    """Run num_episodes seeded episodes (seeds 0..num_episodes-1) and aggregate."""
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="state", render_mode="human" if render else None
    )
    max_rewards, successes = [], []
    for seed in range(num_episodes):
        max_reward, success = rollout(policy, normalizer, env, seed)
        max_rewards.append(max_reward)
        successes.append(success)
    env.close()

    return {
        "mean_max_reward": float(np.mean(max_rewards)),
        "std_max_reward": float(np.std(max_rewards)),
        "success_rate": float(np.mean(successes)),
        "max_rewards": max_rewards,
    }


def main():

    policy, normalizer, cfg = load_model_and_data()
    results = evaluate(policy, normalizer, num_episodes=100)
    print(f"episodes: {len(results['max_rewards'])}")
    print(f"mean max reward: {results['mean_max_reward']:.3f} +/- {results['std_max_reward']:.3f}")
    print(f"success rate:    {results['success_rate']:.2%}")

if __name__ == "__main__":
    main()
