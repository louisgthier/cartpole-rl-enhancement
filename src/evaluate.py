# src/evaluate.py
import torch
import numpy as np

from src.modified_cartpole import ModifiedCartPoleEnv
from src.dqn_model import DQN

def evaluate_model(model_path: str):
    env = ModifiedCartPoleEnv(render_mode="human")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(state_dim, n_actions)

    # Load the file and check if it's a checkpoint
    loaded = torch.load(model_path, map_location=torch.device("cpu"))
    if isinstance(loaded, dict) and "policy_state" in loaded:
        # If it's a checkpoint, extract the policy state
        policy_net.load_state_dict(loaded["policy_state"])
    else:
        # Otherwise, assume it's a direct state dict
        policy_net.load_state_dict(loaded)
    policy_net.eval()

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = policy_net(state_tensor).max(1)[1].item()
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Total Reward: {total_reward}")
    env.close()