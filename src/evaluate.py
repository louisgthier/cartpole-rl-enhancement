# src/evaluate.py
import torch
import numpy as np

from src.modified_cartpole import ModifiedCartPoleEnv
from src.normalization import NormalizedEnv
from src.dqn_model import DQN, DuelingDQN
import src.config as config

# Map model names to classes
MODEL_MAP = {
    "DQN": DQN,
    "DuelingDQN": DuelingDQN
}

model_cass = MODEL_MAP[config.MODEL_TYPE]

def evaluate_model(model_path: str):
    base_env = ModifiedCartPoleEnv(render_mode="human")
    env = NormalizedEnv(base_env)  # Wrap environment with normalization
    
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = model_cass(state_dim, n_actions)

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

    steps = 0
    
    action_counts = {0: 0, 1: 0, 2: 0}
    
    total_energy_used = 0
    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = policy_net(state_tensor).max(1)[1].item()
            action_counts[action] += 1
        next_state, reward, done, _, info = env.step(action)
        total_energy_used += info["energy_used"]
        steps += 1
        total_reward += reward
        state = next_state
        
        
        if steps % 100 == 0:
            avg_energy_used = total_energy_used / steps
            print(f"Cumulative Reward: {total_reward}, Steps: {steps}, Avg. Energy Used: {avg_energy_used}")
            print(f"Action counts: {action_counts}")

    print(f"Final Reward: {total_reward}, Steps: {steps}, Avg. Energy Used: {total_energy_used / steps}")
    env.close()