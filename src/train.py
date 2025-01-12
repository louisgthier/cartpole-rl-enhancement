# src/train.py
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torchrl.data import PrioritizedReplayBuffer, ListStorage

from src.modified_cartpole import ModifiedCartPoleEnv
from src.dqn_model import DQN, DuelingDQN
from src.replay_buffer import ReplayBuffer
from src.config import SEED

# Hyperparameters
MAX_STEPS = 100000  # For instance, one million steps as a target
BATCH_SIZE = 512
GAMMA = 0.95
EPS_START = 0.2
EPS_END = 0.2
EPS_DECAY = 10000
TARGET_UPDATE = 1000
LEARNING_RATE = 2.5e-4
MEMORY_CAPACITY = BATCH_SIZE * 200
MAX_EPISODE_LENGTH = 1000
BACKUP_INTERVAL = 5000
DETERMINISTIC_ACTIONS = True

# Define checkpoint path
CHECKPOINT_PATH = "experiments/saved_models/checkpoint.pth"

device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
def train_model(run_id: int = None, resume: bool = False):
    env = ModifiedCartPoleEnv()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    policy_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1 - min(step, MAX_STEPS) / MAX_STEPS)

    # memory = ReplayBuffer(MEMORY_CAPACITY)
    memory = PrioritizedReplayBuffer(alpha=0.6, beta=0.4, storage=ListStorage(MEMORY_CAPACITY))

    steps_done = 0
    start_episode = 0

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize TensorBoard writer with run ID if available
    tb_comment = f"Cartpole_DQN_Run_{run_id}" if run_id is not None else "Cartpole_DQN_Resumed"
    tb_writer = SummaryWriter(log_dir=f"runs/cartpole_experiment_{run_id}", comment=tb_comment, flush_secs=30)

    # Resume logic: load checkpoint if resuming
    if resume and os.path.exists(CHECKPOINT_PATH):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        policy_net.load_state_dict(checkpoint['policy_state'])
        target_net.load_state_dict(checkpoint['target_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        steps_done = checkpoint['steps_done']
        start_episode = checkpoint['episode'] + 1
    else:
        print("Starting fresh training.")

    def current_epsilon_threshold(steps_done):
        return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)

    def select_action(state):
        nonlocal steps_done
        eps_threshold = current_epsilon_threshold(steps_done)
        steps_done += 1
        if np.random.rand() < eps_threshold:
            return np.random.randint(n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = policy_net(state_tensor).squeeze(0)
                
                if DETERMINISTIC_ACTIONS:
                    action = q_values.max(0)[1].item()
                else:
                    probabilities = torch.softmax(q_values, dim=0)
                    action = torch.multinomial(probabilities, 1).item()
                
                return action

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return None
        # transitions = memory.sample(BATCH_SIZE)
        # batch = list(zip(*transitions))
        
        # transitions, batch_weights, batch_indices = memory.sample(BATCH_SIZE)
        # batch = list(zip(*transitions))
        
        batch, info = memory.sample(BATCH_SIZE, return_info=True)
        

        state_batch = torch.tensor(np.vstack(batch[0]), dtype=torch.float32, device=device)
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=device).unsqueeze(1)
        next_state_batch = torch.tensor(np.vstack(batch[3]), dtype=torch.float32, device=device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=device).unsqueeze(1)
        
        # Extract sampling weights and indices from the info
        weights = info['_weight'].to(device).unsqueeze(1)
        batch_indices = info['index']

        state_action_values = policy_net(state_batch).gather(1, action_batch)
        next_state_values = target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
        
        # Compute TD errors
        td_errors = (state_action_values - expected_state_action_values).pow(2).detach().cpu().numpy().flatten()
            # Update priorities in the replay buffer
        new_priorities = td_errors + 1e-6  # Add a small constant to avoid zero priority
        memory.update_priority(batch_indices, new_priorities)

        # loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values)
        
        # Compute loss with importance sampling weights for Prioritized Replay Buffer
        loss = (torch.nn.functional.mse_loss(state_action_values, expected_state_action_values, reduction='none') * weights).mean()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        tb_writer.add_scalar("Info/LearningRate", scheduler.get_last_lr()[0], steps_done)

        return loss.item()

    try:
        episode = start_episode
        while steps_done < MAX_STEPS:
            state, _ = env.reset(seed=random.randint(0, 100000))
            total_reward = 0
            for t in range(1, MAX_EPISODE_LENGTH):
                action = select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                # memory.push(state, action, reward, next_state, done)
                memory.add((
                    torch.tensor(state, dtype=torch.float32),
                    torch.tensor(action, dtype=torch.int64),
                    torch.tensor(reward, dtype=torch.float32),
                    torch.tensor(next_state, dtype=torch.float32),
                    torch.tensor(done, dtype=torch.float32)
                )) # Prioritized Replay Buffer
                
                state = next_state

                loss_value = optimize_model()
                if loss_value is not None:
                    tb_writer.add_scalar("Info/Loss", loss_value, steps_done)

                # Backup every BACKUP_INTERVAL steps
                if steps_done % BACKUP_INTERVAL == 0:
                    checkpoint = {
                        'policy_state': policy_net.state_dict(),
                        'target_state': target_net.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'steps_done': steps_done,
                        'episode': episode,
                    }
                    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
                    torch.save(checkpoint, CHECKPOINT_PATH)
                    print(f"Checkpoint saved at step {steps_done}")
                    
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if done:
                    break

            print(f"Episode {episode}, Total Reward: {total_reward}, Steps: {t}, Total steps: {steps_done}, Epsilon threshold: {current_epsilon_threshold(steps_done):.4f}")
            tb_writer.add_scalar("Info/Reward", total_reward, steps_done)
            tb_writer.add_scalar("Info/Epsilon Threshold", current_epsilon_threshold(steps_done), steps_done)
            
            episode += 1
    except KeyboardInterrupt:
        print("Training interrupted! Saving checkpoint...")
        checkpoint = {
            'policy_state': policy_net.state_dict(),
            'target_state': target_net.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'steps_done': steps_done,
            'episode': episode,  # The last completed episode before interruption
        }
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"Checkpoint saved at step {steps_done}, episode {episode}")
    finally:
        tb_writer.close()
        env.close()