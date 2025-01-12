# src/dqn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import SEED

# Set seed for reproducibility
torch.manual_seed(SEED)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        hidden_units = 32
        hidden_layers = 2
        
        self.fc = nn.ModuleList(
            [nn.Linear(input_dim, hidden_units)] +
            [nn.Linear(hidden_units, hidden_units) for _ in range(hidden_layers - 1)] +
            [nn.Linear(hidden_units, output_dim)]
        )
    def forward(self, x):
        for layer in self.fc[:-1]:
            x = F.relu(layer(x))
        x = self.fc[-1](x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        # Common feature layer
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Combine value and advantage streams to compute Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))