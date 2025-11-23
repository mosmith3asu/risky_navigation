import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class LinearAgent:
    def __init__(self, state_dim, action_dim, goal_dim, sequence_len=1, lr=1e-3, device=None,
                 action_low=None, action_high=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_len = sequence_len
        input_dim = (state_dim + action_dim) * sequence_len + goal_dim
        self.model = LinearModel(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.action_low = torch.tensor(action_low if action_low is not None else [0.0, -np.pi/6],
                                       dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(action_high if action_high is not None else [2.0, np.pi/6],
                                        dtype=torch.float32, device=self.device)
    
    def train_step(self, state_sequences, action_sequences, goals, target_actions):
        self.model.train()
        self.optimizer.zero_grad()
        
        if isinstance(state_sequences, np.ndarray):
            state_sequences = torch.tensor(state_sequences, dtype=torch.float32, device=self.device)
            action_sequences = torch.tensor(action_sequences, dtype=torch.float32, device=self.device)
            goals = torch.tensor(goals, dtype=torch.float32, device=self.device)
            target_actions = torch.tensor(target_actions, dtype=torch.float32, device=self.device)
        
        batch_size = state_sequences.shape[0]
        state_seq_flat = state_sequences.reshape(batch_size, -1)
        action_seq_flat = action_sequences.reshape(batch_size, -1)
        inputs = torch.cat([state_seq_flat, action_seq_flat, goals], dim=1)
        
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, target_actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_action(self, state_sequence, action_sequence, goal):
        self.model.eval()
        with torch.no_grad():
            if state_sequence.ndim == 2:
                state_sequence = state_sequence.unsqueeze(0)
                action_sequence = action_sequence.unsqueeze(0)
                goal = goal.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            
            batch_size = state_sequence.shape[0]
            state_seq_flat = state_sequence.reshape(batch_size, -1)
            action_seq_flat = action_sequence.reshape(batch_size, -1)
            inputs = torch.cat([state_seq_flat, action_seq_flat, goal], dim=1)
            
            action = self.model(inputs)
            action = torch.clamp(action, self.action_low, self.action_high)
            return action.squeeze(0) if squeeze else action
    
    def save(self, filepath):
        """Save model weights and action bounds."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'action_low': self.action_low.cpu().numpy(),
            'action_high': self.action_high.cpu().numpy()
        }, filepath)
    
    def load(self, filepath):
        """Load model weights and action bounds."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.action_low = torch.tensor(checkpoint['action_low'], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(checkpoint['action_high'], dtype=torch.float32, device=self.device)
