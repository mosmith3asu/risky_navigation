import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TransformerModel(nn.Module):
    """Transformer for temporal sequence modeling in action prediction.
    
    Designed for time-series forecasting: [s_t-k, ..., s_t] → a_t
    Uses self-attention to capture temporal dependencies across state history.
    
    If sequence_len=1, falls back to deep MLP (more efficient for single states).
    """
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, sequence_len=1):
        super().__init__()
        self.sequence_len = sequence_len
        self.d_model = d_model
        
        if sequence_len > 1:
            # True transformer for sequences
            self.input_proj = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, 
                dim_feedforward=4*d_model, 
                dropout=dropout, 
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.output_proj = nn.Linear(d_model, output_dim)
        else:
            # Fall back to MLP for single states (more efficient)
            layers = []
            hidden_dim = d_model
            
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [batch, sequence_len, input_dim] for sequences
               [batch, input_dim] for single states
        Returns:
            action: [batch, output_dim]
        """
        if self.sequence_len > 1:
            # Ensure x is [batch, sequence_len, input_dim]
            if x.ndim == 2:
                x = x.unsqueeze(1)  # Single state → [batch, 1, input_dim]
            
            # Project to d_model dimension
            x = self.input_proj(x)  # [batch, seq_len, d_model]
            
            # Self-attention over temporal sequence
            x = self.transformer(x)  # [batch, seq_len, d_model]
            
            # Use only last timestep for action prediction
            x = x[:, -1, :]  # [batch, d_model]
            
            # Project to action space
            return self.output_proj(x)  # [batch, action_dim]
        else:
            # MLP forward pass
            return self.network(x)

class TransformerAgent:
    def __init__(self, state_dim, action_dim, goal_dim=None, d_model=64, nhead=4, num_layers=2, dropout=0.1, lr=1e-3, device=None,
                 action_low=None, action_high=None, sequence_len=1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_len = sequence_len
        self.model = TransformerModel(state_dim, action_dim, d_model, nhead, num_layers, dropout, sequence_len).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Action bounds for safety
        self.action_low = torch.tensor(action_low if action_low is not None else [0.0, -np.pi/6],
                                       dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(action_high if action_high is not None else [2.0, np.pi/6],
                                        dtype=torch.float32, device=self.device)
    
    def train_step(self, states, actions, goals, expert_actions):
        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = self.loss_fn(predictions, expert_actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_action(self, state, goal=None):
        self.model.eval()
        with torch.no_grad():
            if state.ndim == 1:
                state = state.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            action = self.model(state)
            # Enforce action bounds
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
