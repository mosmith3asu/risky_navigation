import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class AutoEncoder(nn.Module):
    """Deep MLP with bottleneck for behavioral cloning.
    
    Note: This is technically an MLP with a bottleneck layer, not a true autoencoder.
    True autoencoders reconstruct their input. This maps state -> action via a
    compressed representation (latent_dim).
    """
    def __init__(self, input_dim, output_dim, latent_dim=64, hidden_dims=None, 
                 activation='ReLU', dropout=0.0, batch_norm=False):
        super().__init__()
        hidden_dims = hidden_dims or [128]
        act_fn = {'ReLU': nn.ReLU(), 'ELU': nn.ELU(), 'GELU': nn.GELU()}.get(activation, nn.ReLU())
        
        # Encoder: state -> compressed representation
        encoder_layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(dims[i+1]))
            encoder_layers.append(act_fn)
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: compressed representation -> action
        decoder_layers = []
        dims = [latent_dim] + list(reversed(hidden_dims)) + [output_dim]
        for i in range(len(dims) - 1):
            decoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                if batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(dims[i+1]))
                decoder_layers.append(act_fn)
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

class AutoEncoderAgent:
    def __init__(self, state_dim, action_dim, goal_dim=None, latent_dim=64, hidden_dims=None,
                 activation='ReLU', dropout=0.0, batch_norm=False, lr=1e-3, device=None,
                 action_low=None, action_high=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncoder(
            state_dim, action_dim, latent_dim, hidden_dims,
            activation, dropout, batch_norm
        ).to(self.device)
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
