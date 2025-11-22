import torch
import torch.nn as nn
import torch.optim as optim

class AutoEncoder(nn.Module):
    """AutoEncoder for behavioral cloning: (state+goal) -> action"""
    def __init__(self, input_dim, output_dim, latent_dim=64, hidden_dims=None, 
                 activation='ReLU', dropout=0.0, batch_norm=False):
        super().__init__()
        
        hidden_dims = hidden_dims or [128]
        
        # Activation function
        act_fn = {'ReLU': nn.ReLU(), 'ELU': nn.ELU(), 'GELU': nn.GELU()}.get(activation, nn.ReLU())
        
        # Encoder: input -> hidden_dims -> latent
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
        
        # Decoder: latent -> reversed(hidden_dims) -> output
        decoder_layers = []
        dims = [latent_dim] + list(reversed(hidden_dims)) + [output_dim]
        for i in range(len(dims) - 1):
            decoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on output
                if batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(dims[i+1]))
                decoder_layers.append(act_fn)
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass: (state+goal) -> action"""
        return self.decoder(self.encoder(x))

class AutoEncoderAgent:
    """AutoEncoder agent for behavioral cloning"""
    def __init__(self, state_dim, action_dim, goal_dim=None, latent_dim=64, hidden_dims=None,
                 activation='ReLU', dropout=0.0, batch_norm=False, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use state only (goal info is in state[4:6])
        input_dim = state_dim
        
        self.model = AutoEncoder(
            input_dim, action_dim, latent_dim, hidden_dims,
            activation, dropout, batch_norm
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def train_step(self, states, actions, goals, expert_actions):
        """Train step: predict expert actions from states only.
        Args:
            states: [batch, state_dim]
            actions: Ignored (for compatibility)
            goals: Ignored (goal info in state[4:6])
            expert_actions: [batch, action_dim] - target actions
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Use state only
        predictions = self.model(states)
        loss = self.loss_fn(predictions, expert_actions)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_action(self, state, goal=None):
        """Predict action from state only.
        Args:
            state: [state_dim] or [batch, state_dim]
            goal: Ignored (kept for API compatibility)
        Returns:
            action: [action_dim] or [batch, action_dim]
        """
        self.model.eval()
        with torch.no_grad():
            # Handle single sample
            if state.ndim == 1:
                state = state.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            
            action = self.model(state)
            
            return action.squeeze(0) if squeeze else action
