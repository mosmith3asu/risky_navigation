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
    def __init__(self, state_dim, action_dim, goal_dim, latent_dim=64, hidden_dims=None,
                 activation='ReLU', dropout=0.0, batch_norm=False, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Input is state+goal concatenated
        input_dim = state_dim + goal_dim
        
        self.model = AutoEncoder(
            input_dim, action_dim, latent_dim, hidden_dims,
            activation, dropout, batch_norm
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def train_step(self, states, actions, goals, expert_actions):
        """Train step: predict expert actions from states+goals.
        Args:
            states: [batch, state_dim]
            actions: Ignored (for compatibility)
            goals: [batch, goal_dim]
            expert_actions: [batch, action_dim] - target actions
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Concatenate state + goal as input
        inputs = torch.cat([states, goals], dim=-1)
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, expert_actions)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_action(self, state, goal):
        """Predict action from state and goal.
        Args:
            state: [state_dim] or [batch, state_dim]
            goal: [goal_dim] or [batch, goal_dim]
        Returns:
            action: [action_dim] or [batch, action_dim]
        """
        self.model.eval()
        with torch.no_grad():
            # Handle single sample
            if state.ndim == 1:
                state = state.unsqueeze(0)
                goal = goal.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            
            inputs = torch.cat([state, goal], dim=-1)
            action = self.model(inputs)
            
            return action.squeeze(0) if squeeze else action
