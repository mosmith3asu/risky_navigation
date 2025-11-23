import torch
import torch.nn as nn
import torch.optim as optim

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=64, hidden_dims=None, 
                 activation='ReLU', dropout=0.0, batch_norm=False):
        super().__init__()
        hidden_dims = hidden_dims or [128]
        act_fn = {'ReLU': nn.ReLU(), 'ELU': nn.ELU(), 'GELU': nn.GELU()}.get(activation, nn.ReLU())
        
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
                 activation='ReLU', dropout=0.0, batch_norm=False, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncoder(
            state_dim, action_dim, latent_dim, hidden_dims,
            activation, dropout, batch_norm
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
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
            return action.squeeze(0) if squeeze else action
