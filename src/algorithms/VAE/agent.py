import torch
import torch.nn as nn
import torch.optim as optim

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, z):
        h = self.activation(self.fc1(z))
        h = self.activation(self.fc2(h))
        return self.fc3(h)

class VAEAgent:
    def __init__(self, state_dim, action_dim, goal_dim, latent_dim=32, hidden_dim=128, beta=1.0, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = state_dim + goal_dim
        self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dim).to(self.device)
        self.decoder = VAEDecoder(latent_dim, action_dim, hidden_dim).to(self.device)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        self.beta = beta
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def train_step(self, states, actions, goals, expert_actions):
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        inputs = torch.cat([states, goals], dim=-1)
        mu, logvar = self.encoder(inputs)
        z = self.reparameterize(mu, logvar)
        predictions = self.decoder(z)
        recon_loss = nn.functional.mse_loss(predictions, expert_actions)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_action(self, state, goal):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            if state.ndim == 1:
                state = state.unsqueeze(0)
                goal = goal.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            inputs = torch.cat([state, goal], dim=-1)
            mu, _ = self.encoder(inputs)
            action = self.decoder(mu)
            return action.squeeze(0) if squeeze else action
