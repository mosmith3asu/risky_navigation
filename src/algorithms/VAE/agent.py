import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
    def __init__(self, state_dim, action_dim, goal_dim, sequence_len=1, latent_dim=32, hidden_dim=128, beta=1.0, lr=1e-3, device=None,
                 action_low=None, action_high=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_len = sequence_len
        input_dim = (state_dim + action_dim) * sequence_len + goal_dim
        self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dim).to(self.device)
        self.decoder = VAEDecoder(latent_dim, action_dim, hidden_dim).to(self.device)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        self.beta = beta
        
        self.action_low = torch.tensor(action_low if action_low is not None else [0.0, -np.pi/6],
                                       dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(action_high if action_high is not None else [2.0, np.pi/6],
                                        dtype=torch.float32, device=self.device)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def train_step(self, state_sequences, action_sequences, goals, target_actions):
        self.encoder.train()
        self.decoder.train()
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
        
        mu, logvar = self.encoder(inputs)
        z = self.reparameterize(mu, logvar)
        predictions = self.decoder(z)
        
        recon_loss = nn.functional.mse_loss(predictions, target_actions, reduction='sum') / state_sequences.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / state_sequences.size(0)
        loss = recon_loss + self.beta * kl_loss
        
        loss.backward()
        self.optimizer.step()
        
        self.last_recon_loss = recon_loss.item()
        self.last_kl_loss = kl_loss.item()
        
        return loss.item()
    
    def predict_action(self, state_sequence, action_sequence, goal):
        self.encoder.eval()
        self.decoder.eval()
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
            
            mu, _ = self.encoder(inputs)
            action = self.decoder(mu)
            action = torch.clamp(action, self.action_low, self.action_high)
            return action.squeeze(0) if squeeze else action
                prev_action = prev_action.unsqueeze(0) if prev_action.ndim == 1 else prev_action
                goal = goal.unsqueeze(0) if goal.ndim == 1 else goal
                squeeze = True
            else:
                squeeze = False
            inputs = torch.cat([state, prev_action, goal], dim=1)
            mu, _ = self.encoder(inputs)
            action = self.decoder(mu)
            action = torch.clamp(action, self.action_low, self.action_high)
            return action.squeeze(0) if squeeze else action
    
    def save(self, filepath):
        """Save encoder, decoder weights and action bounds."""
        torch.save({
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'action_low': self.action_low.cpu().numpy(),
            'action_high': self.action_high.cpu().numpy()
        }, filepath)
    
    def load(self, filepath):
        """Load encoder, decoder weights and action bounds."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.decoder.load_state_dict(checkpoint['decoder_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.action_low = torch.tensor(checkpoint['action_low'], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(checkpoint['action_high'], dtype=torch.float32, device=self.device)
