# Variational AutoEncoder for next-action prediction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

class VAEEncoder(nn.Module):
    """Encoder network for VAE that outputs mean and log variance for latent distribution"""
    
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads for mean and log variance
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

class VAEDecoder(nn.Module):
    """Decoder network for VAE that reconstructs the output from latent representation"""
    
    def __init__(self, latent_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        return self.decoder(z)

class VAEAgent:
    """
    Variational AutoEncoder agent for next-action prediction.
    
    The VAE learns a probabilistic latent representation of the state-action-goal space
    and can generate diverse next actions while providing uncertainty estimates.
    """
    
    def __init__(self, state_dim, action_dim, goal_dim, latent_dim=64, hidden_dim=128, 
                 lr=1e-3, beta=1.0, device='cpu'):
        """
        Initialize VAE agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space  
            goal_dim: Dimension of goal space
            latent_dim: Dimension of latent space
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            beta: KL divergence weight (beta-VAE parameter)
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.device = device
        
        # Input is state + current_action + goal, output is next_action
        input_dim = state_dim + action_dim + goal_dim
        output_dim = action_dim
        
        # Initialize encoder and decoder
        self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dim).to(device)
        self.decoder = VAEDecoder(latent_dim, output_dim, hidden_dim).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
        )
        
        # For tracking training
        self.training_step = 0
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, states, actions, goals):
        """Forward pass through VAE"""
        # Concatenate inputs
        inputs = torch.cat([states, actions, goals], dim=-1)
        
        # Encode
        mu, logvar = self.encoder(inputs)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar
    
    def compute_loss(self, states, actions, goals, next_actions):
        """Compute VAE loss (reconstruction + KL divergence)"""
        reconstruction, mu, logvar = self.forward(states, actions, goals)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, next_actions, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / states.size(0)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def train_step(self, states, actions, goals, next_actions):
        """Single training step"""
        self.encoder.train()
        self.decoder.train()
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        goals = torch.tensor(goals, dtype=torch.float32, device=self.device)
        next_actions = torch.tensor(next_actions, dtype=torch.float32, device=self.device)
        
        # Forward pass and compute loss
        loss_dict = self.compute_loss(states, actions, goals, next_actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict['loss'].backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss_dict['loss'].item()
    
    def predict_next_action(self, state, action, goal, n_samples=10):
        """
        Predict next action with uncertainty estimation.
        
        Args:
            state: Current state
            action: Current action
            goal: Goal
            n_samples: Number of samples for uncertainty estimation
            
        Returns:
            mean_prediction: Mean of predicted next actions
            std_prediction: Standard deviation of predictions
        """
        self.encoder.eval()
        self.decoder.eval()
        
        # Prepare inputs
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        
        if state.ndim == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            goal = goal.unsqueeze(0)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                inputs = torch.cat([state, action, goal], dim=-1)
                mu, logvar = self.encoder(inputs)
                z = self.reparameterize(mu, logvar)
                pred = self.decoder(z)
                predictions.append(pred)
        
        # Stack predictions and compute statistics
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Convert to numpy
        mean_pred = mean_pred.squeeze().detach().cpu().numpy()
        std_pred = std_pred.squeeze().detach().cpu().numpy()
        
        return mean_pred, std_pred
    
    def validate(self, val_states, val_actions, val_goals, val_next_actions):
        """Validate the model"""
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            states = torch.tensor(val_states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(val_actions, dtype=torch.float32, device=self.device)
            goals = torch.tensor(val_goals, dtype=torch.float32, device=self.device)
            next_actions = torch.tensor(val_next_actions, dtype=torch.float32, device=self.device)
            
            loss_dict = self.compute_loss(states, actions, goals, next_actions)
            
        return loss_dict['loss'].item()
    
    def sample_from_prior(self, n_samples=1):
        """Sample from the prior distribution"""
        self.decoder.eval()
        
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.decoder(z)
            
        return samples.detach().cpu().numpy()
    
    def get_latent_representation(self, states, actions, goals):
        """Get latent representation of inputs"""
        self.encoder.eval()
        
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            goals = torch.tensor(goals, dtype=torch.float32, device=self.device)
            
            inputs = torch.cat([states, actions, goals], dim=-1)
            mu, logvar = self.encoder(inputs)
            
        return mu.detach().cpu().numpy(), logvar.detach().cpu().numpy()
    
    def save(self, path):
        """Save the model"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'beta': self.beta
        }, path)
    
    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']
        self.beta = checkpoint['beta']