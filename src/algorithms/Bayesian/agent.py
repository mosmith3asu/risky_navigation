# Bayesian neural network for next-action prediction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        # Weight mean and log variance parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-9, 0.1))
        
        # Bias mean and log variance parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).normal_(-9, 0.1))
        
        # Prior distribution parameters
        self.prior_std = prior_std
        self.weight_prior = Normal(0, prior_std)
        self.bias_prior = Normal(0, prior_std)
        
        # Initialize kl divergence
        self.kl = 0
    
    def forward(self, x):
        # Sample weights and biases from posterior during training
        weight_var = torch.exp(self.weight_logvar)
        weight = Normal(self.weight_mu, weight_var.sqrt())
        
        bias_var = torch.exp(self.bias_logvar)
        bias = Normal(self.bias_mu, bias_var.sqrt())
        
        # Sample weights for forward pass
        weight_sample = weight.rsample()
        bias_sample = bias.rsample()
        
        # Calculate KL divergence between posterior and prior
        weight_kl = kl_divergence(weight, self.weight_prior).sum()
        bias_kl = kl_divergence(bias, self.bias_prior).sum()
        self.kl = weight_kl + bias_kl
        
        # Perform linear transformation
        return F.linear(x, weight_sample, bias_sample)
    
    def reset_kl(self):
        self.kl = 0


class BayesianEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, latent_dim * 2)
        self.fc2 = BayesianLinear(latent_dim * 2, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    @property
    def kl(self):
        return self.fc1.kl + self.fc2.kl
    
    def reset_kl(self):
        self.fc1.reset_kl()
        self.fc2.reset_kl()


class BayesianDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = BayesianLinear(latent_dim, latent_dim * 2)
        self.fc2 = BayesianLinear(latent_dim * 2, output_dim)
        
        # Learn the log variance of the output distribution
        self.log_var = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mean = self.fc2(x)
        return mean, self.log_var.exp()
    
    @property
    def kl(self):
        return self.fc1.kl + self.fc2.kl
    
    def reset_kl(self):
        self.fc1.reset_kl()
        self.fc2.reset_kl()


class BayesianActionPredictor:
    def __init__(self, state_dim, action_dim, goal_dim, latent_dim=64, lr=1e-3, kl_weight=0.01, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = state_dim + action_dim + goal_dim
        
        # Initialize encoder and decoder
        self.encoder = BayesianEncoder(input_dim, latent_dim).to(self.device)
        self.decoder = BayesianDecoder(latent_dim, action_dim).to(self.device)
        
        # KL divergence weight (beta) for the ELBO loss
        self.kl_weight = kl_weight
        
        # Optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        
        # Number of prediction samples for uncertainty estimation
        self.n_samples = 10
    
    def train_step(self, state, action, goal, next_action):
        """Train the model on a batch of data"""
        self.encoder.train()
        self.decoder.train()
        
        # Reset KL for this batch
        self.encoder.reset_kl()
        self.decoder.reset_kl()
        
        # Move data to device
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        next_action = torch.tensor(next_action, dtype=torch.float32, device=self.device)
        
        # Concatenate inputs
        inputs = torch.cat([state, action, goal], dim=-1)
        
        # Forward pass
        z = self.encoder(inputs)
        mean, var = self.decoder(z)
        
        # Calculate negative log likelihood loss (Gaussian likelihood)
        nll = 0.5 * (((next_action - mean)**2 / var) + torch.log(var)).sum(dim=1).mean()
        
        # Calculate KL divergence
        kl = self.encoder.kl + self.decoder.kl
        
        # ELBO loss = NLL + KL
        loss = nll + self.kl_weight * kl
        
        # Backward pass and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'nll': nll.item(),
            'kl': kl.item()
        }
    
    def predict_next_action(self, state, action, goal):
        """Predict the next action with uncertainty"""
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
        
        inputs = torch.cat([state, action, goal], dim=-1)
        
        # Get samples for uncertainty estimation
        predictions = []
        variances = []
        
        for _ in range(self.n_samples):
            z = self.encoder(inputs)
            mean, var = self.decoder(z)
            predictions.append(mean)
            variances.append(var)
        
        # Stack samples
        predictions = torch.stack(predictions)
        variances = torch.stack(variances)
        
        # Calculate predictive mean and variance
        pred_mean = predictions.mean(dim=0)
        pred_var = (predictions.var(dim=0) + variances.mean(dim=0))
        
        # Convert to numpy
        pred_mean = pred_mean.detach().cpu().numpy()
        pred_std = torch.sqrt(pred_var).detach().cpu().numpy()
        
        return pred_mean, pred_std
    
    def save(self, path):
        """Save the model"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'kl_weight': self.kl_weight
        }, path)
    
    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.kl_weight = checkpoint['kl_weight']
