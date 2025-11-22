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
                 lr=1e-3, beta=1.0, weight_decay=0.0, optimizer_type='Adam', device='cpu'):
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
            weight_decay: L2 regularization
            optimizer_type: Type of optimizer
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.device = device
        
        # For behavioral cloning: state_dim includes state+goal concatenated
        input_dim = state_dim
        output_dim = action_dim
        
        # Initialize encoder and decoder
        self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dim).to(device)
        self.decoder = VAEDecoder(latent_dim, output_dim, hidden_dim).to(device)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # For tracking training
        self.training_step = 0
        self.best_val_loss = float('inf')
        self.best_state_dict = None
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, states, goals):
        """Forward pass through VAE for RL: input is state+goal, output is action."""
        # Concatenate state and goal (no current action)
        inputs = torch.cat([states, goals], dim=-1)
        
        # Encode
        mu, logvar = self.encoder(inputs)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar
    
    def compute_loss(self, states, goals, expert_actions):
        """Compute VAE loss for behavioral cloning."""
        reconstruction, mu, logvar = self.forward(states, goals)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, expert_actions, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / states.size(0)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def train_step(self, states, actions, goals, expert_actions):
        """Single training step for behavioral cloning.
        Args:
            states: Current states
            actions: Not used in RL mode (kept for compatibility)
            goals: Goal positions
            expert_actions: Expert actions to imitate
        """
        self.encoder.train()
        self.decoder.train()
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        goals = torch.tensor(goals, dtype=torch.float32, device=self.device)
        expert_actions = torch.tensor(expert_actions, dtype=torch.float32, device=self.device)
        
        # Forward pass and compute loss
        loss_dict = self.compute_loss(states, goals, expert_actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict['loss'].backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss_dict['loss'].item()
    
    def predict_next_action(self, state, action, goal, n_samples=10):
        """Legacy method - redirects to predict_action."""
        mean_pred, std_pred = self.predict_action(state, goal, n_samples)
        return mean_pred, std_pred
    
    def predict_action(self, state, goal, n_samples=10):
        """
        Predict action with uncertainty estimation.
        
        Args:
            state: Current state
            goal: Goal
            n_samples: Number of samples for uncertainty estimation
            
        Returns:
            mean_prediction: Mean of predicted actions
            std_prediction: Standard deviation of predictions
        """
        self.encoder.eval()
        self.decoder.eval()
        
        # Prepare inputs
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        
        if state.ndim == 1:
            state = state.unsqueeze(0)
            goal = goal.unsqueeze(0)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                inputs = torch.cat([state, goal], dim=-1)
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
    
    def validate(self, val_states, val_actions, val_goals, val_expert_actions):
        """Validate the model."""
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            states = torch.tensor(val_states, dtype=torch.float32, device=self.device)
            goals = torch.tensor(val_goals, dtype=torch.float32, device=self.device)
            expert_actions = torch.tensor(val_expert_actions, dtype=torch.float32, device=self.device)
            
            loss_dict = self.compute_loss(states, goals, expert_actions)
            
        return loss_dict['loss'].item()
    
    def sample_from_prior(self, n_samples=1):
        """Sample from the prior distribution"""
        self.decoder.eval()
        
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.decoder(z)
            
        return samples.detach().cpu().numpy()
    
    def get_latent_representation(self, states, goals):
        """Get latent representation of inputs"""
        self.encoder.eval()
        
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            goals = torch.tensor(goals, dtype=torch.float32, device=self.device)
            
            inputs = torch.cat([states, goals], dim=-1)
            mu, logvar = self.encoder(inputs)
            
        return mu.detach().cpu().numpy(), logvar.detach().cpu().numpy()
    
    def _setup_optimizer(self):
        """Setup optimizer based on type."""
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        elif self.optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def update_hyperparameters(self, lr=None, beta=None, weight_decay=None, optimizer_type=None):
        """Update hyperparameters and reinitialize optimizer."""
        if lr is not None:
            self.lr = lr
        if beta is not None:
            self.beta = beta
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if optimizer_type is not None:
            self.optimizer_type = optimizer_type
        
        self._setup_optimizer()
    
    def grid_search_hyperparameters(self, train_data, val_data, param_grid, num_epochs=10):
        """
        Perform grid search over hyperparameters.
        
        Args:
            train_data: Training data (states, actions, goals, next_actions)
            val_data: Validation data
            param_grid: Dictionary of hyperparameters to search
            num_epochs: Number of epochs to train for each configuration
            
        Returns:
            dict: Best hyperparameters and corresponding validation loss
        """
        from itertools import product
        import copy
        
        best_params = None
        best_val_loss = float('inf')
        results = []
        
        # Generate all combinations of hyperparameters
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        for params in product(*param_values):
            param_dict = dict(zip(param_names, params))
            print(f"Testing parameters: {param_dict}")
            
            # Reset model with new parameters
            self._reset_model()
            
            # Update architecture if needed
            if 'latent_dim' in param_dict:
                self.latent_dim = param_dict['latent_dim']
            if 'hidden_dim' in param_dict:
                self.hidden_dim = param_dict['hidden_dim']
            
            # Recreate model with new architecture
            input_dim = self.state_dim  # state_dim includes state+goal
            output_dim = self.action_dim
            self.encoder = VAEEncoder(input_dim, self.latent_dim, self.hidden_dim).to(self.device)
            self.decoder = VAEDecoder(self.latent_dim, output_dim, self.hidden_dim).to(self.device)
            
            # Update other hyperparameters
            self.update_hyperparameters(**{k: v for k, v in param_dict.items() 
                                         if k in ['lr', 'beta', 'weight_decay', 'optimizer_type']})
            
            # Train with current parameters
            train_losses, val_losses = self._train_epochs(train_data, val_data, num_epochs)
            final_val_loss = val_losses[-1]
            
            results.append({
                'params': param_dict,
                'val_loss': final_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            })
            
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_params = param_dict
                self.best_state_dict = {
                    'encoder': copy.deepcopy(self.encoder.state_dict()),
                    'decoder': copy.deepcopy(self.decoder.state_dict())
                }
        
        # Load best model
        if self.best_state_dict is not None:
            # Reset with best architecture
            if 'latent_dim' in best_params:
                self.latent_dim = best_params['latent_dim']
            if 'hidden_dim' in best_params:
                self.hidden_dim = best_params['hidden_dim']
            
            input_dim = self.state_dim  # state_dim includes state+goal
            output_dim = self.action_dim
            self.encoder = VAEEncoder(input_dim, self.latent_dim, self.hidden_dim).to(self.device)
            self.decoder = VAEDecoder(self.latent_dim, output_dim, self.hidden_dim).to(self.device)
            
            self.encoder.load_state_dict(self.best_state_dict['encoder'])
            self.decoder.load_state_dict(self.best_state_dict['decoder'])
            self.update_hyperparameters(**{k: v for k, v in best_params.items() 
                                         if k in ['lr', 'beta', 'weight_decay', 'optimizer_type']})
        
        return {
            'best_params': best_params,
            'best_val_loss': best_val_loss,
            'all_results': results
        }
    
    def bayesian_optimization_tune(self, train_data, val_data, num_trials=50, num_epochs=10):
        """
        Perform Bayesian optimization for hyperparameter tuning using Optuna.
        
        Args:
            train_data: Training data
            val_data: Validation data
            num_trials: Number of optimization trials
            num_epochs: Epochs per trial
            
        Returns:
            dict: Best hyperparameters and optimization history
        """
        try:
            import optuna
        except ImportError:
            print("Optuna not installed. Using grid search instead.")
            # Fallback to grid search
            param_grid = {
                'lr': [1e-4, 1e-3, 1e-2],
                'latent_dim': [32, 64, 128],
                'beta': [0.1, 1.0, 4.0],
                'hidden_dim': [64, 128, 256]
            }
            return self.grid_search_hyperparameters(train_data, val_data, param_grid, num_epochs)
        
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            latent_dim = trial.suggest_categorical('latent_dim', [16, 32, 64, 128, 256])
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
            beta = trial.suggest_float('beta', 0.1, 10.0, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'SGD', 'RMSprop'])
            
            # Reset model with new parameters
            self.latent_dim = latent_dim
            self.hidden_dim = hidden_dim
            
            input_dim = self.state_dim  # state_dim includes state+goal
            output_dim = self.action_dim
            self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dim).to(self.device)
            self.decoder = VAEDecoder(latent_dim, output_dim, hidden_dim).to(self.device)
            
            self.update_hyperparameters(lr=lr, beta=beta, weight_decay=weight_decay, 
                                       optimizer_type=optimizer_type)
            
            # Train and get validation loss
            _, val_losses = self._train_epochs(train_data, val_data, num_epochs)
            
            return val_losses[-1]
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=num_trials)
        
        # Apply best parameters
        best_params = study.best_params
        self.latent_dim = best_params['latent_dim']
        self.hidden_dim = best_params['hidden_dim']
        
        input_dim = self.state_dim  # state_dim includes state+goal
        output_dim = self.action_dim
        self.encoder = VAEEncoder(input_dim, self.latent_dim, self.hidden_dim).to(self.device)
        self.decoder = VAEDecoder(self.latent_dim, output_dim, self.hidden_dim).to(self.device)
        
        self.update_hyperparameters(
            lr=best_params['lr'],
            beta=best_params['beta'],
            weight_decay=best_params['weight_decay'],
            optimizer_type=best_params['optimizer_type']
        )
        
        return {
            'best_params': best_params,
            'best_val_loss': study.best_value,
            'study': study
        }
    
    def learning_rate_schedule(self, train_data, val_data, scheduler_type='StepLR', 
                              initial_lr=1e-3, num_epochs=50, **scheduler_kwargs):
        """
        Train with learning rate scheduling.
        
        Args:
            train_data: Training data
            val_data: Validation data  
            scheduler_type: Type of scheduler ('StepLR', 'ExponentialLR', 'CosineAnnealingLR')
            initial_lr: Initial learning rate
            num_epochs: Number of epochs to train
            **scheduler_kwargs: Additional scheduler parameters
            
        Returns:
            dict: Training history and final metrics
        """
        self.lr = initial_lr
        self._setup_optimizer()
        
        # Setup scheduler
        if scheduler_type == 'StepLR':
            step_size = scheduler_kwargs.get('step_size', 10)
            gamma = scheduler_kwargs.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'ExponentialLR':
            gamma = scheduler_kwargs.get('gamma', 0.95)
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        elif scheduler_type == 'CosineAnnealingLR':
            T_max = scheduler_kwargs.get('T_max', num_epochs)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        train_losses, val_losses = self._train_epochs(train_data, val_data, num_epochs, scheduler)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses)
        }
    
    def beta_annealing_schedule(self, train_data, val_data, num_epochs=50, 
                               annealing_type='linear', max_beta=None):
        """
        Train with beta annealing schedule for VAE.
        
        Args:
            train_data: Training data
            val_data: Validation data
            num_epochs: Number of epochs to train
            annealing_type: Type of annealing ('linear', 'cosine', 'cyclic')
            max_beta: Maximum beta value (default: current beta)
            
        Returns:
            dict: Training history and final metrics
        """
        if max_beta is None:
            max_beta = self.beta
        
        train_states, train_actions, train_goals, train_next_actions = train_data
        val_states, val_actions, val_goals, val_next_actions = val_data
        
        train_losses = []
        val_losses = []
        beta_values = []
        
        for epoch in range(num_epochs):
            # Update beta based on annealing schedule
            if annealing_type == 'linear':
                self.beta = (epoch / num_epochs) * max_beta
            elif annealing_type == 'cosine':
                self.beta = max_beta * (1 - np.cos(np.pi * epoch / num_epochs)) / 2
            elif annealing_type == 'cyclic':
                cycle_length = num_epochs // 4
                cycle_pos = epoch % cycle_length
                self.beta = (cycle_pos / cycle_length) * max_beta
            
            beta_values.append(self.beta)
            
            # Training
            train_loss = self.train_step(train_states, train_actions, train_goals, train_next_actions)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_states, val_actions, val_goals, val_next_actions)
            val_losses.append(val_loss)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'beta_values': beta_values,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses)
        }
    
    def _reset_model(self):
        """Reset model to initial state."""
        input_dim = self.state_dim  # state_dim includes state+goal
        output_dim = self.action_dim
        self.encoder = VAEEncoder(input_dim, self.latent_dim, self.hidden_dim).to(self.device)
        self.decoder = VAEDecoder(self.latent_dim, output_dim, self.hidden_dim).to(self.device)
        self.training_step = 0
    
    def _train_epochs(self, train_data, val_data, num_epochs, scheduler=None):
        """Internal method to train for specified epochs."""
        train_states, train_actions, train_goals, train_next_actions = train_data
        val_states, val_actions, val_goals, val_next_actions = val_data
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_step(train_states, train_actions, train_goals, train_next_actions)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_states, val_actions, val_goals, val_next_actions)
            val_losses.append(val_loss)
            
            # Step scheduler if provided
            if scheduler is not None:
                scheduler.step()
        
        return train_losses, val_losses
    
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