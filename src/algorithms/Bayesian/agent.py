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


class BayesianAgent:
    def __init__(self, state_dim, action_dim, goal_dim, latent_dim=64, lr=1e-3, kl_weight=0.01, 
                 prior_std=1.0, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store dimensions for model reconstruction
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.prior_std = prior_std
        
        input_dim = state_dim + action_dim + goal_dim
        
        # Initialize encoder and decoder
        self.encoder = BayesianEncoder(input_dim, latent_dim).to(self.device)
        self.decoder = BayesianDecoder(latent_dim, action_dim).to(self.device)
        
        # KL divergence weight (beta) for the ELBO loss
        self.kl_weight = kl_weight
        
        # Optimizer
        self._setup_optimizer()
        
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
    
    def update_hyperparameters(self, lr=None, kl_weight=None, prior_std=None):
        """Update hyperparameters and reinitialize optimizer."""
        if lr is not None:
            self.lr = lr
        if kl_weight is not None:
            self.kl_weight = kl_weight
        if prior_std is not None:
            self.prior_std = prior_std
            # Recreate layers with new prior
            self._reset_model()
        
        self._setup_optimizer()
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)
    
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
            
            # Update parameters
            if 'latent_dim' in param_dict:
                self.latent_dim = param_dict['latent_dim']
                input_dim = self.state_dim + self.action_dim + self.goal_dim
                self.encoder = BayesianEncoder(input_dim, self.latent_dim).to(self.device)
                self.decoder = BayesianDecoder(self.latent_dim, self.action_dim).to(self.device)
            
            # Update other hyperparameters
            self.update_hyperparameters(**{k: v for k, v in param_dict.items() 
                                         if k in ['lr', 'kl_weight', 'prior_std']})
            
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
        if hasattr(self, 'best_state_dict'):
            # Reset with best architecture
            if 'latent_dim' in best_params:
                self.latent_dim = best_params['latent_dim']
                input_dim = self.state_dim + self.action_dim + self.goal_dim
                self.encoder = BayesianEncoder(input_dim, self.latent_dim).to(self.device)
                self.decoder = BayesianDecoder(self.latent_dim, self.action_dim).to(self.device)
            
            self.encoder.load_state_dict(self.best_state_dict['encoder'])
            self.decoder.load_state_dict(self.best_state_dict['decoder'])
            self.update_hyperparameters(**{k: v for k, v in best_params.items() 
                                         if k in ['lr', 'kl_weight', 'prior_std']})
        
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
                'kl_weight': [0.001, 0.01, 0.1],
                'prior_std': [0.1, 1.0, 2.0]
            }
            return self.grid_search_hyperparameters(train_data, val_data, param_grid, num_epochs)
        
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128, 256])
            kl_weight = trial.suggest_float('kl_weight', 1e-4, 1e-1, log=True)
            prior_std = trial.suggest_float('prior_std', 0.1, 5.0)
            
            # Reset model with new parameters
            self.latent_dim = latent_dim
            input_dim = self.state_dim + self.action_dim + self.goal_dim
            self.encoder = BayesianEncoder(input_dim, latent_dim).to(self.device)
            self.decoder = BayesianDecoder(latent_dim, self.action_dim).to(self.device)
            
            self.update_hyperparameters(lr=lr, kl_weight=kl_weight, prior_std=prior_std)
            
            # Train and get validation loss
            _, val_losses = self._train_epochs(train_data, val_data, num_epochs)
            
            return val_losses[-1]
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=num_trials)
        
        # Apply best parameters
        best_params = study.best_params
        self.latent_dim = best_params['latent_dim']
        input_dim = self.state_dim + self.action_dim + self.goal_dim
        self.encoder = BayesianEncoder(input_dim, self.latent_dim).to(self.device)
        self.decoder = BayesianDecoder(self.latent_dim, self.action_dim).to(self.device)
        
        self.update_hyperparameters(
            lr=best_params['lr'],
            kl_weight=best_params['kl_weight'],
            prior_std=best_params['prior_std']
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
    
    def kl_annealing_schedule(self, train_data, val_data, num_epochs=50, 
                             annealing_type='linear', max_kl_weight=None):
        """
        Train with KL divergence annealing schedule.
        
        Args:
            train_data: Training data
            val_data: Validation data
            num_epochs: Number of epochs to train
            annealing_type: Type of annealing ('linear', 'cosine', 'cyclic')
            max_kl_weight: Maximum KL weight (default: current kl_weight)
            
        Returns:
            dict: Training history and final metrics
        """
        if max_kl_weight is None:
            max_kl_weight = self.kl_weight
        
        train_states, train_actions, train_goals, train_next_actions = train_data
        val_states, val_actions, val_goals, val_next_actions = val_data
        
        train_losses = []
        val_losses = []
        kl_weights = []
        
        for epoch in range(num_epochs):
            # Update KL weight based on annealing schedule
            if annealing_type == 'linear':
                self.kl_weight = (epoch / num_epochs) * max_kl_weight
            elif annealing_type == 'cosine':
                self.kl_weight = max_kl_weight * (1 - np.cos(np.pi * epoch / num_epochs)) / 2
            elif annealing_type == 'cyclic':
                cycle_length = num_epochs // 4
                cycle_pos = epoch % cycle_length
                self.kl_weight = (cycle_pos / cycle_length) * max_kl_weight
            
            kl_weights.append(self.kl_weight)
            
            # Training
            train_loss = self.train_step(train_states, train_actions, train_goals, train_next_actions)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_states, val_actions, val_goals, val_next_actions)
            val_losses.append(val_loss)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'kl_weights': kl_weights,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses)
        }
    
    def _reset_model(self):
        """Reset model to initial state."""
        input_dim = self.state_dim + self.action_dim + self.goal_dim
        self.encoder = BayesianEncoder(input_dim, self.latent_dim).to(self.device)
        self.decoder = BayesianDecoder(self.latent_dim, self.action_dim).to(self.device)
    
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
