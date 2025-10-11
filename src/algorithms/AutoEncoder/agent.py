import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
import copy

class AutoEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, latent_dim=64, hidden_dims=None, 
                 activation='ReLU', dropout=0.0, batch_norm=False):
        super().__init__()
        
        input_dim = state_dim + action_dim + goal_dim
        output_dim = action_dim
        
        if hidden_dims is None:
            hidden_dims = [128, latent_dim]
        
        # Select activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'ELU':
            self.activation = nn.ELU()
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(self.activation)
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims[:-1])) + [output_dim]
        prev_dim = latent_dim
        
        for i, hidden_dim in enumerate(hidden_dims_reversed):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims_reversed) - 1:  # Don't add activation/dropout to output layer
                if batch_norm and i < len(hidden_dims_reversed) - 2:
                    decoder_layers.append(nn.BatchNorm1d(hidden_dim))
                decoder_layers.append(self.activation)
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, state, action, goal):
        """Get latent representation."""
        x = torch.cat([state, action, goal], dim=-1)
        return self.encoder(x)

class AutoEncoderAgent:
    def __init__(self, state_dim, action_dim, goal_dim, latent_dim=64, hidden_dims=None,
                 activation='ReLU', dropout=0.0, batch_norm=False, lr=1e-3, weight_decay=0.0,
                 optimizer_type='Adam', device=None):
        """
        AutoEncoder agent for next-action prediction.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            goal_dim (int): Dimension of goal space
            latent_dim (int): Dimension of latent bottleneck
            hidden_dims (list): Hidden layer dimensions
            activation (str): Activation function type
            dropout (float): Dropout probability
            batch_norm (bool): Use batch normalization
            lr (float): Learning rate
            weight_decay (float): L2 regularization
            optimizer_type (str): Optimizer type
            device (str): Device to use
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [128, latent_dim]
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        
        self.model = AutoEncoder(
            state_dim, action_dim, goal_dim, latent_dim, hidden_dims,
            activation, dropout, batch_norm
        ).to(self.device)
        
        self._setup_optimizer()
        self.loss_fn = nn.MSELoss()
        
        # Track training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_state_dict = None
    
    def _setup_optimizer(self):
        """Setup optimizer based on type."""
        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        elif self.optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def update_hyperparameters(self, lr=None, weight_decay=None, optimizer_type=None, dropout=None):
        """Update hyperparameters and reinitialize optimizer."""
        if lr is not None:
            self.lr = lr
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if optimizer_type is not None:
            self.optimizer_type = optimizer_type
        if dropout is not None:
            self.dropout = dropout
            # Recreate model with new dropout
            self._reset_model()
        
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
                self.hidden_dims = [128, self.latent_dim]
            if 'activation' in param_dict:
                self.activation = param_dict['activation']
            if 'dropout' in param_dict:
                self.dropout = param_dict['dropout']
            if 'batch_norm' in param_dict:
                self.batch_norm = param_dict['batch_norm']
            
            # Recreate model with new architecture
            self.model = AutoEncoder(
                self.state_dim, self.action_dim, self.goal_dim, 
                self.latent_dim, self.hidden_dims, self.activation, 
                self.dropout, self.batch_norm
            ).to(self.device)
            
            # Update optimizer parameters
            self.update_hyperparameters(**{k: v for k, v in param_dict.items() 
                                         if k in ['lr', 'weight_decay', 'optimizer_type']})
            
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
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
        
        # Load best model
        if self.best_state_dict is not None:
            # Reset with best architecture
            if 'latent_dim' in best_params:
                self.latent_dim = best_params['latent_dim']
                self.hidden_dims = [128, self.latent_dim]
            if 'activation' in best_params:
                self.activation = best_params['activation']
            if 'dropout' in best_params:
                self.dropout = best_params['dropout']
            if 'batch_norm' in best_params:
                self.batch_norm = best_params['batch_norm']
            
            self.model = AutoEncoder(
                self.state_dim, self.action_dim, self.goal_dim,
                self.latent_dim, self.hidden_dims, self.activation,
                self.dropout, self.batch_norm
            ).to(self.device)
            
            self.model.load_state_dict(self.best_state_dict)
            self.update_hyperparameters(**{k: v for k, v in best_params.items() 
                                         if k in ['lr', 'weight_decay', 'optimizer_type']})
        
        return {
            'best_params': best_params,
            'best_val_loss': best_val_loss,
            'all_results': results
        }
    
    def architecture_search(self, train_data, val_data, latent_dims=[32, 64, 128], 
                           activations=['ReLU', 'ELU', 'GELU'], num_epochs=10):
        """
        Search for optimal architecture.
        
        Args:
            train_data: Training data
            val_data: Validation data
            latent_dims: List of latent dimensions to try
            activations: List of activation functions to try
            num_epochs: Epochs per configuration
            
        Returns:
            dict: Best architecture configuration
        """
        param_grid = {
            'latent_dim': latent_dims,
            'activation': activations,
            'dropout': [0.0, 0.1, 0.2],
            'batch_norm': [False, True]
        }
        
        return self.grid_search_hyperparameters(train_data, val_data, param_grid, num_epochs)
    
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
                'dropout': [0.0, 0.1, 0.2],
                'activation': ['ReLU', 'ELU']
            }
            return self.grid_search_hyperparameters(train_data, val_data, param_grid, num_epochs)
        
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128, 256])
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            activation = trial.suggest_categorical('activation', ['ReLU', 'ELU', 'GELU', 'LeakyReLU'])
            batch_norm = trial.suggest_categorical('batch_norm', [True, False])
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'SGD', 'RMSprop'])
            
            # Reset model with new parameters
            self.latent_dim = latent_dim
            self.hidden_dims = [128, latent_dim]
            self.activation = activation
            self.dropout = dropout
            self.batch_norm = batch_norm
            
            self.model = AutoEncoder(
                self.state_dim, self.action_dim, self.goal_dim,
                latent_dim, self.hidden_dims, activation, dropout, batch_norm
            ).to(self.device)
            
            self.update_hyperparameters(lr=lr, weight_decay=weight_decay, optimizer_type=optimizer_type)
            
            # Train and get validation loss
            _, val_losses = self._train_epochs(train_data, val_data, num_epochs)
            
            return val_losses[-1]
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=num_trials)
        
        # Apply best parameters
        best_params = study.best_params
        self.latent_dim = best_params['latent_dim']
        self.hidden_dims = [128, self.latent_dim]
        self.activation = best_params['activation']
        self.dropout = best_params['dropout']
        self.batch_norm = best_params['batch_norm']
        
        self.model = AutoEncoder(
            self.state_dim, self.action_dim, self.goal_dim,
            self.latent_dim, self.hidden_dims, self.activation,
            self.dropout, self.batch_norm
        ).to(self.device)
        
        self.update_hyperparameters(
            lr=best_params['lr'],
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
    
    def _reset_model(self):
        """Reset model to initial state."""
        self.model = AutoEncoder(
            self.state_dim, self.action_dim, self.goal_dim,
            self.latent_dim, self.hidden_dims, self.activation,
            self.dropout, self.batch_norm
        ).to(self.device)
        self.train_losses = []
        self.val_losses = []
    
    def _train_epochs(self, train_data, val_data, num_epochs, scheduler=None):
        """Internal method to train for specified epochs."""
        train_states, train_actions, train_goals, train_next_actions = train_data
        val_states, val_actions, val_goals, val_next_actions = val_data
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = self.train_step(train_states, train_actions, train_goals, train_next_actions)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = self.validate(val_states, val_actions, val_goals, val_next_actions)
                val_losses.append(val_loss)
            
            # Step scheduler if provided
            if scheduler is not None:
                scheduler.step()

        return train_losses, val_losses
    
    def train_step(self, states, actions, goals, next_actions):
        """Single training step."""
        self.optimizer.zero_grad()
        
        predicted_actions = self.model(states, actions, goals)
        loss = self.loss_fn(predicted_actions, next_actions)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, states, actions, goals, next_actions):
        """Validation step."""
        predicted_actions = self.model(states, actions, goals)
        loss = self.loss_fn(predicted_actions, next_actions)
        return loss.item()
    
    def predict(self, states, actions, goals):
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            return self.model(states, actions, goals)
    
    def get_latent_representation(self, states, actions, goals):
        """Get latent representation from encoder."""
        self.model.eval()
        with torch.no_grad():
            return self.model.encode(states, actions, goals)
