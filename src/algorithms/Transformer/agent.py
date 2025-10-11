import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    Adds information about the position of tokens in the sequence.
    """
    def __init__(self, d_model, max_len=10):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    """
    Transformer model for next-action prediction.
    Uses self-attention to capture complex relationships between state, action, and goal.
    """
    def __init__(self, state_dim, action_dim, goal_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.d_model = d_model
        
        # Input embeddings
        self.state_embedding = nn.Linear(state_dim, d_model)
        self.action_embedding = nn.Linear(action_dim, d_model)
        self.goal_embedding = nn.Linear(goal_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output head
        self.output_layer = nn.Linear(d_model, action_dim)
        
    def forward(self, state, action, goal):
        """
        Forward pass through the transformer model.
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            goal: [batch_size, goal_dim]
            
        Returns:
            next_action_pred: [batch_size, action_dim]
        """
        batch_size = state.shape[0]
        
        # Create embeddings for each input
        state_emb = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, d_model]
        action_emb = self.action_embedding(action).unsqueeze(1)  # [batch_size, 1, d_model]
        goal_emb = self.goal_embedding(goal).unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Concatenate embeddings to form a sequence
        sequence = torch.cat([state_emb, action_emb, goal_emb], dim=1)  # [batch_size, 3, d_model]
        
        # Add positional encoding
        sequence = self.pos_encoder(sequence)
        
        # Apply transformer encoder
        output = self.transformer_encoder(sequence)  # [batch_size, 3, d_model]
        
        # Use the full context output (all 3 tokens) for prediction
        # We could also just use the last token, but using the mean across tokens
        # captures all the information
        output = output.mean(dim=1)  # [batch_size, d_model]
        
        # Project to action space
        next_action_pred = self.output_layer(output)  # [batch_size, action_dim]
        
        return next_action_pred

class TransformerAgent:
    """
    Agent class that wraps the transformer model.
    Provides methods for training, prediction, and evaluation.
    """
    def __init__(self, state_dim, action_dim, goal_dim, d_model=64, nhead=4, num_layers=2, 
                 dropout=0.1, lr=1e-3, weight_decay=0.0, optimizer_type='Adam', device=None):
        """
        Initialize the transformer agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            goal_dim: Dimension of the goal
            d_model: Dimension of the model (embedding size)
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            lr: Learning rate
            weight_decay: L2 regularization
            optimizer_type: Type of optimizer
            device: 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store hyperparameters for model reconstruction
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        
        self.model = TransformerModel(state_dim, action_dim, goal_dim, d_model, nhead, 
                                     num_layers, dropout).to(self.device)
        self._setup_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.loss_fn = nn.MSELoss()
        
        # Track training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_state_dict = None
    
    def train_step(self, state, action, goal, next_action):
        """
        Perform a single training step.
        
        Args:
            state: Current state (batch)
            action: Current action (batch)
            goal: Goal position (batch)
            next_action: Next action to predict (batch)
            
        Returns:
            float: Loss value
        """
        self.model.train()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        next_action = torch.tensor(next_action, dtype=torch.float32, device=self.device)
        
        pred = self.model(state, action, goal)
        loss = self.loss_fn(pred, next_action)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients (common in transformers)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def predict_next_action(self, state, action, goal):
        """
        Predict the next action given current state, action and goal.
        
        Args:
            state: Current state
            action: Current action
            goal: Goal position
            
        Returns:
            numpy.ndarray: Predicted next action
        """
        self.model.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            goal = torch.tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            pred = self.model(state, action, goal)
            return pred.cpu().numpy().squeeze(0)
    
    def validate(self, states, actions, goals, next_actions):
        """
        Validate the model on a validation set.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            goals: Batch of goals
            next_actions: Batch of next actions to predict
            
        Returns:
            float: Validation loss
        """
        self.model.eval()
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            goals = torch.tensor(goals, dtype=torch.float32, device=self.device)
            next_actions = torch.tensor(next_actions, dtype=torch.float32, device=self.device)
            
            preds = self.model(states, actions, goals)
            val_loss = self.loss_fn(preds, next_actions).item()
            
            # Update scheduler based on validation loss
            self.scheduler.step(val_loss)
            
            return val_loss
    
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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
    
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
            if 'd_model' in param_dict:
                self.d_model = param_dict['d_model']
            if 'nhead' in param_dict:
                self.nhead = param_dict['nhead']
            if 'num_layers' in param_dict:
                self.num_layers = param_dict['num_layers']
            if 'dropout' in param_dict:
                self.dropout = param_dict['dropout']
            
            # Recreate model with new architecture
            self.model = TransformerModel(
                self.state_dim, self.action_dim, self.goal_dim, 
                self.d_model, self.nhead, self.num_layers, self.dropout
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
            if 'd_model' in best_params:
                self.d_model = best_params['d_model']
            if 'nhead' in best_params:
                self.nhead = best_params['nhead']
            if 'num_layers' in best_params:
                self.num_layers = best_params['num_layers']
            if 'dropout' in best_params:
                self.dropout = best_params['dropout']
            
            self.model = TransformerModel(
                self.state_dim, self.action_dim, self.goal_dim,
                self.d_model, self.nhead, self.num_layers, self.dropout
            ).to(self.device)
            
            self.model.load_state_dict(self.best_state_dict)
            self.update_hyperparameters(**{k: v for k, v in best_params.items() 
                                         if k in ['lr', 'weight_decay', 'optimizer_type']})
        
        return {
            'best_params': best_params,
            'best_val_loss': best_val_loss,
            'all_results': results
        }
    
    def architecture_search(self, train_data, val_data, d_models=[32, 64, 128], 
                           nheads=[2, 4, 8], num_layers_list=[1, 2, 3], num_epochs=10):
        """
        Search for optimal transformer architecture.
        
        Args:
            train_data: Training data
            val_data: Validation data
            d_models: List of model dimensions to try
            nheads: List of attention head counts to try
            num_layers_list: List of layer counts to try
            num_epochs: Epochs per configuration
            
        Returns:
            dict: Best architecture configuration
        """
        param_grid = {
            'd_model': d_models,
            'nhead': nheads,
            'num_layers': num_layers_list,
            'dropout': [0.0, 0.1, 0.2]
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
                'd_model': [32, 64, 128],
                'nhead': [2, 4, 8],
                'dropout': [0.0, 0.1, 0.2]
            }
            return self.grid_search_hyperparameters(train_data, val_data, param_grid, num_epochs)
        
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            d_model = trial.suggest_categorical('d_model', [32, 64, 128, 256])
            nhead = trial.suggest_categorical('nhead', [2, 4, 8])
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'SGD', 'RMSprop'])
            
            # Reset model with new parameters
            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            self.dropout = dropout
            
            self.model = TransformerModel(
                self.state_dim, self.action_dim, self.goal_dim,
                d_model, nhead, num_layers, dropout
            ).to(self.device)
            
            self.update_hyperparameters(lr=lr, weight_decay=weight_decay, optimizer_type=optimizer_type)
            
            # Train and get validation loss
            _, val_losses = self._train_epochs(train_data, val_data, num_epochs)
            
            return val_losses[-1]
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=num_trials)
        
        # Apply best parameters
        best_params = study.best_params
        self.d_model = best_params['d_model']
        self.nhead = best_params['nhead']
        self.num_layers = best_params['num_layers']
        self.dropout = best_params['dropout']
        
        self.model = TransformerModel(
            self.state_dim, self.action_dim, self.goal_dim,
            self.d_model, self.nhead, self.num_layers, self.dropout
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
        elif scheduler_type == 'ReduceLROnPlateau':
            patience = scheduler_kwargs.get('patience', 5)
            factor = scheduler_kwargs.get('factor', 0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                           patience=patience, factor=factor)
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
        self.model = TransformerModel(
            self.state_dim, self.action_dim, self.goal_dim,
            self.d_model, self.nhead, self.num_layers, self.dropout
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
            train_loss = self.train_step(train_states, train_actions, train_goals, train_next_actions)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_states, val_actions, val_goals, val_next_actions)
            val_losses.append(val_loss)
            
            # Step scheduler if provided
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        
        return train_losses, val_losses
    
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'hyperparams': {
                'd_model': self.model.d_model,
                'nhead': self.model.transformer_encoder.layers[0].self_attn.num_heads,
                'num_layers': len(self.model.transformer_encoder.layers),
            }
        }, path)
    
    def load(self, path):
        """
        Load the model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Handle loading scheduler if it exists in the checkpoint
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
