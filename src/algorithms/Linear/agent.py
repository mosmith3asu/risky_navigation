import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
import copy

class LinearModel(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, weight_decay=0.0):
        super().__init__()
        # For behavioral cloning: state_dim includes state+goal concatenated
        input_dim = state_dim
        self.linear = nn.Linear(input_dim, action_dim)
        self.weight_decay = weight_decay
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, state_goal):
        """For RL: input is state+goal, output is action."""
        action_pred = self.linear(state_goal)
        return action_pred

class LinearAgent:
    def __init__(self, state_dim, action_dim, goal_dim, lr=1e-3, weight_decay=0.0, 
                 optimizer_type='Adam', device=None):
        """
        Linear regression agent for next-action prediction.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            goal_dim (int): Dimension of the goal
            lr (float): Learning rate for optimizer
            weight_decay (float): L2 regularization strength
            optimizer_type (str): 'Adam', 'SGD', or 'RMSprop'
            device (str): 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        
        self.model = LinearModel(state_dim, action_dim, goal_dim, weight_decay).to(self.device)
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
    
    def update_hyperparameters(self, lr=None, weight_decay=None, optimizer_type=None):
        """Update hyperparameters and reinitialize optimizer."""
        if lr is not None:
            self.lr = lr
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
            self.update_hyperparameters(**param_dict)
            
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
            self.model.load_state_dict(self.best_state_dict)
            self.update_hyperparameters(**best_params)
        
        return {
            'best_params': best_params,
            'best_val_loss': best_val_loss,
            'all_results': results
        }
    
    def bayesian_optimization_tune(self, train_data, val_data, n_trials=20, num_epochs=10):
        """
        Use Bayesian optimization for hyperparameter tuning (requires optuna).
        
        Args:
            train_data: Training data
            val_data: Validation data  
            n_trials: Number of optimization trials
            num_epochs: Epochs per trial
            
        Returns:
            dict: Best hyperparameters found
        """
        try:
            import optuna
        except ImportError:
            print("Optuna not installed. Please install with: pip install optuna")
            return None
        
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            weight_decay = trial.suggest_float('weight_decay', 0, 1e-2)
            optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'SGD', 'RMSprop'])
            
            # Reset and configure model
            self._reset_model()
            self.update_hyperparameters(lr=lr, weight_decay=weight_decay, optimizer_type=optimizer_type)
            
            # Train and evaluate
            _, val_losses = self._train_epochs(train_data, val_data, num_epochs)
            return val_losses[-1]
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Apply best parameters
        best_params = study.best_params
        self.update_hyperparameters(**best_params)
        
        return {
            'best_params': best_params,
            'best_val_loss': study.best_value,
            'study': study
        }
    
    def learning_rate_schedule(self, scheduler_type='cosine', **kwargs):
        """
        Add learning rate scheduling.
        
        Args:
            scheduler_type: 'cosine', 'step', 'exponential', or 'plateau'
            **kwargs: Scheduler-specific parameters
        """
        if scheduler_type == 'cosine':
            T_max = kwargs.get('T_max', 50)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == 'step':
            step_size = kwargs.get('step_size', 10)
            gamma = kwargs.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'exponential':
            gamma = kwargs.get('gamma', 0.95)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        elif scheduler_type == 'plateau':
            patience = kwargs.get('patience', 5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _reset_model(self):
        """Reset model to initial state."""
        self.model = LinearModel(self.state_dim, self.action_dim, self.goal_dim).to(self.device)
        self.train_losses = []
        self.val_losses = []
    
    def _train_epochs(self, train_data, val_data, num_epochs):
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
            
            # Update learning rate if scheduler exists
            if hasattr(self, 'scheduler'):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
        
        return train_losses, val_losses
    
    def train_step(self, state, action, goal, expert_action):
        """
        Perform a single training step for behavioral cloning.
        
        Args:
            state: Current state (batch)
            action: Not used in RL mode (kept for compatibility)
            goal: Goal position (batch)
            expert_action: Expert action to imitate (batch)
            
        Returns:
            float: Loss value
        """
        self.model.train()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        expert_action = torch.tensor(expert_action, dtype=torch.float32, device=self.device)
        
        # Concatenate state and goal
        state_goal = torch.cat([state, goal], dim=-1)
        pred = self.model(state_goal)
        loss = self.loss_fn(pred, expert_action)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_next_action(self, state, action, goal):
        """Legacy method - redirects to predict_action."""
        return self.predict_action(state, goal)
    
    def predict_action(self, state, goal):
        """
        Predict action given current state and goal.
        
        Args:
            state: Current state
            goal: Goal position
            
        Returns:
            numpy.ndarray: Predicted action
        """
        self.model.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            goal = torch.tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            state_goal = torch.cat([state, goal], dim=-1)
            pred = self.model(state_goal)
            return pred.cpu().numpy().squeeze(0)
    
    def validate(self, states, actions, goals, expert_actions):
        """
        Validate the model on a validation set.
        
        Args:
            states: Batch of states
            actions: Not used in RL mode
            goals: Batch of goals
            expert_actions: Batch of expert actions to predict
            
        Returns:
            float: Validation loss
        """
        self.model.eval()
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            goals = torch.tensor(goals, dtype=torch.float32, device=self.device)
            expert_actions = torch.tensor(expert_actions, dtype=torch.float32, device=self.device)
            
            state_goal = torch.cat([states, goals], dim=-1)
            preds = self.model(state_goal)
            val_loss = self.loss_fn(preds, expert_actions).item()
            
            return val_loss
    
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
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
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
