import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearModel(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):
        super().__init__()
        input_dim = state_dim + action_dim + goal_dim
        self.linear = nn.Linear(input_dim, action_dim)
    
    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        next_action_pred = self.linear(x)
        return next_action_pred

class LinearAgent:
    def __init__(self, state_dim, action_dim, goal_dim, lr=1e-3, device=None):
        """
        Linear regression agent for next-action prediction.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            goal_dim (int): Dimension of the goal
            lr (float): Learning rate for optimizer
            device (str): 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LinearModel(state_dim, action_dim, goal_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Track training metrics
        self.train_losses = []
        self.val_losses = []
    
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
