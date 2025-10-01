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
                 dropout=0.1, lr=1e-3, device=None):
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
            device: 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerModel(state_dim, action_dim, goal_dim, d_model, nhead, 
                                     num_layers, dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
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
