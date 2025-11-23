import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        self.prior_std = prior_std
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3)
    
    def forward(self, x, deterministic=False):
        if deterministic:
            # Use only mean weights during inference (no sampling)
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)
        else:
            # Sample weights during training
            weight_std = torch.log1p(torch.exp(self.weight_rho))
            bias_std = torch.log1p(torch.exp(self.bias_rho))
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
            return nn.functional.linear(x, weight, bias)

class BayesianNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, prior_std=1.0):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim, prior_std)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim, prior_std)
        self.fc3 = BayesianLinear(hidden_dim, output_dim, prior_std)
        self.activation = nn.ReLU()
    
    def forward(self, x, deterministic=False):
        x = self.activation(self.fc1(x, deterministic))
        x = self.activation(self.fc2(x, deterministic))
        return self.fc3(x, deterministic)

class BayesianAgent:
    def __init__(self, state_dim, action_dim, goal_dim, sequence_len=1, hidden_dim=128, prior_std=1.0, lr=1e-3, device=None,
                 action_low=None, action_high=None, kl_weight=1e-5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_len = sequence_len
        input_dim = (state_dim + action_dim) * sequence_len + goal_dim
        self.model = BayesianNetwork(input_dim, action_dim, hidden_dim, prior_std).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.prior_std = prior_std
        self.kl_weight = kl_weight
        
        self.action_low = torch.tensor(action_low if action_low is not None else [0.0, -np.pi/6], 
                                       dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(action_high if action_high is not None else [2.0, np.pi/6],
                                        dtype=torch.float32, device=self.device)
    
    def train_step(self, state_sequences, action_sequences, goals, target_actions):
        self.model.train()
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
        
        predictions = self.model(inputs, deterministic=False)
        
        likelihood_loss = self.loss_fn(predictions, target_actions)
        
        kl_loss = 0.0
        for name, module in self.model.named_modules():
            if isinstance(module, BayesianLinear):
                weight_std = torch.log1p(torch.exp(module.weight_rho))
                weight_kl = 0.5 * torch.sum(
                    (weight_std**2 + module.weight_mu**2) / (self.prior_std**2) 
                    - 1 + 2 * np.log(self.prior_std) - 2 * torch.log(weight_std)
                )
                
                bias_std = torch.log1p(torch.exp(module.bias_rho))
                bias_kl = 0.5 * torch.sum(
                    (bias_std**2 + module.bias_mu**2) / (self.prior_std**2)
                    - 1 + 2 * np.log(self.prior_std) - 2 * torch.log(bias_std)
                )
                
                kl_loss += weight_kl + bias_kl
        
        kl_loss = kl_loss / states.size(0)
        loss = likelihood_loss + self.kl_weight * kl_loss
        
        loss.backward()
        self.optimizer.step()
        
        self.last_likelihood_loss = likelihood_loss.item()
        self.last_kl_loss = kl_loss.item()
        
        return loss.item()
    
    def predict_action(self, state_sequence, action_sequence, goal):
        self.model.eval()
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
            
            action = self.model(inputs, deterministic=True)
            action = torch.clamp(action, self.action_low, self.action_high)
            return action.squeeze(0) if squeeze else action
    
    def save(self, filepath):
        """Save model weights and action bounds."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'action_low': self.action_low.cpu().numpy(),
            'action_high': self.action_high.cpu().numpy()
        }, filepath)
    
    def load(self, filepath):
        """Load model weights and action bounds."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.action_low = torch.tensor(checkpoint['action_low'], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(checkpoint['action_high'], dtype=torch.float32, device=self.device)
