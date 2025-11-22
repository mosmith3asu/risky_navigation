import torch
import torch.nn as nn
import torch.optim as optim

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
    
    def forward(self, x):
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
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

class BayesianAgent:
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=128, prior_std=1.0, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = state_dim + goal_dim
        self.model = BayesianNetwork(input_dim, action_dim, hidden_dim, prior_std).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def train_step(self, states, actions, goals, expert_actions):
        self.model.train()
        self.optimizer.zero_grad()
        inputs = torch.cat([states, goals], dim=-1)
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, expert_actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_action(self, state, goal):
        self.model.eval()
        with torch.no_grad():
            if state.ndim == 1:
                state = state.unsqueeze(0)
                goal = goal.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            inputs = torch.cat([state, goal], dim=-1)
            action = self.model(inputs)
            return action.squeeze(0) if squeeze else action
