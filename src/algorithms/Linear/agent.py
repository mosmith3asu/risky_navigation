import torch
import torch.nn as nn
import torch.optim as optim

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class LinearAgent:
    def __init__(self, state_dim, action_dim, goal_dim, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = state_dim + goal_dim
        self.model = LinearModel(input_dim, action_dim).to(self.device)
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
