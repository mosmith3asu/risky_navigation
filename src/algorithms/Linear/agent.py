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
    def __init__(self, state_dim, action_dim, goal_dim=None, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Use only state_dim (goal info is in state[4:6])
        input_dim = state_dim
        self.model = LinearModel(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def train_step(self, states, actions, goals, expert_actions):
        self.model.train()
        self.optimizer.zero_grad()
        # Use state only (no goal concatenation)
        predictions = self.model(states)
        loss = self.loss_fn(predictions, expert_actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_action(self, state, goal=None):
        # goal parameter kept for API compatibility but not used
        self.model.eval()
        with torch.no_grad():
            if state.ndim == 1:
                state = state.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            action = self.model(state)
            return action.squeeze(0) if squeeze else action
