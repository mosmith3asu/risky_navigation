import torch
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = self.output_proj(x.squeeze(1))
        return x

class TransformerAgent:
    def __init__(self, state_dim, action_dim, goal_dim=None, d_model=64, nhead=4, num_layers=2, dropout=0.1, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = state_dim  # State only
        self.model = TransformerModel(input_dim, action_dim, d_model, nhead, num_layers, dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def train_step(self, states, actions, goals, expert_actions):
        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.model(states)  # State only
        loss = self.loss_fn(predictions, expert_actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_action(self, state, goal=None):
        self.model.eval()
        with torch.no_grad():
            if state.ndim == 1:
                state = state.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            action = self.model(state)
            return action.squeeze(0) if squeeze else action
