import torch
import torch.nn as nn
import torch.optim as optim

class ActionAutoEncoder(nn.Module):
	def __init__(self, state_dim, action_dim, goal_dim, latent_dim=64):
		super().__init__()
		input_dim = state_dim + action_dim + goal_dim
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, latent_dim),
			nn.ReLU(),
		)
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, latent_dim),
			nn.ReLU(),
			nn.Linear(latent_dim, action_dim),
		)

	def forward(self, state, action, goal):
		x = torch.cat([state, action, goal], dim=-1)
		z = self.encoder(x)
		next_action_pred = self.decoder(z)
		return next_action_pred


class AutoEncoderAgent:
	def __init__(self, state_dim, action_dim, goal_dim, latent_dim=64, lr=1e-3, device=None):
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.model = ActionAutoEncoder(state_dim, action_dim, goal_dim, latent_dim).to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
		self.loss_fn = nn.MSELoss()

	def train_step(self, state, action, goal, next_action):
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
		self.model.eval()
		with torch.no_grad():
			state = torch.tensor(state, dtype=torch.float32, device=self.device)
			action = torch.tensor(action, dtype=torch.float32, device=self.device)
			goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
			pred = self.model(state, action, goal)
		return pred.cpu().numpy()

	def save(self, path):
		torch.save(self.model.state_dict(), path)

	def load(self, path):
		self.model.load_state_dict(torch.load(path, map_location=self.device))
