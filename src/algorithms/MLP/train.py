import numpy as np
import os
from tqdm import trange
from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.algorithms.AutoEncoder.agent import AutoEncoderAgent
from src.utils.file_management import save_pickle, load_pickle
from src.utils.logger import Logger

def collect_data(env, num_episodes=100, max_steps=200):
	data = []
	for ep in trange(num_episodes, desc='Collecting data'):
		state = env.reset()
		goal = env.goal.copy() if hasattr(env, 'goal') else np.zeros(2)
		for t in range(max_steps):
			action = env.action_space.sample()
			next_state, reward, done, info = env.step(action)
			# For next_action prediction, sample a new action for the next state
			next_action = env.action_space.sample() if not done else np.zeros_like(action)
			data.append({
				'state': state.copy(),
				'action': action.copy(),
				'goal': goal.copy(),
				'next_action': next_action.copy(),
			})
			state = next_state
			if done:
				break
	return data

def prepare_arrays(data):
	states = np.stack([d['state'] for d in data])
	actions = np.stack([d['action'] for d in data])
	goals = np.stack([d['goal'] for d in data])
	next_actions = np.stack([d['next_action'] for d in data])
	return states, actions, goals, next_actions

def main():
	# --- Config ---
	num_episodes = 200
	max_steps = 200
	latent_dim = 64
	lr = 1e-3
	batch_size = 128
	num_epochs = 20
	dataset_path = 'autoencoder_dataset.pickle'
	model_path = 'autoencoder_model.pth'

	# --- Env & Data ---
	env = ContinuousNavigationEnv()
	logger = Logger(env)

	if os.path.exists(dataset_path):
		data = load_pickle(dataset_path)
	else:
		data = collect_data(env, num_episodes=num_episodes, max_steps=max_steps)
		save_pickle(data, dataset_path)

	states, actions, goals, next_actions = prepare_arrays(data)
	state_dim = states.shape[1]
	action_dim = actions.shape[1]
	goal_dim = goals.shape[1]

	# --- Model ---
	agent = AutoEncoderAgent(state_dim, action_dim, goal_dim, latent_dim=latent_dim, lr=lr)

	# --- Training ---
	num_samples = states.shape[0]
	indices = np.arange(num_samples)
	losses = []
	for epoch in range(num_epochs):
		np.random.shuffle(indices)
		epoch_loss = 0
		for i in range(0, num_samples, batch_size):
			idx = indices[i:i+batch_size]
			batch_loss = agent.train_step(states[idx], actions[idx], goals[idx], next_actions[idx])
			epoch_loss += batch_loss * len(idx)
		epoch_loss /= num_samples
		losses.append(epoch_loss)
		print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.6f}")

	agent.save(model_path)

	# --- Plot training loss ---
	import matplotlib.pyplot as plt
	plt.figure()
	plt.plot(losses)
	plt.xlabel('Epoch')
	plt.ylabel('MSE Loss')
	plt.title('AutoEncoder Training Loss')
	plt.show()

if __name__ == '__main__':
	main()
