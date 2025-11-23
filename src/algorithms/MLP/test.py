import numpy as np
from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.algorithms.AutoEncoder.agent import AutoEncoderAgent
from src.utils.logger import Logger

def main():
	model_path = 'autoencoder_model.pth'
	env = ContinuousNavigationEnv()
	logger = Logger(env)

	# Infer dimensions
	dummy_state = env.reset()
	state_dim = dummy_state.shape[0]
	action_dim = env.action_space.shape[0]
	goal_dim = env.goal.shape[0] if hasattr(env, 'goal') else 2

	agent = AutoEncoderAgent(state_dim, action_dim, goal_dim)
	agent.load(model_path)

	num_episodes = 10
	max_steps = 200
	for ep in range(num_episodes):
		state = env.reset()
		goal = env.goal.copy() if hasattr(env, 'goal') else np.zeros(goal_dim)
		ep_reward = 0.0
		states_seq = [state[:2]]
		actions_seq = []
		for t in range(max_steps):
			# For first step, take random action
			if t == 0:
				action = env.action_space.sample()
			else:
				# Use model to predict next action
				action = agent.predict_next_action(state, prev_action, goal)
				action = np.clip(action, env.action_space.low, env.action_space.high)
			next_state, reward, done, info = env.step(action)
			actions_seq.append(action)
			states_seq.append(next_state[:2])
			ep_reward += reward
			prev_action = action
			state = next_state
			if done:
				terminal_cause = info.get('reason', 'done')
				break
		logger.log(ep_reward, states_seq, terminal_cause=terminal_cause)
		print(f"Episode {ep+1}: Reward={ep_reward:.2f}, Terminal={terminal_cause}")
		logger.draw()

if __name__ == '__main__':
	main()
