import numpy as np
import matplotlib.pyplot as plt
from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.algorithms.Linear.agent import LinearAgent
from src.utils.logger import Logger

def evaluate_mse(agent, env, num_episodes=10, max_steps=200):
    """
    Evaluate the Mean Squared Error between predicted and actual actions.
    
    Args:
        agent: Trained agent
        env: Environment
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        
    Returns:
        float: Average MSE
    """
    mse_values = []
    
    for ep in range(num_episodes):
        state = env.reset()
        goal = env.goal.copy() if hasattr(env, 'goal') else np.zeros(2)
        
        for t in range(max_steps):
            # For evaluation, we take a random action
            action = env.action_space.sample()
            
            # Get next state and sample a new action as ground truth
            next_state, reward, done, info = env.step(action)
            next_action_ground_truth = env.action_space.sample() if not done else np.zeros_like(action)
            
            # Predict next action
            next_action_pred = agent.predict_next_action(state, action, goal)
            
            # Calculate MSE for this step
            mse = np.mean((next_action_pred - next_action_ground_truth)**2)
            mse_values.append(mse)
            
            state = next_state
            
            if done:
                break
    
    return np.mean(mse_values)

def main():
    model_path = 'linear_model.pth'
    env = ContinuousNavigationEnv()
    logger = Logger(env)

    # Infer dimensions
    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.goal.shape[0] if hasattr(env, 'goal') else 2

    # Initialize and load agent
    agent = LinearAgent(state_dim, action_dim, goal_dim)
    agent.load(model_path)
    
    # Print model architecture
    print("Linear Model Architecture:")
    print(agent.model)
    
    # Evaluate prediction accuracy
    print("Evaluating prediction accuracy...")
    mse = evaluate_mse(agent, env, num_episodes=10)
    print(f"Average Mean Squared Error: {mse:.6f}")

    # Test the agent on a few episodes
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

    # Plot training and validation losses if available
    if hasattr(agent, 'train_losses') and hasattr(agent, 'val_losses'):
        if len(agent.train_losses) > 0 and len(agent.val_losses) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(agent.train_losses, label='Training Loss')
            plt.plot(agent.val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('linear_test_loss.png')
            plt.close()
            print("Loss plot saved as linear_test_loss.png")

if __name__ == '__main__':
    main()
