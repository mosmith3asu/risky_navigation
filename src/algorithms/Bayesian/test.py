import numpy as np
import matplotlib.pyplot as plt
from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.algorithms.Bayesian.agent import BayesianActionPredictor
from src.utils.logger import Logger

def visualize_uncertainty(predictions, uncertainties, actual=None, title="Action Predictions with Uncertainty"):
    """
    Visualize the predictions with uncertainty bands
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(predictions))
    
    # Plot for throttle (first dimension)
    ax.plot(x, predictions[:, 0], label="Predicted Throttle", color='blue')
    ax.fill_between(x, 
                    predictions[:, 0] - uncertainties[:, 0], 
                    predictions[:, 0] + uncertainties[:, 0], 
                    alpha=0.3, color='blue',
                    label="Throttle Uncertainty (±1σ)")
    
    # Plot for steering (second dimension)
    ax.plot(x, predictions[:, 1], label="Predicted Steering", color='green')
    ax.fill_between(x, 
                    predictions[:, 1] - uncertainties[:, 1], 
                    predictions[:, 1] + uncertainties[:, 1], 
                    alpha=0.3, color='green',
                    label="Steering Uncertainty (±1σ)")
    
    # Plot actual actions if provided
    if actual is not None:
        ax.scatter(x, actual[:, 0], label="Actual Throttle", color='darkblue', marker='x')
        ax.scatter(x, actual[:, 1], label="Actual Steering", color='darkgreen', marker='x')
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Action Value")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    model_path = 'bayesian_model.pth'
    env = ContinuousNavigationEnv()
    logger = Logger(env)

    # Infer dimensions
    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.goal.shape[0] if hasattr(env, 'goal') else 2

    # Load model
    agent = BayesianActionPredictor(state_dim, action_dim, goal_dim)
    agent.load(model_path)
    
    # Number of samples to generate for uncertainty visualization
    agent.n_samples = 30  # Increase for better uncertainty estimates
    
    # Test parameters
    num_episodes = 5
    max_steps = 200
    
    # Keep track of all predictions and uncertainties
    all_episode_predictions = []
    all_episode_uncertainties = []
    all_episode_rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        goal = env.goal.copy() if hasattr(env, 'goal') else np.zeros(goal_dim)
        ep_reward = 0.0
        states_seq = [state[:2]]
        actions_seq = []
        
        # For this episode's visualization
        episode_predictions = []
        episode_uncertainties = []
        
        prev_action = env.action_space.sample()  # Initial action is random
        
        for t in range(max_steps):
            # Use model to predict next action with uncertainty
            pred_mean, pred_std = agent.predict_next_action(state, prev_action, goal)
            
            # Store prediction and uncertainty
            episode_predictions.append(pred_mean[0])  # First dimension because pred_mean has shape [batch_size, action_dim]
            episode_uncertainties.append(pred_std[0])
            
            # Execute action
            action = np.clip(pred_mean[0], env.action_space.low, env.action_space.high)
            next_state, reward, done, info = env.step(action)
            actions_seq.append(action)
            states_seq.append(next_state[:2])
            ep_reward += reward
            prev_action = action
            state = next_state
            
            # Optional: render environment
            # env.render()
            
            if done:
                terminal_cause = info.get('reason', 'done')
                break
        
        # Convert lists to numpy arrays for visualization
        episode_predictions = np.array(episode_predictions)
        episode_uncertainties = np.array(episode_uncertainties)
        
        # Store episode data
        all_episode_predictions.append(episode_predictions)
        all_episode_uncertainties.append(episode_uncertainties)
        all_episode_rewards.append(ep_reward)
        
        # Log episode results
        logger.log(ep_reward, states_seq, terminal_cause=terminal_cause)
        print(f"Episode {ep+1}: Reward={ep_reward:.2f}, Steps={t+1}, Terminal={terminal_cause}")
        
        # Visualize trajectory
        logger.draw()
        
        # Visualize predictions and uncertainties for this episode
        visualize_uncertainty(
            episode_predictions, 
            episode_uncertainties, 
            title=f"Episode {ep+1}: Action Predictions with Uncertainty"
        )
    
    # Print average reward
    avg_reward = np.mean(all_episode_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    # Optional: compute average uncertainty
    avg_uncertainty_throttle = np.mean([ep_unc[:, 0].mean() for ep_unc in all_episode_uncertainties])
    avg_uncertainty_steering = np.mean([ep_unc[:, 1].mean() for ep_unc in all_episode_uncertainties])
    
    print(f"Average uncertainty - Throttle: {avg_uncertainty_throttle:.4f}, Steering: {avg_uncertainty_steering:.4f}")

if __name__ == '__main__':
    main()
