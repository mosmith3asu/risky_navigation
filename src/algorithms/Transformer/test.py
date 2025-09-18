import numpy as np
import matplotlib.pyplot as plt
import torch
from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.algorithms.Transformer.agent import TransformerAgent
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

def visualize_attention(agent, state, action, goal, save_path=None):
    """
    Visualize attention weights from the transformer model.
    
    Args:
        agent: Transformer agent
        state: Current state
        action: Current action
        goal: Goal position
        save_path: Path to save the visualization (optional)
    """
    # This works if we've modified the model to expose attention weights
    # Here we provide a simple example assuming we have access to attention weights
    
    try:
        # This is a placeholder - in a real implementation, we'd need to 
        # modify the model to store and return attention weights
        agent.model.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            action_t = torch.tensor(action, dtype=torch.float32, device=agent.device).unsqueeze(0)
            goal_t = torch.tensor(goal, dtype=torch.float32, device=agent.device).unsqueeze(0)
            
            # Get attention weights (this is a placeholder)
            # In a real implementation, we would need to modify the model
            # to return attention weights
            # Here, we just create a dummy 3x3 attention matrix for visualization
            dummy_attention = np.array([
                [0.5, 0.3, 0.2],  # State attention to state, action, goal
                [0.3, 0.6, 0.1],  # Action attention to state, action, goal
                [0.2, 0.1, 0.7]   # Goal attention to state, action, goal
            ])
            
            # Plot attention weights
            plt.figure(figsize=(8, 6))
            plt.imshow(dummy_attention, cmap='viridis')
            plt.colorbar()
            plt.xticks([0, 1, 2], ['State', 'Action', 'Goal'])
            plt.yticks([0, 1, 2], ['State', 'Action', 'Goal'])
            plt.title('Transformer Attention Weights (Example)')
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
            print("Attention visualization created (example only)")
    except Exception as e:
        print(f"Failed to visualize attention: {e}")
        print("Note: This is just a placeholder - actual attention visualization requires model modification.")

def main():
    model_path = 'transformer_model.pth'
    env = ContinuousNavigationEnv()
    logger = Logger(env)

    # Infer dimensions
    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.goal.shape[0] if hasattr(env, 'goal') else 2

    # Try to load model with hyperparameters from the checkpoint
    try:
        checkpoint = torch.load(model_path)
        hyperparams = checkpoint.get('hyperparams', {})
        d_model = hyperparams.get('d_model', 64)
        nhead = hyperparams.get('nhead', 4)
        num_layers = hyperparams.get('num_layers', 2)
        print(f"Loaded hyperparameters: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    except Exception as e:
        print(f"Could not load hyperparameters from checkpoint: {e}")
        d_model = 64
        nhead = 4
        num_layers = 2
        print(f"Using default hyperparameters: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")

    # Initialize and load agent
    agent = TransformerAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        goal_dim=goal_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    agent.load(model_path)
    
    # Print model architecture
    print("Transformer Model Architecture:")
    print(agent.model)
    
    # Evaluate prediction accuracy
    print("Evaluating prediction accuracy...")
    mse = evaluate_mse(agent, env, num_episodes=10)
    print(f"Average Mean Squared Error: {mse:.6f}")

    # Optional: Visualize attention (placeholder)
    dummy_state = env.reset()
    dummy_action = env.action_space.sample()
    dummy_goal = env.goal.copy()
    visualize_attention(agent, dummy_state, dummy_action, dummy_goal, 'transformer_attention.png')

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
            plt.savefig('transformer_test_loss.png')
            plt.close()
            print("Loss plot saved as transformer_test_loss.png")

if __name__ == '__main__':
    main()
