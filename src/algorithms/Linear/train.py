import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.algorithms.Linear.agent import LinearAgent
from src.utils.file_management import save_pickle, load_pickle
from src.utils.logger import Logger

def collect_data(env, num_episodes=100, max_steps=200):
    """
    Collect data from random trajectories in the environment.
    
    Args:
        env: Environment instance
        num_episodes: Number of episodes to collect data from
        max_steps: Maximum steps per episode
        
    Returns:
        list: Dictionary containing state, action, goal, next_action
    """
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
    """
    Convert data list to numpy arrays.
    
    Args:
        data: List of dictionaries with state, action, goal, next_action
        
    Returns:
        tuple: Arrays of states, actions, goals, next_actions
    """
    states = np.stack([d['state'] for d in data])
    actions = np.stack([d['action'] for d in data])
    goals = np.stack([d['goal'] for d in data])
    next_actions = np.stack([d['next_action'] for d in data])
    return states, actions, goals, next_actions

def split_data(states, actions, goals, next_actions, val_ratio=0.2):
    """
    Split data into training and validation sets.
    
    Args:
        states: Array of states
        actions: Array of actions
        goals: Array of goals
        next_actions: Array of next actions
        val_ratio: Validation set ratio
        
    Returns:
        tuple: Training and validation data arrays
    """
    n = states.shape[0]
    indices = np.random.permutation(n)
    val_size = int(n * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_states = states[train_indices]
    train_actions = actions[train_indices]
    train_goals = goals[train_indices]
    train_next_actions = next_actions[train_indices]
    
    val_states = states[val_indices]
    val_actions = actions[val_indices]
    val_goals = goals[val_indices]
    val_next_actions = next_actions[val_indices]
    
    return (train_states, train_actions, train_goals, train_next_actions), \
           (val_states, val_actions, val_goals, val_next_actions)

def plot_losses(train_losses, val_losses, save_path=None):
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    # --- Config ---
    num_episodes = 200
    max_steps = 200
    lr = 1e-3
    batch_size = 128
    num_epochs = 20
    val_ratio = 0.2
    dataset_path = 'linear_dataset.pickle'
    model_path = 'linear_model.pth'
    
    # --- Environment setup ---
    env = ContinuousNavigationEnv()
    logger = Logger(env)
    
    # --- Data collection ---
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        data = load_pickle(dataset_path)
    else:
        print(f"Collecting data from {num_episodes} episodes")
        data = collect_data(env, num_episodes, max_steps)
        save_pickle(data, dataset_path)
        print(f"Dataset saved to {dataset_path}")
    
    states, actions, goals, next_actions = prepare_arrays(data)
    (train_states, train_actions, train_goals, train_next_actions), \
    (val_states, val_actions, val_goals, val_next_actions) = \
        split_data(states, actions, goals, next_actions, val_ratio)
    
    print(f"Training set size: {len(train_states)}, Validation set size: {len(val_states)}")
    
    # --- Model setup ---
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    goal_dim = goals.shape[1]
    agent = LinearAgent(state_dim, action_dim, goal_dim, lr=lr)
    
    # --- Training ---
    train_losses = []
    val_losses = []
    num_batches = len(train_states) // batch_size
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Shuffle training data
        indices = np.random.permutation(len(train_states))
        train_states_shuffled = train_states[indices]
        train_actions_shuffled = train_actions[indices]
        train_goals_shuffled = train_goals[indices]
        train_next_actions_shuffled = train_next_actions[indices]
        
        # Train for one epoch
        epoch_loss = 0.0
        for batch_idx in trange(num_batches, desc=f"Epoch {epoch+1}/{num_epochs}"):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_states = train_states_shuffled[start_idx:end_idx]
            batch_actions = train_actions_shuffled[start_idx:end_idx]
            batch_goals = train_goals_shuffled[start_idx:end_idx]
            batch_next_actions = train_next_actions_shuffled[start_idx:end_idx]
            
            loss = agent.train_step(batch_states, batch_actions, batch_goals, batch_next_actions)
            epoch_loss += loss
        
        # Compute average training loss
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validate
        val_loss = agent.validate(val_states, val_actions, val_goals, val_next_actions)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")
    
    # --- Save model ---
    agent.train_losses = train_losses
    agent.val_losses = val_losses
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    # --- Plot losses ---
    plot_losses(train_losses, val_losses, 'linear_training_loss.png')
    print("Training complete!")

if __name__ == '__main__':
    main()
