#!/usr/bin/env python3
"""
Training script for Variational AutoEncoder (VAE) agent.

This script demonstrates how to train a VAE for next-action prediction
in the risky navigation environment.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.algorithms.VAE.agent import VAEAgent
from src.utils.file_management import save_pickle
from src.utils.logger import Logger

def collect_training_data(env, num_episodes=500, max_steps_per_episode=100):
    """Collect training data from random trajectories."""
    print("Collecting training data...")
    data = []
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        goal = env.goal.copy() if hasattr(env, 'goal') else np.zeros(2)
        
        for step in range(max_steps_per_episode):
            # Take random action
            action = env.action_space.sample()
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # For next action, we'll use another random action as ground truth
            # In practice, this could be from expert demonstrations or optimal policy
            next_action = env.action_space.sample() if not done else np.zeros_like(action)
            
            # Store transition
            data.append({
                'state': state.copy(),
                'action': action.copy(),
                'goal': goal.copy(),
                'next_action': next_action.copy(),
                'reward': reward,
                'done': done
            })
            
            state = next_state
            if done:
                break
    
    print(f"Collected {len(data)} transitions")
    return data

def prepare_data_for_training(data, validation_ratio=0.2):
    """Prepare data for training."""
    # Extract arrays
    states = np.array([d['state'] for d in data])
    actions = np.array([d['action'] for d in data])
    goals = np.array([d['goal'] for d in data])
    next_actions = np.array([d['next_action'] for d in data])
    
    # Split into training and validation
    n_samples = len(data)
    n_val = int(n_samples * validation_ratio)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Split data
    train_data = {
        'states': states[train_indices],
        'actions': actions[train_indices],
        'goals': goals[train_indices],
        'next_actions': next_actions[train_indices]
    }
    
    val_data = {
        'states': states[val_indices],
        'actions': actions[val_indices],
        'goals': goals[val_indices],
        'next_actions': next_actions[val_indices]
    }
    
    print(f"Training samples: {len(train_data['states'])}")
    print(f"Validation samples: {len(val_data['states'])}")
    
    return train_data, val_data

def train_vae_agent(agent, train_data, val_data, num_epochs=100, batch_size=64):
    """Train the VAE agent."""
    print(f"Starting training for {num_epochs} epochs...")
    
    train_losses = []
    val_losses = []
    
    n_train_samples = len(train_data['states'])
    n_batches = n_train_samples // batch_size
    
    for epoch in range(num_epochs):
        # Training phase
        agent.encoder.train()
        agent.decoder.train()
        epoch_loss = 0.0
        
        # Shuffle training data
        indices = np.random.permutation(n_train_samples)
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, n_train_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_states = train_data['states'][batch_indices]
            batch_actions = train_data['actions'][batch_indices]
            batch_goals = train_data['goals'][batch_indices]
            batch_next_actions = train_data['next_actions'][batch_indices]
            
            # Train step
            loss = agent.train_step(batch_states, batch_actions, batch_goals, batch_next_actions)
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss = agent.validate(
            val_data['states'], val_data['actions'], 
            val_data['goals'], val_data['next_actions']
        )
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
    
    return train_losses, val_losses

def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate the trained agent."""
    print("Evaluating agent...")
    
    episode_rewards = []
    prediction_errors = []
    
    for episode in range(num_episodes):
        state = env.reset()
        goal = env.goal.copy() if hasattr(env, 'goal') else np.zeros(2)
        episode_reward = 0.0
        episode_errors = []
        
        for step in range(100):  # Max 100 steps per episode
            if step == 0:
                action = env.action_space.sample()
            else:
                # Use agent to predict next action
                pred_action, pred_std = agent.predict_next_action(state, prev_action, goal)
                action = pred_action
                
                # Clip action to valid range
                action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Calculate prediction error (against random next action as baseline)
            if step > 0:
                ground_truth_next = env.action_space.sample()
                pred_next, _ = agent.predict_next_action(state, action, goal)
                error = np.mean((pred_next - ground_truth_next)**2)
                episode_errors.append(error)
            
            prev_action = action
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        if episode_errors:
            prediction_errors.extend(episode_errors)
    
    avg_reward = np.mean(episode_rewards)
    avg_mse = np.mean(prediction_errors) if prediction_errors else float('inf')
    
    print(f"Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"  Average MSE: {avg_mse:.6f}")
    
    return avg_reward, avg_mse

def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved to {save_path}")
    
    plt.show()

def main():
    """Main training function."""
    # Configuration
    config = {
        'num_episodes': 300,
        'max_steps_per_episode': 100,
        'num_epochs': 50,
        'batch_size': 64,
        'latent_dim': 32,
        'hidden_dim': 128,
        'lr': 1e-3,
        'beta': 1.0,  # Beta-VAE parameter
        'validation_ratio': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("VAE Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize environment
    env = ContinuousNavigationEnv()
    state = env.reset()
    
    # Get dimensions
    state_dim = state.shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.goal.shape[0] if hasattr(env, 'goal') else 2
    
    print(f"Environment dimensions:")
    print(f"  State: {state_dim}")
    print(f"  Action: {action_dim}")
    print(f"  Goal: {goal_dim}")
    print()
    
    # Initialize agent
    agent = VAEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        lr=config['lr'],
        beta=config['beta'],
        device=config['device']
    )
    
    print(f"VAE Agent initialized on {config['device']}")
    print(f"Encoder parameters: {sum(p.numel() for p in agent.encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in agent.decoder.parameters()):,}")
    print()
    
    # Collect data
    data = collect_training_data(
        env, 
        num_episodes=config['num_episodes'],
        max_steps_per_episode=config['max_steps_per_episode']
    )
    
    # Prepare data
    train_data, val_data = prepare_data_for_training(
        data, validation_ratio=config['validation_ratio']
    )
    
    # Train agent
    train_losses, val_losses = train_vae_agent(
        agent, train_data, val_data,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size']
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, 'vae_training_curves.png')
    
    # Evaluate agent
    avg_reward, avg_mse = evaluate_agent(agent, env, num_episodes=20)
    
    # Save agent
    model_path = f'vae_agent_model.pth'
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save training data
    save_pickle({
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'avg_reward': avg_reward,
        'avg_mse': avg_mse
    }, 'vae_training_results.pickle')
    
    print("\nTraining completed successfully!")
    print(f"Final Results:")
    print(f"  Train Loss: {train_losses[-1]:.6f}")
    print(f"  Val Loss: {val_losses[-1]:.6f}")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"  Average MSE: {avg_mse:.6f}")

if __name__ == "__main__":
    main()