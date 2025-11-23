#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm, trange

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))


from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.env.layouts import read_layout_dict
from src.algorithms.VAE.agent import VAEAgent
from src.utils.file_management import save_pickle, load_pickle
from src.utils.logger import Logger
from src.utils.visibility_graph import VisibilityGraph

def collect_training_data(env, vgraph, num_episodes=500, max_steps_per_episode=100):
    print("Collecting expert training data...")
    data = []
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        goal = env.goal.copy()
        prev_action = np.zeros(env.action_space.shape[0])
        
        for step in range(max_steps_per_episode):
            current_pos = state[:2]
            current_theta = state[2]
            
            _, path = vgraph(current_pos)
            if len(path) > 1:
                target = np.array(path[1])
            else:
                target = goal
            
            direction = target - current_pos
            desired_theta = np.arctan2(direction[1], direction[0])
            angle_diff = desired_theta - current_theta
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            
            steering = np.clip(angle_diff * 2.0, env.action_space.low[1], env.action_space.high[1])
            dist_to_target = np.linalg.norm(direction)
            
            if dist_to_target < env.goal_radius * 3:
                throttle = env.action_space.high[0] * 0.3
            else:
                throttle = env.action_space.high[0] * 0.8
            
            action = np.array([throttle, steering])
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            next_state, reward, done, info = env.step(action)
            
            data.append({
                'state': state.copy(),
                'prev_action': prev_action.copy(),
                'action': action.copy(),
                'goal': goal.copy(),
            })
            
            prev_action = action
            state = next_state
            if done:
                break
    
    print(f"Collected {len(data)} transitions")
    return data

def prepare_data_for_training(data, validation_ratio=0.2):
    states = np.array([d['state'] for d in data])
    prev_actions = np.array([d['prev_action'] for d in data])
    actions = np.array([d['action'] for d in data])
    goals = np.array([d['goal'] for d in data])
    
    n_samples = len(data)
    n_val = int(n_samples * validation_ratio)
    
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_data = {
        'states': states[train_indices],
        'prev_actions': prev_actions[train_indices],
        'actions': actions[train_indices],
        'goals': goals[train_indices]
    }
    
    val_data = {
        'states': states[val_indices],
        'prev_actions': prev_actions[val_indices],
        'actions': actions[val_indices],
        'goals': goals[val_indices]
    }
    
    print(f"Training samples: {len(train_data['states'])}")
    print(f"Validation samples: {len(val_data['states'])}")
    
    return train_data, val_data

def train_vae_agent(agent, train_data, val_data, num_epochs=100, batch_size=64):
    print(f"Starting training for {num_epochs} epochs...")
    
    train_losses = []
    val_losses = []
    
    n_train_samples = len(train_data['states'])
    n_batches = n_train_samples // batch_size
    
    for epoch in range(num_epochs):
        agent.encoder.train()
        agent.decoder.train()
        epoch_loss = 0.0
        
        indices = np.random.permutation(n_train_samples)
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, n_train_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_states = train_data['states'][batch_indices]
            batch_prev_actions = train_data['prev_actions'][batch_indices]
            batch_actions = train_data['actions'][batch_indices]
            batch_goals = train_data['goals'][batch_indices]
            
            loss = agent.train_step(batch_states, batch_prev_actions, batch_goals, batch_actions)
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        with torch.no_grad():
            agent.encoder.eval()
            agent.decoder.eval()
            val_states_t = torch.tensor(val_data['states'], dtype=torch.float32, device=agent.device)
            val_prev_actions_t = torch.tensor(val_data['prev_actions'], dtype=torch.float32, device=agent.device)
            val_actions_t = torch.tensor(val_data['actions'], dtype=torch.float32, device=agent.device)
            val_goals_t = torch.tensor(val_data['goals'], dtype=torch.float32, device=agent.device)
            inputs = torch.cat([val_states_t, val_prev_actions_t, val_goals_t], dim=1)
            mu, _ = agent.encoder(inputs)
            predictions = agent.decoder(mu)
            val_loss = torch.nn.functional.mse_loss(predictions, val_actions_t).item()
        
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return train_losses, val_losses
    
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
    print("Evaluating agent...")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        goal = env.goal.copy()
        episode_reward = 0.0
        prev_action = np.zeros(env.action_space.shape[0])
        
        for step in range(100):
            state_t = torch.tensor(state, dtype=torch.float32, device=agent.device)
            prev_action_t = torch.tensor(prev_action, dtype=torch.float32, device=agent.device)
            goal_t = torch.tensor(goal, dtype=torch.float32, device=agent.device)
            
            action = agent.predict_action(state_t, prev_action_t, goal_t)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            prev_action = action
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average Reward: {avg_reward:.3f}")
    
    return avg_reward

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
    config = {
        'num_episodes': 300,
        'max_steps_per_episode': 100,
        'num_epochs': 50,
        'batch_size': 64,
        'latent_dim': 32,
        'hidden_dim': 128,
        'lr': 1e-3,
        'beta': 1.0,
        'validation_ratio': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("VAE Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    layout_dict = read_layout_dict('example0')
    env = ContinuousNavigationEnv(**layout_dict)
    
    vgraph = VisibilityGraph(env.goal, env.obstacles, env.bounds, resolution=(20, 20))
    
    dataset_path = 'vae_expert_dataset.pickle'
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        data = load_pickle(dataset_path)
    else:
        data = collect_training_data(
            env, vgraph,
            num_episodes=config['num_episodes'],
            max_steps_per_episode=config['max_steps_per_episode']
        )
        save_pickle(data, dataset_path)
    
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.goal.shape[0]
    
    print(f"Environment dimensions:")
    print(f"  State: {state_dim}")
    print(f"  Action: {action_dim}")
    print(f"  Goal: {goal_dim}")
    print()
    
    agent = VAEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        lr=config['lr'],
        beta=config['beta'],
        device=config['device'],
        action_low=env.action_space.low,
        action_high=env.action_space.high
    )
    
    print(f"VAE Agent initialized on {config['device']}")
    print(f"Encoder parameters: {sum(p.numel() for p in agent.encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in agent.decoder.parameters()):,}")
    print()
    
    train_data, val_data = prepare_data_for_training(
        data, validation_ratio=config['validation_ratio']
    )
    
    train_losses, val_losses = train_vae_agent(
        agent, train_data, val_data,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size']
    )
    
    plot_training_curves(train_losses, val_losses, 'vae_training_curves.png')
    
    avg_reward = evaluate_agent(agent, env, num_episodes=20)
    
    model_path = f'vae_agent_model.pth'
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    save_pickle({
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'avg_reward': avg_reward
    }, 'vae_training_results.pickle')
    
    print("\nTraining completed!")
    print(f"Final Results:")
    print(f"  Train Loss: {train_losses[-1]:.6f}")
    print(f"  Val Loss: {val_losses[-1]:.6f}")
    print(f"  Average Reward: {avg_reward:.3f}")

if __name__ == "__main__":
    main()