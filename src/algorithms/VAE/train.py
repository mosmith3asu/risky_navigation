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

def collect_training_data(env, vgraph, num_episodes=500, max_steps_per_episode=100, sequence_len=1):
    print(f"Collecting expert training data with sequence_len={sequence_len}...")
    episodes = []
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        goal = env.goal.copy()
        prev_action = np.zeros(env.action_space.shape[0])
        episode_data = []
        
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
            
            episode_data.append({
                'state': state.copy(),
                'action': action.copy(),
            })
            
            prev_action = action
            state = next_state
            if done:
                break
        
        if len(episode_data) > 0:
            episodes.append({'transitions': episode_data, 'goal': goal})
    
    # Create sequences from episodes
    data = []
    for episode in episodes:
        transitions = episode['transitions']
        goal = episode['goal']
        
        for i in range(len(transitions)):
            start_idx = max(0, i - sequence_len + 1)
            seq_transitions = transitions[start_idx:i+1]
            
            while len(seq_transitions) < sequence_len:
                seq_transitions.insert(0, {'state': np.zeros_like(transitions[0]['state']), 
                                           'action': np.zeros_like(transitions[0]['action'])})
            
            state_seq = np.array([t['state'] for t in seq_transitions[:-1]] + [transitions[i]['state']])
            action_seq = np.array([t['action'] for t in seq_transitions[:-1]] + [np.zeros_like(transitions[i]['action'])])
            target_action = transitions[i]['action']
            
            data.append({
                'state_sequences': state_seq,
                'action_sequences': action_seq,
                'target_action': target_action,
                'goal': goal,
            })
    
    print(f"Collected {len(data)} sequence samples from {num_episodes} episodes")
    return data

def prepare_data_for_training(data, validation_ratio=0.2):
    state_sequences = np.array([d['state_sequences'] for d in data])
    action_sequences = np.array([d['action_sequences'] for d in data])
    target_actions = np.array([d['target_action'] for d in data])
    goals = np.array([d['goal'] for d in data])
    
    n_samples = len(data)
    n_val = int(n_samples * validation_ratio)
    
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_data = {
        'state_sequences': state_sequences[train_indices],
        'action_sequences': action_sequences[train_indices],
        'target_actions': target_actions[train_indices],
        'goals': goals[train_indices]
    }
    
    val_data = {
        'state_sequences': state_sequences[val_indices],
        'action_sequences': action_sequences[val_indices],
        'target_actions': target_actions[val_indices],
        'goals': goals[val_indices]
    }
    
    print(f"Training samples: {len(train_data['state_sequences'])}")
    print(f"Validation samples: {len(val_data['state_sequences'])}")
    
    return train_data, val_data

def train_vae_agent(agent, train_data, val_data, num_epochs=100, batch_size=64):
    print(f"Starting training for {num_epochs} epochs...")
    
    train_losses = []
    val_losses = []
    
    n_train_samples = len(train_data['state_sequences'])
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
            
            batch_state_seqs = train_data['state_sequences'][batch_indices]
            batch_action_seqs = train_data['action_sequences'][batch_indices]
            batch_target_actions = train_data['target_actions'][batch_indices]
            batch_goals = train_data['goals'][batch_indices]
            
            loss = agent.train_step(batch_state_seqs, batch_action_seqs, batch_goals, batch_target_actions)
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        with torch.no_grad():
            agent.encoder.eval()
            agent.decoder.eval()
            val_state_seqs_t = torch.tensor(val_data['state_sequences'], dtype=torch.float32, device=agent.device)
            val_action_seqs_t = torch.tensor(val_data['action_sequences'], dtype=torch.float32, device=agent.device)
            val_target_actions_t = torch.tensor(val_data['target_actions'], dtype=torch.float32, device=agent.device)
            val_goals_t = torch.tensor(val_data['goals'], dtype=torch.float32, device=agent.device)
            
            batch_size_val = val_state_seqs_t.shape[0]
            state_seq_flat = val_state_seqs_t.reshape(batch_size_val, -1)
            action_seq_flat = val_action_seqs_t.reshape(batch_size_val, -1)
            inputs = torch.cat([state_seq_flat, action_seq_flat, val_goals_t], dim=1)
            
            mu, _ = agent.encoder(inputs)
            predictions = agent.decoder(mu)
            val_loss = torch.nn.functional.mse_loss(predictions, val_target_actions_t).item()
        
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
    sequence_len = agent.sequence_len if hasattr(agent, 'sequence_len') else 1
    
    for episode in range(num_episodes):
        state = env.reset()
        goal = env.goal.copy()
        episode_reward = 0.0
        
        state_history = [np.zeros_like(state) for _ in range(sequence_len)]
        action_history = [np.zeros(env.action_space.shape[0]) for _ in range(sequence_len)]
        
        for step in range(100):
            state_history.append(state.copy())
            state_history.pop(0)
            
            state_seq = np.array(state_history)
            action_seq = np.array(action_history)
            
            state_seq_t = torch.tensor(state_seq, dtype=torch.float32, device=agent.device)
            action_seq_t = torch.tensor(action_seq, dtype=torch.float32, device=agent.device)
            goal_t = torch.tensor(goal, dtype=torch.float32, device=agent.device)
            
            action = agent.predict_action(state_seq_t, action_seq_t, goal_t)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            action_history.append(action.copy())
            action_history.pop(0)
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
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
        'sequence_len': 5,
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
    
    dataset_path = f"vae_expert_dataset_seq{config['sequence_len']}.pickle"
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        data = load_pickle(dataset_path)
    else:
        data = collect_training_data(
            env, vgraph,
            num_episodes=config['num_episodes'],
            max_steps_per_episode=config['max_steps_per_episode'],
            sequence_len=config['sequence_len']
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
        sequence_len=config['sequence_len'],
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