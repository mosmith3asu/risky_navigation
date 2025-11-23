import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange
import torch

from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.env.layouts import read_layout_dict
from src.algorithms.Bayesian.agent import BayesianAgent
from src.utils.file_management import save_pickle, load_pickle
from src.utils.logger import Logger
from src.utils.visibility_graph import VisibilityGraph

def collect_data(env, vgraph, num_episodes=100, max_steps=200, sequence_len=1):
    episodes = []
    for ep in trange(num_episodes, desc='Collecting expert data'):
        state = env.reset()
        goal = env.goal.copy()
        prev_action = np.zeros(env.action_space.shape[0])
        episode_data = []
        
        for t in range(max_steps):
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
    
    return data

def prepare_arrays(data):
    state_sequences = np.stack([d['state_sequences'] for d in data])
    action_sequences = np.stack([d['action_sequences'] for d in data])
    target_actions = np.stack([d['target_action'] for d in data])
    goals = np.stack([d['goal'] for d in data])
    return state_sequences, action_sequences, target_actions, goals

def plot_losses(train_losses, val_losses):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(train_losses)
    ax[0].set_title('Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[1].plot(val_losses)
    ax[1].set_title('Validation Loss')
    ax[1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('bayesian_training_loss.png')
    plt.close()

def main():
    num_episodes = 200
    max_steps = 200
    sequence_len = 5
    hidden_dim = 128
    lr = 1e-3
    kl_weight = 1e-5
    batch_size = 128
    num_epochs = 20
    val_ratio = 0.2
    dataset_path = f'bayesian_expert_dataset_seq{sequence_len}.pickle'
    model_path = f'bayesian_model_seq{sequence_len}.pth'

    layout_dict = read_layout_dict('example0')
    env = ContinuousNavigationEnv(**layout_dict)
    logger = Logger(env)
    
    vgraph = VisibilityGraph(env.goal, env.obstacles, env.bounds, resolution=(20, 20))

    if os.path.exists(dataset_path):
        data = load_pickle(dataset_path)
        print(f"Loaded dataset from {dataset_path}")
    else:
        print(f"Collecting expert dataset with sequence_len={sequence_len}...")
        data = collect_data(env, vgraph, num_episodes=num_episodes, max_steps=max_steps, sequence_len=sequence_len)
        save_pickle(data, dataset_path)
        print(f"Saved dataset to {dataset_path}")

    state_sequences, action_sequences, target_actions, goals = prepare_arrays(data)
    state_dim = state_sequences.shape[2]
    action_dim = target_actions.shape[1]
    goal_dim = goals.shape[1]
    
    num_samples = state_sequences.shape[0]
    num_train = int((1 - val_ratio) * num_samples)
    
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_state_seqs = state_sequences[train_indices]
    train_action_seqs = action_sequences[train_indices]
    train_target_actions = target_actions[train_indices]
    train_goals = goals[train_indices]
    
    val_state_seqs = state_sequences[val_indices]
    val_action_seqs = action_sequences[val_indices]
    val_target_actions = target_actions[val_indices]
    val_goals = goals[val_indices]

    agent = BayesianAgent(
        state_dim, action_dim, goal_dim,
        sequence_len=sequence_len,
        hidden_dim=hidden_dim, 
        lr=lr,
        kl_weight=kl_weight,
        action_low=env.action_space.low,
        action_high=env.action_space.high
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(num_epochs):
        train_indices_shuffled = np.random.permutation(len(train_state_seqs))
        epoch_loss = 0.0
        num_batches = len(train_state_seqs) // batch_size
        
        for i in range(0, len(train_indices_shuffled), batch_size):
            if i + batch_size > len(train_indices_shuffled):
                break
            idx = train_indices_shuffled[i:i+batch_size]
            loss = agent.train_step(
                train_state_seqs[idx], 
                train_action_seqs[idx], 
                train_goals[idx], 
                train_target_actions[idx]
            )
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        with torch.no_grad():
            agent.model.eval()
            val_state_seqs_t = torch.tensor(val_state_seqs, dtype=torch.float32, device=agent.device)
            val_action_seqs_t = torch.tensor(val_action_seqs, dtype=torch.float32, device=agent.device)
            val_target_actions_t = torch.tensor(val_target_actions, dtype=torch.float32, device=agent.device)
            val_goals_t = torch.tensor(val_goals, dtype=torch.float32, device=agent.device)
            
            batch_size_val = val_state_seqs_t.shape[0]
            state_seq_flat = val_state_seqs_t.reshape(batch_size_val, -1)
            action_seq_flat = val_action_seqs_t.reshape(batch_size_val, -1)
            inputs = torch.cat([state_seq_flat, action_seq_flat, val_goals_t], dim=1)
            
            predictions = agent.model(inputs, deterministic=True)
            val_loss = torch.nn.functional.mse_loss(predictions, val_target_actions_t).item()
            agent.model.train()
        
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            agent.save(model_path)
    
    print(f"Training completed! Best val loss: {best_val_loss:.6f}")
    plot_losses(train_losses, val_losses)

if __name__ == '__main__':
    main()
