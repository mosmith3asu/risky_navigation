import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.env.layouts import read_layout_dict
from src.algorithms.Linear.agent import LinearAgent
from src.utils.file_management import save_pickle, load_pickle
from src.utils.logger import Logger
from src.utils.visibility_graph import VisibilityGraph

def collect_expert_data(env, vgraph, num_episodes=100, max_steps=200, sequence_len=1):
    """
    Collect expert demonstration data using visibility graph policy.
    
    Args:
        env: Environment instance
        vgraph: VisibilityGraph instance for optimal navigation
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        sequence_len: Length of temporal sequences (1 = no history)
        
    Returns:
        list: Transitions with state_sequences, action_sequences, expert_action, goal
    """
    episodes = []
    for ep in trange(num_episodes, desc='Collecting expert data'):
        state = env.reset()
        goal = env.goal.copy()
        prev_action = np.zeros(env.action_space.shape[0])
        episode_data = []
        
        for t in range(max_steps):
            current_pos = state[:2]
            current_theta = state[2]
            
            path = vgraph.shortest_path(current_pos, goal)
            if path is not None and len(path) > 1:
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
            # Get sequence history
            start_idx = max(0, i - sequence_len + 1)
            seq_transitions = transitions[start_idx:i+1]
            
            # Pad if necessary
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
    """
    Convert data list to numpy arrays.
    
    Args:
        data: List of dictionaries with state_sequences, action_sequences, target_action, goal
        
    Returns:
        tuple: Arrays of state_sequences, action_sequences, target_actions, goals
    """
    state_sequences = np.stack([d['state_sequences'] for d in data])
    action_sequences = np.stack([d['action_sequences'] for d in data])
    target_actions = np.stack([d['target_action'] for d in data])
    goals = np.stack([d['goal'] for d in data])
    return state_sequences, action_sequences, target_actions, goals

def split_data(state_sequences, action_sequences, target_actions, goals, val_ratio=0.2):
    """
    Split data into training and validation sets.
    
    Args:
        state_sequences: Array of state sequences
        action_sequences: Array of action sequences
        target_actions: Array of target actions
        goals: Array of goals
        val_ratio: Validation set ratio
        
    Returns:
        tuple: Training and validation data arrays
    """
    n = state_sequences.shape[0]
    indices = np.random.permutation(n)
    val_size = int(n * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_state_seqs = state_sequences[train_indices]
    train_action_seqs = action_sequences[train_indices]
    train_target_actions = target_actions[train_indices]
    train_goals = goals[train_indices]
    
    val_state_seqs = state_sequences[val_indices]
    val_action_seqs = action_sequences[val_indices]
    val_target_actions = target_actions[val_indices]
    val_goals = goals[val_indices]
    
    return (train_state_seqs, train_action_seqs, train_target_actions, train_goals), \
           (val_state_seqs, val_action_seqs, val_target_actions, val_goals)

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
    num_episodes = 200
    max_steps = 200
    sequence_len = 5  # Temporal history length
    lr = 1e-3
    batch_size = 128
    num_epochs = 20
    val_ratio = 0.2
    dataset_path = f'linear_expert_dataset_seq{sequence_len}.pickle'
    model_path = f'linear_model_seq{sequence_len}.pth'
    
    layout_dict = read_layout_dict('example0')
    env = ContinuousNavigationEnv(**layout_dict)
    logger = Logger(env)
    
    vgraph = VisibilityGraph(env.goal, env.obstacles, env.bounds, resolution=(20, 20))
    
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        data = load_pickle(dataset_path)
    else:
        print(f"Collecting expert data from {num_episodes} episodes with sequence_len={sequence_len}")
        data = collect_expert_data(env, vgraph, num_episodes, max_steps, sequence_len)
        save_pickle(data, dataset_path)
        print(f"Dataset saved to {dataset_path}")
    
    state_sequences, action_sequences, target_actions, goals = prepare_arrays(data)
    (train_state_seqs, train_action_seqs, train_target_actions, train_goals), \
    (val_state_seqs, val_action_seqs, val_target_actions, val_goals) = \
        split_data(state_sequences, action_sequences, target_actions, goals, val_ratio)
    
    print(f"Training set size: {len(train_state_seqs)}, Validation set size: {len(val_state_seqs)}")
    print(f"Sequence shape: {train_state_seqs.shape}")
    
    state_dim = state_sequences.shape[2]  # (batch, seq_len, state_dim)
    action_dim = target_actions.shape[1]
    goal_dim = goals.shape[1]
    agent = LinearAgent(state_dim, action_dim, goal_dim, sequence_len=sequence_len, lr=lr, 
                       action_low=env.action_space.low, action_high=env.action_space.high)
    
    train_losses = []
    val_losses = []
    num_batches = len(train_state_seqs) // batch_size
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        indices = np.random.permutation(len(train_state_seqs))
        train_state_seqs_shuffled = train_state_seqs[indices]
        train_action_seqs_shuffled = train_action_seqs[indices]
        train_target_actions_shuffled = train_target_actions[indices]
        train_goals_shuffled = train_goals[indices]
        
        epoch_loss = 0.0
        for batch_idx in trange(num_batches, desc=f"Epoch {epoch+1}/{num_epochs}"):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_state_seqs = train_state_seqs_shuffled[start_idx:end_idx]
            batch_action_seqs = train_action_seqs_shuffled[start_idx:end_idx]
            batch_target_actions = train_target_actions_shuffled[start_idx:end_idx]
            batch_goals = train_goals_shuffled[start_idx:end_idx]
            
            loss = agent.train_step(batch_state_seqs, batch_action_seqs, batch_goals, batch_target_actions)
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        with torch.no_grad():
            agent.model.eval()
            val_state_seqs_t = torch.tensor(val_state_seqs, dtype=torch.float32, device=agent.device)
            val_action_seqs_t = torch.tensor(val_action_seqs, dtype=torch.float32, device=agent.device)
            val_target_actions_t = torch.tensor(val_target_actions, dtype=torch.float32, device=agent.device)
            val_goals_t = torch.tensor(val_goals, dtype=torch.float32, device=agent.device)
            
            batch_size = val_state_seqs_t.shape[0]
            state_seq_flat = val_state_seqs_t.reshape(batch_size, -1)
            action_seq_flat = val_action_seqs_t.reshape(batch_size, -1)
            inputs = torch.cat([state_seq_flat, action_seq_flat, val_goals_t], dim=1)
            
            predictions = agent.model(inputs)
            val_loss = agent.loss_fn(predictions, val_target_actions_t).item()
            agent.model.train()
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")
    
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    plot_losses(train_losses, val_losses, 'linear_training_loss.png')
    print("Training complete!")

if __name__ == '__main__':
    main()
