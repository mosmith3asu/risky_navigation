import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.env.layouts import read_layout_dict
from src.algorithms.Transformer.agent import TransformerAgent
from src.utils.file_management import save_pickle, load_pickle
from src.utils.logger import Logger
from src.utils.visibility_graph import VisibilityGraph

def collect_data(env, vgraph, num_episodes=100, max_steps=200):
    data = []
    for ep in trange(num_episodes, desc='Collecting expert data'):
        state = env.reset()
        goal = env.goal.copy()
        prev_action = np.zeros(env.action_space.shape[0])
        
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
    
    return data

def prepare_arrays(data):
    states = np.stack([d['state'] for d in data])
    prev_actions = np.stack([d['prev_action'] for d in data])
    actions = np.stack([d['action'] for d in data])
    goals = np.stack([d['goal'] for d in data])
    return states, prev_actions, actions, goals

def split_data(states, prev_actions, actions, goals, val_ratio=0.2):
    n = states.shape[0]
    indices = np.random.permutation(n)
    val_size = int(n * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_states = states[train_indices]
    train_prev_actions = prev_actions[train_indices]
    train_actions = actions[train_indices]
    train_goals = goals[train_indices]
    
    val_states = states[val_indices]
    val_prev_actions = prev_actions[val_indices]
    val_actions = actions[val_indices]
    val_goals = goals[val_indices]
    
    return (train_states, train_prev_actions, train_actions, train_goals), \
           (val_states, val_prev_actions, val_actions, val_goals)

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
    d_model = 64
    nhead = 4
    num_layers = 2
    dropout = 0.1
    sequence_len = 1
    lr = 1e-3
    batch_size = 64
    num_epochs = 30
    val_ratio = 0.2
    dataset_path = 'transformer_expert_dataset.pickle'
    model_path = 'transformer_model.pth'
    
    layout_dict = read_layout_dict('example0')
    env = ContinuousNavigationEnv(**layout_dict)
    logger = Logger(env)
    
    vgraph = VisibilityGraph(env.goal, env.obstacles, env.bounds, resolution=(20, 20))
    
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        data = load_pickle(dataset_path)
    else:
        print(f"Collecting expert data from {num_episodes} episodes")
        data = collect_data(env, vgraph, num_episodes, max_steps)
        save_pickle(data, dataset_path)
        print(f"Dataset saved to {dataset_path}")
    
    states, prev_actions, actions, goals = prepare_arrays(data)
    (train_states, train_prev_actions, train_actions, train_goals), \
    (val_states, val_prev_actions, val_actions, val_goals) = \
        split_data(states, prev_actions, actions, goals, val_ratio)
    
    print(f"Training set size: {len(train_states)}, Validation set size: {len(val_states)}")
    
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    goal_dim = goals.shape[1]
    
    agent = TransformerAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        goal_dim=goal_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        sequence_len=sequence_len,
        lr=lr,
        action_low=env.action_space.low,
        action_high=env.action_space.high
    )
    
    train_losses = []
    val_losses = []
    num_batches = len(train_states) // batch_size
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        indices = np.random.permutation(len(train_states))
        train_states_shuffled = train_states[indices]
        train_prev_actions_shuffled = train_prev_actions[indices]
        train_actions_shuffled = train_actions[indices]
        train_goals_shuffled = train_goals[indices]
        
        epoch_loss = 0.0
        for batch_idx in trange(num_batches, desc=f"Epoch {epoch+1}/{num_epochs}"):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_states = train_states_shuffled[start_idx:end_idx]
            batch_prev_actions = train_prev_actions_shuffled[start_idx:end_idx]
            batch_actions = train_actions_shuffled[start_idx:end_idx]
            batch_goals = train_goals_shuffled[start_idx:end_idx]
            
            loss = agent.train_step(batch_states, batch_prev_actions, batch_goals, batch_actions)
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        with torch.no_grad():
            agent.model.eval()
            val_states_t = torch.tensor(val_states, dtype=torch.float32, device=agent.device)
            val_prev_actions_t = torch.tensor(val_prev_actions, dtype=torch.float32, device=agent.device)
            val_actions_t = torch.tensor(val_actions, dtype=torch.float32, device=agent.device)
            val_goals_t = torch.tensor(val_goals, dtype=torch.float32, device=agent.device)
            inputs = torch.cat([val_states_t, val_prev_actions_t, val_goals_t], dim=-1)
            predictions = agent.model(inputs)
            val_loss = agent.loss_fn(predictions, val_actions_t).item()
            agent.model.train()
        
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            agent.save(model_path)
    
    plot_losses(train_losses, val_losses, 'transformer_training_loss.png')
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("Training complete!")

if __name__ == '__main__':
    main()
