import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange
import torch

from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.algorithms.Bayesian.agent import BayesianActionPredictor
from src.utils.file_management import save_pickle, load_pickle
from src.utils.logger import Logger

def collect_data(env, num_episodes=100, max_steps=200):
    """Collect state, action, goal, next_action data from the environment"""
    data = []
    for ep in trange(num_episodes, desc='Collecting data'):
        state = env.reset()
        goal = env.goal.copy() if hasattr(env, 'goal') else np.zeros(2)
        for t in range(max_steps):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            # For next_action prediction, sample a new action
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
    """Convert data dict to numpy arrays"""
    states = np.stack([d['state'] for d in data])
    actions = np.stack([d['action'] for d in data])
    goals = np.stack([d['goal'] for d in data])
    next_actions = np.stack([d['next_action'] for d in data])
    return states, actions, goals, next_actions

def plot_losses(losses):
    """Plot training losses"""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot total loss
    ax[0].plot([l['loss'] for l in losses])
    ax[0].set_title('Total Loss')
    ax[0].set_xlabel('Epoch')
    
    # Plot NLL
    ax[1].plot([l['nll'] for l in losses])
    ax[1].set_title('Negative Log Likelihood')
    ax[1].set_xlabel('Epoch')
    
    # Plot KL divergence
    ax[2].plot([l['kl'] for l in losses])
    ax[2].set_title('KL Divergence')
    ax[2].set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.show()

def main():
    # --- Config ---
    num_episodes = 200
    max_steps = 200
    latent_dim = 64
    lr = 1e-3
    kl_weight = 0.01  # KL divergence weight in ELBO loss
    batch_size = 128
    num_epochs = 20
    dataset_path = 'bayesian_dataset.pickle'
    model_path = 'bayesian_model.pth'
    
    # Split ratio for train/validation
    train_ratio = 0.8

    # --- Env & Data ---
    env = ContinuousNavigationEnv()
    logger = Logger(env)

    if os.path.exists(dataset_path):
        data = load_pickle(dataset_path)
        print(f"Loaded dataset from {dataset_path}")
    else:
        print("Collecting new dataset...")
        data = collect_data(env, num_episodes=num_episodes, max_steps=max_steps)
        save_pickle(data, dataset_path)
        print(f"Saved dataset to {dataset_path}")

    # Prepare data arrays
    states, actions, goals, next_actions = prepare_arrays(data)
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    goal_dim = goals.shape[1]
    
    # Split data into train and validation
    num_samples = states.shape[0]
    num_train = int(train_ratio * num_samples)
    
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_states = states[train_indices]
    train_actions = actions[train_indices]
    train_goals = goals[train_indices]
    train_next_actions = next_actions[train_indices]
    
    val_states = states[val_indices]
    val_actions = actions[val_indices]
    val_goals = goals[val_indices]
    val_next_actions = next_actions[val_indices]

    # --- Model ---
    agent = BayesianActionPredictor(
        state_dim, action_dim, goal_dim, 
        latent_dim=latent_dim, 
        lr=lr,
        kl_weight=kl_weight
    )

    # --- Training ---
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training
        train_indices = np.random.permutation(len(train_states))
        epoch_losses = []
        
        for i in range(0, len(train_indices), batch_size):
            idx = train_indices[i:i+batch_size]
            loss_dict = agent.train_step(
                train_states[idx], 
                train_actions[idx], 
                train_goals[idx], 
                train_next_actions[idx]
            )
            epoch_losses.append(loss_dict)
        
        # Calculate average training loss
        avg_loss = {k: np.mean([loss[k] for loss in epoch_losses]) for k in epoch_losses[0].keys()}
        train_losses.append(avg_loss)
        
        # Validation
        val_loss_dicts = []
        with torch.no_grad():
            for i in range(0, len(val_indices), batch_size):
                end = min(i + batch_size, len(val_indices))
                batch_size_actual = end - i
                val_batch_indices = np.arange(i, end)
                
                # Get validation batch
                val_batch_states = val_states[val_batch_indices]
                val_batch_actions = val_actions[val_batch_indices]
                val_batch_goals = val_goals[val_batch_indices]
                val_batch_next_actions = val_next_actions[val_batch_indices]
                
                # Forward pass to compute validation loss
                mean, std = agent.predict_next_action(val_batch_states, val_batch_actions, val_batch_goals)
                mse = ((mean - val_batch_next_actions) ** 2).mean()
                val_loss_dicts.append({'mse': mse})
        
        # Calculate average validation loss
        val_loss = np.mean([loss['mse'] for loss in val_loss_dicts])
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | " +
              f"Train Loss: {avg_loss['loss']:.6f} " +
              f"(NLL: {avg_loss['nll']:.6f}, KL: {avg_loss['kl']:.6f}) | " +
              f"Val MSE: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            agent.save(model_path)
            print(f"Saved best model with validation MSE: {val_loss:.6f}")
    
    print("Training completed!")
    
    # Plot training losses
    plot_losses(train_losses)
    
    # Plot validation MSE
    plt.figure()
    plt.plot(val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Validation MSE')
    plt.title('Validation Error')
    plt.show()

if __name__ == '__main__':
    main()
