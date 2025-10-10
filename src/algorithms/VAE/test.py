#!/usr/bin/env python3
"""
Test script for Variational AutoEncoder (VAE) agent.

This script demonstrates how to test and analyze a trained VAE agent.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.algorithms.VAE.agent import VAEAgent
from src.utils.file_management import load_pickle

def test_prediction_accuracy(agent, env, num_episodes=50):
    """Test the prediction accuracy of the VAE agent."""
    print("Testing prediction accuracy...")
    
    prediction_errors = []
    uncertainties = []
    
    for episode in range(num_episodes):
        state = env.reset()
        goal = env.goal.copy() if hasattr(env, 'goal') else np.zeros(2)
        
        for step in range(50):  # Test for 50 steps per episode
            action = env.action_space.sample()
            
            # Get prediction with uncertainty
            pred_next_action, pred_std = agent.predict_next_action(state, action, goal)
            
            # Ground truth next action (random for this test)
            ground_truth_next = env.action_space.sample()
            
            # Calculate error
            error = np.mean((pred_next_action - ground_truth_next)**2)
            uncertainty = np.mean(pred_std)
            
            prediction_errors.append(error)
            uncertainties.append(uncertainty)
            
            # Step environment
            next_state, _, done, _ = env.step(action)
            state = next_state
            
            if done:
                break
    
    avg_error = np.mean(prediction_errors)
    avg_uncertainty = np.mean(uncertainties)
    
    print(f"Prediction Test Results:")
    print(f"  Average MSE: {avg_error:.6f}")
    print(f"  Average Uncertainty: {avg_uncertainty:.6f}")
    print(f"  Error Std: {np.std(prediction_errors):.6f}")
    print(f"  Uncertainty Std: {np.std(uncertainties):.6f}")
    
    return prediction_errors, uncertainties

def test_latent_space_coverage(agent, env, num_samples=1000):
    """Test the latent space coverage and diversity."""
    print("Testing latent space coverage...")
    
    states = []
    actions = []
    goals = []
    latent_mus = []
    latent_logvars = []
    
    # Collect diverse samples
    for _ in range(num_samples):
        state = env.reset()
        goal = env.goal.copy() if hasattr(env, 'goal') else np.zeros(2)
        action = env.action_space.sample()
        
        # Get latent representation
        mu, logvar = agent.get_latent_representation(
            state.reshape(1, -1), action.reshape(1, -1), goal.reshape(1, -1)
        )
        
        states.append(state)
        actions.append(action)
        goals.append(goal)
        latent_mus.append(mu[0])
        latent_logvars.append(logvar[0])
    
    latent_mus = np.array(latent_mus)
    latent_logvars = np.array(latent_logvars)
    
    # Analyze latent space
    latent_std = np.mean(np.exp(0.5 * latent_logvars))
    latent_diversity = np.std(latent_mus, axis=0).mean()
    
    print(f"Latent Space Analysis:")
    print(f"  Average latent std: {latent_std:.6f}")
    print(f"  Latent diversity: {latent_diversity:.6f}")
    print(f"  Latent dimensions with high variance: {np.sum(np.std(latent_mus, axis=0) > 0.5)}")
    
    return latent_mus, latent_logvars

def test_generation_quality(agent, num_samples=100):
    """Test the quality of generated actions from the prior."""
    print("Testing generation quality...")
    
    # Sample from prior
    generated_actions = agent.sample_from_prior(num_samples)
    
    # Analyze generated actions
    action_mean = np.mean(generated_actions, axis=0)
    action_std = np.std(generated_actions, axis=0)
    action_range = np.ptp(generated_actions, axis=0)  # Peak-to-peak range
    
    print(f"Generated Actions Analysis:")
    print(f"  Mean: {action_mean}")
    print(f"  Std: {action_std}")
    print(f"  Range: {action_range}")
    
    return generated_actions

def visualize_latent_space(latent_mus, save_path=None):
    """Visualize the latent space using dimensionality reduction."""
    print("Visualizing latent space...")
    
    if latent_mus.shape[1] > 2:
        # Use PCA first to reduce to reasonable dimensions, then t-SNE
        if latent_mus.shape[1] > 50:
            pca = PCA(n_components=50)
            latent_reduced = pca.fit_transform(latent_mus)
        else:
            latent_reduced = latent_mus
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latent_2d = tsne.fit_transform(latent_reduced)
    else:
        latent_2d = latent_mus
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=10)
    plt.title('Latent Space Visualization (t-SNE)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent space visualization saved to {save_path}")
    
    plt.show()

def compare_with_deterministic(agent, env, num_episodes=20):
    """Compare VAE performance with deterministic prediction."""
    print("Comparing with deterministic prediction...")
    
    vae_rewards = []
    deterministic_rewards = []
    
    for episode in range(num_episodes):
        # Test VAE
        state = env.reset()
        goal = env.goal.copy() if hasattr(env, 'goal') else np.zeros(2)
        vae_reward = 0.0
        
        for step in range(100):
            if step == 0:
                action = env.action_space.sample()
            else:
                pred_action, _ = agent.predict_next_action(state, prev_action, goal)
                action = np.clip(pred_action, env.action_space.low, env.action_space.high)
            
            next_state, reward, done, _ = env.step(action)
            vae_reward += reward
            prev_action = action
            state = next_state
            
            if done:
                break
        
        vae_rewards.append(vae_reward)
        
        # Test deterministic (random) baseline
        state = env.reset()
        deterministic_reward = 0.0
        
        for step in range(100):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            deterministic_reward += reward
            state = next_state
            
            if done:
                break
        
        deterministic_rewards.append(deterministic_reward)
    
    vae_avg = np.mean(vae_rewards)
    det_avg = np.mean(deterministic_rewards)
    
    print(f"Performance Comparison:")
    print(f"  VAE Average Reward: {vae_avg:.3f} ± {np.std(vae_rewards):.3f}")
    print(f"  Random Average Reward: {det_avg:.3f} ± {np.std(deterministic_rewards):.3f}")
    print(f"  Improvement: {((vae_avg - det_avg) / abs(det_avg) * 100):.1f}%")
    
    return vae_rewards, deterministic_rewards

def main():
    """Main testing function."""
    print("VAE Agent Testing")
    print("=" * 50)
    
    # Check if model exists
    model_path = 'vae_agent_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please run train.py first to train the model.")
        return
    
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
    
    # Initialize and load agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = VAEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        latent_dim=32,  # Should match training config
        hidden_dim=128,
        lr=1e-3,
        beta=1.0,
        device=device
    )
    
    agent.load(model_path)
    print(f"Model loaded from {model_path}")
    print()
    
    # Run tests
    print("Running comprehensive tests...")
    print()
    
    # Test 1: Prediction accuracy
    prediction_errors, uncertainties = test_prediction_accuracy(agent, env)
    print()
    
    # Test 2: Latent space coverage
    latent_mus, latent_logvars = test_latent_space_coverage(agent, env)
    print()
    
    # Test 3: Generation quality
    generated_actions = test_generation_quality(agent)
    print()
    
    # Test 4: Performance comparison
    vae_rewards, det_rewards = compare_with_deterministic(agent, env)
    print()
    
    # Visualizations
    print("Creating visualizations...")
    
    # Visualize latent space
    visualize_latent_space(latent_mus, 'vae_latent_space.png')
    
    # Plot uncertainty vs error correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(uncertainties, prediction_errors, alpha=0.6, s=10)
    plt.xlabel('Prediction Uncertainty')
    plt.ylabel('Prediction Error (MSE)')
    plt.title('Uncertainty vs Prediction Error')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(uncertainties, prediction_errors)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig('vae_uncertainty_vs_error.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot generated actions distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(generated_actions[:, 0], bins=30, alpha=0.7, density=True)
    plt.title('Generated Action Dimension 1')
    plt.xlabel('Action Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(generated_actions[:, 1], bins=30, alpha=0.7, density=True)
    plt.title('Generated Action Dimension 2')
    plt.xlabel('Action Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vae_generated_actions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Testing completed successfully!")
    print("Check the generated plots for detailed analysis.")

if __name__ == "__main__":
    main()