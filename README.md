# Risky Navigation

Reinforcement learning for navigation with obstacles

Working algorithm

```bash
./src/algorithms/A2C/train.py
```

<img width="996" height="397" alt="image" src="https://github.com/user-attachments/assets/c124e5eb-c2ca-429d-aca0-f4d5f4ea5a94" />
Hyperparamters are not optimized...


---

# Neural Network Approaches for Action Prediction

## Overview
This project implements multiple neural network architectures for predicting operator actions in a 2D navigation environment with communication delays. The goal is to predict actions based on current state and goal position - critical for space teleoperation scenarios with signal blackouts.

## Main Components
- **Environment:** `src/env/continuous_nav_env.py` – Simulates the world, agent, obstacles, and goal.
- **Algorithms:**
  - **Linear Agent:** `src/algorithms/Linear/agent.py` – Simple linear regression baseline
  - **Bayesian Agent:** `src/algorithms/Bayesian/agent.py` – Probabilistic neural network with uncertainty estimates
  - **VAE Agent:** `src/algorithms/VAE/agent.py` – Variational autoencoder for action prediction
  - **Transformer Agent:** `src/algorithms/Transformer/agent.py` – Self-attention over temporal sequences
- **Logger:** `src/utils/logger.py` – Plots rewards and trajectories.
- **File Management:** `src/utils/file_management.py` – Saves/loads datasets and models.

## Workflow
1. **Data Collection:**
	- The environment is run with an expert policy (visibility graph) to generate (state, action, goal) samples.
	- These are saved as a dataset for behavioral cloning.
2. **Training:**
	- Each algorithm is trained to predict actions given states and goals, using MSE loss (+ KL divergence for VAE/Bayesian).
	- Grid search optimizes hyperparameters for each architecture.
3. **Testing:**
	- Trained models are evaluated in the environment using predicted actions.
	- Success rate, average reward, and prediction accuracy are measured.

## How to Run
```bash
# Run the complete algorithm comparison notebook
jupyter notebook algorithm_comparison.ipynb

# Or train individual algorithms
python src/algorithms/Linear/train.py
python src/algorithms/Bayesian/train.py
python src/algorithms/VAE/train.py
python src/algorithms/Transformer/train.py
```

## Key Notes
- All models use supervised learning (behavioral cloning) for action prediction
- Bayesian and VAE provide uncertainty estimates critical for safe autonomous operation
- Transformer supports temporal sequences for modeling action history
- The environment generates synthetic expert demonstrations for training

## Research Context (FURI Project)
This work addresses NASA's communication delay challenges in space exploration:
- Predict operator actions during signal blackouts (Mars: 22 min delay)
- Enable rovers to continue productive work autonomously
- Maintain operator risk preferences (future: CPT integration)
- Critical for deep-space missions with extended communication delays

## Tips
- If you see missing package errors, install them with `pip install <package-name>`.
- Review the code comments and algorithm-specific READMEs for guidance.
- Grid search configs are in `algorithm_comparison.ipynb`

---

For more details, see the README files in each subfolder.
