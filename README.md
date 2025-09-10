# Risky Navigation

Reinforcement learning for navigation with obstacles

Working algorithm

```bash
./src/algorithms/A2C/train.py
```

<img width="996" height="397" alt="image" src="https://github.com/user-attachments/assets/c124e5eb-c2ca-429d-aca0-f4d5f4ea5a94" />
Hyperparamters are not optimized...


---

# AutoEncoder Next-Action Prediction Module

## Overview
This module uses an autoencoder neural network to predict the next action for an agent in a 2D navigation environment, given its current state, previous action, and goal.

## Main Components
- **Environment:** `src/env/continuous_nav_env.py` – Simulates the world, agent, obstacles, and goal.
- **AutoEncoder Agent:** `src/algorithms/AutoEncoder/agent.py` – Neural network for next-action prediction.
- **Training Script:** `src/algorithms/AutoEncoder/train.py` – Collects data, trains the autoencoder.
- **Testing Script:** `src/algorithms/AutoEncoder/test.py` – Evaluates the trained model.
- **Logger:** `src/utils/logger.py` – Plots rewards and trajectories.
- **File Management:** `src/utils/file_management.py` – Saves/loads datasets and models.

## Workflow
1. **Data Collection:**
	- The environment is run with random actions to generate (state, action, goal, next_action) samples.
	- These are saved as a dataset.
2. **Training:**
	- The autoencoder is trained to predict the next action given (state, action, goal), using MSE loss.
3. **Testing:**
	- The trained autoencoder is used to select actions in the environment, replacing random action selection.

## How to Run
```bash
# Train the autoencoder
python src/algorithms/AutoEncoder/train.py

# Test the trained autoencoder
python src/algorithms/AutoEncoder/test.py
```

## Key Notes
- The environment is used to generate synthetic data for training.
- The autoencoder is trained using supervised learning (MSE loss).
- During testing, the autoencoder replaces random action selection.
- No explicit normalization/regularization is done by default (add if needed).

## Tips
- If you see missing package errors, install them with `pip install <package-name>`.
- Review the code comments and this README for guidance.

---

For more details, see the README files in each subfolder.
