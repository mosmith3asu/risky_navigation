# continuous_nav_env.py

This file defines the `ContinuousNavigationEnv` class, which simulates a 2D navigation environment for an agent (like an RC car).

## Key Features
- **State:** Includes position (x, y), heading, velocity, distance to goal, and distance to nearest obstacle.
- **Action:** Throttle (speed) and steering angle (direction).
- **step(action):** Updates the agent's state, checks for collisions, computes reward, and returns (next_state, reward, done, info).
- **reset():** Starts a new episode.
- **render():** Visualizes the environment, obstacles, goal, and agent.

## Usage
- Used for both data collection (with random actions) and for evaluating trained agents (with learned actions).
- Returns all necessary information for reinforcement learning or imitation learning tasks.

See the main README for workflow and integration details.
