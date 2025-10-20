# A2C (Advantage Actor-Critic) Module

This folder contains implementations of the A2C reinforcement learning algorithm for navigation tasks.

## Files
- `agent.py`, `agent_BU.py`, `agent_BU_WORKING.py`: Define A2C agent classes, including actor and critic networks, and training logic.
- `train.py`, `train_vec.py`: Scripts to train the A2C agent in the navigation environment. `train_vec.py` supports vectorized (parallel) environments for faster training.
- `__init__.py`: Marks this as a Python package.

## How A2C Works
- **Actor:** Learns a policy to select actions that maximize expected reward.
- **Critic:** Estimates the value (expected future reward) of each state.
- **On-policy:** Learns from actions taken by the current policy.
- **No replay buffer:** Updates are made using the most recent data.
- **Parallelism:** Can run multiple environments in parallel for more stable and efficient learning.


---

## Analogy: Student and Teacher

Imagine a student (the actor) learning to play a game, and a teacher (the critic) giving feedback after each move:

- **Student (Actor):** Decides what move to make at each step.
- **Teacher (Critic):** Watches and gives a score for each move, based on how good it is for winning the game.
- If the move is good, the teacher encourages the student to do it again. If not, the teacher discourages it.
- Over time, the student learns to make better moves, and the teacher gets better at scoring situations.

In A2C, the actor and critic are neural networks trained together: the actor to maximize rewards, the critic to accurately evaluate states.

---

## Workflow
1. The agent observes the current state.
2. The actor network selects an action.
3. The environment returns the next state and reward.
4. The critic network evaluates the state.
5. The agent updates both networks to improve future decisions.

## Usage
- Run `train.py` or `train_vec.py` to train the A2C agent.
- Review the code and comments for more details on hyperparameters and customization.

See the main project README for a comparison with other algorithms.
