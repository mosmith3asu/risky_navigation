# DDPG (Deep Deterministic Policy Gradient) Module

This folder contains implementations of the DDPG reinforcement learning algorithm for continuous control tasks.

## Files
- `ddpg_agent.py`: Defines the DDPG agent, including actor/critic networks, replay buffer, and update logic.
- `ddpg_train.py`: Script to train the DDPG agent in the navigation environment.
- `models/`: Stores trained model weights and pickled files for saving/loading agent state.
- `__init__.py`: Marks this as a Python package.

## How DDPG Works
- **Actor:** Outputs deterministic continuous actions for a given state.
- **Critic:** Estimates the Q-value (expected return) for state-action pairs.
- **Replay Buffer:** Stores past experiences for off-policy learning.
- **Target Networks:** Stabilize learning by slowly updating copies of the actor and critic.
- **Off-policy:** Learns from experiences sampled from the replay buffer, not just the current policy.
- **Exploration:** Adds noise to actions for exploration.


---

## Analogy: Race Car Driver, Coach, and Replay Camera

Imagine a race car driver (the actor) who must decide exactly how much to turn the wheel and press the gas at every moment (continuous actions):

- **Driver (Actor):** Makes decisions on steering and throttle.
- **Coach (Critic):** Watches and scores each maneuver, telling the driver how good it was for getting around the track.
- **Replay Camera (Replay Buffer):** Records all past races. The driver and coach can review and learn from any past moment, not just the latest lap.
- **Shadow Team (Target Networks):** A second, slower-moving driver and coach provide stable advice, so learning is less shaky.

In DDPG, the actor and critic learn from both current and past experiences, with extra stability from target networks, just like a driver and coach reviewing race footage and getting advice from a shadow team.

---

## Workflow
1. The agent observes the current state.
2. The actor network outputs a continuous action.
3. The environment returns the next state and reward.
4. The experience is stored in the replay buffer.
5. The agent samples random batches from the buffer to update the actor and critic networks.
6. Target networks are updated slowly for stability.

## Usage
- Run `ddpg_train.py` to train the DDPG agent.
- Review the code and comments for more details on hyperparameters and customization.

See the main project README for a comparison with other algorithms.
