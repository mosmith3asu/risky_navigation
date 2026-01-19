# Linear Regression for Temporal Action Prediction

Simple linear baseline for temporal action prediction in teleoperation with communication delays.

## Problem

Predict operator actions using temporal history to compensate for communication delays:
- **Input**: State sequence, action sequence, goal
- **Output**: Next action prediction
- **Use case**: Robot predicts delayed human action using temporal patterns

## Technical Workflow

### Input
- **State Sequence**: History of states `[state_{t-n}, ..., state_t]`
  - Each state: Position, velocity, heading, obstacle distances
- **Action Sequence**: History of actions `[action_{t-n}, ..., action_{t-1}]`
- **Goal Vector**: Target position
- **sequence_len**: Controls temporal history length
  - `sequence_len=1`: Single previous state/action (baseline)
  - `sequence_len>1`: Temporal sequences for history-based prediction

### Processing
```python
# Flatten sequences into single vector
state_flat = state_sequence.reshape(batch, -1)  # (batch, seq_len*state_dim)
action_flat = action_sequence.reshape(batch, -1)  # (batch, seq_len*action_dim)
input = concat([state_flat, action_flat, goal])  # (batch, seq_len*(state+action)+goal)

action_t = W * input + b
```

Single linear layer maps flattened temporal context to action space.

### Training
- **Data**: Expert demonstrations from visibility graph optimal policy
- **Sequences**: Created using sliding window over episodes
- **Loss**: MSE between predicted and expert actions

### Output
- **Action**: (throttle, steering) for RC car dynamics
- **Type**: Deterministic prediction

## Model Architecture

```python
class LinearModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear = nn.Linear(state_dim, action_dim)
    
    def forward(self, state):
        return self.linear(state)
```

## Hyperparameters

- `sequence_len`: Temporal history length ⚠️ **Critical** (default: 5)
  - Controls how many previous (state, action) pairs are used
  - `sequence_len=1`: No temporal modeling
  - `sequence_len>1`: Temporal sequence modeling
- `lr`: Learning rate (default: 1e-3)
- `batch_size`: Batch size for training (default: 128)
- `num_epochs`: Number of training epochs (default: 20)

## Usage

```bash
# Train
python -m src.algorithms.Linear.train

# Test
python -m src.algorithms.Linear.test
```