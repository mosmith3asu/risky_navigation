# Linear Regression for Action Prediction

Simple linear baseline for temporal action prediction in teleoperation with communication delays.

## Problem

Predict operator actions to compensate for communication delays:
- **Input**: Current state, previous action, goal
- **Output**: Current action prediction
- **Use case**: Robot predicts delayed human action to continue safe navigation

## Technical Workflow

### Input
- **State Vector**: Position, velocity, heading, obstacle distances
- **Previous Action**: Action from previous timestep (autoregressive)
- **Goal Vector**: Target position

### Processing
```python
action_t = W * [state_t, action_{t-1}, goal] + b
```

Single linear layer maps concatenated input to action space.

### Training
- **Data**: Expert demonstrations from visibility graph optimal policy
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