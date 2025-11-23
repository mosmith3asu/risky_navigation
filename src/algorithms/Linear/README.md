# Linear Regression for Action Prediction

Simple linear regression baseline for predicting operator actions in a 2D navigation environment.

## Technical Workflow

### Input
- **State Vector** (dim: state_dim): Current position, velocity, obstacles
- **Goal Vector** (dim: goal_dim): Target position
- **Combined Input**: `[state, goal]` concatenated (dim: state_dim + goal_dim)

### Processing
1. **Linear Transformation**: Single fully-connected layer
   ```python
   action = W * [state, goal] + b
   ```
   - `W`: Weight matrix (action_dim × input_dim)
   - `b`: Bias vector (action_dim)

2. **Training**: Supervised learning with MSE loss
   ```python
   loss = MSE(predicted_action, expert_action)
   ```

### Output
- **Action Vector** (dim: action_dim): Predicted action (throttle, steering)
- **Type**: Deterministic point prediction (no uncertainty)

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