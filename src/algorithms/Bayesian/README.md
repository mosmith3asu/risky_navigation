# Bayesian Neural Network for Action Prediction

Bayesian neural network for temporal action prediction with epistemic uncertainty quantification.

## Problem

Predict operator actions with uncertainty estimates for safe teleoperation:
- **Input**: Current state, previous action, goal
- **Output**: Action prediction + uncertainty
- **Use case**: Robot knows when predictions are unreliable during delays

## Technical Workflow

### Input
- **State Vector**: Position, velocity, heading, obstacle distances
- **Previous Action**: Action from previous timestep
- **Goal Vector**: Target position

### Processing
```python
# Weights are distributions, not fixed values
for each layer:
    weight ~ N(μ_w, σ_w²)
    bias ~ N(μ_b, σ_b²)

action_t = network([state_t, action_{t-1}, goal])
```

Bayesian layers use weight distributions to capture model uncertainty.

### Training
- **Data**: Expert demonstrations from visibility graph
- **Loss**: ELBO = Reconstruction loss + KL divergence
- **Output**: Mean prediction (deterministic mode)

### Output
- **Action**: (throttle, steering)
- **Uncertainty**: Epistemic uncertainty from weight distributions

## Model Architecture

```python
class BayesianNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = BayesianLinear(state_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)
    
    def predict_with_uncertainty(self, state, n_samples=10):
        predictions = [self.forward(state) for _ in range(n_samples)]
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        return mean, std
```

## Hyperparameters

- `hidden_dim`: Hidden layer size (default: 64)
- `lr`: Learning rate (default: 1e-3)
- `kl_weight`: Weight for KL divergence term (default: 0.01)
- `n_samples`: Number of samples for uncertainty estimation (default: 10)
- `batch_size`: Batch size for training (default: 128)
- `num_epochs`: Number of training epochs (default: 50)

## Usage

```bash
# Train
python -m src.algorithms.Bayesian.train

# Test with uncertainty visualization
python -m src.algorithms.Bayesian.test
```