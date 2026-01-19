# Bayesian Neural Network for Temporal Action Prediction

Bayesian neural network for temporal action prediction with epistemic uncertainty quantification.

## Problem

Predict operator actions with uncertainty estimates using temporal history:
- **Input**: State sequence, action sequence, goal
- **Output**: Action prediction + uncertainty
- **Use case**: Robot knows when predictions are unreliable during delays

## Technical Workflow

### Input
- **State Sequence**: History of states `[state_{t-n}, ..., state_t]`
  - Each state: Position, velocity, heading, obstacle distances
- **Action Sequence**: History of actions `[action_{t-n}, ..., action_{t-1}]`
- **Goal Vector**: Target position
- **sequence_len**: Controls temporal history length

### Processing
```python
# Flatten sequences
state_flat = state_sequence.reshape(batch, -1)
action_flat = action_sequence.reshape(batch, -1)
input = concat([state_flat, action_flat, goal])

# Weights are distributions, not fixed values
for each layer:
    weight ~ N(μ_w, σ_w²)
    bias ~ N(μ_b, σ_b²)

action_t = bayesian_network(input)
```

Bayesian layers use weight distributions to capture model uncertainty over temporal patterns.

### Training
- **Data**: Expert demonstrations from visibility graph
- **Sequences**: Created using sliding window over episodes
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

- `sequence_len`: Temporal history length ⚠️ **Critical** (default: 5)
  - Controls how many previous (state, action) pairs are used
  - `sequence_len=1`: No temporal modeling
  - `sequence_len>1`: Temporal sequence modeling with uncertainty
- `hidden_dim`: Hidden layer size (default: 128, range: [64, 128, 256])
- `prior_std`: Prior standard deviation for weight distributions (default: 1.0, range: [0.5, 1.0, 2.0])
- `kl_weight`: Weight for KL divergence term (default: 1e-5, range: [1e-5, 1e-4, 1e-3])
- `lr`: Learning rate (default: 1e-3, range: [1e-3, 5e-4, 1e-4])
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