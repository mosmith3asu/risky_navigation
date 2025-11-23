# Bayesian Neural Network for Action Prediction

Bayesian neural network for action prediction with uncertainty quantification.

## Technical Workflow

### Input
- **State Vector** (dim: state_dim): Current position, velocity, obstacles
- **Goal Vector** (dim: goal_dim): Target position  
- **Combined Input**: `[state, goal]` concatenated (dim: state_dim + goal_dim)

### Processing
1. **Bayesian Layers**: Neural network with probabilistic weights
   - Each weight is a distribution (mean μ, std σ) instead of fixed value
   - Sample weights from distribution during forward pass
   
2. **Multiple Forward Passes**: Sample N times to get distribution of predictions
   ```python
   for i in range(n_samples):
       weights_i = sample(weight_distribution)
       action_i = forward(state, weights_i)
   ```

3. **Training**: Variational inference with ELBO loss
   ```python
   loss = -log_likelihood(action|state) + KL(q(w)||p(w))
   ```
   - First term: how well model fits data
   - Second term: regularization (how much distributions differ from priors)

### Output
- **Mean Action** (dim: action_dim): Average of sampled predictions
- **Uncertainty** (dim: action_dim): Standard deviation of sampled predictions
- **Type**: Probabilistic prediction with epistemic uncertainty

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