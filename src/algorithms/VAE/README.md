# Variational AutoEncoder (VAE) for Next-Action Prediction

## Overview

The Variational AutoEncoder (VAE) is a generative model that learns a probabilistic latent representation of the state-action-goal space to predict next actions. Unlike deterministic approaches, the VAE provides uncertainty estimates and can generate diverse action predictions.

## Architecture

### Core Components

1. **Encoder Network** (`VAEEncoder`)
   - Maps input (state + current_action + goal) to latent distribution parameters
   - Outputs mean (μ) and log-variance (log σ²) for latent variables
   - Architecture: FC layers with ReLU activations

2. **Decoder Network** (`VAEDecoder`)
   - Maps latent samples back to action space
   - Reconstructs next actions from latent representation
   - Architecture: FC layers with ReLU activations

3. **Reparameterization Trick**
   - Enables gradient flow through stochastic sampling
   - z = μ + ε × σ, where ε ~ N(0,1)

## Mathematical Foundation

### Loss Function
The VAE optimizes a combination of reconstruction and regularization losses:

```
Total Loss = Reconstruction Loss + β × KL Divergence

Reconstruction Loss = MSE(predicted_action, ground_truth_action)
KL Loss = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
``` 

### Beta-VAE
The β parameter controls the trade-off between reconstruction quality and latent space regularization:
- β = 1: Standard VAE
- β > 1: Emphasizes disentangled representations
- β < 1: Prioritizes reconstruction accuracy

## Key Features

### 1. Uncertainty Quantification
```python
# Get prediction with uncertainty
mean_action, std_action = agent.predict_next_action(state, action, goal, n_samples=10)
```

### 2. Generative Capabilities
```python
# Sample diverse actions from learned prior
random_actions = agent.sample_from_prior(n_samples=100)
```

### 3. Latent Space Analysis
```python
# Analyze latent representations
mu, logvar = agent.get_latent_representation(states, actions, goals)
```

## Comparison with Other Approaches

| Aspect | VAE | AutoEncoder | Bayesian | Linear |
|--------|-----|-------------|----------|--------|
| **Probabilistic** | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| **Uncertainty** | ✅ High | ❌ None | ✅ High | ❌ None |
| **Generative** | ✅ Yes | ❌ No | ❌ Limited | ❌ No |
| **Interpretability** | 🔶 Medium | 🔶 Medium | 🔶 Medium | ✅ High |
| **Training Speed** | 🔶 Medium | ✅ Fast | ❌ Slow | ✅ Fast |
| **Memory Usage** | 🔶 Medium | 🔶 Medium | ❌ High | ✅ Low |

## Advantages

1. **Probabilistic Predictions**: Provides uncertainty estimates for decision-making
2. **Generative Modeling**: Can sample diverse, plausible actions
3. **Latent Structure**: Learns meaningful representations of the action space
4. **Controllable Diversity**: β parameter allows tuning exploration vs exploitation
5. **Anomaly Detection**: Can identify unusual state-action combinations

## Limitations

1. **Training Complexity**: Requires careful tuning of β parameter
2. **Computational Overhead**: Multiple samples needed for uncertainty estimation
3. **Mode Collapse**: May fail to capture full action distribution
4. **Posterior Collapse**: Latent variables may become uninformative
5. **Hyperparameter Sensitivity**: Performance depends on architecture choices

## Use Cases

### When to Use VAE:
- **Exploration Tasks**: When diverse action generation is beneficial
- **Uncertainty-Critical Domains**: When knowing prediction confidence is important
- **Safety Applications**: When conservative actions based on uncertainty are needed
- **Data Analysis**: When understanding action space structure is valuable

### When NOT to Use VAE:
- **Real-time Systems**: When computational efficiency is critical
- **Deterministic Tasks**: When single optimal actions are sufficient
- **Limited Data**: When insufficient data for learning latent distributions
- **Simple Domains**: When linear relationships dominate

## Implementation Details

### Model Architecture
```python
Input Dim: state_dim + action_dim + goal_dim
Hidden Layers: 128 → 128 → 128
Latent Dim: 32 (configurable)
Output Dim: action_dim
```

### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Beta (KL weight)**: 1.0
- **Batch Size**: 64
- **Epochs**: 50

### Key Hyperparameters
- `latent_dim`: Controls representation capacity
- `beta`: Balances reconstruction vs regularization
- `n_samples`: Number of samples for uncertainty estimation
- `hidden_dim`: Network capacity

## Performance Characteristics

### Computational Complexity
- **Training**: O(n × hidden_dim²) per sample
- **Inference**: O(latent_dim × hidden_dim) × n_samples
- **Memory**: ~37K parameters (default config)

### Expected Performance
- **Prediction Accuracy**: Moderate (focuses on diversity over precision)
- **Uncertainty Calibration**: Good (well-calibrated confidence estimates)
- **Generation Quality**: High (diverse, realistic actions)
- **Training Stability**: Good (with proper β tuning)

## Usage Examples

### Basic Training
```python
from src.algorithms.VAE.agent import VAEAgent

agent = VAEAgent(
    state_dim=8, action_dim=2, goal_dim=2,
    latent_dim=32, beta=1.0, device='cuda'
)

# Train on batch
loss = agent.train_step(states, actions, goals, next_actions)
```

### Prediction with Uncertainty
```python
# Get multiple samples for uncertainty estimation
mean_pred, std_pred = agent.predict_next_action(
    state, action, goal, n_samples=20
)

# Use uncertainty for action selection
if np.mean(std_pred) > threshold:
    action = conservative_action()
else:
    action = mean_pred
```

### Latent Space Exploration
```python
# Analyze learned representations
mu, logvar = agent.get_latent_representation(states, actions, goals)

# Generate diverse actions
diverse_actions = agent.sample_from_prior(n_samples=100)
```

## Research Applications

The VAE approach is particularly valuable for:

1. **Behavioral Cloning**: Learning diverse policies from demonstrations
2. **Exploration Strategies**: Generating diverse actions for environment exploration
3. **Safety Research**: Understanding prediction uncertainty for safe navigation
4. **Policy Regularization**: Preventing overfitting to specific action patterns

## Future Improvements

Potential enhancements include:
- **Conditional VAE**: Incorporating additional context variables
- **Hierarchical VAE**: Multi-level action representations
- **Normalizing Flows**: More flexible posterior distributions
- **Adversarial Training**: Improving generation quality with GANs

## References

1. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes.
2. Higgins, I., et al. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework.
3. Sohn, K., et al. (2015). Learning structured output representation using deep conditional generative models.