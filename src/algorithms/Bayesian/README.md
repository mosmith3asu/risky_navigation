# Bayesian Next-Action Prediction

This module uses Bayesian neural networks to predict the next action for an agent in a 2D navigation environment, with uncertainty quantification.

## Overview

Unlike deterministic models (like the AutoEncoder), Bayesian neural networks provide not just predictions but also uncertainty estimates. This is particularly useful in risky navigation scenarios where knowing when the model is uncertain can help avoid potentially dangerous actions.

## Files
- `agent.py`: Defines the Bayesian neural network for next-action prediction.
- `train.py`: Collects data from the environment and trains the Bayesian model.
- `test.py`: Evaluates the trained model, visualizing predictions with uncertainty.
- `__init__.py`: Marks this as a Python package.

## How it Works

1. **Data Collection:**
   - Similar to the AutoEncoder approach, we collect (state, action, goal, next_action) tuples.
   
2. **Bayesian Neural Network:**
   - Instead of regular neural network layers with fixed weights, we use Bayesian layers.
   - Each weight is represented by a probability distribution rather than a single value.
   - During training and inference, we sample from these distributions to get the weights.
   
3. **Training:**
   - We train the model using variational inference, which approximates the true Bayesian posterior.
   - The loss function is the Evidence Lower Bound (ELBO), which combines:
     - Negative log likelihood (how well the model fits the data)
     - KL divergence (how much the learned distributions differ from the priors)
   
4. **Prediction:**
   - For each input, we sample from the weight distributions multiple times.
   - This gives us multiple different predictions, which we use to compute:
     - The mean (our best guess at the next action)
     - The standard deviation (our uncertainty about that guess)

## Analogy: The Cautious Navigator

Imagine a navigator who doesn't just tell you "turn right here," but says "I'm 80% confident you should turn right, but there's a 20% chance you should go straight."

- **Regular AutoEncoder:** Like a navigator who always gives you a single direction, even when unsure.
- **Bayesian Approach:** Like a navigator who tells you both the most likely direction AND how confident they are about it.

## Uncertainty in Action

The uncertainty information provided by the Bayesian approach is valuable in several ways:

1. **Safety:** When the model is uncertain, it can lead to more cautious behavior.
2. **Exploration:** High uncertainty areas can be targeted for additional data collection.
3. **Robustness:** The model can avoid making overconfident predictions in unfamiliar situations.

## Hyperparameter Tuning

Key parameters to tune include:

- **Latent Dimension (`latent_dim`)**: Size of the bottleneck.
- **Learning Rate (`lr`)**: Speed of optimization.
- **KL Weight (`kl_weight`)**: How much to penalize complex distributions (higher = simpler model).
- **Number of Samples (`n_samples`)**: How many times to sample for uncertainty estimation (higher = more accurate uncertainty).

## Usage

```python
# Train the Bayesian model
python src/algorithms/Bayesian/train.py

# Test the model with uncertainty visualization
python src/algorithms/Bayesian/test.py
```

## Bayesian vs. AutoEncoder Approach

| Feature | AutoEncoder | Bayesian |
|---------|-------------|----------|
| Output | Point prediction | Mean prediction + uncertainty |
| Training | Standard backprop | Variational inference |
| Complexity | Lower | Higher |
| Computation | Faster | Slower |
| Use case | When you need speed & simplicity | When you need uncertainty & robustness |

## References

- Blundell, C., et al. (2015). Weight Uncertainty in Neural Networks.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.