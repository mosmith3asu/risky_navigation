# MLP (Multi-Layer Perceptron) Module

**Note:** This module was previously named "AutoEncoder" but is NOT a true autoencoder. It's an MLP with a bottleneck architecture that predicts actions from states, not reconstructs inputs.

This folder contains the code for the MLP-based action prediction system.

## Files
- `agent.py`: Defines the MLP neural network with bottleneck architecture and training logic.
- `train.py`: Collects data from the environment and trains the MLP.
- `test.py`: Evaluates the trained MLP in the environment.

## How it Works
1. **Data Collection:**
   - The environment is run with an expert policy (visibility graph).
   - For each step, (state, goal, action) is recorded for behavioral cloning.
2. **Training:**
   - The MLP is trained to predict actions given (state, goal).
   - Loss is computed using Mean Squared Error (MSE) between predicted and actual actions.
3. **Testing:**
   - The trained MLP is used to select actions in the environment.
   - Performance is visualized using the logger.


## How the MLP Predicts Actions (NOT Reconstruction)

**Important Distinction:** Unlike a true autoencoder that reconstructs its input, this MLP predicts actions from states.

1. **Input Construction:**
   - The MLP receives the current state and goal, concatenated into a single input vector.

2. **Encoder (Feature Extraction):**
   - A series of neural network layers compress the input into a lower-dimensional latent representation (bottleneck).
   - This captures the most important features for action prediction.

3. **Decoder (Action Generation):**
   - Another set of neural network layers expands the latent representation into a prediction for the action.

4. **Prediction:** 
   - The output is the predicted action (e.g., throttle and steering values).

5. **Learning:**
   - During training, the model's prediction is compared to the expert action from the dataset using Mean Squared Error (MSE) loss.
   - The model's weights are updated via backpropagation to minimize this loss.

**Summary:**
- The MLP learns a mapping: `(state, goal) → action`
- It's a feedforward network for supervised learning (behavioral cloning)
- The bottleneck architecture compresses information, but the task is action prediction, not input reconstruction

**Diagram:**

```
state      goal
  |         |
  +----+----+
       |
 [Concatenate]
       |
   [Encoder]
   (128 → 64)
       |
   [Latent]
    (bottleneck)
       |
   [Decoder]
   (64 → 128)
       |
  Predicted action
  (throttle, steering)
```


---

## Analogy: How the MLP Works in This Project

Think of the MLP as a skilled operator learning from an expert:

- **Training Data:** The expert's demonstrations showing (state, goal) → action pairs.
- **Encoder (Pattern Recognition):** Identifies key features in the current situation (position, obstacles, goal direction).
- **Bottleneck (Compressed Understanding):** A compact representation of "what matters most" for decision-making.
- **Decoder (Action Generation):** Translates the understanding into specific control commands (throttle, steering).
- **Goal:** The predicted action should match the expert's action as closely as possible.

**Why NOT a True Autoencoder?**
- A true autoencoder reconstructs its input: `input → latent → reconstructed_input`
- This MLP predicts actions: `(state, goal) → latent → action`
- The architecture has encoder/decoder components, but the task is supervised prediction, not unsupervised reconstruction


## Hyperparameter Tuning for MLPs

Tuning hyperparameters is key to getting good results with neural networks. Here's how to do it in this codebase:

### What to Tune

- **Latent Dimension (`latent_dim`)**: Size of the bottleneck. Too small = underfitting, too large = overfitting.
- **Hidden Dimensions (`hidden_dims`)**: Number of units in each layer. More capacity = better fit, but risk of overfitting.
- **Dropout (`dropout`)**: Regularization to prevent overfitting. Range: 0.0 - 0.5.
- **Learning Rate (`lr`)**: How fast the model learns. Too high = unstable, too low = slow.
- **Batch Size (`batch_size`)**: Number of samples per training step. Larger = smoother, smaller = noisier.
- **Number of Epochs (`num_epochs`)**: How many times the model sees the dataset. Too few = underfit, too many = overfit.

### Where to Change

- Use the grid search in `algorithm_comparison.ipynb`:
   ```python
   GRID_SEARCH_CONFIGS = {
       'MLP': {
           'latent_dim': [16, 32, 64],
           'hidden_dims': [[128, 64], [256, 128], [256, 128, 64]],
           'dropout': [0.0, 0.1, 0.2],
           'lr': [1e-3, 5e-4, 1e-4]
       }
   }
   ```
- Or edit the model architecture in `agent.py` for custom changes.

### How to Tune

1. Change one parameter at a time and observe the effect on training loss and performance.
2. Keep a log of what you try and the results.
3. Use validation data if possible to check for overfitting.
4. For advanced tuning, consider using libraries like Optuna or Ray Tune.

### Typical Values

| Hyperparameter | Where to Change         | Typical Values         |
|----------------|------------------------|-----------------------|
| latent_dim     | `train.py`, `agent.py` | 16, 32, 64, 128       |
| lr             | `train.py`             | 1e-4, 1e-3, 1e-2      |
| batch_size     | `train.py`             | 32, 64, 128, 256      |
| num_epochs     | `train.py`             | 10, 20, 50, 100       |
| layers/units   | `agent.py`             | 1-3 layers, 32-256 units |

---

## Usage
- Run `train.py` to train the autoencoder.
- Run `test.py` to evaluate the trained model.

See the main project README for more details and workflow.
