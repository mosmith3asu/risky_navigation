# AutoEncoder Module

This folder contains the code for the autoencoder-based next-action prediction system.

## Files
- `agent.py`: Defines the autoencoder neural network and training logic.
- `train.py`: Collects data from the environment and trains the autoencoder.
- `test.py`: Evaluates the trained autoencoder in the environment.
- `__init__.py`: Marks this as a Python package.

## How it Works
1. **Data Collection:**
   - The environment is run with random actions.
   - For each step, (state, action, goal, next_action) is recorded.
2. **Training:**
   - The autoencoder is trained to predict the next action given (state, action, goal).
   - Loss is computed using Mean Squared Error (MSE) between predicted and actual next action.
3. **Testing:**
   - The trained autoencoder is used to select actions in the environment.
   - Performance is visualized using the logger.


## How the AutoEncoder Predicts the Next Action

1. **Input Construction:**
   - The autoencoder receives the current state, previous action, and goal, concatenated into a single input vector.

2. **Encoder:**
   - The encoder (a series of neural network layers) compresses this input into a lower-dimensional latent representation, capturing the most important features for prediction.

3. **Decoder:**
   - The decoder (another set of neural network layers) expands the latent representation back into a prediction for the next action.

4. **Prediction:**
   - The output of the decoder is the predicted next action (e.g., throttle and steering values).

5. **Learning:**
   - During training, the model's prediction is compared to the actual next action from the dataset using Mean Squared Error (MSE) loss.
   - The model's weights are updated to minimize this loss, so it gets better at predicting next actions over time.

**Summary:**
- The autoencoder learns a mapping: (state, action, goal) → next_action
- It does this by recognizing patterns in the data, not by reasoning like a human.

**Diagram:**

```
state   action   goal
  |       |       |
  +-------+-------+
          |
   [Concatenate]
          |
      [Encoder]
          |
      [Latent]
          |
      [Decoder]
          |
  Predicted next_action
```


---

## Analogy: How the AutoEncoder Works in This Project

Think of the autoencoder as a two-person team working with a book:

- **Original Data (Book):** The agent's current state, previous action, and goal are like the full book.
- **Encoder (Summarizer):** Reads the book and writes a short summary (the latent vector).
- **Bottleneck (Summary):** The compressed, essential information about the situation.
- **Decoder (Rewriter):** Takes the summary and tries to write the next action the agent should take (like rewriting the next page of the book).
- **Goal:** The predicted next action (rewritten page) should match the real next action (original page) as closely as possible.

In this project, the autoencoder is not reconstructing the original input, but is reconstructing (predicting) the next action that should be taken, given the current situation. It learns the “essence” of how to act in the environment, just like a good summary captures the essence of a book.


## Hyperparameter Tuning for AutoEncoders

Tuning hyperparameters is key to getting good results with autoencoders. Here’s how to do it in this codebase:

### What to Tune

- **Latent Dimension (`latent_dim`)**: Size of the bottleneck. Too small = underfitting, too large = overfitting.
- **Learning Rate (`lr`)**: How fast the model learns. Too high = unstable, too low = slow.
- **Batch Size (`batch_size`)**: Number of samples per training step. Larger = smoother, smaller = noisier.
- **Number of Epochs (`num_epochs`)**: How many times the model sees the dataset. Too few = underfit, too many = overfit.
- **Network Architecture**: Number of layers/units in encoder/decoder. More = more capacity, but risk of overfitting.
- **Activation Functions**: E.g., ReLU, LeakyReLU, Tanh.

### Where to Change

- Edit the config section at the top of `src/algorithms/AutoEncoder/train.py`:
   ```python
   # --- Config ---
   num_episodes = 200
   max_steps = 200
   latent_dim = 64
   lr = 1e-3
   batch_size = 128
   num_epochs = 20
   ```
- Edit the model architecture in `agent.py` if you want to add/remove layers or change units.

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
