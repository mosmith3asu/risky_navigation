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

## Usage
- Run `train.py` to train the autoencoder.
- Run `test.py` to evaluate the trained model.

See the main project README for more details and workflow.
