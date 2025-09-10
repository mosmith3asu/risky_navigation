# logger.py

This file provides the `Logger` class for visualizing training and evaluation results.

## Features
- Plots episode rewards (mean and standard deviation).
- Plots agent trajectories in the environment.
- Helps track agent performance during training and testing.

## Usage
- Create a `Logger` instance with the environment.
- Call `logger.log()` to record results after each episode.
- Call `logger.draw()` to update the plots.

See the main README for more details on how this fits into the workflow.
