# Transformer for Temporal Action Prediction

Transformer model for temporal action prediction using self-attention over action sequences.

## Problem

Predict operator actions using temporal patterns from action history:
- **Input**: State sequence, action sequence, goal
- **Output**: Action prediction based on temporal context
- **Use case**: Capture temporal dependencies in human control patterns

## Technical Workflow

### Input
- **State Sequence**: History of states `[state_{t-n}, ..., state_t]`
  - Each state: Position, velocity, heading, obstacle distances
- **Action Sequence**: History of actions `[action_{t-n}, ..., action_{t-1}]`
- **Goal Vector**: Target position (replicated across sequence)
- **sequence_len**: Controls temporal context ⚠️ **Critical for Transformer**
  - `sequence_len=1`: No temporal modeling (falls back to feedforward)
  - `sequence_len>1`: True self-attention over temporal sequences

### Processing
```python
if sequence_len > 1:
    # Concatenate state and action at each timestep
    state_action = concat([state_sequence, action_sequence], dim=-1)
    # Replicate goal across sequence
    goal_expanded = goal.unsqueeze(1).expand(batch, seq_len, goal_dim)
    inputs = concat([state_action, goal_expanded], dim=-1)
    # Shape: (batch, seq_len, state_dim + action_dim + goal_dim)
    
    # Project to embedding space
    x = input_projection(inputs)  # (batch, seq_len, d_model)
    
    # Self-attention across time
    attention_output = transformer_encoder(x)  # (batch, seq_len, d_model)
    
    # Use last timestep for prediction
    action_t = output_layer(attention_output[:, -1, :])  # (batch, action_dim)
else:
    # Fallback to feedforward network
    inputs = concat([state, prev_action, goal])  # (batch, input_dim)
    action_t = feedforward_network(inputs)  # (batch, action_dim)
```

Transformer uses self-attention to weigh importance of different timesteps in action history.

### Training
- **Data**: Expert demonstrations from visibility graph
- **Sequences**: Created using sliding window over episodes
- **Loss**: MSE between predicted and expert actions
- **Key**: sequence_len determines if true temporal modeling is used

### Output
- **Action**: (throttle, steering) prediction
- **Type**: Deterministic with temporal awareness via attention

## Model Architecture

```python
class TransformerModel(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # Input embedding
        self.embedding = nn.Linear(state_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layer
        self.fc_out = nn.Linear(d_model, action_dim)
    
    def forward(self, state_sequence):
        # state_sequence: (batch, sequence_len, state_dim)
        x = self.embedding(state_sequence)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        action = self.fc_out(x[:, -1, :])  # Use last time step
        return action
```

## Hyperparameters

- `d_model`: Embedding dimensionality (default: 64, range: [32, 64, 128])
- `nhead`: Number of attention heads (default: 4, range: [4, 8])
- `num_layers`: Transformer encoder layers (default: 2, range: [2, 3, 4])
- `sequence_len`: Length of input sequence ⚠️ **Critical** (default: 1, range: [1, 5, 10])
  - `sequence_len=1`: No temporal modeling (feedforward network)
  - `sequence_len>1`: True temporal sequence modeling
- `dropout`: Dropout probability (default: 0.1, range: [0.0, 0.1, 0.2])
- `lr`: Learning rate (default: 1e-3, range: [1e-3, 5e-4, 1e-4])
- `batch_size`: Batch size for training (default: 128)
- `num_epochs`: Number of training epochs (default: 50)

## Usage

```bash
# Train
python -m src.algorithms.Transformer.train

# Test with attention visualization
python -m src.algorithms.Transformer.test
``` 