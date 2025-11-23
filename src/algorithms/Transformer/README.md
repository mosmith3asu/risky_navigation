# Transformer for Temporal Action Prediction

Transformer model for temporal action prediction using self-attention over action sequences.

## Problem

Predict operator actions using temporal patterns from action history:
- **Input**: Current state, previous action(s), goal
- **Output**: Action prediction based on temporal context
- **Use case**: Capture temporal dependencies in human control patterns

## Technical Workflow

### Input
- **State Vector**: Position, velocity, heading, obstacle distances
- **Previous Action(s)**: Recent action history (sequence_len determines history length)
- **Goal Vector**: Target position

### Processing
```python
# Embed inputs
x = embedding([state_t, action_{t-1}, goal])

# Self-attention across time
attention_output = transformer_encoder(x)

# Predict action
action_t = output_layer(attention_output)
```

Transformer uses self-attention to weigh importance of different timesteps.

### Training
- **Data**: Expert demonstrations from visibility graph
- **Loss**: MSE between predicted and expert actions
- **sequence_len**: Controls temporal context (1=no history, >1=temporal modeling)

### Output
- **Action**: (throttle, steering) prediction
- **Type**: Deterministic with temporal awareness

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