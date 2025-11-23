# Transformer for Temporal Action Prediction

Transformer model for action prediction using temporal sequences of states and self-attention mechanisms.

## Technical Workflow

### Input
- **State Sequence** (dim: sequence_len × state_dim): History of past states
- **Goal Vector** (dim: goal_dim): Target position
- **Combined Input**: Sequence of `[state_t-k, ..., state_t]` where each includes goal information

### Processing
1. **Input Embeddings**: Map each state to higher-dimensional space
   ```python
   embeddings = [Linear(state_i) for state_i in sequence]
   # Output: sequence_len × d_model
   ```

2. **Positional Encoding**: Add temporal position information
   ```python
   pos_encoded = embeddings + positional_encoding
   # Sinusoidal encoding: PE(pos, 2i) = sin(pos/10000^(2i/d_model))
   ```

3. **Self-Attention**: Compute relationships between time steps
   ```python
   Q, K, V = Linear(pos_encoded) for each head
   attention_scores = softmax(Q @ K.T / sqrt(d_k))
   attended = attention_scores @ V
   # Output: sequence_len × d_model
   ```

4. **Feed-Forward Networks**: Process attention outputs
   ```python
   output = FFN(LayerNorm(attended + residual))
   ```

5. **Action Prediction**: Map final representation to action
   ```python
   action = Linear(output[-1])  # Use last time step
   # Output: action_dim
   ```

6. **Training**: MSE loss on predicted actions
   ```python
   loss = MSE(predicted_action, expert_action)
   ```

### Output
- **Action Vector** (dim: action_dim): Predicted action based on sequence
- **Attention Weights** (dim: nhead × sequence_len × sequence_len): Which past states matter most
- **Type**: Deterministic prediction with temporal context

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