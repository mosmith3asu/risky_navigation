# Variational Autoencoder (VAE) for Temporal Action Prediction

VAE for temporal action prediction with latent space modeling of action sequences.

## Problem

Predict operator actions while learning compressed latent representations of temporal patterns:
- **Input**: State sequence, action sequence, goal
- **Output**: Action prediction from learned distribution
- **Use case**: Model temporal action variability and generate diverse behaviors

## Technical Workflow

### Input
- **State Sequence**: History of states `[state_{t-n}, ..., state_t]`
  - Each state: Position, velocity, heading, obstacle distances
- **Action Sequence**: History of actions `[action_{t-n}, ..., action_{t-1}]`
- **Goal Vector**: Target position
- **sequence_len**: Controls temporal history length

### Processing
```python
# Flatten sequences
state_flat = state_sequence.reshape(batch, -1)
action_flat = action_sequence.reshape(batch, -1)
input = concat([state_flat, action_flat, goal])

# Encode temporal patterns to latent distribution
mu, log_var = encoder(input)
z = mu + exp(0.5 * log_var) * epsilon

# Decode to action
action_t = decoder(z)
```

VAE learns a compressed latent space representation of temporal state-action patterns.

### Training
- **Data**: Expert demonstrations from visibility graph
- **Sequences**: Created using sliding window over episodes
- **Loss**: ELBO = Reconstruction loss + β * KL divergence
- **β**: Controls regularization strength (β-VAE)

### Output
- **Action**: (throttle, steering) from latent sampling
- **Latent**: Mean μ and variance σ² in latent space

## Model Architecture

```python
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

## Hyperparameters

- `sequence_len`: Temporal history length ⚠️ **Critical** (default: 5)
  - Controls how many previous (state, action) pairs are used
  - `sequence_len=1`: No temporal modeling
  - `sequence_len>1`: Temporal latent space modeling
- `latent_dim`: Latent space dimensionality (default: 32, range: [16, 32, 64])
- `hidden_dim`: Hidden layer size (default: 128, range: [64, 128, 256])
- `beta`: Weight for KL divergence term (default: 1.0, range: [0.5, 1.0, 2.0])
- `lr`: Learning rate (default: 1e-3, range: [1e-3, 5e-4, 1e-4])
- `batch_size`: Batch size for training (default: 64)
- `num_epochs`: Number of training epochs (default: 50)

## Usage

```bash
# Train
python -m src.algorithms.VAE.train

# Test with latent space visualization
python -m src.algorithms.VAE.test
```