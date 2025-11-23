# Variational Autoencoder (VAE) for Action Prediction

VAE for temporal action prediction with latent space modeling.

## Problem

Predict operator actions while learning compressed latent representations:
- **Input**: Current state, previous action, goal
- **Output**: Action prediction from learned distribution
- **Use case**: Model action variability and generate diverse behaviors

## Technical Workflow

### Input
- **State Vector**: Position, velocity, heading, obstacle distances
- **Previous Action**: Action from previous timestep
- **Goal Vector**: Target position

### Processing
```python
# Encode to latent distribution
mu, log_var = encoder([state_t, action_{t-1}, goal])
z = mu + exp(0.5 * log_var) * epsilon

# Decode to action
action_t = decoder(z)
```

VAE learns a compressed latent space representation of the state-action mapping.

### Training
- **Data**: Expert demonstrations from visibility graph
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

- `latent_dim`: Latent space dimensionality (default: 32)
- `hidden_dim`: Hidden layer size (default: 128)
- `beta`: Weight for KL divergence term (default: 1.0)
- `lr`: Learning rate (default: 1e-3)
- `batch_size`: Batch size for training (default: 64)
- `num_epochs`: Number of training epochs (default: 50)

## Usage

```bash
# Train
python -m src.algorithms.VAE.train

# Test with latent space visualization
python -m src.algorithms.VAE.test
```