# Variational Autoencoder (VAE) for Action Prediction

Variational Autoencoder for probabilistic action prediction with latent space modeling.

## Technical Workflow

### Input
- **State Vector** (dim: state_dim): Current position, velocity, obstacles
- **Goal Vector** (dim: goal_dim): Target position
- **Combined Input**: `[state, goal]` concatenated (dim: state_dim + goal_dim)

### Processing
1. **Encoder**: Maps input to latent distribution
   ```python
   mu, log_var = encoder([state, goal])
   # mu: mean of latent distribution (dim: latent_dim)
   # log_var: log variance of latent distribution (dim: latent_dim)
   ```

2. **Reparameterization**: Sample from latent distribution
   ```python
   std = exp(0.5 * log_var)
   eps = randn_like(std)  # Random noise
   z = mu + eps * std     # Latent sample (dim: latent_dim)
   ```

3. **Decoder**: Maps latent sample to action prediction
   ```python
   action = decoder(z)  # (dim: action_dim)
   ```

4. **Training**: ELBO loss (reconstruction + regularization)
   ```python
   reconstruction_loss = MSE(predicted_action, expert_action)
   kl_loss = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
   loss = reconstruction_loss + beta * kl_loss
   ```

### Output
- **Action Prediction** (dim: action_dim): Sampled from learned distribution
- **Latent Distribution**: Mean μ and variance σ² of latent space
- **Type**: Probabilistic prediction with aleatoric uncertainty (can sample diverse actions)

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