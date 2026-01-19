
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


def _as_batch(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Ensure x has batch dimension. Returns (xb, had_batch)."""
    if x.dim() == 1:
        return x.unsqueeze(0), False
    return x, True


# -----------------------------
# Networks
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()

        ACTIVATION = nn.LeakyReLU
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), ACTIVATION()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(nn.Module):
    """Q(s,a) -> scalar"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1, hidden_dim=hidden_dim, n_hidden=n_hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        s_b, _ = _as_batch(s)
        a_b, _ = _as_batch(a)
        x = torch.cat([s_b, a_b], dim=-1)
        return self.q(x)  # [B,1]


class TanhGaussianPolicy(nn.Module):
    """
    Gaussian policy with tanh squashing, then affine map to env action bounds.

    Returns:
      action (env-scaled), log_prob (w.r.t env-scaled action), mean_action (env-scaled)
    """
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim

        self.backbone = MLP(state_dim, hidden_dim, hidden_dim=hidden_dim, n_hidden=n_hidden - 1) if n_hidden > 1 else None
        trunk_in = hidden_dim if self.backbone is not None else state_dim

        self.mu = nn.Linear(trunk_in, action_dim)
        self.log_std = nn.Linear(trunk_in, action_dim)

        # action scaling buffers
        action_low = action_low.detach()
        action_high = action_high.detach()
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def _trunk(self, s: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            return s
        return self.backbone(s)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_b, _ = _as_batch(s)
        h = self._trunk(s_b)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparameterized action sample.
        """
        s_b, had_batch = _as_batch(s)
        mu, log_std = self.forward(s_b)
        std = log_std.exp()

        eps = torch.randn_like(mu)
        u = mu + std * eps  # pre-tanh
        a = torch.tanh(u)   # in [-1,1]

        # affine to env bounds
        action = a * self.action_scale + self.action_bias
        mean_action = torch.tanh(mu) * self.action_scale + self.action_bias

        # log_prob with tanh correction
        # base gaussian log prob
        log_prob = -0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # [B,1]

        # tanh correction: log |det da/du| = sum log(1 - tanh(u)^2)
        log_prob -= torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        # affine correction: constant w.r.t state but correct density in env units
        log_prob -= torch.log(self.action_scale + 1e-8).sum().view(1, 1)

        if not had_batch:
            action = action.squeeze(0)
            mean_action = mean_action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        return action, log_prob, mean_action

