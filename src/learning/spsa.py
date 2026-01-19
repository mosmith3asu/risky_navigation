import math
from typing import Callable, Optional, Union

import torch
from torch.optim import Optimizer


class SPSA(Optimizer):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    Notes
    -----
    - Uses two function evaluations per step (theta + c_k*delta, theta - c_k*delta).
    - The closure MUST compute and return the loss (a scalar Tensor) and MUST NOT call backward().
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,          # "a" in a_k schedule
        perturb: float = 5e-2,     # "c" in c_k schedule
        alpha: float = 0.602,      # common SPSA choice
        gamma: float = 0.101,      # common SPSA choice
        A: float = 10.0,           # stability constant in a_k
        seed: Optional[int] = None,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if perturb <= 0:
            raise ValueError(f"perturb must be > 0, got {perturb}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if A < 0:
            raise ValueError(f"A must be >= 0, got {A}")

        defaults = dict(lr=lr, perturb=perturb, alpha=alpha, gamma=gamma, A=A)
        super().__init__(params, defaults)

        self._k = 0
        self._gen = None
        if seed is not None:
            # Generator helps make the Rademacher perturbations reproducible.
            self._gen = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
            self._gen.manual_seed(int(seed))

    @torch.no_grad()
    def step(self, closure: Callable[[], Union[torch.Tensor, float]]):
        if closure is None or not callable(closure):
            raise ValueError("SPSA.step requires a callable closure that returns a scalar loss Tensor/float.")

        self._k += 1

        # Compute scheduled step sizes
        group0 = self.param_groups[0]
        a = float(group0["lr"])
        c = float(group0["perturb"])
        alpha = float(group0["alpha"])
        gamma = float(group0["gamma"])
        A = float(group0["A"])

        a_k = a / ((self._k + A) ** alpha)
        c_k = c / (self._k ** gamma)

        # Sample perturbations (Rademacher ±1) for each parameter tensor
        deltas = []
        params_flat = []
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue
                if not isinstance(p, torch.Tensor):
                    raise TypeError(f"Parameter must be a torch.Tensor, got {type(p)}")
                if p.numel() == 0:
                    raise ValueError("Encountered an empty parameter tensor (numel == 0).")
                if not p.is_floating_point():
                    raise TypeError(f"SPSA requires floating point parameters, got dtype={p.dtype}")
                if not p.is_leaf:
                    raise ValueError("SPSA expects leaf parameter tensors (typical model.parameters()).")

                # delta has same shape as p (dimension check)
                delta = torch.randint(
                    low=0, high=2, size=p.shape, device=p.device, generator=self._gen
                ).to(dtype=p.dtype)
                delta = delta.mul_(2).add_(-1)  # {0,1} -> {-1,+1}
                if delta.shape != p.shape:
                    raise RuntimeError("Internal error: delta shape mismatch with parameter shape.")
                deltas.append(delta)
                params_flat.append(p)

        if len(params_flat) == 0:
            raise ValueError("No parameters provided to SPSA optimizer.")

        # Evaluate loss(theta + c_k * delta)
        for p, d in zip(params_flat, deltas):
            p.add_(c_k * d)
        loss_plus = closure()
        loss_plus = self._as_scalar_loss(loss_plus)

        # Evaluate loss(theta - c_k * delta) (from theta + c_k*d go to theta - c_k*d by subtracting 2*c_k*d)
        for p, d in zip(params_flat, deltas):
            p.add_(-2.0 * c_k * d)
        loss_minus = closure()
        loss_minus = self._as_scalar_loss(loss_minus)

        # Restore parameters to original theta
        for p, d in zip(params_flat, deltas):
            p.add_(c_k * d)

        # SPSA gradient estimate and parameter update:
        # g_hat = (loss_plus - loss_minus) / (2*c_k) * delta^{-1} ; with delta in {-1,+1}, delta^{-1} = delta
        diff = (loss_plus - loss_minus) / (2.0 * c_k)
        if not torch.isfinite(diff):
            raise FloatingPointError(f"Non-finite SPSA diff encountered: {diff.item()}")

        for p, d in zip(params_flat, deltas):
            # p <- p - a_k * (diff * d)
            p.add_(-a_k * diff * d)

        return loss_plus

    @staticmethod
    def _as_scalar_loss(x: Union[torch.Tensor, float]) -> torch.Tensor:
        if isinstance(x, (float, int)):
            x = torch.tensor(float(x))
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Closure must return a scalar Tensor/float, got {type(x)}")
        if x.numel() != 1:
            raise ValueError(f"Closure must return a scalar (numel==1). Got shape={tuple(x.shape)}")
        x = x.reshape(())
        if not torch.isfinite(x):
            raise FloatingPointError(f"Non-finite loss encountered: {x.item()}")
        return x

import copy
from typing import Callable, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def compare_spsa_to_standard_optimizers(
    make_model: Optional[Callable[[], nn.Module]] = None,
    n_steps: int = 300,
    n_samples: int = 2048,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, List[float]]:
    """
    Compare SPSA against standard PyTorch optimizers on a simple regression task and plot loss curves.

    Requirements
    ------------
    - Assumes the SPSA optimizer class is already defined in scope (from your prior snippet):
        opt = SPSA(model.parameters(), lr=..., perturb=..., ...)

    Returns
    -------
    history: dict[str, list[float]]
        Mapping from optimizer name to list of loss values per step.
    """
    # -----------------------------
    # Basic argument validation
    # -----------------------------
    if not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError(f"n_steps must be a positive int, got {n_steps}")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int, got {n_samples}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive int, got {batch_size}")
    if batch_size > n_samples:
        raise ValueError(f"batch_size ({batch_size}) cannot exceed n_samples ({n_samples})")

    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Data: linear ground truth + noise
    # -----------------------------
    d_in = 8
    d_out = 1
    X = torch.randn(n_samples, d_in, device=device)
    w_true = torch.randn(d_in, d_out, device=device)
    y = X @ w_true + 0.1 * torch.randn(n_samples, d_out, device=device)

    # Dimension checks (as requested)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, D), got shape {tuple(X.shape)}")
    if y.ndim != 2:
        raise ValueError(f"y must be 2D (N, O), got shape {tuple(y.shape)}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same N. Got {X.shape[0]} vs {y.shape[0]}")
    if X.shape[1] != d_in or y.shape[1] != d_out:
        raise ValueError("Internal dimension mismatch; check d_in/d_out definitions.")

    loss_fn = nn.MSELoss()

    # -----------------------------
    # Model factory (same init for all optimizers)
    # -----------------------------
    if make_model is None:
        def make_model():
            return nn.Sequential(
                nn.Linear(d_in, 32),
                nn.Tanh(),
                nn.Linear(32, d_out),
            )

    base_model = make_model().to(device)
    base_state = copy.deepcopy(base_model.state_dict())

    # -----------------------------
    # Optimizer configurations
    # -----------------------------
    # NOTE: SPSA hyperparameters are not directly comparable to gradient-based optimizers.
    #       You will likely want to tune lr/perturb for your problem size/noise level.
    optim_specs: List[Tuple[str, Callable[[nn.Module], torch.optim.Optimizer]]] = [
        ("SGD",   lambda m: torch.optim.SGD(m.parameters(), lr=5e-2, momentum=0.9)),
        ("Adam",  lambda m: torch.optim.Adam(m.parameters(), lr=1e-2)),
        ("RMSprop", lambda m: torch.optim.RMSprop(m.parameters(), lr=1e-2)),
        # ("SPSA",  lambda m: SPSA(m.parameters(), lr=2e-2, perturb=1e-2, seed=seed)),
        ("SPSA", lambda m: SPSA(m.parameters(), seed=seed)),
    ]

    history: Dict[str, List[float]] = {}

    # -----------------------------
    # Training loops
    # -----------------------------
    for name, make_opt in optim_specs:
        model = make_model().to(device)
        model.load_state_dict(copy.deepcopy(base_state))
        model.train()

        opt = make_opt(model)
        losses: List[float] = []

        # Fixed minibatch schedule for fairness across optimizers
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        for t in range(n_steps):
            idx = torch.randint(0, n_samples, (batch_size,), generator=g, device=device)
            xb = X.index_select(0, idx)
            yb = y.index_select(0, idx)

            if name == "SPSA":
                # SPSA closure: must return scalar loss, and MUST NOT call backward()
                def closure():
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    # Closure must return a scalar
                    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                        raise ValueError(f"Closure must return a scalar Tensor, got {type(loss)} shape={getattr(loss, 'shape', None)}")
                    return loss

                loss_t = opt.step(closure)
                losses.append(float(loss_t.item()))
            else:
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)

                # Scalar check
                if loss.ndim != 0:
                    raise ValueError(f"Loss must be scalar. Got shape {tuple(loss.shape)}")

                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

        history[name] = losses

    # -----------------------------
    # Plot convergence
    # -----------------------------
    plt.figure()
    for name, losses in history.items():
        plt.plot(losses, label=name)
    plt.xlabel("Step")
    plt.ylabel("Loss (MSE)")
    plt.title("Optimizer loss convergence")
    plt.legend(frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return history


# Example:
if __name__ == "__main__":

    history = compare_spsa_to_standard_optimizers(n_steps=400, n_samples=4096, batch_size=256, seed=0)

# def example_function(x: torch.Tensor) -> torch.Tensor:
#     """ Example objective function: f(theta) = (theta - 3)^2 """
#     x += 0.1 * torch.randn_like(x)
#     return (x - 3.0).pow(2).sum() #+ noise
#
# # -----------------------------
# # Example usage
# # -----------------------------
# if __name__ == "__main__":
#     model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
#     opt = SPSA(model.parameters(), lr=1e-2, perturb=1e-3, seed=0)
#
#     x = torch.randn(64, 4)
#     y = example_function(x)# torch.randn(64, 1)
#
#     def closure():
#         # NOTE: Do NOT call backward() here.
#         pred = model(x)
#         loss = torch.mean((pred - y) ** 2)
#         return loss
#
#     for _ in range(100):
#         loss = opt.step(closure)
#         print(f"Loss: {loss.item():.6f}")
