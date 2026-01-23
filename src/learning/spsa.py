import math
from typing import Callable, Optional, Union

import numpy as np
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support, _device_has_foreach_support
from torch import Tensor
import copy
from typing import Callable, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
from collections import deque

"""https://www.jhuapl.edu/spsa/Pages/MATLAB.htm"""


class SPSA_logger(Optimizer):
    def __init__(
            self,
            params,
            # lr: float = 1e-2,  # "a" in a_k schedule
            # perturb: float = 5e-2,  # "c" in c_k schedule
            # lr: float = 5e-4,  # "a" in a_k schedule
            # perturb: float = 5e-5,  # "c" in c_k schedule
            # lr: float = 2e-3,  # "a" in a_k schedule
            # perturb: float = 1e-3,  # "c" in c_k schedule
            #
            # alpha: float = 0.602,  # common SPSA choice
            # gamma: float = 0.101,  # common SPSA choice
            # A: float = 5.0,  # stability constant in a_k
            # n: int = 50,  # number of iterations
            # clip_grad_norm: Optional[float] = 2000,
            # log_losses: bool = False,
            # seed: Optional[int] = None,
            lr: float = 2e-3,  # "a" in a_k schedule
            perturb: float = 1e-3,  # "c" in c_k schedule

            alpha: float = 0.602,  # common SPSA choice
            gamma: float = 0.101,  # common SPSA choice
            A: float = 5.0,  # stability constant in a_k
            n: int = 200,  # number of iterations
            clip_grad_norm: Optional[float] = 2000,
            log_losses: bool = False,
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

        self.lr = lr
        self.perturb = perturb
        self.alpha = alpha
        self.gamma = gamma
        self.A = A
        self.n = n
        self.clip_grad_norm = clip_grad_norm

        if log_losses:
            warnings.warn("Enabling loss logging will increase memory usage.", stacklevel=2)
        self._loss_log = deque(maxlen=10) if log_losses else None
        self.fig = None
        self.ax = None

        defaults = dict(lr=lr, perturb=perturb, alpha=alpha, gamma=gamma, A=A)
        super().__init__(params, defaults)

        self.flat_params = self._flatten_params()

    def _flatten_params(self):
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

                params_flat.append(p)
        return params_flat

    def _purtabations(self, params_flat):
        n_params = len(params_flat)
        dtype = params_flat[0].dtype
        device = params_flat[0].device
        deltas = 2 * torch.randint(low=0, high=2, size=(n_params,), device=device).to(dtype=dtype) - 1
        assert torch.all((deltas == 1) | (deltas == -1)), "Deltas must be ±1"
        return deltas

    @torch.no_grad()
    def step(self, loss_eval: Callable[[], Union[torch.Tensor, float]]):
        # loss = self.loss_fun
        loss = loss_eval
        a = self.lr
        c = self.perturb

        theta_flat = self.flat_params
        dtype = theta_flat[0].dtype
        device = theta_flat[0].device
        # print(f"SPSA optimizing {n_params} parameters for {self.n} iterations.")
        step_losses = np.empty(self.n, dtype=np.float32)

        for k in range(self.n):
            # print(f"SPSA iteration {k+1}/{self.n}")

            ak = a / (k + 1 + self.A) ** self.alpha
            ck = c / (k + 1) ** self.gamma

            deltas = []
            for theta in theta_flat:
                d = 2 * torch.randint(low=0, high=2, size=theta.shape, device=device).to(dtype=dtype) - 1
                deltas.append(d)
                assert d.shape == theta.shape, "Delta shape mismatch"
                assert torch.all((d == 1) | (d == -1)), "Deltas must be ±1"

            # thetaplus = theta + ck * delta
            for theta, d in zip(theta_flat, deltas):
                theta.add_(ck * d)
            yplus = loss()
            yplus = self._as_scalar_loss(yplus)

            # thetaminus = theta - ck * delta
            for theta, d in zip(theta_flat, deltas):
                theta.add_(-2 * ck * d)
            yminus = loss()
            yminus = self._as_scalar_loss(yminus)

            # restore theta
            for theta, d in zip(theta_flat, deltas):
                theta.add_(ck * d)

            # gradient estimate
            # diff =  (yplus - yminus) / (2 * ck )
            ghat = [(yplus - yminus) / (2 * ck * d) for d in deltas]

            # clip gradients norms
            if self.clip_grad_norm is not None:
                self._clip_grad_norm(ghat, self.clip_grad_norm)

            assert torch.isfinite(yplus), f"Non-finite SPSA yplus encountered"
            assert torch.isfinite(yminus), f"Non-finite SPSA yminus encountered"
            assert all([torch.all(torch.isfinite(g)) for g in ghat]), f"Non-finite SPSA diff encountered"

            # print(f"Iteration {k+1}/{self.n}, Loss+: {yplus.item():.6f}, Loss-: {yminus.item():.6f}")

            # theta = theta - ak * ghat
            for theta, g in zip(theta_flat, ghat):
                theta.add_(-ak * g)

            if self._loss_log is not None:
                step_losses[k] = loss().item()

            # ---- early stop check on current params ----
            # curr = self._as_scalar_loss(loss()).item()
            # if curr < best - tol:
            #     best = curr
            #     bad = 0
            #     if restore_best:
            #         best_params = [p.detach().clone() for p in theta_flat]
            # else:
            #     bad += 1
            #
            # if (k + 1) >= min_iters and bad >= patience:
            #         break
        if self._loss_log is not None:
            self._loss_log.append(step_losses)
        l = loss()
        assert torch.isfinite(l), f"Non-finite SPSA final loss encountered"
        return l

    def plot_losses(self):
        if self._loss_log is None:
            raise RuntimeError("Loss logging was not enabled.")
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlabel("SPSA Iteration")
            self.ax.set_ylabel("Loss")
            self.ax.set_title(f"SPSA Loss {self.defaults}")
        self.ax.clear()
        for l in self._loss_log:
            self.ax.plot(l, color='gray', alpha=0.3)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @torch.no_grad()
    def _clip_grad_norm(self,
                        grads, max_norm: float, norm_type: float = 2.0,
                        error_if_nonfinite: bool = False, foreach: Optional[bool] = None) -> torch.Tensor:
        r"""Clip the gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float): max norm of the gradients
            norm_type (float): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)
            foreach (bool): use the faster foreach-based implementation.
                If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
                fall back to the slow implementation for other device types.
                Default: ``None``

        Returns:
            Total norm of the parameter gradients (viewed as a single vector).
        """
        # if isinstance(parameters, torch.Tensor):
        #     parameters = [parameters]
        # grads = [p.grad for p in parameters if p.grad is not None]
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if len(grads) == 0:
            return torch.tensor(0.)
        first_device = grads[0].device
        grouped_grads: Dict[Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]] \
            = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

        norms: List[Tensor] = []
        for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
            if (
                    (foreach is None and _has_foreach_support(device_grads, device))
                    or (foreach and _device_has_foreach_support(device))
            ):
                norms.extend(torch._foreach_norm(device_grads, norm_type))
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

        total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
            if (
                    (foreach is None and _has_foreach_support(device_grads, device))
                    or (foreach and _device_has_foreach_support(device))
            ):
                torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                clip_coef_clamped_device = clip_coef_clamped.to(device)
                for g in device_grads:
                    g.mul_(clip_coef_clamped_device)

        return total_norm

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


class SPSA(Optimizer):
    """
    Speed-optimized SPSA:
      - Reuses preallocated delta buffers (no per-iter allocations)
      - Uses foreach ops for perturbation steps (fewer Python loops / kernel launches)
      - Exploits 1/delta == delta for Rademacher ±1 deltas (removes tensor divisions)
      - Fast global-norm clipping specialized for SPSA gradient structure
      - Avoids GPU sync by keeping y+/y- and scalars as 0-dim tensors on-device
    """
    def __init__(
        self,
        params,
            # lr: float = 2e-3,
            # perturb: float = 1e-3,
            # alpha: float = 0.602,
            # gamma: float = 0.101,
            # A: float = 0.0,
            # n: int = 50,
            # clip_grad_norm: Optional[float] = 5,
            lr: float = 5e-2,
            perturb: float = 1e-3,
            alpha: float = 0.602,
            gamma: float = 0.101,
            A: float = 1.0,
            n: int = 10,
            clip_grad_norm: Optional[float] = 10,

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
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")

        self.lr = float(lr)
        self.perturb = float(perturb)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.A = float(A)
        self.n = int(n)
        self.clip_grad_norm = None if clip_grad_norm is None else float(clip_grad_norm)

        defaults = dict(lr=lr, perturb=perturb, alpha=alpha, gamma=gamma, A=A)
        super().__init__(params, defaults)

        self.flat_params: List[torch.Tensor] = self._flatten_params()

        # Precompute for fast SPSA gradient-norm clipping:
        # ghat = s * delta (elementwise), delta ∈ {±1} => ||ghat||_2 = |s| * sqrt(total_numel)
        self._sqrt_numel: float = math.sqrt(sum(p.numel() for p in self.flat_params))

        # Delta buffers (allocated lazily; rebuilt if device/dtype changes)
        self._delta_bufs: Optional[List[torch.Tensor]] = None
        self._buf_device: Optional[torch.device] = None
        self._buf_dtype: Optional[torch.dtype] = None

    def _flatten_params(self) -> List[torch.Tensor]:
        params_flat: List[torch.Tensor] = []
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
                params_flat.append(p)

        if not params_flat:
            raise ValueError("No parameters found for SPSA.")

        # Foreach ops require same device/dtype across the list.
        dev0, dt0 = params_flat[0].device, params_flat[0].dtype
        for p in params_flat[1:]:
            if p.device != dev0:
                raise ValueError("All parameters must be on the same device for this SPSA implementation.")
            if p.dtype != dt0:
                raise ValueError("All parameters must share the same dtype for this SPSA implementation.")
        return params_flat

    def _ensure_delta_bufs(self) -> None:
        p0 = self.flat_params[0]
        if (
            self._delta_bufs is None
            or self._buf_device != p0.device
            or self._buf_dtype != p0.dtype
            or len(self._delta_bufs) != len(self.flat_params)
        ):
            self._delta_bufs = [torch.empty_like(p) for p in self.flat_params]
            self._buf_device = p0.device
            self._buf_dtype = p0.dtype

    @staticmethod
    def _loss_to_scalar_tensor(
        x: Union[torch.Tensor, float, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                raise ValueError(f"Closure must return a scalar (numel==1). Got shape={tuple(x.shape)}")
            return x.reshape(())
        if isinstance(x, (float, int)):
            return torch.tensor(float(x), device=device, dtype=dtype)
        raise TypeError(f"Closure must return a scalar Tensor/float, got {type(x)}")

    @staticmethod
    def _foreach_add_(tensors: List[torch.Tensor], others: List[torch.Tensor], alpha: float = 1) -> None:
        try:
            torch._foreach_add_(tensors, others, alpha=alpha)
        except Exception:
            for t, o in zip(tensors, others):
                t.add_(o, alpha=alpha)

    @torch.no_grad()
    def step(self, loss_eval: Callable[[], Union[torch.Tensor, float]]):
        theta = self.flat_params
        device, dtype = theta[0].device, theta[0].dtype

        self._ensure_delta_bufs()
        deltas = self._delta_bufs  # type: ignore[assignment]

        a = self.lr
        c = self.perturb
        eps = 1e-12

        for k in range(self.n):
            kp1 = k + 1.0
            ak = a / math.pow(kp1 + self.A, self.alpha)
            ck = c / math.pow(kp1, self.gamma)

            # In-place Rademacher deltas: {-1, +1} with no allocations
            # random_(0,2) gives {0,1}; map -> {-1,+1}
            for d in deltas:
                d.random_(0, 2).mul_(2.0).sub_(1.0)

            # theta -> theta + ck*delta
            self._foreach_add_(theta, deltas, alpha=ck)
            yplus = self._loss_to_scalar_tensor(loss_eval(), device=device, dtype=dtype)

            # theta + ck*delta -> theta - ck*delta
            self._foreach_add_(theta, deltas, alpha=-2.0 * ck)
            yminus = self._loss_to_scalar_tensor(loss_eval(), device=device, dtype=dtype)


            # GRAD-NORM 1 #########################################
            # Scalar factor: s = (y+ - y-) / (2*ck)
            # And since delta ∈ {±1}, 1/delta == delta, so ghat = s * delta (no divisions).
            s = (yplus - yminus) / (2.0 * ck)

            # Fast SPSA-specific global norm clipping (L2):
            # ||ghat||_2 = |s| * sqrt(total_numel)
            if self.clip_grad_norm is not None:
                total_norm = s.abs() * self._sqrt_numel
                scale = (self.clip_grad_norm / (total_norm + eps)).clamp(max=1.0)
                s = s * scale

            ## Current params are at theta_minus = theta - ck*delta.
            # Want: theta_new = theta - ak*(s*delta).
            # So from theta_minus add: (ck - ak*s)*delta  ==> theta - ck*delta + (ck - ak*s)*delta
            coef = ck - ak * s  # 0-dim tensor on device (avoids GPU sync)
            for p, d in zip(theta, deltas):
                p.addcmul_(d, coef)  # p += d * coef (broadcast coef)


            # GRAD-NORM 2 #########################################
            # s = (yplus - yminus) / (2.0 * ck)
            # ghat = [s * d for d in deltas]
            #
            # if self.clip_grad_norm is not None:
            #     _compiled_clip_grad_norm(ghat, self.clip_grad_norm)
            #
            # # restore theta
            # self._foreach_add_(theta, deltas, alpha=ck)
            # for p, g in zip(theta, ghat):
            #     p.add_(- ak * g)  # add gradient step


        l = self._loss_to_scalar_tensor(loss_eval(), device=device, dtype=dtype)
        if not torch.isfinite(l).item():
            raise FloatingPointError(f"Non-finite SPSA final loss encountered: {l.item()}")
        return l

    @torch.no_grad()
    def step_hist(self, loss_eval: Callable[[], Union[torch.Tensor, float]]):
        theta = self.flat_params
        device, dtype = theta[0].device, theta[0].dtype
        loss_history = np.empty(self.n)

        self._ensure_delta_bufs()
        deltas = self._delta_bufs  # type: ignore[assignment]

        a = self.lr
        c = self.perturb
        eps = 1e-12
        theta_max = -float('inf')

        for k in range(self.n):
            kp1 = k + 1.0
            ak = a / math.pow(kp1 + self.A, self.alpha)
            ck = c / math.pow(kp1, self.gamma)

            # In-place Rademacher deltas: {-1, +1} with no allocations
            # random_(0,2) gives {0,1}; map -> {-1,+1}
            for d in deltas:
                d.random_(0, 2).mul_(2.0).sub_(1.0)

            # theta -> theta + ck*delta
            self._foreach_add_(theta, deltas, alpha=ck)
            yplus = self._loss_to_scalar_tensor(loss_eval(), device=device, dtype=dtype)

            # theta + ck*delta -> theta - ck*delta
            self._foreach_add_(theta, deltas, alpha=-2.0 * ck)
            yminus = self._loss_to_scalar_tensor(loss_eval(), device=device, dtype=dtype)

            # GRAD-NORM 1 #########################################
            # Scalar factor: s = (y+ - y-) / (2*ck)
            # And since delta ∈ {±1}, 1/delta == delta, so ghat = s * delta (no divisions).
            s = (yplus - yminus) / (2.0 * ck)

            # Fast SPSA-specific global norm clipping (L2):
            # ||ghat||_2 = |s| * sqrt(total_numel)
            if self.clip_grad_norm is not None:
                total_norm = s.abs() * self._sqrt_numel
                scale = (self.clip_grad_norm / (total_norm + eps)).clamp(max=1.0)
                s = s * scale

            ## Current params are at theta_minus = theta - ck*delta.
            # Want: theta_new = theta - ak*(s*delta).
            # So from theta_minus add: (ck - ak*s)*delta  ==> theta - ck*delta + (ck - ak*s)*delta
            coef = ck - ak * s  # 0-dim tensor on device (avoids GPU sync)
            for p, d in zip(theta, deltas):
                p.addcmul_(d, coef)  # p += d * coef (broadcast coef)

            # GRAD-NORM 2 #########################################
            # s = (yplus - yminus) / (2.0 * ck)
            # ghat = [s * d for d in deltas]
            #
            # if self.clip_grad_norm is not None:
            #     _compiled_clip_grad_norm(ghat, self.clip_grad_norm)
            #
            # # restore theta
            # self._foreach_add_(theta, deltas, alpha=ck)
            # for p, g in zip(theta, ghat):
            #     p.add_(- ak * g)  # add gradient step

            # find max value of theta
            # theta_max = max(theta_max, max([p.max().item() for p in theta]))

            loss_history[k] = self._loss_to_scalar_tensor(loss_eval(), device=device, dtype=dtype).item()

        # print("Max theta value during SPSA:", theta_max)
        return loss_history


class SPSA_OG(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 2e-3,  # "a" in a_k schedule
            perturb: float = 1e-3,  # "c" in c_k schedule
            alpha: float = 0.602,  # common SPSA choice
            gamma: float = 0.101,  # common SPSA choice
            A: float = 0.0,  # stability constant in a_k
            n: int = 10,  # number of iterations
            clip_grad_norm: Optional[float] = 2000,
    ):
        if lr <= 0:  raise ValueError(f"lr must be > 0, got {lr}")
        if perturb <= 0:  raise ValueError(f"perturb must be > 0, got {perturb}")
        if not (0.0 < alpha <= 1.0): raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not (0.0 < gamma <= 1.0):   raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if A < 0:  raise ValueError(f"A must be >= 0, got {A}")

        self.lr = lr
        self.perturb = perturb
        self.alpha = alpha
        self.gamma = gamma
        self.A = A
        self.n = n
        self.clip_grad_norm = clip_grad_norm
        defaults = dict(lr=lr, perturb=perturb, alpha=alpha, gamma=gamma, A=A)
        super().__init__(params, defaults)
        self.flat_params = self._flatten_params()


    def _flatten_params(self):
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

                params_flat.append(p)
        return params_flat

    def _purtabations(self, params_flat):
        n_params = len(params_flat)
        dtype = params_flat[0].dtype
        device = params_flat[0].device
        deltas = 2 * torch.randint(low=0, high=2, size=(n_params,), device=device).to(dtype=dtype) - 1
        assert torch.all((deltas == 1) | (deltas == -1)), "Deltas must be ±1"
        return deltas

    @torch.no_grad()
    def step(self, loss_eval: Callable[[], Union[torch.Tensor, float]]):
        a = self.lr
        c = self.perturb

        theta_flat = self.flat_params
        dtype = theta_flat[0].dtype
        device = theta_flat[0].device

        for k in range(self.n):
            ak = a / (k + 1 + self.A) ** self.alpha
            ck = c / (k + 1) ** self.gamma

            deltas = []
            for theta in theta_flat:
                d = 2 * torch.randint(low=0, high=2, size=theta.shape, device=device).to(dtype=dtype) - 1
                deltas.append(d)
                assert d.shape == theta.shape, "Delta shape mismatch"
                assert torch.all((d == 1) | (d == -1)), "Deltas must be ±1"

            # thetaplus = theta + ck * delta
            for theta, d in zip(theta_flat, deltas):
                theta.add_(ck * d)
            yplus = loss_eval()
            yplus = self._as_scalar_loss(yplus)

            # thetaminus = theta - ck * delta
            for theta, d in zip(theta_flat, deltas):
                theta.add_(-2 * ck * d)
            yminus = loss_eval()
            yminus = self._as_scalar_loss(yminus)


            # gradient estimate ---------------------
            # diff =  (yplus - yminus) / (2 * ck )
            ghat = [(yplus - yminus) / (2 * ck * d) for d in deltas]

            # clip gradients norms
            if self.clip_grad_norm is not None:
                self._clip_grad_norm(ghat, self.clip_grad_norm)

            # Sanity checks
            # assert torch.isfinite(yplus), f"Non-finite SPSA yplus encountered"
            # assert torch.isfinite(yminus), f"Non-finite SPSA yminus encountered"
            # assert all([torch.all(torch.isfinite(g)) for g in ghat]), f"Non-finite SPSA diff encountered"


            # update theta ---------------------
            # # restore theta
            # for theta, d in zip(theta_flat, deltas):
            #     theta.add_(ck * d)

            # # theta = theta - ak * ghat
            # for theta, g in zip(theta_flat, ghat):
            #     theta.add_(-ak * g)

            for theta, d, g in zip(theta_flat, deltas, ghat):
                # theta.add_(ck * d) # restore theta
                # theta.add_(-ak * g) # theta = theta - ak * ghat
                theta.add_(ck * d - ak * g)  # do both in one step


        l = loss_eval()
        assert torch.isfinite(l), f"Non-finite SPSA final loss encountered"
        return l

    @torch.no_grad()
    def step_hist(self, loss_eval: Callable[[], Union[torch.Tensor, float]]):
        loss_hist = np.empty(self.n)
        a = self.lr
        c = self.perturb

        theta_flat = self.flat_params
        dtype = theta_flat[0].dtype
        device = theta_flat[0].device

        for k in range(self.n):
            ak = a / (k + 1 + self.A) ** self.alpha
            ck = c / (k + 1) ** self.gamma

            deltas = []
            for theta in theta_flat:
                d = 2 * torch.randint(low=0, high=2, size=theta.shape, device=device).to(dtype=dtype) - 1
                deltas.append(d)
                assert d.shape == theta.shape, "Delta shape mismatch"
                assert torch.all((d == 1) | (d == -1)), "Deltas must be ±1"

            # thetaplus = theta + ck * delta
            for theta, d in zip(theta_flat, deltas):
                theta.add_(ck * d)
            yplus = loss_eval()
            yplus = self._as_scalar_loss(yplus)

            # thetaminus = theta - ck * delta
            for theta, d in zip(theta_flat, deltas):
                theta.add_(-2 * ck * d)
            yminus = loss_eval()
            yminus = self._as_scalar_loss(yminus)

            # gradient estimate ---------------------
            # diff =  (yplus - yminus) / (2 * ck )
            ghat = [(yplus - yminus) / (2 * ck * d) for d in deltas]

            # clip gradients norms
            if self.clip_grad_norm is not None:
                self._clip_grad_norm(ghat, self.clip_grad_norm)

            # Sanity checks
            # assert torch.isfinite(yplus), f"Non-finite SPSA yplus encountered"
            # assert torch.isfinite(yminus), f"Non-finite SPSA yminus encountered"
            # assert all([torch.all(torch.isfinite(g)) for g in ghat]), f"Non-finite SPSA diff encountered"

            # update theta ---------------------
            # # restore theta
            # for theta, d in zip(theta_flat, deltas):
            #     theta.add_(ck * d)

            # # theta = theta - ak * ghat
            # for theta, g in zip(theta_flat, ghat):
            #     theta.add_(-ak * g)

            for theta, d, g in zip(theta_flat, deltas, ghat):
                # theta.add_(ck * d) # restore theta
                # theta.add_(-ak * g) # theta = theta - ak * ghat
                theta.add_(ck * d - ak * g)  # do both in one step

            loss_hist[k] = self._as_scalar_loss(loss_eval()).item()
        return loss_hist
        # l = loss_eval()
        # assert torch.isfinite(l), f"Non-finite SPSA final loss encountered"
        # return l

    @torch.no_grad()
    def _clip_grad_norm(self, grads, max_norm: float) -> torch.Tensor:
        r"""Clip the gradient norm of an iterable of parameters.
        re-implements compiled version of torch.nn.utils.clip_grad_norm_ for use in SPSA
        Returns:
            Total norm of the parameter gradients (viewed as a single vector).
        """
        return _compiled_clip_grad_norm(grads, max_norm)


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

#
# @torch.jit.script
@torch.compile(fullgraph=False)
def _compiled_clip_grad_norm(
        grads, max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None) -> torch.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    # if isinstance(parameters, torch.Tensor):
    #     parameters = [parameters]
    # grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]] \
        = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    norms: List[Tensor] = []
    for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        if (
                (foreach is None and _has_foreach_support(device_grads, device))
                or (foreach and _device_has_foreach_support(device))
        ):
            norms.extend(torch._foreach_norm(device_grads, norm_type))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        if (
                (foreach is None and _has_foreach_support(device_grads, device))
                or (foreach and _device_has_foreach_support(device))
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)

    return total_norm

class DualModel:
    def __init__(self,d_in,d_out):
        self.m1 = nn.Sequential(
                nn.Linear(d_in, 32),
                nn.Tanh(),
                nn.Linear(32, d_out),
            )

        self.m2 = nn.Sequential(
                nn.Linear(d_in, 32),
                nn.Tanh(),
                nn.Linear(32, d_out),
            )

    def loss(self, x, yb, each_loss=False):
        q1 = self.m1(x)
        q2 = self.m2(x)
        l1 = F.smooth_l1_loss(q1, yb)
        l2 = F.smooth_l1_loss(q2, yb)

        # l1 = F.mse_loss(q1, yb)
        # l2 = F.mse_loss(q2, yb)

        return l1 + l2 if not each_loss else (l1,l2)

    def __call__(self, *args, **kwargs):
        out1 = self.m1(*args, **kwargs)
        out2 = self.m2(*args, **kwargs)
        return torch.min(out1 + out2)

    def to(self, device):
        self.m1.to(device)
        self.m2.to(device)
        return self

    def state_dict(self):
        return {
            'm1': self.m1.state_dict(),
            'm2': self.m2.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.m1.load_state_dict(state_dict['m1'])
        self.m2.load_state_dict(state_dict['m2'])

    def parameters(self):
        return list(self.m1.parameters()) + list(self.m2.parameters())

    def train(self):
        self.m1.train()
        self.m2.train()

    def eval(self):
        self.m1.eval()
        self.m2.eval()

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


    # -----------------------------
    # Model factory (same init for all optimizers)
    # -----------------------------
    if make_model is None:
        def make_model():
            return DualModel(d_in, d_out)
        # def make_model():
        #     return nn.Sequential(
        #         nn.Linear(d_in, 32),
        #         nn.Tanh(),
        #         nn.Linear(32, d_out),
        #     )

    base_model = make_model().to(device)
    base_state = copy.deepcopy(base_model.state_dict())
    loss_fn = nn.MSELoss()


    # -----------------------------
    # Optimizer configurations
    # -----------------------------
    # NOTE: SPSA hyperparameters are not directly comparable to gradient-based optimizers.
    #       You will likely want to tune lr/perturb for your problem size/noise level.
    optim_specs: List[Tuple[str, Callable[[nn.Module], torch.optim.Optimizer]]] = [
        ("SGD",   lambda m: torch.optim.SGD(m.parameters(), lr=5e-2, momentum=0.9)),
        ("Adam",  lambda m: torch.optim.Adam(m.parameters(), lr=1e-2)),
        ("RMSprop", lambda m: torch.optim.RMSprop(m.parameters(), lr=1e-2)),
        ("SPSA",  lambda m: SPSA(m.parameters(), lr=2e-2, perturb=1e-2)),
        ("SPSA-1step",  lambda m: SPSA(m.parameters(), lr=2e-2, perturb=1e-2, n=1)),
        ("SPSA_OG-1step", lambda m: SPSA_OG(m.parameters(), lr=2e-2, perturb=1e-2, n=1)),
        # ("SPSA", lambda m: SPSA(m.parameters(), seed=seed)),
    ]

    history: Dict[str, List[float]] = {}
    history_l1: Dict[str, List[float]] = {}
    history_l2: Dict[str, List[float]] = {}
    # -----------------------------
    # Training loops
    # -----------------------------
    for name, make_opt in optim_specs:
        model = make_model().to(device)
        model.load_state_dict(copy.deepcopy(base_state))
        model.train()

        opt = make_opt(model)
        losses: List[float] = []
        l1, l2 = [], []


        # Fixed minibatch schedule for fairness across optimizers
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        for t in range(n_steps):
            idx = torch.randint(0, n_samples, (batch_size,), generator=g, device=device)
            xb = X.index_select(0, idx)
            yb = y.index_select(0, idx)

            if "SPSA" in name:
                # SPSA closure: must return scalar loss, and MUST NOT call backward()
                def closure():
                    # pred = model(xb)
                    # loss = loss_fn(pred, yb)
                    loss = model.loss(xb, yb)
                    # Closure must return a scalar
                    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                        raise ValueError(f"Closure must return a scalar Tensor, got {type(loss)} shape={getattr(loss, 'shape', None)}")
                    return loss

                loss_t = opt.step(closure)
                _l1,_l2 = model.loss(xb, yb, each_loss=True)
                l1.append(float(_l1.item()))
                l2.append(float(_l2.item()))

                losses.append(float(loss_t.item()))
            else:
                opt.zero_grad(set_to_none=True)
                # pred = model(xb)
                # loss = loss_fn(pred, yb)
                loss = model.loss(xb, yb)

                # Scalar check
                if loss.ndim != 0:
                    raise ValueError(f"Loss must be scalar. Got shape {tuple(loss.shape)}")

                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

        history[name] = losses
        if len(l1)> 0:
            history_l2[name] = l2
            history_l1[name] = l1



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

    plt.figure()
    for name in history_l1.keys():
        plt.plot(history_l1[name], label=name + "_l1")
        plt.plot(history_l2[name], label=name + "_l2")
    plt.xlabel("Step")
    plt.ylabel("Loss (MSE)")
    plt.title("Optimizer loss convergence")
    plt.legend(frameon=False)
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    return history


from learning.SAC.sac_model import QNetwork, TanhGaussianPolicy
from learning.utils import ProspectReplayBuffer, Schedule, soft_update, hard_update, Logger
from src.env.continous_nav_env_delay_comp import ContinousNavEnv

class DummyAgent():
    def __init__(self, env,
        batch_size: int = 256,
        replay_sz: int = 30_000,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_lr: float = 1e-4, # 3e-4,
        q_lr: float =  5e-4,
        alpha_init: float = 0.2,
        alpha_lr: float = 3e-4, #  1e-4,
        automatic_entropy_tuning: bool = True,
        grad_clip: Optional[float] = None,

        # target_entropy: Optional[float] = None,
        target_entropy_scale: float = 0.98,
        randstart_sched = (1,0.25,250),
        rshape_sched = (1,1,500),
        # rshape_epis: int = 1000,
        # random_start_epis: int = 1000,
        warmup_steps: int = 2_000,
        updates_per_step: int = 1,
        # num_hidden_layers: int = 5,
        # size_hidden_layers: int = 256,
        num_hidden_layers: int = 5,
        size_hidden_layers: int = 256,
        loads: Optional[str] = None):

        seed = 80
        self.env = env

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.a_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        self.a_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=self.device)

        self.q1 = QNetwork(self.state_dim, self.action_dim, hidden_dim=size_hidden_layers,
                           n_hidden=num_hidden_layers).to(self.device)
        self.q2 = QNetwork(self.state_dim, self.action_dim, hidden_dim=size_hidden_layers,
                           n_hidden=num_hidden_layers).to(self.device)
        self.q1_targ = QNetwork(self.state_dim, self.action_dim, hidden_dim=size_hidden_layers,
                                n_hidden=num_hidden_layers).to(self.device)
        self.q2_targ = QNetwork(self.state_dim, self.action_dim, hidden_dim=size_hidden_layers,
                                n_hidden=num_hidden_layers).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        # self.q_opt = SPSA(list(self.q1.parameters()) + list(self.q2.parameters()))
        self.q_opt = SPSA(list(self.q1.parameters()) + list(self.q2.parameters()))
        # self.q_opt = SPSA_OG(list(self.q1.parameters()) + list(self.q2.parameters()))
        # self.q_opt = SPSA_logger(list(self.q1.parameters()) + list(self.q2.parameters()), log_losses=True)
        # self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=q_lr)
        # self.q_opt = torch.optim.RMSprop(list(self.q1.parameters()) + list(self.q2.parameters()), lr=q_lr)

        self.policy = TanhGaussianPolicy(
            self.state_dim,
            self.action_dim,
            action_low  = self.a_low,
            action_high = self.a_high,
            hidden_dim  = size_hidden_layers,
            n_hidden    = num_hidden_layers,
        ).to(self.device)

        self.replay = ProspectReplayBuffer(
            self.state_dim,
            self.action_dim,
            n_samples=env.n_samples,
            size=replay_sz,
            device=self.device,
            seed =seed
        )
        self.replay.load('./SAC/debug_replay.pkl')
        self.alpha = torch.tensor(alpha_init, dtype=torch.float32, device=self.device)
        self.gamma = gamma
        self.q_loss_fn = F.smooth_l1_loss
        # self.q_loss_fn = F.mse_loss

    def test(self, n_updates=1000, batch_size=256):
        update_losses = []

        update_loss_hists = np.empty([n_updates, self.q_opt.n]) if isinstance(self.q_opt, (SPSA,SPSA_logger,SPSA_OG)) else []
        plt.ion()

        o1, a1, r_prospects, o2, d_prospects = self.replay.sample(batch_size)

        for _update in range(n_updates):
            print(f"\rUpdate step: {_update}", end='')
            # o1, a1, r_prospects, o2, d_prospects = self.replay.sample(batch_size)



            # o1, a1, r_prospects, o2, d_prospects = self.replay.sample(batch_size)

            # unpack prospects: r_vals, r_probs both [B, N]
            r_vals, r_probs = [r_prospects[:, i, :] for i in range(2)]


            # robust done shape handling
            d = d_prospects
            if d.dim() == 3 and d.shape[1] == 1:
                d = d.squeeze(1)
            if d.dim() == 2 and d.shape[1] == 1:
                d = d.expand(-1, r_vals.shape[1])

            # scale rewards to [reward_min_step, reward_max_step]
            rmax,rmin = self.env.reward_max_step, self.env.reward_min_step
            r_vals = (r_vals - rmin) / (rmax - rmin)  # scale to [0,1]
            # print(f'r_vals min/max: {r_vals.min().item():.3f}/{r_vals.max().item():.3f}')

            # ---------------------- target ----------------------
            y = self._get_td_target_expectations(o1, a1, o2, r_vals, r_probs, d)  # [B]



            if isinstance(self.q_opt, (SPSA,SPSA_logger,SPSA_OG)):
                def closure():
                    q1 = self.q1(o1, a1).squeeze(-1)  # [B]
                    q2 = self.q2(o1, a1).squeeze(-1)  # [B]
                    loss = self.q_loss_fn(q1, y) + self.q_loss_fn(q2, y)
                    return loss

                loss_q = self.q_opt.step(closure)
                update_losses.append(float(loss_q.item()))

                # try:
                #     loss_q = self.q_opt.step(closure)
                #     update_losses.append(float(loss_q.item()))
                #     self.q_opt.plot_losses()
                # except:
                #     loss_hist = self.q_opt.step_hist(closure)
                #     update_loss_hists[_update,:] = loss_hist
                #     update_losses.append(float(loss_hist[-1]))
                #     # assert torch.isfinite(loss_q).all(), "Non-finite critic loss!"
                #     # assert torch.isfinite(y).all(), "Non-finite td-target values!"
            else:
                q1 = self.q1(o1, a1).squeeze(-1)  # [B]
                q2 = self.q2(o1, a1).squeeze(-1)  # [B]
                loss_q = self.q_loss_fn(q1, y) + self.q_loss_fn(q2, y)
                # assert torch.isfinite(q1).all(), "Non-finite Q1 values!"
                # assert torch.isfinite(q2).all(), "Non-finite Q2 values!"
                assert torch.isfinite(loss_q).all(), "Non-finite critic loss!"
                self.q_opt.zero_grad(set_to_none=True)
                loss_q.backward()
                self.q_opt.step()
                update_losses.append(float(loss_q.item()))

        # if isinstance(self.q_opt, (SPSA, SPSA_logger, SPSA_OG)):
        #     plt.figure()
        #     for lhist in update_loss_hists[:min(100, n_updates)]:
        #         plt.plot(lhist, color='gray', alpha=0.3)
        #     plt.xlabel("SPSA Inner Step")
        #     plt.ylabel("Loss")
        #     plt.title("Critic Loss Convergence per SPSA Step")
        #     plt.grid(True)

        #
        # # plot loss curve
        plt.figure()
        plt.plot(update_losses, label="Critic Loss")
        plt.xlabel("Outer  Step")
        plt.ylabel("Loss")
        plt.title(f"Critic Loss Convergence: {self.q_opt.__class__.__name__}")
        plt.legend(frameon=False)
        plt.grid(True)
        plt.tight_layout()
        plt.ioff()
        plt.show()


    def _get_td_target_expectations(self,o1,a1,o2,r_vals,r_probs,d):
        with torch.no_grad():
            a2, logp2, _ = self.policy.sample(o2)  # a2 [B,A], logp2 [B,1]
            q1_t = self.q1_targ(o2, a2)
            q2_t = self.q2_targ(o2, a2)
            min_q_t = torch.min(q1_t, q2_t).squeeze(-1)  # [B]
            ent_term = (self.alpha.detach() * logp2).squeeze(-1)  # [B]
            target_v = min_q_t - ent_term  # [B]
            td_targets = r_vals + (1.0 - d) * self.gamma * target_v.unsqueeze(1)  # [B,N]
            assert torch.sum(r_probs, dim=1).allclose(
                torch.ones(r_probs.shape[0], device=self.device)), "Reward probs do not sum to 1!"
            y = self._risk_measure(td_targets, r_probs)  # [B]
        return y

    def _risk_measure(self, vals: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return torch.mean(vals, dim=1)


def test_replay():
    layout = 'spath'
    env_config = {}
    env_config['dynamics_belief'] = {
        'b_min_lin_vel': (0.0, 1e-6),
        'b_max_lin_vel': (1.5, 0.5),
        'b_max_lin_acc': (0.5, 0.2),
        'b_max_rot_vel': (math.pi / 2.0, math.pi / 6.0)
        # 'b_min_lin_vel': (0.0, 1e-6),
        # 'b_max_lin_vel': (1.0, 1e-6),
        # 'b_max_lin_acc': (0.5, 1e-6),
        # 'b_max_rot_vel': (np.pi / 4, 1e-6)
    }

    env_config['dt'] = 0.1
    env_config['delay_steps'] = 10  # (two-way delay)  delay_time = delay_steps * dt
    env_config['n_samples'] = 500  # 50
    env_config['vgraph_resolution'] = 50
    env_config['max_steps'] = 600

    # OG Config
    agent_config = {}
    env = ContinousNavEnv.from_layout(layout, **env_config)
    agent = DummyAgent(env)

    agent.test()





# Example:
if __name__ == "__main__":
    test_replay()

    # history = compare_spsa_to_standard_optimizers(n_steps=400, n_samples=4096, batch_size=256, seed=0)

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
