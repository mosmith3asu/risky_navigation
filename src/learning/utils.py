import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
class OUNoiseBounded:
    """
    Ornstein–Uhlenbeck process with bounded state.

    x_{t+dt} = x_t + theta*(mu - x_t)*dt + sigma*sqrt(dt)*N(0, I)

    Parameters
    ----------
    shape : tuple or int
        Shape of the noise state (e.g., action dimension).
    mu : float or array-like, default 0.0
        Long-run mean (broadcastable to `shape`).
    theta : float, default 0.15
        Mean-reversion rate.
    sigma : float or array-like, default 0.2
        Diffusion scale (broadcastable to `shape`).
    dt : float, default 1e-2
        Time step.
    x0 : None or array-like
        Initial state (bounded). If None, starts at `mu` squashed/trimmed into bounds.
    low : float or array-like, default -1.0
        Lower bound(s).
    high : float or array-like, default 1.0
        Upper bound(s).
    mode : {"clip", "reflect", "squash"}, default "reflect"
        Bounding mode:
          - "clip": clip state into [low, high]
          - "reflect": reflect back into [low, high] if overshoot occurs
          - "squash": evolve latent z (unbounded OU), then x = low + (high-low)*(tanh(z)+1)/2
    seed : int or None
        Random seed.

    Notes
    -----
    - For "squash" mode, the visible state `x` is always in [low, high] but internally the OU runs on `z`.
    - For "reflect" mode, large overshoots are handled without bias using modulo reflection.

    Methods
    -------
    reset(x0=None): Reset the process state.
    sample(): Step once and return the new bounded state.
    __call__(): Alias of `sample()`.
    state: Property to inspect current bounded state.
    """

    def __init__(self, shape,
                 mu=0.0, theta=0.15, sigma=0.2, dt=1e-2,
                 sigma_min=0.05, sigma_decay=.995,
                 x0=np.array([0,0]),
                 low=(-1.0,0.0), high=(1.0, 1.0),
                 mode="reflect",
                 device = 'cpu',
                 decay_freq= None,
                 seed=None):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.theta = float(theta)
        self.dt = float(dt)
        self.mu = np.broadcast_to(np.asarray(mu, dtype=float), self.shape)
        self.sigma = np.broadcast_to(np.asarray(sigma, dtype=float), self.shape)
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.device = device
        self.low = np.broadcast_to(np.asarray(low, dtype=float), self.shape)
        self.high = np.broadcast_to(np.asarray(high, dtype=float), self.shape)

        self._sigma0 = self.sigma.copy()
        self.decay_freq = decay_freq # if none, decay on reset
        self.decay_count = 1

        if np.any(self.high <= self.low):
            raise ValueError("All elements of `high` must be greater than `low`.")

        self.mode = str(mode).lower()
        if self.mode not in {"clip", "reflect", "squash"}:
            raise ValueError("mode must be 'clip', 'reflect', or 'squash'.")

        # self.rng = np.random.default_rng(seed)

        # State variables
        self._x = None           # visible (bounded) state
        self._z = None           # latent (for squash mode)
        self._x0 = x0
        self.reset()

    # ---------- Public API ----------
    def reset_sigma(self):
        self.sigma = self._sigma0.copy()
        self.decay_count = 1

    def reset(self):
        x0 = self._x0
        """Reset the process. If x0 is None, start from mu (bounded)."""
        if self.mode == "squash":
            # initialize latent z so that tanh(z) maps near mu within bounds
            # invert the squash around midpoint:
            mid = (self.low + self.high) / 2.0
            half = (self.high - self.low) / 2.0
            target = np.clip((self.mu - mid) / np.maximum(half, 1e-12), -0.999999, 0.999999)
            self._z = np.arctanh(target)
            self._x = self._squash(self._z) if x0 is None else self._clip_into_bounds(np.asarray(x0, float))
        else:
            start = self.mu if x0 is None else np.asarray(x0, dtype=float)
            self._x = self._clip_into_bounds(start)
            self._z = None

        # Decay sigma exploration (on reset)
        if self.decay_freq is None:
            dsig = - self.sigma * (1 - self.sigma_decay)
            sigma = self.sigma + (dsig)
            self.sigma = np.max([self.sigma_min*np.ones_like(sigma), sigma],axis=0)

        return self._x.copy()

    def sample(self):
        """Advance one step and return the new bounded state."""
        # n = self.rng.standard_normal(self.shape)
        n = np.random.standard_normal(self.shape)
        if self.mode == "squash":
            # OU on latent z, then squash to bounded x
            x = self._z.copy()
            self._z = x + self.theta * (self.mu - x) + self.sigma * n
            # self._z = self._z + self.theta * (0.0 - self._z) * self.dt + self.sigma * np.sqrt(self.dt) * n
            self._x = self._squash(self._z)
        else:
            # OU directly on bounded x, then bound via clip or reflect
            # x = self._x + self.theta * (self.mu - self._x) * self.dt + self.sigma * np.sqrt(self.dt) * n
            x = self._x.copy()
            x = x + self.theta * (self.mu - x) + self.sigma * n


            if self.mode == "clip":
                self._x = self._clip_into_bounds(x)
            elif self.mode == "reflect":
                self._x = self._reflect_into_bounds(x)

        # Decay sigma exploration (on sample)
        if self.decay_freq is not None and (self.decay_count % self.decay_freq == 0):
            dsig = - self.sigma * (1 - self.sigma_decay)
            sigma = self.sigma + (dsig)
            self.sigma = np.max([self.sigma_min*np.ones_like(sigma), sigma],axis=0)
            self.decay_count = 1
        else:
            self.decay_count += 1


        # return torch.tensor(self._x.copy(), dtype=torch.float32, device=self.device)
        return torch.tensor(self._x.copy(), dtype=torch.float32, device=self.device)

    def __call__(self):
        return self.sample()

    @property
    def state(self):
        """Current bounded state (copy)."""
        return None if self._x is None else self._x.copy()

    # ---------- Utilities ----------

    def _clip_into_bounds(self, x):
        return np.clip(x, self.low, self.high)

    def _reflect_into_bounds(self, x):
        """
        Vectorized reflection into [low, high] handling large overshoots:
        Maps x using sawtooth reflection with period 2*width.
        """
        width = self.high - self.low
        # Avoid /0 if any widths are zero (already checked above, but guard anyway)
        width = np.maximum(width, 1e-12)
        y = (x - self.low) % (2.0 * width)
        y_ref = np.where(y <= width, self.low + y, self.high - (y - width))
        return y_ref

    def _squash(self, z):
        """Map latent z in R smoothly to [low, high] via tanh."""
        half = (self.high - self.low) / 2.0
        mid = (self.high + self.low) / 2.0
        return mid + half * np.tanh(z)

    def __repr__(self):
        return self.__str__()
    def __str__(self):

        s = f"OUNoiseBounded(shape={self.shape}, mode='{self.mode}')\n" \
                f"\t| mu={self.mu}\n " \
                f"\t| theta={self.theta}\n" \
                f"\t| sigma={self.sigma}\n, " \
                f"\t| sigma_decay={self.sigma_decay}\n" \
                f"\t| decay_freq:{self.decay_freq }\n" \
                f"\t| sigma_min={self.sigma_min}\n" \
                f"\t| low={self.low}, high={self.high}\n"
        return s

class OUNoise:
    """
    Ornstein–Uhlenbeck noise for exploration.
    Handles both single env and vectorized envs by broadcasting on shape.
    - theta: rate of mean reversion (returns noise to mu)
    - sigma: scale of noise
    """
    def __init__(self, shape, mu=0.0, theta=0.15, sigma=0.5, n_envs=1,
                 sigma_min = 0.05, device='cpu', sigma_decay=.995,
                 decay_freq= None,
                 sigma_sched = None,
                 **kwargs):


        self.action_dim = shape
        self.shape = shape
        self.mu = mu
        self.theta = theta

        self._sigma0 = sigma
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.decay_freq = decay_freq # if none, decay on reset
        self.decay_count = 1

        self.sigma_sched = sigma_sched
        if self.sigma_sched is not None:
            self.sigma_sched['xstart'] = self.sigma
            self.sigma_sched['xend'] = self.sigma_min
            self.sigma_sched = Schedule(**self.sigma_sched)
            # self.sigma_sched = Schedule(xstart=sigma,
            #                             xend=sigma_min,
            #                             horizon=1_000,
            #                             step_event='sample',
            #                             mode='exponential',
            #                             exp_gain=2.0
            #                             )


        # self.dt = dt
        self.n_envs = n_envs
        self.device = device
        self.size = (self.n_envs, self.action_dim)
        self.reset()

    def reset(self,i=None):
        if i is None:
            self.state = self.mu * torch.zeros(self.n_envs, self.action_dim, device=self.device)
        else:
            self.state[i] = 0

        # Decay sigma exploration (on reset)
        if self.decay_freq is None:
            self.decay_sigma()

    def sample(self, increment_decay=True):
        x = self.state
        # dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0,1,self.size,device=self.device)
        dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0, 1, self.size, device=self.device)
        self.state = x + dx

        # Decay sigma exploration (on sample)
        if self.decay_freq is not None and increment_decay:
            if self.decay_count % self.decay_freq == 0:
                self.decay_sigma()
                self.decay_count = 1
            else:
                self.decay_count += 1

        return self.state

    def simulate(self,T=600):
        n_trials = 3
        self.reset()
        # self.theta = 1
        # self.sigma = 0.05
        fig,axs = plt.subplots(self.action_dim,n_trials)

        for trial in range(n_trials):
            data = np.zeros((T, self.action_dim))
            # xy = np.array([[0.0,0.0]])

            for t in range(T):
                xy = np.array([[0.0, 0.0]])
                xy += self.sample().cpu().numpy()
                data[t,:] = xy
            # plot data
            for d in range(self.action_dim):
                axs[d,trial].plot(data[:,d])
                axs[d,trial].set_title(f'OU Noise Dimension {["x","y"][d]}')
                axs[d,trial].hlines(0, xmin=0, xmax=T, colors='k', linestyles='dashed', alpha=0.5)
                axs[d,trial].set_xlabel('Timestep')


        plt.show()

    def reset_sigma(self):
        self.sigma = self._sigma0
        self.decay_count = 1

    def decay_sigma(self):
        if self.sigma_sched is not None:
            self.sigma = self.sigma_sched.sample(advance=True)
            return
        dsig = - self.sigma * (1 - self.sigma_decay)
        sigma = self.sigma + (dsig)
        self.sigma = np.max([self.sigma_min * np.ones_like(sigma), sigma], axis=0)

    def __repr__(self):
        return self.__str__()
    def __str__(self):

        s = f"\nOUNoise(shape={self.shape})\n" \
                f"\t| mu={self.mu}\n " \
                f"\t| theta={self.theta}\n" \
                f"\t| sigma={self.sigma}\n, " \
                f"\t| sigma_decay={self.sigma_decay}\n" \
                f"\t| decay_freq:{self.decay_freq }\n" \
                f"\t| sigma_min={self.sigma_min}\n"
        return s

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=int(1e6), device='cpu'):
        self.device = device
        self.size = size
        self.ptr = 0
        self.full = False
        self.s = torch.zeros((size, obs_dim),   dtype=torch.float32, device=device)
        self.a = torch.zeros((size, act_dim),   dtype=torch.float32, device=device)
        self.r = torch.zeros((size, 1),         dtype=torch.float32, device=device)
        self.s2 = torch.zeros((size, obs_dim),  dtype=torch.float32, device=device)
        self.d = torch.zeros((size, 1),         dtype=torch.float32, device=device)

    def add(self, s, a, r, s2, d):
        n = s.shape[0] if s.ndim == 2 else 1
        # ensure batched
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).view(-1, 1)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).view(-1, 1)

        idxs = (self.ptr + torch.arange(n, device=self.device)) % self.size
        self.s[idxs] = s
        self.a[idxs] = a
        self.r[idxs] = r
        self.s2[idxs] = s2
        self.d[idxs] = d

        self.ptr = (self.ptr + n) % self.size
        if self.ptr == 0:
            self.full = True

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch_size):
        max_idx = self.size if self.full else self.ptr
        idxs = torch.randint(0, max_idx, (batch_size,), device=self.device)
        return self.s[idxs], self.a[idxs], self.r[idxs], self.s2[idxs], self.d[idxs]

class ProspectReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, n_samples, size=int(1e6), device='cpu'):
        super().__init__(obs_dim, act_dim, size, device)
        prospect_dim = 2 # [reward, prob]
        self.r = torch.zeros((size, prospect_dim, n_samples), dtype=torch.float32, device=device)
        self.d = torch.zeros((size, 1, n_samples), dtype=torch.float32, device=device)

    def add(self, s, a, r_prospects, s2, d_prospects):
        n = s.shape[0] if s.ndim == 2 else 1
        # ensure batched
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r_prospects = torch.as_tensor(r_prospects, dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d_prospects = torch.as_tensor(d_prospects, dtype=torch.float32, device=self.device).view(-1, 1)



        idxs = (self.ptr + torch.arange(n, device=self.device)) % self.size
        self.s[idxs] = s
        self.a[idxs] = a
        self.r[idxs] = r_prospects
        self.s2[idxs] = s2
        self.d[idxs] = d_prospects.reshape(1,-1)

        self.ptr = (self.ptr + n) % self.size
        if self.ptr == 0:
            self.full = True

class Schedule:
    """
    RL-friendly schedule that evolves a scalar from xstart -> xend over a fixed horizon.

    Modes
    -----
    - 'linear':        linear interpolation from xstart to xend.
    - 'exponential':   geometric interpolation (handles signs and zeros robustly).
    - 'decay':         hyperbolic decay toward xend: x_t = xend + (xstart-xend)/(1+decay_rate*t)

    Step Events
    -----------
    - 'reset':  advance one schedule step when reset() is called (e.g., per-episode schedule).
    - 'step':   advance one schedule step when step() is called (e.g., per-environment step).
    - 'sample': advance one schedule step when sample() is called (e.g., per-optimizer step).
    - None:     schedule does not auto-advance; you can manually call advance(n).
    """

    def __init__(self, xstart, xend, horizon,
                 step_event='sample', mode='exponential',
                 decay_rate=None,exp_gain=1.0, name=''):
        assert step_event in ['reset', 'sample', None], f"Unknown step_event [{step_event}] must be 'reset', 'sample', or None"
        assert mode in ['exponential', 'linear', 'decay','manual']
        if mode == 'decay':
            assert decay_rate is not None, "decay_rate must be specified for 'decay' mode"
        assert horizon >= 0, "horizon must be >= 0"
        self.is_trivial = horizon == 0

        if decay_rate is not None and mode=='decay' and decay_rate >= 0.5:
            warnings.warn(f"High (possibly inverse) decay_rate={decay_rate} detected. "
                          f"Try values decay_rate<= 0.01 for smoother decay. ")

        self.name = name
        self._step_event = step_event
        self._mode = mode
        self.horizon = int(horizon) if not self.is_trivial else 100

        self._x0 = float(xstart) if not self.is_trivial else float(xend)
        self._xf = float(xend)
        self.state = float(xstart)

        self._decay_rate = decay_rate
        self._exp_gain = exp_gain
        self._schedule_step = 0
        self._schedule = self._precomputed_schedule()

    # ---------- internals ----------
    def _precomputed_schedule(self):
        """Precompute and return a numpy array of length `horizon`."""

        T = self.horizon
        if T == 1:
            return np.array([self._x0], dtype=float)
        elif self.is_trivial:
            return self._xf * np.ones(T)

        if self._mode == 'linear':
            # inclusive endpoints across T points
            sched = np.linspace(self._x0, self._xf, T, dtype=float)

        elif self._mode == 'exponential':
            # Geometric interpolation that is robust to zeros and sign changes.
            # If xstart and xend have the same sign and both non-zero:
            if self._x0 != 0.0 and self._xf != 0.0 and np.sign(self._x0) == np.sign(self._xf):
                # ratio goes 0..1 across T points
                r = np.linspace(0.0, 1.0, T, dtype=float)
                r = np.power(r, 1/self._exp_gain)
                sched = self._x0 * ((self._xf / self._x0) ** r)
            else:
                # Fall back to smooth exponential easing via exp(-k * t)
                # x_t = x_f + (x_0 - x_f) * exp(-k * t), t in [0,1]
                k = 5.0
                tau = np.linspace(0.0, 1.0, T, dtype=float)
                sched = self._xf + (self._x0 - self._xf) * np.exp(-k * tau)

        elif self._mode == 'decay':
            # Hyperbolic decay toward x_f. At t=0 => x0; as t grows, -> x_f.
            # t = np.arange(T, dtype=float)
            t = np.array([1], dtype=float)
            sched = self._xf + (self._x0 - self._xf) / (1.0 + self._decay_rate * t)
            # continue decay until self.xf is reached


        else:
            raise ValueError(f"Unknown mode {self._mode}")

        return sched.astype(float)

    # ---------- helpers ----------
    def _clamp_step(self):
        if self._schedule_step < 0:
            self._schedule_step = 0

        if self._mode == 'decay':
            # allow unbounded growth of step for decay mode
            pass
        elif self._schedule_step >= self.horizon:
            self._schedule_step = self.horizon - 1
        # else:
        #     raise ValueError(f"Unknown clamp mode for {self._mode}")


    def _apply_state_from_step(self):
        self._clamp_step()

        if self._mode == 'decay':
            t = float(self._schedule_step)
            self.state = self._xf + (self._x0 - self._xf) / (1.0 + self._decay_rate * t)
        else:
            self.state = float(self._schedule[self._schedule_step])

    def advance(self, n=1):
        """Manually advance the schedule by n steps (can be negative)."""
        self._schedule_step = int(self._schedule_step + n)
        self._apply_state_from_step()
        return self.state

    # ---------- API ----------
    def reset(self):
        """Resets the schedule to initial state; optionally advances on reset-event."""
        self._schedule_step = 0
        self._apply_state_from_step()
        if self._step_event == 'reset':
            # Advance one step so next episode uses the next scheduled value
            self.advance(1)
        return self.state

    # def step(self):
    #     """Advance schedule if step_event == 'step'."""
    #     if self._step_event == 'step':
    #         self.advance(1)
    #     return self.state

    def sample(self,at = None, advance = True):
        """Return current value; advance if step_event == 'sample'."""
        if at is not None:
            return self.at(at)

        val = self.state
        if self._step_event == 'sample' and advance:
            self.advance(1)
        return val

    def value(self):
        """Current scheduled value without advancing."""
        return self.state

    def at(self, idx):
        """Peek the schedule value at absolute index idx (clamped)."""
        if self._mode == 'decay':
            t = float(idx)
            return self._xf + (self._x0 - self._xf) / (1.0 + self._decay_rate * t)

        i = int(np.clip(idx, 0, self.horizon - 1))
        return float(self._schedule[i])

    def plot(self, ax= None, T=None, show=True, label=None):
        """Plots the schedule over T steps with a vertical line at current schedule step."""
        label = self._mode if label is None else label

        T = self.horizon if T is None else int(T)
        T = max(1, T)
        # Extend/trim displayed schedule to length T (repeat last value if T > horizon)
        if self._mode == 'decay':
            t = np.arange(T, dtype=float)
            y = self._xf + (self._x0 - self._xf) / (1.0 + self._decay_rate * t)
        elif T <= self.horizon:
            y = self._schedule[:T]
        else:
            pad = np.full(T - self.horizon, self._schedule[-1], dtype=float)
            y = np.concatenate([self._schedule, pad], axis=0)

        x = np.arange(T)
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(x, y, linewidth=2, label=label)
        # vertical line at current step (clamped to T-1)
        v = int(np.clip(self._schedule_step, 0, T - 1))
        ax.axvline(v, linestyle='--')
        ax.set_title(f"Schedule ({self._mode}, step_event={self._step_event})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        # ax.tight_layout()
        if show:
            plt.show()
        # plt.figure()
        # plt.plot(x, y, linewidth=2)
        # # vertical line at current step (clamped to T-1)
        # v = int(np.clip(self._schedule_step, 0, T - 1))
        # plt.axvline(v, linestyle='--')
        # plt.title(f"Schedule ({self._mode}, step_event={self._step_event})")
        # plt.xlabel("Step")
        # plt.ylabel("Value")
        # plt.tight_layout()
        # plt.show()

    # ---------- dunder ----------mode={self._mode}
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = f"Schedule(name = {self.name}) \n" \
            f"\t| step_event={self._step_event} \n" \
            f"\t| horizon={0 if self.is_trivial else self.horizon}\n" \
            f"\t| x0={self._x0} --> xf={self._xf}\n" \
            f"\t| mode={self._mode} "

        if self._mode == 'decay':
            s += f" - decay_rate={self._decay_rate}"
        elif self._mode == 'exponential':
            s += f"- exp_gain={self._exp_gain}"
        return s





def main():
    fig,axs = plt.subplots(1,2, figsize=(12,9))
    sched = Schedule(xstart=0.5, xend=0.05, horizon=1000, mode='exponential',exp_gain=2.0)
    sched.plot(axs[0], show =False, T=1000, label='exponential0', )
    print(sched)

    # sched = Schedule(xstart=1.0, xend=0.1, horizon=600, mode='exponential',exp_gain=2.0)
    # sched.plot(axs[0], show=False, T=800, label='exponential1')
    #
    # sched = Schedule(xstart=1.0, xend=0.1, horizon=600, mode='linear')
    # sched.plot(axs[0], show =False, T=800)
    #
    # sched = Schedule(xstart=1.0, xend=0.1, horizon=600, decay_rate=0.005, mode='decay')
    # sched.plot(axs[0], show =False, T=800)
    axs[0].legend()
    plt.show()
    pass

if __name__ == '__main__':
    main()

# class OUNoise:
#     """
#     Ornstein–Uhlenbeck noise for exploration.
#     Handles both single env and vectorized envs by broadcasting on shape.
#     - theta: rate of mean reversion (returns noise to mu)
#     - sigma: scale of noise
#     """
#     def __init__(self, shape, mu=0.0, theta=0.15, sigma=0.5, n_envs=1,
#                  sigma_min = 0.05, sigma_decay=.995, device='cpu', decay_freq= None,
#                  precompute = True, precompute_steps=600,
#                  precompute_low=None, precompute_high=None,
#                  **kwargs):
#
#
#         self.action_dim = shape
#         self.shape = shape
#         self.mu = mu
#         self.theta = theta
#
#         self._sigma0 = sigma
#         self.sigma = sigma
#         self.sigma_min = sigma_min
#         self.sigma_decay = sigma_decay
#         self.decay_freq = decay_freq # if none, decay on reset
#         self.decay_count = 1
#
#
#         # self.dt = dt
#         self.n_envs = n_envs
#         self.device = device
#         self.size = (self.n_envs, self.action_dim)
#         self.state = self.mu * torch.zeros(self.n_envs, self.action_dim, device=self.device)
#
#         # Precomputing and scaling
#         self.precompute = precompute
#         low = precompute_low
#         high = precompute_high
#         self.low = np.broadcast_to(np.asarray(low, dtype=float), self.shape) if low is not None else None
#         self.high = np.broadcast_to(np.asarray(high, dtype=float), self.shape) if high is not None else None
#         self._precompute_steps = precompute_steps
#
#         self._precomputed_noise = None
#         self._precompute_step = 0
#         if precompute:
#             self._precomputed_noise = self._precompute_noise()
#
#
#         # Reset
#
#         self.reset()
#
#
#
#     def _precompute_noise(self):
#         assert self.precompute, "Precompute flag is false. Should not be calling this method."
#         precomputed_noise = torch.zeros((self._precompute_steps, self.n_envs, self.action_dim), device=self.device)
#         for t in range(self._precompute_steps):
#             noise = self.sample()
#             precomputed_noise[t] = noise
#         # Scale precomputed noise into bounds if specified
#         if self.low is not None and self.high is not None:
#             dim_max = precomputed_noise.max()
#             dim_min = precomputed_noise.min()
#             # Scale each dimension separately
#             for d in range(self.action_dim):
#                 # dim_max = self.precomputed_noise[:,:,d].max()
#                 # dim_min = self.precomputed_noise[:,:,d].min()
#                 scale = (self.high[d] - self.low[d]) / (dim_max - dim_min + 1e-12)
#                 precomputed_noise[:,:,d] = self.low[d] + (precomputed_noise[:,:,d] - dim_min) * scale
#         return precomputed_noise
#
#     def reset(self,i=None):
#         if i is None:
#             self.state = self.mu * torch.zeros(self.n_envs, self.action_dim, device=self.device)
#         else:
#             self.state[i] = 0
#
#         # Decay sigma exploration (on reset)
#         if self.decay_freq is None:
#             dsig = - self.sigma * ( 1-self.sigma_decay)
#             sigma = self.sigma + (dsig)/self.n_envs
#             self.sigma = max(self.sigma_min, sigma)
#
#         # Handle precomputation
#
#         self._precomputed_noise = None
#         self._precompute_step = 0
#         if self.precompute:
#             self._precomputed_noise = self._precompute_noise()
#
#     def sample(self):
#         if self.precompute and self._precomputed_noise is not None:
#             assert self._precompute_step <= self._precompute_steps, "Precomputed steps exceeded."
#             noise = self._precomputed_noise[self._precompute_step]
#             self._precompute_step += 1
#             return noise
#
#
#
#         x = self.state
#         # dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0,1,self.size,device=self.device)
#         dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0, 1, self.size, device=self.device)
#         self.state = x + dx
#
#         # Decay sigma exploration (on sample)
#         if self.decay_freq is not None:
#             if self.decay_count % self.decay_freq == 0:
#                 dsig = - self.sigma * (1 - self.sigma_decay)
#                 sigma = self.sigma + (dsig)
#                 self.sigma = np.max([self.sigma_min * np.ones_like(sigma), sigma], axis=0)
#                 self.decay_count = 1
#             else:
#                 self.decay_count += 1
#
#         return self.state
#
#     def simulate(self,T=600):
#         n_trials = 3
#         self.reset()
#         # self.theta = 1
#         # self.sigma = 0.05
#         fig,axs = plt.subplots(self.action_dim,n_trials)
#
#         for trial in range(n_trials):
#             self.reset()
#             data = np.zeros((T, self.action_dim))
#             # xy = np.array([[0.0,0.0]])
#
#             for t in range(T):
#                 xy = np.array([[0.0, 0.0]])
#                 xy += self.sample().cpu().numpy()
#                 data[t,:] = xy
#             # plot data
#             for d in range(self.action_dim):
#                 axs[d,trial].plot(data[:,d])
#                 axs[d,trial].set_title(f'OU Noise Dimension {["x","y"][d]}')
#                 axs[d,trial].hlines(0, xmin=0, xmax=T, colors='k', linestyles='dashed', alpha=0.5)
#                 axs[d,trial].set_xlabel('Timestep')
#
#
#         plt.show()
#
#     def reset_sigma(self):
#         self.sigma = self._sigma0
#         self.decay_count = 1
#     def __repr__(self):
#         return self.__str__()
#     def __str__(self):
#
#         s = f"OUNoise(shape={self.shape})\n" \
#                 f"\t| mu={self.mu}\n " \
#                 f"\t| theta={self.theta}\n" \
#                 f"\t| sigma={self.sigma}\n, " \
#                 f"\t| sigma_decay={self.sigma_decay}\n" \
#                 f"\t| decay_freq:{self.decay_freq }\n" \
#                 f"\t| sigma_min={self.sigma_min}\n"\
#                 f"\t| precompute={self.precompute}\n"
#         return s
