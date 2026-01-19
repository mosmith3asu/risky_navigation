import numpy as np
from numba import njit, float64, int64, boolean
from numba.experimental import jitclass
import math
import warnings
# from scipy.stats import norm
from numba import njit, prange
# Parameter indices (for readability)
MIN_LIN_VEL = 0
MAX_LIN_VEL = 1
MAX_LIN_ACC = 2
MAX_ROT_VEL = 3
AX_NOISE = 4
AY_NOISE = 5

# -----------------------------
# Helper: normal pdf (JIT-safe)
# -----------------------------

@njit
def _logsumexp_1d(a):
    """Numba-safe logsumexp for 1D arrays."""
    n = a.shape[0]
    if n == 0:
        raise ValueError("logsumexp input is empty")

    # max
    m = a[0]
    for i in range(1, n):
        if a[i] > m:
            m = a[i]

    # sum exp(a - m)
    s = 0.0
    for i in range(n):
        s += np.exp(a[i] - m)

    return m + np.log(s)


@njit
def diag_gaussian_weights(Xs, mu, std, normalize=True, min_std=1e-12, check_finite=True):
    """
    Numba-compatible weights for a diagonal multivariate Gaussian.

    Parameters
    ----------
    Xs : (N, D) float array
    mu : (D,) float array
    std : (D,) float array, must be > 0
    normalize : if True, weights sum to 1 across samples
    min_std : rejects extremely small std values for numerical safety
    check_finite : if True, rejects NaN/inf in inputs

    Returns
    -------
    weights : (N,) float array
    logpdf  : (N,) float array
    """
    # ---- Dimension checks ----
    if Xs.ndim != 2:
        raise ValueError("Xs must be 2D (N, D)")
    if mu.ndim != 1:
        raise ValueError("mu must be 1D (D,)")
    if std.ndim != 1:
        raise ValueError("std must be 1D (D,)")

    # ikeep = []
    # for i in range(std.shape[0]):
    #     if not (std[i] < 1 / np.sqrt(2 * np.pi)):
    #         ikeep.append(i)
    # assert len(ikeep) >0, "All stds are too small"
    #
    # Xs = np.array([Xs[i] for i in ikeep])
    # mu = np.array([mu[i] for i in ikeep])
    # std = np.array([std[i] for i in ikeep])

    N = Xs.shape[0]
    D = Xs.shape[1]

    if N <= 0:
        raise ValueError("Xs must have at least one sample (N > 0)")
    if mu.shape[0] != D:
        raise ValueError("mu length must match Xs.shape[1]")
    if std.shape[0] != D:
        raise ValueError("std length must match Xs.shape[1]")

    # ---- Value checks ----
    for d in range(D):
        sd = std[d]
        if sd <= 0.0:
            raise ValueError("std must be > 0")
        if sd < min_std:
            raise ValueError("std has entries < min_std")
        if check_finite:
            if not np.isfinite(mu[d]):
                raise ValueError("mu contains NaN/inf")
            if not np.isfinite(sd):
                raise ValueError("std contains NaN/inf")

    if check_finite:
        for i in range(N):
            for d in range(D):
                if not np.isfinite(Xs[i, d]):
                    raise ValueError("Xs contains NaN/inf")

    # ---- Precompute constants ----
    # log_norm_const = sum log(std_d)
    log_norm_const = 0.0
    for d in range(D):
        log_norm_const += np.log(std[d])

    c = D * np.log(2.0 * np.pi) + 2.0 * log_norm_const  # scalar

    # ---- Compute logpdf per sample ----
    logpdf = np.empty(N, dtype=np.float64)
    for i in range(N):
        quad = 0.0
        for d in range(D):
            z = (Xs[i, d] - mu[d]) / std[d]
            quad += z * z
        logpdf[i] = -0.5 * (quad + c)

    # ---- Convert to weights ----
    weights = np.empty(N, dtype=np.float64)
    if normalize:
        lse = _logsumexp_1d(logpdf)
        for i in range(N):
            weights[i] = np.exp(logpdf[i] - lse)
    else:
        for i in range(N):
            weights[i] = np.exp(logpdf[i])

    return weights, logpdf

@njit
def normal_pdf(x, mu, std):
    """pdf of normal distribution at x with mean mu and std dev std. Handles small std safely."""
    assert std > 0.0, "std must be positive in normal_pdf"
    z = (x - mu) / std
    denom = std * np.sqrt(2 * np.pi)
    p = np.exp(- 0.5 * z ** 2) / denom
    return p

@njit
def normal_pdf_vec(xs, mu, std):
    n = xs.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = normal_pdf(xs[i], mu, std)
    return out

# ======================================================
# JIT class spec (avoid dicts/properties; use arrays)
# ======================================================
spec = [
    ('n_samples', int64),
    ('n_params', int64),

    # Belief (mu/std) for each param (4 params total)
    ('belief_mu', float64[:]),   # shape (4,)
    ('belief_std', float64[:]),  # shape (4,)

    # Sampled parameters for each of the 4 params (shape (4, n_samples))
    ('samples', float64[:, :]),

    # Cached per-param views (current samples) for convenience (length n_samples)
    ('min_lin_vel', float64[:]),
    ('max_lin_vel', float64[:]),
    ('max_lin_acc', float64[:]),
    ('max_rot_vel', float64[:]),
    ('ax_noise', float64[:]),
    ('ay_noise', float64[:]),

    # Probability per dynamics sample
    ('robot_prob', float64[:]),

    # "True" params (use belief means as the true values, matching original)
    ('true_min_lin_vel', float64),
    ('true_max_lin_vel', float64),
    ('true_max_lin_acc', float64),
    ('true_max_rot_vel', float64),
    ('true_ax_noise', float64),
    ('true_ay_noise', float64),
]

@jitclass(spec)
class Belief_FetchRobotMDP_Compiled:
    """
    Numba-compiled version of Belief_FetchRobotMDP.

    Parameters (tuples are (mu, std)):
      - b_min_lin_vel: (mu, std)
      - b_max_lin_vel: (mu, std)
      - b_max_lin_acc: (mu, std)
      - b_max_rot_vel: (mu, std)
    """
    def __init__(self,
                 n_samples=50,
                 b_min_lin_vel=(-0.25, 1e-6),
                 b_max_lin_vel=(1.5, 0.5),
                 b_max_lin_acc=(0.5, 0.2),
                 b_max_rot_vel=(math.pi/2.0, math.pi/6.0),
                 b_ax_noise= (0.0, 0.1),
                 b_ay_noise = (0.0, 0.1)
                 # b_ax_noise=(0.0, 0.05),
                 # b_ay_noise=(0.0, 0.05)
                 ):

        self.n_samples = n_samples
        self.n_params = 6

        self.belief_mu = np.empty(self.n_params, dtype=np.float64)
        self.belief_std = np.empty(self.n_params, dtype=np.float64)

        self.belief_mu[MIN_LIN_VEL] = b_min_lin_vel[0]
        self.belief_std[MIN_LIN_VEL] = b_min_lin_vel[1]

        self.belief_mu[MAX_LIN_VEL] = b_max_lin_vel[0]
        self.belief_std[MAX_LIN_VEL] = b_max_lin_vel[1]

        self.belief_mu[MAX_LIN_ACC] = b_max_lin_acc[0]
        self.belief_std[MAX_LIN_ACC] = b_max_lin_acc[1]

        self.belief_mu[MAX_ROT_VEL] = b_max_rot_vel[0]
        self.belief_std[MAX_ROT_VEL] = b_max_rot_vel[1]

        self.belief_mu[AX_NOISE] = b_ax_noise[0]
        self.belief_std[AX_NOISE] = b_ax_noise[1]

        self.belief_mu[AY_NOISE] = b_ay_noise[0]
        self.belief_std[AY_NOISE] = b_ay_noise[1]


        # True (mean) parameters
        self.true_min_lin_vel = self.belief_mu[MIN_LIN_VEL]
        self.true_max_lin_vel = self.belief_mu[MAX_LIN_VEL]
        self.true_max_lin_acc = self.belief_mu[MAX_LIN_ACC]
        self.true_max_rot_vel = self.belief_mu[MAX_ROT_VEL]
        self.true_ax_noise = self.belief_mu[AX_NOISE]
        self.true_ay_noise = self.belief_mu[AY_NOISE]


        # Allocate arrays
        self.samples = np.empty((self.n_params, n_samples), dtype=np.float64)
        self.robot_prob = np.full(n_samples, np.nan, dtype=np.float64)

        self.min_lin_vel  = self.samples[MIN_LIN_VEL, :]
        self.max_lin_vel  = self.samples[MAX_LIN_VEL, :]
        self.max_lin_acc  = self.samples[MAX_LIN_ACC, :]
        self.max_rot_vel  = self.samples[MAX_ROT_VEL, :]
        self.ax_noise     = self.samples[AX_NOISE, :]
        self.ay_noise     = self.samples[AY_NOISE, :]



        # Initialize with one resample so fields are defined
        self.resample_dynamics()

    # ----------------------------------------------
    # Resample dynamics + update per-sample prob
    # ----------------------------------------------
    def resample_dynamics(self):
        # Sample each param value
        n = self.n_samples
        for k in range(self.n_params):
            mu = self.belief_mu[k]
            std = self.belief_std[k]
            if std == 0.0:
                self.samples[k, :] = mu
            else:
                # Gaussian samples
                self.samples[k, :] = np.random.normal(mu, std, n)


        # get weights
        deterministic_thresh = 1e-3 # 1 / np.sqrt(2 * np.pi)
        irandom = np.where(self.belief_std  > deterministic_thresh)[0]

        p, _ = diag_gaussian_weights(self.samples.T[:,irandom], self.belief_mu[irandom], self.belief_std[irandom])


        # Update cache views (already aliasing self.samples rows)
        # (No-op, but keeps intent clear)
        self.min_lin_vel  = self.samples[MIN_LIN_VEL, :]
        self.max_lin_vel = self.samples[MAX_LIN_VEL, :]
        self.max_lin_acc = self.samples[MAX_LIN_ACC, :]
        self.max_rot_vel = self.samples[MAX_ROT_VEL, :]
        self.ax_noise    = self.samples[AX_NOISE, :]
        self.ay_noise    = self.samples[AY_NOISE, :]


        # Bound sampled params to physically valid ranges
        for i in range(n):
            if self.samples[MIN_LIN_VEL, i] >0:
                self.min_lin_vel[i] = 0.0
            if self.samples[MAX_LIN_VEL, i] < 0:
                self.max_lin_vel[i] = 0.0
            if self.samples[MAX_LIN_ACC, i] < 0:
                self.max_lin_acc[i] = 0.0
            if self.samples[MAX_ROT_VEL, i] < 0:
                self.max_rot_vel[i] = 0.0

        # Sanity check prob
        assert np.all(np.isfinite(p))
        assert np.all(p <= 1.0)
        self.robot_prob = p

    # ----------------------------------------------
    # Vectorized dynamics step (matches original)
    # Xt: (N,4)  [x, y, v, theta]
    # action: (2,) [jx, jy] in [-1,1]
    # dt: float
    # is_true_state: bool
    # ----------------------------------------------
    def step_true(self, Xt, action, dt):
        return self.step(Xt, action, dt, is_true_state=True)


    def step(self, Xt, action, dt, is_true_state=False):
        # Derive N
        N = self.n_samples
        if is_true_state:
            N = 1

        # Ensure Xt is (N,4)
        # if Xt.ndim == 1:
        #     Xt = Xt.reshape(1, 4)
        # if Xt.shape[0] == 1:
        #     Xt = np.repeat(Xt ,N).reshape(N, 4)
        assert Xt.ndim == 2, f"Xt must be 2D array, got shape {Xt.shape}"
        assert Xt.shape[1] == 4, f"Xt must have shape (N,4), got shape {Xt.shape}"
        if Xt.shape[0] == 1:
            # R, C = X.shape
            # out = np.empty((R * N, C), dtype=X.dtype)

            # Xt = np.repeat(Xt.T ,N).T
            # Xt = np.tile(Xt ,(N,1))
            # Xt = np.repeat(Xt, N, axis=0)
            # Xt = np.array([Xt for _ in range(N)])
            Xt = self.repeat_axis0_2d(Xt, N)

        # Unpack action
        jx = action[0]
        jy = action[1]

        # Select parameter sets
        if is_true_state:
            # Use "true" (mean) params as scalars, then broadcast
            min_lin = np.full(N, self.true_min_lin_vel)
            max_lin = np.full(N, self.true_max_lin_vel)
            max_acc = np.full(N, self.true_max_lin_acc)
            max_rot = np.full(N, self.true_max_rot_vel)
        else:
            # Use sampled arrays
            min_lin = self.min_lin_vel
            max_lin = self.max_lin_vel
            max_acc = self.max_lin_acc
            max_rot = self.max_rot_vel

        # Unpack current state
        # x, y not needed individually before update; we use vector form
        v = Xt[:, 2].copy()
        theta = Xt[:, 3].copy()

        # Velocity reference based on sign of jy
        # (scalar condition jy) chooses which vector to use
        if jy < 0.0:
            vel_ref = np.abs(min_lin)
        else:
            vel_ref = max_lin

        # Acceleration towards target vel_ref, limited by max_acc
        vdot_raw = jy * vel_ref - v
        # clamp magnitude by max_acc
        # vdot = sign(vdot_raw) * min(|vdot_raw|, max_acc)
        abs_vdot = np.abs(vdot_raw)
        clamp = np.minimum(abs_vdot, max_acc)
        vdot = np.empty_like(vdot_raw)
        for i in range(N):
            if vdot_raw[i] >= 0.0:
                vdot[i] = clamp[i]
            else:
                vdot[i] = -clamp[i]

        thetadot = jx * max_rot

        # Integrate
        theta = theta + thetadot * dt
        v = v + vdot * dt

        # Kinematics
        xdot = v * np.cos(theta)
        ydot = v * np.sin(theta)

        # Assemble Xdot and update
        Xdot = np.empty_like(Xt)
        Xdot[:, 0] = xdot
        Xdot[:, 1] = ydot
        Xdot[:, 2] = vdot
        Xdot[:, 3] = thetadot

        Xtt = Xt + Xdot * dt
        return Xtt

    def step_biased(self, Xt, action, dt, is_true_state=False, noise_std=0.05):
        # Derive N
        N = self.n_samples

        if is_true_state: N = 1
        assert Xt.ndim == 2, f"Xt must be 2D array, got shape {Xt.shape}"
        assert Xt.shape[1] == 4, f"Xt must have shape (N,4), got shape {Xt.shape}"
        if Xt.shape[0] == 1:
            Xt = self.repeat_axis0_2d(Xt, N)

        # Unpack action
        # jx = np.clip(action[0] + action_bias[:,0],-1.0, 1.0)
        # jy = np.clip(action[1] + action_bias[:,1],-1.0, 1.0)
        jx = action[0] + self.ax_noise
        jy = action[1] + self.ay_noise
        # !!! do not need to clamp here, simulating uncertainty in dynamics which allows jx>1 or < -1

        # reflection instead of clipping ---------
        # jx[jx > 1.0]  += - 2 * (jx[jx > 1.0]  - 1)
        # jy[jy > 1.0]  += - 2 * (jy[jy > 1.0]  - 1)
        # jx[jx < -1.0] += - 2 * (jx[jx < -1.0] + 1)
        # jy[jy < -1.0] += - 2 * (jy[jy < -1.0] + 1)
        # standard clipping -------------
        # jx = np.clip(action[0] + self.ax_noise, -1.0, 1.0)
        # jy = np.clip(action[1] + self.ay_noise, -1.0, 1.0)
        # assert np.all(jx <= 1.0) and np.all(jx >= -1.0), "jx out of bounds after bias"
        # assert np.all(jy <= 1.0) and np.all(jy >= -1.0), "jy out of bounds after bias"

        # Select parameter sets
        if is_true_state:
            # Use "true" (mean) params as scalars, then broadcast
            min_lin = np.full(N, self.true_min_lin_vel)
            max_lin = np.full(N, self.true_max_lin_vel)
            max_acc = np.full(N, self.true_max_lin_acc)
            max_rot = np.full(N, self.true_max_rot_vel)
        else:
            # Use sampled arrays
            min_lin = self.min_lin_vel
            max_lin = self.max_lin_vel
            max_acc = self.max_lin_acc
            max_rot = self.max_rot_vel

        # Unpack current state
        # x, y not needed individually before update; we use vector form
        v = Xt[:, 2].copy()
        theta = Xt[:, 3].copy()

        # Velocity reference based on sign of jy
        # (scalar condition jy) chooses which vector to use
        vel_ref = (jy < 0.0) * np.abs(min_lin) + (jy >= 0.0) * max_lin

        # Acceleration towards target vel_ref, limited by max_acc
        vdot_raw = jy * vel_ref - v
        # clamp magnitude by max_acc
        # vdot = sign(vdot_raw) * min(|vdot_raw|, max_acc)
        abs_vdot = np.abs(vdot_raw)
        clamp = np.minimum(abs_vdot, max_acc)
        vdot = np.empty_like(vdot_raw)
        for i in range(N):
            if vdot_raw[i] >= 0.0:
                vdot[i] = clamp[i]
            else:
                vdot[i] = -clamp[i]

        thetadot = jx * max_rot

        # Integrate
        theta = theta + thetadot * dt
        v = v + vdot * dt

        # Kinematics
        xdot = v * np.cos(theta)
        ydot = v * np.sin(theta)

        # Assemble Xdot and update
        Xdot = np.empty_like(Xt)
        Xdot[:, 0] = xdot
        Xdot[:, 1] = ydot
        Xdot[:, 2] = vdot
        Xdot[:, 3] = thetadot

        Xtt = Xt + Xdot * dt
        return Xtt

    def repeat_axis0_2d(self, X, N):
        """
        Numba replacement for np.repeat(X, N, axis=0) when X is 2D.
        Repeats each row of X exactly N times.
        """
        new_X = np.empty((X.shape[0]*N, X.shape[1]), dtype=X.dtype)
        for i in range(X.shape[0]):
            for j in range(N):
                new_X[i*N + j, :] = X[i, :]
        return new_X
        # if N <= 0:
        #     return np.empty((0, X.shape[1]), dtype=X.dtype)
        #
        # R, C = X.shape
        # out = np.empty((R * N, C), dtype=X.dtype)
        #
        # # Parallelize across original rows (safe: rows don't overlap)
        # for i in range(R):
        #     # copy row i into positions [i*N ... i*N+N-1]
        #     base = i * N
        #     for k in range(N):
        #         dest_row = base + k
        #         # manual copy avoids slicing/broadcast edge cases in nopython
        #         for j in range(C):
        #             out[dest_row, j] = X[i, j]
        # return out
    # ----------------------------------------------
    # Convenience getters (since properties aren’t JIT)
    # ----------------------------------------------
    def get_robot_prob(self):
        return self.robot_prob

    def get_belief(self):
        # returns (mu, std) arrays of shape (4,)
        return self.belief_mu, self.belief_std

    # def __str__(self):
    #     s = "Belief_FetchRobotMDP_Compiled:"
    #     s += f"\n  n_samples: {self.n_samples}"
    #     s += f"\n  belief_mu: {self.belief_mu}"
    #     s += f"\n  belief_std: {self.belief_std}"
    #     s += f"\n  true parameters:"
    #     s += f"\n    true_min_lin_vel: {self.true_min_lin_vel}"
    #     s += f"\n    true_max_lin_vel: {self.true_max_lin_vel}"
    #     s += f"\n    true_max_lin_acc: {self.true_max_lin_acc}"
    #     s += f"\n    true_max_rot_vel: {self.true_max_rot_vel}"
    #     return s




# -------------------------------------------------------------------
# (Optional) Thin Python wrapper to keep near-original construction
# -------------------------------------------------------------------
def make_belief_fetchrobot_mdp(
        n_samples=50,
        b_min_lin_vel=(0.0, 1e-6),
        b_max_lin_vel=(2.0, 1.0),
        b_max_lin_acc=(0.5, 0.2),
        b_max_rot_vel=(math.pi/2.0, math.pi/6.0),
):
    """
    Returns a compiled Belief_FetchRobotMDP_Num instance.
    """
    return Belief_FetchRobotMDP_Compiled(
        n_samples,
        b_min_lin_vel,
        b_max_lin_vel,
        b_max_lin_acc,
        b_max_rot_vel
    )



@njit(inline='always', fastmath=True, cache=True)
def ray_circle_first_t(px, py, ux, uy, cx, cy, r, eps=1e-12):
    # Solve |(p + t u) - c|^2 = r^2, pick smallest t >= eps
    ocx = px - cx
    ocy = py - cy
    # a = 1 because u is unit
    b = 2.0 * (ux * ocx + uy * ocy)
    c0 = ocx * ocx + ocy * ocy - r * r
    disc = b * b - 4.0 * c0
    tmin = np.inf
    if disc >= 0.0:
        sqrtD = np.sqrt(disc)  # disc already clamped by >=0
        t1 = (-b - sqrtD) * 0.5
        t2 = (-b + sqrtD) * 0.5
        if t1 >= eps and t1 < tmin:
            tmin = t1
        if t2 >= eps and t2 < tmin:
            tmin = t2
    return tmin

@njit(inline='always', fastmath=True, cache=True)
def ray_aabb_first_t(px, py, ux, uy, xmin, xmax, ymin, ymax, eps=1e-12):
    tmin = np.inf

    # Vertical sides: x = xmin, xmax
    if np.abs(ux) > eps:
        t = (xmin - px) / ux
        if t >= eps:
            y_at = py + t * uy
            if y_at >= (ymin - eps) and y_at <= (ymax + eps):
                if t < tmin: tmin = t
        t = (xmax - px) / ux
        if t >= eps:
            y_at = py + t * uy
            if y_at >= (ymin - eps) and y_at <= (ymax + eps):
                if t < tmin: tmin = t

    # Horizontal sides: y = ymin, ymax
    if np.abs(uy) > eps:
        t = (ymin - py) / uy
        if t >= eps:
            x_at = px + t * ux
            if x_at >= (xmin - eps) and x_at <= (xmax + eps):
                if t < tmin: tmin = t
        t = (ymax - py) / uy
        if t >= eps:
            x_at = px + t * ux
            if x_at >= (xmin - eps) and x_at <= (xmax + eps):
                if t < tmin: tmin = t

    return tmin

# ---------- batched, parallel lidar kernel ----------

@njit(parallel=True, fastmath=True, cache=True)
def lidar_kernel(X, thetas,  # shapes: (S,4), (R,)
                 circ_C, circ_R,   # (Nc,2), (Nc,)
                 rect_CWH          # (Nr,4) -> (cx,cy,w,h)
                ):
    S = X.shape[0]
    R = thetas.shape[0]
    Nc = circ_R.shape[0]
    Nr = rect_CWH.shape[0]
    out = np.empty((S, R), dtype=np.float32)

    for s in prange(S):
        px = X[s, 0]
        py = X[s, 1]
        theta = X[s, 3]  # x, y, v, θ

        for i in range(R):
            ang = theta + thetas[i]
            ux = np.cos(ang)
            uy = np.sin(ang)

            tbest = np.inf

            # circles
            for k in range(Nc):
                cx = circ_C[k, 0]
                cy = circ_C[k, 1]
                r  = circ_R[k]
                t = ray_circle_first_t(px, py, ux, uy, cx, cy, r)
                if t < tbest:
                    tbest = t

            # axis-aligned rectangles
            for k in range(Nr):
                cx = rect_CWH[k, 0]
                cy = rect_CWH[k, 1]
                w  = rect_CWH[k, 2]
                h  = rect_CWH[k, 3]
                xmin = cx - 0.5 * w
                xmax = cx + 0.5 * w
                ymin = cy - 0.5 * h
                ymax = cy + 0.5 * h

                t = ray_aabb_first_t(px, py, ux, uy, xmin, xmax, ymin, ymax)
                if t < tbest:
                    tbest = t

            # return distance (norm of the hit vector), or -1 if no hit
            out[s, i] = -1.0 if not np.isfinite(tbest) else np.float32(tbest)

    return out

# ---------- convenience wrapper class ----------

class Compiled_LidarFun:
    """
    Fast lidar: preprocess obstacles to arrays and run a Numba-parallel kernel.
    obstacles: list of dicts with either:
      {'type':'circle', 'center':(x,y), 'radius':r}
      {'type':'rect',   'center':(x,y), 'width':w, 'height':h}
      (anything not 'circle' is treated as rect for backward-compat)
    """
    def __init__(self, obstacles, n_rays):
        self.n_rays = int(n_rays)

        # Precompute beam offsets
        self.δrays = np.linspace(-np.pi, np.pi, self.n_rays, endpoint=False).astype(np.float64)

        # Preprocess obstacles into Numba-friendly arrays
        circ_C = []
        circ_R = []
        rect_CWH = []

        for obs in obstacles:
            if obs.get('type', 'rect') == 'circle':
                circ_C.append(obs['center'])
                circ_R.append(float(obs['radius']))
            else:
                cx, cy = obs['center']
                rect_CWH.append((float(cx), float(cy), float(obs['width']), float(obs['height'])))

        # Store as contiguous arrays
        self.circ_C = np.ascontiguousarray(circ_C, dtype=np.float64) if len(circ_C) else np.zeros((0,2), np.float64)
        self.circ_R = np.ascontiguousarray(circ_R, dtype=np.float64) if len(circ_R) else np.zeros((0,),  np.float64)
        self.rect_CWH = np.ascontiguousarray(rect_CWH, dtype=np.float64) if len(rect_CWH) else np.zeros((0,4), np.float64)

        # Optional: trigger compilation once with a tiny dummy call (warm-up)
        _X = np.zeros((1,4), dtype=np.float64)
        _ = lidar_kernel(_X, self.δrays, self.circ_C, self.circ_R, self.rect_CWH)

    def __call__(self, X):
        """
        X: (S,4) with columns [x, y, v, θ]
        returns: (S, n_rays) distances (float32); -1 if no intersection
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        drays = lidar_kernel(X, self.δrays, self.circ_C, self.circ_R, self.rect_CWH)
        return drays


def main():
    robot = Belief_FetchRobotMDP_Compiled()
    robot.resample_dynamics()

    Xt = np.array([[0,0,0,0]],dtype=np.float32)
    # Xt = Xt.repeat(robot.n_samples, axis=0)
    action = np.array([0.5, 0.5])
    dt = 0.1
    for _ in range(10):
        Xt = robot.step_true(Xt, action, dt)
        print(Xt)


if __name__ == "__main__":
    main()



# class Compiled_LidarFun:
#     def __init__(self, obstacles, n_rays):
#         self.obstacles = obstacles
#         self.n_rays = n_rays
#         self.δrays = np.linspace(-np.pi, np.pi, self.n_rays, endpoint=False)  # lidar beam angles
#
#
#     def __call__(self, X):
#
#         drays = np.zeros([X.shape[0],self.n_rays],dtype=np.float32)
#
#         for s, _X in enumerate(X):
#             x, y, v, θ = _X
#
#             for i, δbeam in enumerate(self.δrays):
#                 ray_angle = θ + δbeam
#                 dxy = self.get_ray2obstacle_dist(x, y, ray_angle)
#                 drays[s, i] = np.linalg.norm(dxy)
#
#         return drays
#
#     def get_ray2obstacle_dist(self, x, y, δ):
#         """
#         Cast a ray from (x, y) in direction δ (degrees) and return the 2D vector
#         from (x, y) to the first intersection point with any obstacle.
#         Supports:
#           - obs['type'] == 'circle' with keys: center (x,y), radius
#           - obs['type'] != 'circle' as axis-aligned rectangle with keys:
#               center (x,y), width, height
#         """
#         import numpy as np
#
#         min_t = np.inf
#         for obs in self.obstacles:
#             if obs['type'] == 'circle':
#                 c = np.array(obs['center'], dtype=np.float64)
#                 r = float(obs['radius'])
#                 min_t = get_ray2obstacle_dist_circ(min_t, x, y, δ, c, r)
#
#             else:
#                 # Axis-aligned rectangle
#                 cx, cy = obs['center']
#                 w, h = float(obs['width']), float(obs['height'])
#                 min_t = get_ray2obstacle_dist_rect(min_t, x, y, δ, cx, cy, w, h)
#
#         # assert np.isfinite(min_t), f"Ray does not intersect any obstacle x:{x} y:{y} δ:{δ}"
#
#         if not np.isfinite(min_t): # alread went out of bounds since env is fully enclosed
#             min_t = -1
#
#         u = np.array([np.cos(δ), np.sin(δ)], dtype=np.float64)  # ray direction, |u|=1
#         vec = (u * min_t).astype(np.float32)  # vector from (x,y) to intersection point
#         return vec
#
# @njit
# def get_ray2obstacle_dist_circ(min_t, x, y, δ, c,r, eps = 1e-12):
#     # c = np.array(obs['center'], dtype=np.float64)
#     # r = float(obs['radius'])
#
#
#     p = np.array([x, y], dtype=np.float64)
#     u = np.array([np.cos(δ), np.sin(δ)], dtype=np.float64)  # ray direction, |u|=1
#
#     # Solve ||p + t u - c||^2 = r^2 for t >= 0
#     oc = p - c
#     # a = 1 because u is unit; b = 2 u·oc; c0 = ||oc||^2 - r^2
#     b = 2.0 * np.dot(u, oc)
#     c0 = np.dot(oc, oc) - r * r
#     disc = b * b - 4.0 * c0
#
#     if disc >= 0.0:
#         sqrtD = np.sqrt(max(0.0, disc))
#         t1 = (-b - sqrtD) / 2.0
#         t2 = (-b + sqrtD) / 2.0
#         # Select smallest nonnegative root (intersection going forward)
#         for t in (t1, t2):
#             if t >= eps and t < min_t:
#                 min_t = t
#     return min_t
# #
# @njit
# def get_ray2obstacle_dist_rect(min_t, x, y, δ, cx, cy,w,h, eps=1e-12):
#     p = np.array([x, y], dtype=np.float64)
#     u = np.array([np.cos(δ), np.sin(δ)], dtype=np.float64)  # ray direction, |u|=1
#
#     # Axis-aligned rectangle
#     # cx, cy = obs['center']
#     # w, h = float(obs['width']), float(obs['height'])
#     xmin, xmax = cx - w / 2.0, cx + w / 2.0
#     ymin, ymax = cy - h / 2.0, cy + h / 2.0
#
#     # Intersect with the 4 lines x = xmin/xmax, y = ymin/ymax
#     # Keep hits where the other coordinate lies within segment bounds.
#     ux, uy = u
#     px, py = p
#
#     # Vertical sides
#     if abs(ux) > eps:
#         for x_edge in (xmin, xmax):
#             t = (x_edge - px) / ux
#             if t >= eps:
#                 y_at = py + t * uy
#                 if y_at >= ymin - eps and y_at <= ymax + eps:
#                     if t < min_t:
#                         min_t = t
#
#     # Horizontal sides
#     if abs(uy) > eps:
#         for y_edge in (ymin, ymax):
#             t = (y_edge - py) / uy
#             if t >= eps:
#                 x_at = px + t * ux
#                 if x_at >= xmin - eps and x_at <= xmax + eps:
#                     if t < min_t:
#                         min_t = t
#
#
#     return min_t
#
#
# ##############################