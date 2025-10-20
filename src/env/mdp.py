import numpy as np
from numba import njit, float64, int64, boolean
from numba.experimental import jitclass
import math
import warnings

# Parameter indices (for readability)
MIN_LIN_VEL = 0
MAX_LIN_VEL = 1
MAX_LIN_ACC = 2
MAX_ROT_VEL = 3

# -----------------------------
# Helper: normal pdf (JIT-safe)
# -----------------------------
@njit
def normal_pdf(x, mu, std):
    if std == 0.0:
        # treat degenerate case as probability 1 (matches original code’s intent)
        return 1.0
    inv = 1.0 / (std * math.sqrt(2.0 * math.pi))
    z = (x - mu) / std
    return inv * math.exp(-0.5 * z * z)

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

    # Probability per dynamics sample
    ('robot_prob', float64[:]),

    # "True" params (use belief means as the true values, matching original)
    ('true_min_lin_vel', float64),
    ('true_max_lin_vel', float64),
    ('true_max_lin_acc', float64),
    ('true_max_rot_vel', float64),
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
                 b_min_lin_vel=(0.0, 1e-6),
                 b_max_lin_vel=(1.0, 1.0),
                 b_max_lin_acc=(0.5, 0.2),
                 b_max_rot_vel=(math.pi/2.0, math.pi/6.0)):

        self.n_samples = n_samples

        self.belief_mu = np.empty(4, dtype=np.float64)
        self.belief_std = np.empty(4, dtype=np.float64)

        self.belief_mu[MIN_LIN_VEL] = b_min_lin_vel[0]
        self.belief_std[MIN_LIN_VEL] = b_min_lin_vel[1]

        self.belief_mu[MAX_LIN_VEL] = b_max_lin_vel[0]
        self.belief_std[MAX_LIN_VEL] = b_max_lin_vel[1]

        self.belief_mu[MAX_LIN_ACC] = b_max_lin_acc[0]
        self.belief_std[MAX_LIN_ACC] = b_max_lin_acc[1]

        self.belief_mu[MAX_ROT_VEL] = b_max_rot_vel[0]
        self.belief_std[MAX_ROT_VEL] = b_max_rot_vel[1]

        # True (mean) parameters
        self.true_min_lin_vel = self.belief_mu[MIN_LIN_VEL]
        self.true_max_lin_vel = self.belief_mu[MAX_LIN_VEL]
        self.true_max_lin_acc = self.belief_mu[MAX_LIN_ACC]
        self.true_max_rot_vel = self.belief_mu[MAX_ROT_VEL]

        # Allocate arrays
        self.samples = np.empty((4, n_samples), dtype=np.float64)
        self.robot_prob = np.full(n_samples, np.nan, dtype=np.float64)

        self.min_lin_vel  = self.samples[MIN_LIN_VEL, :]
        self.max_lin_vel  = self.samples[MAX_LIN_VEL, :]
        self.max_lin_acc  = self.samples[MAX_LIN_ACC, :]
        self.max_rot_vel  = self.samples[MAX_ROT_VEL, :]

        # Initialize with one resample so fields are defined
        self.resample_dynamics()

    # ----------------------------------------------
    # Resample dynamics + update per-sample prob
    # ----------------------------------------------
    def resample_dynamics(self):
        n = self.n_samples
        # start as ones (cumulative product across params)
        p = np.ones(n, dtype=np.float64)

        # For each parameter, sample and accumulate probability
        for k in range(4):
            mu = self.belief_mu[k]
            std = self.belief_std[k]

            if std == 0.0:
                # Deterministic
                self.samples[k, :] = mu
                # probability contribution = 1 (matches original)
                # p *= 1  (no-op)
            else:
                # Gaussian samples
                self.samples[k, :] = np.random.normal(mu, std, n)
                # Multiply in pdfs
                p_k = normal_pdf_vec(self.samples[k, :], mu, std)
                for i in range(n):
                    p[i] *= p_k[i]

        # Update cache views (already aliasing self.samples rows)
        # (No-op, but keeps intent clear)
        self.min_lin_vel  = self.samples[MIN_LIN_VEL, :]
        self.max_lin_vel = self.samples[MAX_LIN_VEL, :]
        self.max_lin_acc = self.samples[MAX_LIN_ACC, :]
        self.max_rot_vel = self.samples[MAX_ROT_VEL, :]

        for i in range(self.samples.shape[1]):
            if self.samples[MIN_LIN_VEL, i] >0:
                self.min_lin_vel[i] = 0.0
            if self.samples[MAX_LIN_VEL, i] < 0:
                self.max_lin_vel[i] = 0.0
            if self.samples[MAX_LIN_ACC, i] < 0:
                self.max_lin_acc[i] = 0.0
            if self.samples[MAX_ROT_VEL, i] < 0:
                self.max_rot_vel[i] = 0.0
        # self.max_lin_vel  = np.max(self.samples[MAX_LIN_VEL, :],0)
        # self.max_lin_acc  = np.max(self.samples[MAX_LIN_ACC, :],0)
        # self.max_rot_vel  = np.max(self.samples[MAX_ROT_VEL, :],0)

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
        if Xt.ndim == 1:
            Xt = Xt.reshape(1, 4)
        if Xt.shape[0] == 1:
            Xt = np.repeat(Xt ,N).reshape(N, 4)

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
            vel_ref = min_lin
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


class Compiled_LidarFun:
    def __init__(self, obstacles, n_rays):
        self.obstacles = obstacles
        self.n_rays = n_rays
        self.δrays = np.linspace(-np.pi, np.pi, self.n_rays, endpoint=False)  # lidar beam angles


    def __call__(self, X):
        #TODO: Need to vectorize this function

        drays = np.zeros([X.shape[0],self.n_rays],dtype=np.float32)

        for s, _X in enumerate(X):
            x, y, v, θ = _X

            for i, δbeam in enumerate(self.δrays):
                ray_angle = θ + δbeam
                dxy = self.get_ray2obstacle_dist(x, y, ray_angle)
                drays[s, i] = np.linalg.norm(dxy)
            # for i, δbeam in enumerate(self.δrays):
            #     ray_angle = θ + δbeam
            #     dxy = self.get_ray2obstacle_dist(x, y, ray_angle)
            #     drays[s,i] = np.linalg.norm(dxy)
            #     if np.linalg.norm(dxy) == 0:
            #         print(f"{i} Warning: Lidar ray does not intersect any obstacle.")

        return drays

    def get_ray2obstacle_dist(self, x, y, δ):
        """
        Cast a ray from (x, y) in direction δ (degrees) and return the 2D vector
        from (x, y) to the first intersection point with any obstacle.
        Supports:
          - obs['type'] == 'circle' with keys: center (x,y), radius
          - obs['type'] != 'circle' as axis-aligned rectangle with keys:
              center (x,y), width, height
        """
        import numpy as np

        min_t = np.inf
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                c = np.array(obs['center'], dtype=np.float64)
                r = float(obs['radius'])
                min_t = get_ray2obstacle_dist_circ(min_t, x, y, δ, c, r)

            else:
                # Axis-aligned rectangle
                cx, cy = obs['center']
                w, h = float(obs['width']), float(obs['height'])
                min_t = get_ray2obstacle_dist_rect(min_t, x, y, δ, cx, cy, w, h)

        assert np.isfinite(min_t), f"Ray does not intersect any obstacle x:{x} y:{y} δ:{δ}"
        u = np.array([np.cos(δ), np.sin(δ)], dtype=np.float64)  # ray direction, |u|=1
        vec = (u * min_t).astype(np.float32)  # vector from (x,y) to intersection point
        return vec
#
# class Compiled_LidarFun:
#     def __init__(self, obstacles, n_rays):
#         self.obstacles = obstacles
#         self.n_rays = n_rays
#         self.δrays = np.linspace(-np.pi, np.pi, self.n_rays, endpoint=False)  # lidar beam angles
#
#
#     def __call__(self, X):
#         #TODO: Need to vectorize this function
#
#         drays = np.zeros([X.shape[0],self.n_rays],dtype=np.float32)
#
#         for s, _X in enumerate(X):
#             x, y, v, θ = _X
#             for i, δbeam in enumerate(self.δrays):
#                 ray_angle = θ + δbeam
#                 dxy = get_ray2obstacle_dist(x, y, ray_angle,self.obstacles)
#                 drays[s,i] = np.linalg.norm(dxy)
#                 if np.linalg.norm(dxy) == 0:
#                     print(f"{i} Warning: Lidar ray does not intersect any obstacle.")
#
#         return drays
#
#
@njit
def get_ray2obstacle_dist_circ(min_t, x, y, δ, c,r, eps = 1e-12):
    # c = np.array(obs['center'], dtype=np.float64)
    # r = float(obs['radius'])


    p = np.array([x, y], dtype=np.float64)
    u = np.array([np.cos(δ), np.sin(δ)], dtype=np.float64)  # ray direction, |u|=1

    # Solve ||p + t u - c||^2 = r^2 for t >= 0
    oc = p - c
    # a = 1 because u is unit; b = 2 u·oc; c0 = ||oc||^2 - r^2
    b = 2.0 * np.dot(u, oc)
    c0 = np.dot(oc, oc) - r * r
    disc = b * b - 4.0 * c0

    if disc >= 0.0:
        sqrtD = np.sqrt(max(0.0, disc))
        t1 = (-b - sqrtD) / 2.0
        t2 = (-b + sqrtD) / 2.0
        # Select smallest nonnegative root (intersection going forward)
        for t in (t1, t2):
            if t >= eps and t < min_t:
                min_t = t
    return min_t
#
@njit
def get_ray2obstacle_dist_rect(min_t, x, y, δ, cx, cy,w,h, eps=1e-12):
    p = np.array([x, y], dtype=np.float64)
    u = np.array([np.cos(δ), np.sin(δ)], dtype=np.float64)  # ray direction, |u|=1

    # Axis-aligned rectangle
    # cx, cy = obs['center']
    # w, h = float(obs['width']), float(obs['height'])
    xmin, xmax = cx - w / 2.0, cx + w / 2.0
    ymin, ymax = cy - h / 2.0, cy + h / 2.0

    # Intersect with the 4 lines x = xmin/xmax, y = ymin/ymax
    # Keep hits where the other coordinate lies within segment bounds.
    ux, uy = u
    px, py = p

    # Vertical sides
    if abs(ux) > eps:
        for x_edge in (xmin, xmax):
            t = (x_edge - px) / ux
            if t >= eps:
                y_at = py + t * uy
                if y_at >= ymin - eps and y_at <= ymax + eps:
                    if t < min_t:
                        min_t = t

    # Horizontal sides
    if abs(uy) > eps:
        for y_edge in (ymin, ymax):
            t = (y_edge - py) / uy
            if t >= eps:
                x_at = px + t * ux
                if x_at >= xmin - eps and x_at <= xmax + eps:
                    if t < min_t:
                        min_t = t


    return min_t
#
#
# # @njit
# def get_ray2obstacle_dist( x, y, δ, obstacles):
#     """
#     Cast a ray from (x, y) in direction δ (degrees) and return the 2D vector
#     from (x, y) to the first intersection point with any obstacle.
#     Supports:
#       - obs['type'] == 'circle' with keys: center (x,y), radius
#       - obs['type'] != 'circle' as axis-aligned rectangle with keys:
#           center (x,y), width, height
#     """
#
#     p = np.array([x, y], dtype=np.float64)
#     u = np.array([np.cos(δ), np.sin(δ)], dtype=np.float64)  # ray direction, |u|=1
#     eps = 1e-12
#
#     min_t = np.inf
#
#     for obs in obstacles:
#         if obs['type'] == 'circle':
#             c = np.array(obs['center'], dtype=np.float64)
#             r = float(obs['radius'])
#             min_t = get_ray2obstacle_dist_circ(min_t, x, y, δ, c, r)
#
#             # # Solve ||p + t u - c||^2 = r^2 for t >= 0
#             # oc = p - c
#             # # a = 1 because u is unit; b = 2 u·oc; c0 = ||oc||^2 - r^2
#             # b = 2.0 * np.dot(u, oc)
#             # c0 = np.dot(oc, oc) - r * r
#             # disc = b * b - 4.0 * c0
#             #
#             # if disc >= 0.0:
#             #     sqrtD = np.sqrt(max(0.0, disc))
#             #     t1 = (-b - sqrtD) / 2.0
#             #     t2 = (-b + sqrtD) / 2.0
#             #     # Select smallest nonnegative root (intersection going forward)
#             #     for t in (t1, t2):
#             #         if t >= eps and t < min_t:
#             #             min_t = t
#
#         else:
#             # Axis-aligned rectangle
#             cx, cy = obs['center']
#             w, h = float(obs['width']), float(obs['height'])
#             min_t = get_ray2obstacle_dist_rect(min_t, x, y, δ, cx, cy, w, h)
#             # xmin, xmax = cx - w / 2.0, cx + w / 2.0
#             # ymin, ymax = cy - h / 2.0, cy + h / 2.0
#             #
#             # # Intersect with the 4 lines x = xmin/xmax, y = ymin/ymax
#             # # Keep hits where the other coordinate lies within segment bounds.
#             # ux, uy = u
#             # px, py = p
#             #
#             # # Vertical sides
#             # if abs(ux) > eps:
#             #     for x_edge in (xmin, xmax):
#             #         t = (x_edge - px) / ux
#             #         if t >= eps:
#             #             y_at = py + t * uy
#             #             if y_at >= ymin - eps and y_at <= ymax + eps:
#             #                 if t < min_t:
#             #                     min_t = t
#             #
#             # # Horizontal sides
#             # if abs(uy) > eps:
#             #     for y_edge in (ymin, ymax):
#             #         t = (y_edge - py) / uy
#             #         if t >= eps:
#             #             x_at = px + t * ux
#             #             if x_at >= xmin - eps and x_at <= xmax + eps:
#             #                 if t < min_t:
#             #                     min_t = t
#
#     # assert np.isfinite(min_t), "Ray does not intersect any obstacle."
#
#     if np.isfinite(min_t):
#         warnings.warn("Ray does not intersect any obstacle.")
#         min_t = 0  #"Ray does not intersect any obstacle."
#     vec = (u * min_t).astype(np.float32)  # vector from (x,y) to intersection point
#     return vec


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



#
# class Belief_FetchRobotMDP:
#     def __init__(self,
#                  n_samples=10,
#                  b_min_lin_vel = (0,1e-6),
#                  b_max_lin_vel = (2.0,1.0),
#                  b_max_lin_acc = (0.5,0.2),
#                  b_max_rot_vel = (np.pi/2,np.pi/6),
#
#                  ):
#         """
#         Robot dynamics model with uncertainty in parameters.
#         - b_min_lin_vel: belief_dist of robot dynamics mu,std
#         - b_max_lin_vel: belief_dist of robot dynamics mu,std
#         - b_max_lin_acc: belief_dist of robot dynamics mu,std
#         - b_max_rot_vel: belief_dist of robot dynamics mu,std
#         - n_samples: number of dynamics samples for belief MDP
#         """
#         self.n_samples = n_samples  # number of dynamics samples for belief MDP
#         self._dynamics_belief = { # Robot dynamics parameters
#             'min_lin_vel': b_min_lin_vel,
#             'max_lin_vel': b_max_lin_vel,
#             'max_lin_acc': b_max_lin_acc,
#             'max_rot_vel': b_max_rot_vel
#         }
#
#         # Sampled dynamics parameters ------------------------------------------------------------
#         self._robot_prob = np.full(n_samples, np.nan)  # probability of each dynamics sample (uninitialized)
#         self.min_lin_vel = np.nan   # minim linear velocity in m/s (<0 means reversing)
#         self.max_lin_vel = np.nan   # maximum linear velocity in m/s
#         self.max_lin_acc = np.nan   # maximum linear acceleration in m/s^2
#         self.max_rot_vel = np.nan   # maximum rotational velocity in rad/s
#
#         # True dynamics parameters (for simulating the "real" robot) ---------------------------------------------------
#         self.true_min_lin_vel = b_min_lin_vel[0]  # minim linear velocity in m/s (<0 means reversing)
#         self.true_max_lin_vel = b_max_lin_vel[0]  # maximum linear velocity in m/s
#         self.true_max_lin_acc = b_max_lin_acc[0]
#         self.true_max_rot_vel = b_max_rot_vel[0]
#
#     def step(self, Xt, action, dt,is_true_state=False):
#         """
#         Vectorized dynamics step.
#         Xt: (N,4) array of N robot states [x,y,v,θ]
#         action: (2,) array of joystick commands [jx,jy] in [-1,1]
#         dt: time step in seconds
#
#         """
#         min_lin_vel = self.min_lin_vel if not is_true_state else self.dynamics_belief['min_lin_vel'][0]
#         max_lin_vel = self.max_lin_vel if not is_true_state else self.dynamics_belief['max_lin_vel'][0]
#         max_lin_acc = self.max_lin_acc if not is_true_state else self.dynamics_belief['max_lin_acc'][0]
#         max_rot_vel = self.max_rot_vel if not is_true_state else self.dynamics_belief['max_rot_vel'][0]
#         N = self.n_samples if not is_true_state else 1
#
#
#         # Ensure Xt is (N,4)
#         if len(Xt.shape) == 1:
#             Xt = Xt[np.newaxis, :]
#         if Xt.shape[0]==1:
#             Xt = np.repeat(Xt, N, axis=0)
#         assert Xt.shape == (N, 4), f"Xt shape {Xt.shape} does not match expected {(N,4)}"
#
#
#         # Unpack inputs
#         jx, jy = action  # joystick commands for steering and velocity
#         _, _, v, θ = Xt.T
#
#         vel_ref = min_lin_vel if jy < 0 else max_lin_vel
#         vdot = (jy * vel_ref) - v
#         vdot = np.sign(vdot) * np.min(np.vstack([np.abs(vdot), max_lin_acc]),axis=0)
#         θdot = jx * max_rot_vel
#
#         θ += θdot * dt
#         v += vdot * dt
#
#         # Dynamics equations
#         xdot = v * np.cos(θ)
#         ydot = v * np.sin(θ)
#
#         # Update state
#         Xdot = np.array([xdot, ydot, vdot, θdot]).T
#         assert Xdot.shape == Xt.shape, f"Xdot shape {Xdot.shape} does not match Xt shape {Xt.shape}"
#
#         Xtt = Xt + Xdot * dt
#         return Xtt
#
#     def resample_dynamics(self):
#         """ Resample robot dynamics parameters from belief distribution and update robot_prob."""
#
#         p_dynamics = np.ones(self.n_samples) # cumulative probability of each dynamics sample
#
#         for key, val in self.dynamics_belief.items():
#             mu, std = self.dynamics_belief[key]
#             if std == 0:
#                 samp_vals = np.full(self.n_samples, mu)
#                 p_sample = np.full(self.n_samples, 1)
#             else:
#                 samp_vals = np.random.normal(mu, std, size=self.n_samples) # samples all values at once
#                 p_sample = [norm.pdf(v, loc=mu, scale=std) for v in samp_vals]
#
#             self.__dict__[key] = samp_vals
#             p_dynamics *= p_sample
#         self._robot_prob = p_dynamics
#
#     @property
#     def robot_prob(self):
#         return self._robot_prob
#
#     @property
#     def dynamics_belief(self):
#         return self._dynamics_belief
#
#     @dynamics_belief.setter
#     def dynamics_belief(self, value):
#         if value is not None:
#             assert isinstance(value, dict), "dynamics_belief must be a dictionary of parameter:(mu,std) pairs"
#             for key, val in value.items():
#                 assert hasattr(self, key), f"dynamics_belief key {key} not an attribute of the robot"
#                 assert isinstance(val, (list, tuple)) and len(
#                     val) == 2, f"dynamics_belief value for {key} must be (mu,std)"
#                 mu, std = val
#                 assert isinstance(mu, (int, float)) and isinstance(std, (
#                 int, float)) and std >= 0, f"dynamics_belief value for {key} must be (mu,std) with std>=0"
#         self._dynamics_belief = value
#
