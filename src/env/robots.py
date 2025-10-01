from gym import spaces
import numpy as np
from abc import ABC, abstractmethod
import warnings
from collections import deque

class FetchBase(ABC):
    def __init__(self, start_pos, start_velocity, start_heading, parent_env,
                 min_lin_vel=0.0,  # min_lin_vel = -0.5,
                 max_lin_vel=1.0,
                 max_lin_acc=0.5,  # 1.0
                 max_rot_vel=np.pi / 2,  # 3.0
                 base_state_validator=None):

        self.min_lin_vel = min_lin_vel  # minim linear velocity in m/s (<0 means reversing)
        self.max_lin_vel = max_lin_vel  # maximum linear velocity in m/s
        self.max_lin_acc = max_lin_acc
        self.max_rot_vel = max_rot_vel

        self.start_pos = tuple(start_pos)
        self.start_velocity = start_velocity
        self.start_heading = start_heading

        self.goal = parent_env.goal
        self.bounds = parent_env.bounds
        self.vgraph = parent_env.vgraph
        self.obstacles = parent_env.obstacles
        # self.robot = parent_env.robot

        self.base_state_validator = base_state_validator # function to validate base state during randomization

        self._state_bounds = {}
        self._state_names = []
        self._state_idxs = {}
        self._state = np.empty((0,0), dtype=np.float32)
        self.observation_space = None

        self.add_feature('x', bounds=[self.bounds[0][0], self.bounds[1][0]])
        self.add_feature('y', bounds=[self.bounds[0][1], self.bounds[1][1]])
        self.add_feature('v', bounds=[0, self.max_lin_vel])
        self.add_feature('θ', bounds=[-np.pi, np.pi])

    def set_dynamics(self, min_lin_vel, max_lin_vel, max_lin_acc, max_rot_vel):
        self.min_lin_vel = min_lin_vel
        self.max_lin_vel = max_lin_vel
        self.max_lin_acc = max_lin_acc
        self.max_rot_vel = max_rot_vel

    def step(self, Xt, action, dt):
        jx,jy = action  # joystick commands for steering and velocity
        _, _, v, θ = Xt.T

        # Dynamics equations
        xdot = v * np.cos(θ)
        ydot = v * np.sin(θ)

        vel_ref = -1*self.min_lin_vel if jy<0 else self.max_lin_vel

        vdot = jy * vel_ref - v
        vdot = np.sign(vdot) * min(abs(vdot), self.max_lin_acc)
        θdot = jx * self.max_rot_vel

        # Update state
        Xdot = np.array([xdot, ydot, vdot, θdot])
        assert Xdot.shape == Xt.shape, f"Xdot shape {Xdot.shape} does not match Xt shape {Xt.shape}"

        Xtt = Xt + Xdot * dt
        return Xtt

    def post_init(self):
        low_obs = np.array([bound[0] for bound in self._state_bounds.values()], dtype=np.float32)
        high_obs = np.array([bound[1] for bound in self._state_bounds.values()], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

    @abstractmethod
    def reset(self,*args, **kwargs):
        pass

    @abstractmethod
    def update_state(self, x,y, v, θ, hold_prev=False):
        """ Update the base state features in self._state"""
        pass

    def get_base_startstate(self,random_start=False):
        if not random_start:
            x, y, v, θ = self.start_pos[0], self.start_pos[1], self.start_velocity, self.start_heading
            return x, y, v, θ


        x, y = self.start_pos

        attempt = 0
        for _ in range(50):
            attempt += 1
            _x = np.random.uniform(self.bounds[0][0], self.bounds[1][0])
            _y = np.random.uniform(self.bounds[0][1], self.bounds[1][1])
            if self.base_state_validator is None or \
                    not self.base_state_validator((x,y,None,None)):
                x, y = _x, _y
                break
        if attempt > 49:
            warnings.warn("Failed to find a valid random X Y after 50 attempts. Using last standard.")

        v = np.random.uniform(0, self.robot.max_lin_vel)  # random initial velocity
        _, δGoal = self.get_dist2goal_sphere(x, y, 0)  # distance to goal in meters
        θ = δGoal + np.random.uniform(-np.pi / 4, np.pi / 4)

        return x, y, v, θ

    def _set(self, name, value):
        self._state[self._state_idxs[name]] = value

    def _set_dynmamic(self, name, value):
        self.__dict__[name] = value

    def add_feature(self, name, bounds = (-np.inf, np.inf), init_val=np.nan):
        self._state_idxs[name] = len(self._state_idxs)
        self._state_bounds[name] = bounds
        self._state_names.append(name)
        self._state = np.append(self._state, init_val)

        # create index property
        def idx_prop(self, _name=name):
            return self._state_idxs[_name]

        # create getter
        def getter(self, _name=name):
            return self._state[self._state_idxs[_name]]

        # create setter
        def setter(self, value, _name=name):
            raise ValueError(f"Cannot set {name} directly. Use update_state() instead.")
            # self._state[self._state_idxs[_name]] = value

        # dynamically set attributes on the class
        setattr(self.__class__, f"i{name}", property(idx_prop))
        setattr(self.__class__, name, property(getter, setter))

    def draw_features(self, ax):
        """Draws robot and additional state-specific elements on the given axes."""
        warnings.warn("draw() method called but not implemented in State subclass.")
        path = np.array(self.vgraph.shortest_path(np.array([self.x, self.y]), self.goal))
        ax.plot(path[:, 0], path[:, 1], color='g', linestyle='--', label='Dist to Goal')

        dGoal, δGoal = self.get_dist2goal_sphere(self.x, self.y, self.θ)
        dx = dGoal * np.cos(δGoal + self.θ)
        dy = dGoal * np.sin(δGoal + self.θ)
        # dx,dy = self.get_dist2goal_cart(self.x, self.y)
        # # dx,dy = path[1, :] - path[0, :]
        ax.arrow(self.x, self.y, dx, dy, head_width=0.1, color='g')

    # State feature getters #########################################################

    def get_dist2goal_cart(self, x, y):
        return self.vgraph.dist_xy(x, y)

    def get_dist2goal_sphere(self, x, y, θ):
        # dx,dy = self.get_dist2goal_cart(x, y)
        # return self.cart2sphere(dx,dy, θ)
        return self.vgraph.dist_dθ(x, y, θ)

    def cart2sphere(self, dx, dy, θ):
        """
        Convert Cartesian coordinates (x, y) to spherical coordinates (r, theta).
        r is the distance from the origin, and theta is the angle in radians.
        """
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx) - θ
        return np.array([r, theta], dtype=np.float32)

    ## Properties ##################################################################
    @property
    def observation(self):
        return self._state.copy() if self._state is not None else None
    @property
    def size(self):
        return self.observation_space.shape[0]

    @property
    def base_state(self):
        return self._state[0:4].copy()
    def deepcopy(self):
        new_instance = self.__class__(self.start_pos, self.start_velocity, self.start_heading, None)

        new_instance._state_bounds = self._state_bounds.copy()
        new_instance._state_names = self._state_names.copy()
        new_instance._state_idxs = self._state_idxs.copy()
        new_instance._state = self._state.copy()
        new_instance.observation_space = spaces.Box(self.observation_space.low.copy(),
                                                    self.observation_space.high.copy(),
                                                    dtype=self.observation_space.dtype) if self.observation_space is not None else None
        return new_instance

class Fetch_DistanceHeading(FetchBase):
    """
    Auxiliary Features Added:
        - dGoal: distance to goal
        - δGoal: relative heading to the goal
        - dObst: distance to closest obstacle
        - δObst: relative heading to closest obstacle
    """
    def __init__(self, start_pos,start_velocity,start_heading, parent_env, **kwargs):
        super().__init__(start_pos, start_velocity, start_heading, parent_env, **kwargs)
        self.add_feature('dGoal', bounds = [0     , np.inf]) # distance to goal
        self.add_feature('δGoal', bounds = [-np.pi, np.pi] ) # relative heading to the goal
        self.add_feature('dObst', bounds = [0     , np.inf]) # spherical distance to the closest obstacle,
        self.add_feature('δObst', bounds = [-np.pi, np.pi] ) # relative heading to the closest obstacle
        self.post_init()

    def reset(self,random_start=False, **kwargs):
        """ add auxillary features to self._state """
        x, y, v, θ = self.get_base_startstate(random_start=random_start)
        self.update_state(x, y, v, θ)

        assert not np.any(np.isnan(self._state)), "State contains NaN values after reset. Likley missed definition in subclass."
        return self.observation

    def update_state(self, x,y, v, θ, hold_prev=False):
        dGoal, δGoal = self.get_dist2goal_sphere(x, y, θ)  # dist to goal & relative heading
        dObst, δObst = self.get_dist2obstacle_sphere(x, y, θ)  # dist to closest obstacle & relative heading

        if hold_prev:
            new_state = self.deepcopy()
            new_state.update_state(x, y, v, θ, hold_prev=False)
            return new_state.observation

        else:
            self._set('x', x)
            self._set('y', y)
            self._set('v', v)
            self._set('θ', θ)
            self._set('dGoal', dGoal)
            self._set('δGoal', δGoal)
            self._set('dObst', dObst)
            self._set('δObst', δObst)
            return self.observation

    def get_dist2obstacle_cart(self, x, y):
        """
        Computes the distance to the nearest obstacle.
        """
        xy = np.array([x, y], dtype=np.float32)

        min_dist = float('inf')
        min_x = None
        min_y = None
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                r = obs['radius']
                dx, dy = xy - obs['center']
                dist_to_center = np.linalg.norm(xy - obs['center'])  # - obs['radius']
                dist = dist_to_center - r
                scale = (dist) / dist_to_center

                # get distance to perimeter of circle in x,y
                # Directional components
                dist_x = dx * scale
                dist_y = dy * scale
            else:
                xmin = obs['center'][0] - obs['width'] / 2
                xmax = obs['center'][0] + obs['width'] / 2
                ymin = obs['center'][1] - obs['height'] / 2
                ymax = obs['center'][1] + obs['height'] / 2

                # xedges = np.array([xmin, xmax])
                # yedges = np.array([ymin, ymax])
                corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
                edges = []
                if x < xmin and ymin <= y <= ymax:  edges.append([xmin, y])  # left edge
                if x > xmax and ymin <= y <= ymax: edges.append([xmax, y])  # right edge
                if y < ymin and xmin <= x <= xmax:  edges.append([x, ymin])  # bottom edge
                if y > ymax and xmin <= x <= xmax:  edges.append([x, ymax])

                if len(edges) == 0:
                    checks = corners
                else:
                    checks = np.vstack([corners, np.array(edges)])
                dist_x, dist_y, dist = float('inf'), float('inf'), float('inf')
                for check in checks:
                    _dx, _dy = xy - check
                    _dist = np.linalg.norm([_dx, _dy])
                    if _dist < dist:
                        dist_x = _dx
                        dist_y = _dy
                        dist = _dist

            if dist < min_dist:
                min_dist = dist
                min_x = dist_x
                min_y = dist_y

        assert min_x is not None and min_y is not None, "Distance to obstacle should not be None"
        return -1 * np.array([min_x, min_y], dtype=np.float32)

    def get_dist2obstacle_sphere(self, x, y, θ):
        dx, dy = self.get_dist2obstacle_cart(x, y)
        return self.cart2sphere(dx, dy, θ)

    def draw_features(self, ax):
        super().draw_features(ax)
        dObst, δObst = self.get_dist2obstacle_sphere(self.x, self.y, self.θ)
        dx = dObst * np.cos(δObst + self.θ)
        dy = dObst * np.sin(δObst + self.θ)
        ax.plot([self.x, self.x + dx],
                [self.y, self.y + dy],
                color='r', linestyle='--', label='Dist to Goal')

class Fetch_Lidar(FetchBase):
    def __init__(self, start_pos, start_velocity, start_heading, parent_env, **kwargs):
        super().__init__(start_pos, start_velocity, start_heading, parent_env, **kwargs)

        self.n_rays = kwargs.get('n_rays', 16)
        self.add_feature('dGoal', bounds=[0, np.inf])  # distance to goal
        self.add_feature('δGoal', bounds=[-np.pi, np.pi])  # relative heading to the goal

        self.δrays = np.linspace(-np.pi, np.pi, self.n_rays, endpoint=False)  # lidar beam angles
        for i in range(self.n_rays):
            self.add_feature(f'lidar{i}', bounds=[0, np.inf])
        self.post_init()

    def reset(self, random_start=False, **kwargs):
        """ add auxillary features to self._state """
        x, y, v, θ = self.get_base_startstate(random_start=random_start)
        self.update_state(x, y, v, θ)

        assert not np.any(np.isnan(self._state)), "State contains NaN values after reset. Likley missed definition in subclass."
        return self.observation

    def update_state(self, x, y, v, θ, hold_prev=False):
        dGoal, δGoal = self.get_dist2goal_sphere(x, y, θ)  # dist to goal & relative heading

        if hold_prev:
            new_state = self.deepcopy()
            new_state.update_state(x, y, v, θ, hold_prev=False)
            return new_state.observation

        else:
            self._set('x', x)
            self._set('y', y)
            self._set('v', v)
            self._set('θ', θ)
            self._set('dGoal', dGoal)
            self._set('δGoal', δGoal)

            for i, δbeam in enumerate(self.δrays):
                ray_angle = θ + δbeam
                dlidar = self.get_ray2obstacle_dist(x, y, ray_angle)
                self._set(f'lidar{i}', np.linalg.norm(dlidar))

            return self.observation

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

        p = np.array([x, y], dtype=np.float64)
        u = np.array([np.cos(δ), np.sin(δ)], dtype=np.float64)  # ray direction, |u|=1
        eps = 1e-12

        min_t = np.inf

        for obs in self.obstacles:
            if obs['type'] == 'circle':
                c = np.array(obs['center'], dtype=np.float64)
                r = float(obs['radius'])

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

            else:
                # Axis-aligned rectangle
                cx, cy = obs['center']
                w, h = float(obs['width']), float(obs['height'])
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

        assert np.isfinite(min_t), "Ray does not intersect any obstacle."
        vec = (u * min_t).astype(np.float32)  # vector from (x,y) to intersection point
        return vec

    def draw_features(self, ax):
        super().draw_features(ax)
        for i, δbeam in enumerate(self.δrays):
            ray_angle = self.θ + δbeam
            dlidar = eval(f'self.lidar{i}')
            dx = dlidar * np.cos(ray_angle)
            dy = dlidar * np.sin(ray_angle)

            ax.plot([self.x, self.x + dx],
                    [self.y, self.y + dy],
                    color='r', linestyle='--',
                    label=f'Dist to lidar{i}')


class FetchVec_Wrapper:
    def __init__(self, robot_base, dynamics_belief, n_robots=5):
        """
        Vectorized wrapper for multiple robots with different dynamics sampled from a belief distribution.
        :param robot_base:
        :param dynamics_belief: dict of robot dynamics keys with value as (mean, std)
        :param n_robots:
        """
        self._robot_base = robot_base
        self.n_robots = n_robots
        self.robots = [self._robot_base.deepcopy() for _ in range(n_robots)]
        self.dynamics_belief = dynamics_belief
        self.probs = np.zeros(n_robots) # placeholder for robot probabilities

    def ressample_dynamics(self):
        """Resamples robot dynamics from belief distribution"""
        for robot in self.robots:
            for key, val in self.dynamics_belief.items():
                mu, std = self.dynamics_belief[key]
                v = np.random.normal(mu, std)
                robot._set_dynmamic(key, v)
                raise NotImplementedError
                #TODO: need to find joint prob of p(v_i) ∀ v_i

    def steps(self, Xt, at, dt):
        """Given a single action, compute all next states for each sampled dynamics"""
        assert Xt.shape[0] == self.n_robots, f"Expected Xt shape[0] to be {self.n_robots}, got {Xt.shape[0]}"
        possible_states = []
        for i,robot in enumerate(self.robots):
            Xtti = robot.step(Xt[i], at, dt)
            possible_states.append(Xtti)
        return np.array(possible_states)

    def update_states(self, Xt):
        """ Add auxillary state features (e.g., lidar) to states"""
        states = []
        for i, robot in enumerate(self.robots):
            states.append(robot.update_state(*Xt[i]))

        return np.array(states)

class DelayedOperator:
    def __init__(self, robot_base, dynamics_belief, delay_steps=3):
        """
        Computes n_robot transitions
                from    obs = {s_(t-τ), a_(t-τ),..., a_(t-1)}  + a_t
                to      s_t+1
        """
        self._robot_base = robot_base
        self.robots = FetchVec_Wrapper(robot_base, dynamics_belief)
        self.delay_steps = delay_steps
        self.state_buffer =  deque(maxlen=delay_steps)
        self.action_buffer = deque(maxlen=delay_steps)

    def step(self, delayed_state, at, dt):
        prev_X = delayed_state
        for prev_a in self.action_buffer:
            prev_X = self.robots.steps(prev_X, prev_a, dt)
        Xtt = self.robots.steps(prev_X, at, dt) # next state after current action applied


        return Xtt


    def state_belief(self, obs,dt):
        """calculates samples of the possible current true states given...
        - delayed state observation obs 
        - belief about robot dynamics 
        - past actions since delayed observation
        """

        Xt = np.repeat(obs, self.n_robots, axis=0)
        for at in self.action_buffer:
            Xt = self.robots.step(Xt, at, dt)
        return Xt, self.robots.probs




    @property
    def observed_state(self):
        """Delayed Observation"""
        return self.state_buffer[0]

    @property
    def true_state(self):
        """Current Observation"""
        return self.state_buffer[-1]



#
# class DelayWrapper:
#     def __init__(self, robot, delay_steps=3):
#         self.robot = robot
#         self.robots = []
#         self.delay_steps = delay_steps
#         self.state_buffer =  deque(maxlen=delay_steps)
#         self.action_buffer = deque(maxlen=delay_steps)
#
#
#     def step(self, Xt, at, dt):
#         # self.state_buffer.append(Xt.copy())
#
#         # Apply and Observe delayed state-action
#         a_delayed = self.action_buffer[0]
#
#         # update buffer
#         Xt_new = self.robot.step(self.true_state, a_delayed, dt)
#         self.state_buffer.append(Xt_new.copy())
#         self.action_buffer.append(at.copy())
#
#         return self.observed_state
#
#     def step_belief(self, Xt, at, dt):
#         possible_states = []
#         for robot in self.robots:
#             obs_tt = self.observed_state
#             for tt in range(self.delay_steps):
#                 obs_tt = robot.step(obs_tt, at, dt)
#             possible_states.append(obs_tt.copy())
#             # print(f"obs_tt: {obs_tt}")
#
#
#
#         # self.state_buffer.append(Xt.copy())
#
#         # Apply and Observe delayed state-action
#         a_delayed = self.action_buffer[0]
#
#         # update buffer
#         Xt_new = self.robot.step(self.true_state, a_delayed, dt)
#         self.state_buffer.append(Xt_new.copy())
#         self.action_buffer.append(at.copy())
#
#         return self.observed_state
#
#     @property
#     def observed_state(self):
#         """Delayed Observation"""
#         return self.state_buffer[0]
#
#     @property
#     def true_state(self):
#         """Current Observation"""
#         return self.state_buffer[-1]






