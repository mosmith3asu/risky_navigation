import numpy as np
import matplotlib.pyplot as plt
from src.utils.visibility_graph import VisibilityGraph
from src.env.layouts import read_layout_dict
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
# try:
#     import gymnasium as gym
# except ImportError:
#     import gym  # Fallback for older versions of gym
# from gym import spaces
from abc import ABC, abstractmethod
import matplotlib.image as mpimg
from matplotlib import transforms as mtransforms
import warnings
import matplotlib.image as mpimg
from matplotlib import transforms as mtransforms
from utils.file_management import get_project_root
import os

class StateBase(ABC):
    def __init__(self, start_pos, start_velocity, start_heading, parent_env,
                 base_state_validator=None):
        self.start_pos = tuple(start_pos)
        self.start_velocity = start_velocity
        self.start_heading = start_heading

        self.goal = parent_env.goal
        self.bounds = parent_env.bounds
        self.vgraph = parent_env.vgraph
        self.obstacles = parent_env.obstacles
        self.robot = parent_env.robot

        self.base_state_validator = base_state_validator # function to validate base state during randomization

        self._state_bounds = {}
        self._state_names = []
        self._state_idxs = {}
        self._state = np.empty((0,0), dtype=np.float32)
        self.observation_space = None

        self.add_feature('x', bounds=[self.bounds[0][0], self.bounds[1][0]])
        self.add_feature('y', bounds=[self.bounds[0][1], self.bounds[1][1]])
        self.add_feature('v', bounds=[0, self.robot.max_lin_vel])
        self.add_feature('θ', bounds=[-np.pi, np.pi])

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

class State_DistanceHeading(StateBase):
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

class State_Lidar(StateBase):
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

class ContinuousNavigationEnvBase(gym.Env):
    def __init__(self,
                 goal           = (5.0 , 5.0),
                 bounds         = ((0.0, 0.0) , (10.0, 10.0)),
                 obstacles      = None,
                 start_pos      = (0   , 0),
                 start_heading  = 0.0,
                 start_velocity = 0.0,
                 dt             = 0.1,
                 max_steps      = 600, # 60 sec
                 vgraph = None,
                 **kwargs
                 ):


        start_heading = np.deg2rad(start_heading)  # convert heading to radians

        # Layout parameters ------------------------------------------------------------------------------------------
        self.layout        = kwargs.get('layout')
        self.goal          = tuple(goal)
        self.goal_velocity = kwargs.get('goal_velocity', 0.1)  # goal velocity, if any
        self.bounds        = bounds
        self.obstacles     = self.format_obstacles(obstacles)
        self.obstacles     = self.create_boarder_obstacles(self.obstacles)  # add borders to the environment


        self.goal_radius = kwargs.get('goal_radius', 0.25)  # meters
        self.car_radius = kwargs.get('car_radius', 0.65)  # meters
        self.fig,self.ax = None, None

        # MDP parameters ------------------------------------------------------------------------------------------
        self.dt               = dt
        self.max_steps        = max_steps
        # self.reward_goal      = kwargs.get('reward_goal'   , 10.0)
        # self.reward_collide   = kwargs.get('reward_collide', -20.0)           # penalty for collision
        # self.reward_step      = kwargs.get('reward_step'   , -10/max_steps)   # penalty for being slow
        # self.reward_dist2goal = kwargs.get('reward_dist'   , 0.25)#20/max_steps)    # maximum possible reward being close to goal
        # self.reward_smooth    = kwargs.get('reward_smooth' , 5/max_steps)     # maximum possible reward for smooth trajectory

        self.reward_goal      = kwargs.get('reward_goal'   , 10.0   )
        self.reward_collide   = kwargs.get('reward_collide', -20.0  )  # penalty for collision
        self.reward_step      = kwargs.get('reward_step'   , -0.1   )  # penalty for being slow
        self.reward_dist2goal = kwargs.get('reward_dist'   , 0.1    )  # 20/max_steps)    # maximum possible reward being close to goal
        self.reward_smooth    = kwargs.get('reward_smooth' , 0   )  # maximum possible reward for smooth trajectory

        # self.reward_goal      = kwargs.get('reward_goal'   , 0)
        # self.reward_collide   = kwargs.get('reward_collide', 0)  # penalty for collision
        # self.reward_step      = kwargs.get('reward_step'   , 0)  # penalty for being slow
        # self.reward_dist2goal = kwargs.get('reward_dist'   , 0)  # 20/max_steps)    # maximum possible reward being close to goal
        # self.reward_smooth    = kwargs.get('reward_smooth' , 0)  # maximum possible reward for smooth trajectory

        assert self.reward_dist2goal <= -1*self.reward_step, 'Reward for distance to goal should be less than or equal to timecost' \
                                                   ' to make completing task faster aslways more optimal. '

        if vgraph is None:
            resolution = kwargs.get('vgraph_resolution', (20, 20))  # resolution for visibility graph
            self.vgraph = VisibilityGraph(self.goal, self.obstacles, self.bounds, resolution=resolution)
        else: self.vgraph = vgraph

        robot_dynamics = kwargs.get('robot_dynamics', {})
        self.robot = FetchRobotMDP(**robot_dynamics)  # initialize robot dynamics

        # Action parameters ------------------------------------------------------------------------------------------
        self.action_bounds = {}
        self.action_bounds['jx'] = [-1, 1]  # normalized joystick commands (steering)
        self.action_bounds['jy'] = [-1, 1]  # normalized joystick commands (velocity)

        low_act = np.array([bound[0] for bound in self.action_bounds.values()], dtype=np.float32)
        high_act = np.array([bound[1] for bound in self.action_bounds.values()], dtype=np.float32)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

        self.steps = 0
        self.done = False  # flag to indicate if the episode is done
        # self.nav_state = State_DistanceHeading(start_pos, start_velocity, start_heading, self, base_state_validator=self.check_collision)
        self.nav_state = State_Lidar(start_pos, start_velocity, start_heading, self,base_state_validator=self.check_collision)

    def reset(self,*args,**kwargs):
        self.steps = 0
        self.done = False  # flag to indicate if the episode is done

        is_random_state = kwargs.get('p_rand_state', 0) > np.random.rand()
        state = self.nav_state.reset(random_start=is_random_state)
        return state, {}


    def step(self, action):
        assert not self.done, "Environment is done. Please reset it before stepping."

        info = {}
        self.steps += 1
        prev_state = self.state.copy()  # Store previous state for smoothness calculation

        new_state       = self.resolve_action(self.robot_state, action)  # Update state based on action and robot dynamics
        self.done, info = self.resolve_terminal_state(new_state, info)
        reward, info    = self.resolve_rewards(prev_state, new_state, info)

        # self.state = new_state # set new state

        return self.state, reward, self.done, info

    def resolve_terminal_state(self, state, info):
        done = False
        if self.check_collision(state):  # Check for collisions
            done = True
            info['reason'] = 'collision'
        elif self.check_goal(state):  # Check if goal is reached
            done = True
            info['reason'] = 'goal_reached'
        elif self.steps >= self.max_steps:  # Check if max steps reached
            done = True
            info['reason'] = 'max_steps'
        return done, info

    def resolve_rewards(self, prev_state, new_state, info):
        """
        Resolve the reward based on the previous and new state.
        """
        reward = 0.0

        try:
            dGoal = new_state[self.nav_state.idGoal]
        except:
            warnings.warn("State does not have dGoal feature. Computing manually.")
            x, y, θ = new_state[self.nav_state.ix], new_state[self.nav_state.iy], new_state[self.nav_state.iθ]
            dGoal, _ = self.nav_state.get_dist2goal_sphere(x, y, θ)

        rew_dist2goal = self.reward_dist2goal * (1 - (dGoal / self.vgraph.max_dist))  # progress towards goal reward
        rew_smooth = self.reward_smooth * self.smoothness_score(prev_state, new_state)  # smoothness reward

        reward += rew_dist2goal  # progress towards goal reward
        reward += self.reward_step  # time cost
        reward += rew_smooth

        info['rew_step'] = self.reward_step
        info['rew_dist2goal'] = rew_dist2goal
        info['rew_smooth'] = rew_smooth


        if self.done and "collision" in info['reason']:
            reward += self.reward_collide
        elif self.done and "goal_reached" in info['reason']:
            reward += self.reward_goal

        return reward, info

    def resolve_action(self,robot_state, action):
        x, y, v, θ = self.robot.step(robot_state, action, self.dt)  # Update robot state based on action
        new_state = self.nav_state.update_state(x, y, v, θ)  # Update the navigation state
        return new_state

    def check_collision(self, state):
        xy = np.array([state[self.nav_state.ix], state[self.nav_state.iy]], dtype=np.float32)

        for obs in self.obstacles:
            if obs['type'] == 'circle':
                if np.linalg.norm(xy - obs['center']) <= obs['radius'] + self.car_radius:
                    done = True
                    return done
            else:
                cx, cy = obs['center']
                w, h = obs['width'], obs['height']
                x, y = xy
                if (cx - w / 2 <= x <= cx + w / 2) and (cy - h / 2 <= y <= cy + h / 2):
                    done = True
                    return done

        done = False
        return done

    def check_goal(self, state):
        xy = np.array([state[self.nav_state.ix], state[self.nav_state.iy]], dtype=np.float32)
        # xy = np.array([state[self.sidx['x']], state[self.sidx['y']]], dtype=np.float32)
        v = state[self.nav_state.iv]

        dist_to_goal = np.linalg.norm(xy - np.array(self.goal))
        if dist_to_goal <= self.goal_radius and v < self.goal_velocity:
            return True
        return False

    def render(self, mode='human',ax=None,dist_lines=False):
        obs_alpha= 1
        goal_alpha = 0.5
        if ax is not None:
            self.ax = ax
        elif self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()

        self.ax.clear()
        self.ax.set_xlim(self.bounds[0][0]-self.border_sz/2, self.bounds[1][0]+self.border_sz/2)
        self.ax.set_ylim(self.bounds[0][1]-self.border_sz/2, self.bounds[1][1]+self.border_sz/2)
        self.ax.set_aspect('equal', adjustable='box')

        # Draw assets
        self.draw_obstacles(self.ax)
        self.draw_goal(self.ax)
        self.draw_robot(self.ax)
        if dist_lines:
            self.nav_state.draw_features(self.ax)
            # self.draw_dist_lines(self.ax)

        plt.draw()
        plt.pause(0.0001)

    def draw_robot(self,ax, image_path = "src/env/assets/fetch_robot.png"):
        """insert image of robot from ./assets/fetch_robot.png"""
        project_root = get_project_root()

        image_path = os.path.join(project_root, image_path)

        image_height = self.car_radius
        arrow_len = 1.3 * self.car_radius
        arrow_width = 0.1 * self.car_radius
        arrow_color = 'b'
        x,y,theta = self.x, self.y, self.θ

        # Add arrow to indicate direction
        dx = arrow_len * np.cos(self.θ)
        dy = arrow_len * np.sin(self.θ)
        ax.arrow(self.x, self.y, dx, dy, head_width=arrow_width, color=arrow_color)

        # add image of fetch ########
        img = mpimg.imread(image_path)

        h_px, w_px = img.shape[0], img.shape[1]
        aspect = w_px / h_px if h_px != 0 else 1.0
        image_width = image_height * aspect

        # Build a rotation-around-center transform in data coords
        t = mtransforms.Affine2D().rotate_around(x, y, theta) + ax.transData

        # Place the image via extent in data units
        ax.imshow(
            img,
            extent=[x - image_width / 2, x + image_width / 2,
                    y - image_height / 2, y + image_height / 2],
            transform=t,
            zorder=3,
            interpolation="bilinear",
        )


        # car = plt.Circle((self.x, self.y), self.car_radius, color='blue')
        # ax.add_patch(car)
        # dx = arrow_len * np.cos(self.θ)
        # dy = arrow_len * np.sin(self.θ)
        # ax.arrow(self.x, self.y, dx, dy, head_width=arrow_width, color='black')

    def draw_goal(self, ax, goal_alpha= 0.7):
        goal_circle = plt.Circle(tuple(self.goal), self.goal_radius, color='green', alpha=goal_alpha)
        ax.add_patch(goal_circle)

    def draw_obstacles(self, ax, obs_alpha=1.0):
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                circle = plt.Circle(tuple(obs['center']), obs['radius'], color='k', alpha=obs_alpha)
                ax.add_patch(circle)
            else:
                cx, cy = obs['center']
                w, h = obs['width'], obs['height']
                rect = plt.Rectangle((cx - w/2, cy - h/2), w, h, color='k', alpha=obs_alpha)
                ax.add_patch(rect)

    def render_reward_heatmap(self, ax=None, cmap="coolwarm", vmin=None, vmax=None, show_colorbar=True,block=True):

        # grid = self.vgraph.grid
        # H, W, _ = grid.shape
        resolution = (50,50)
        X = np.linspace(self.bounds[0][0], self.bounds[1][0], resolution[0])
        Y = np.linspace(self.bounds[0][1], self.bounds[1][1], resolution[1])

        # Compute Z by querying dist at each grid point
        Z = np.empty(resolution, dtype=float)

        for c, x in enumerate(X):
            for r, y in enumerate(Y):
                dGoal, δGoal = self.nav_state.get_dist2goal_sphere(x, y, 0)  # distances to goal and obstacles
                rew_dist2goal = self.reward_dist2goal * (1 - (dGoal / self.vgraph.max_dist))  # progress towards goal reward
                Z[r, c] = float(rew_dist2goal)

        # World-coordinate extent for imshow
        xs = X
        ys = Y
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(
            Z,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Visibility-Graph Distance Heatmap")

        self.draw_obstacles(ax)
        self.draw_goal(ax)
        self.draw_robot(ax)

            # Mark goal (if available)
        if hasattr(self.vgraph, "goal") and self.vgraph.goal is not None:
            gx, gy = self.vgraph.goal
            ax.plot(gx, gy, marker="*", markersize=12, linewidth=0, label="goal")
            ax.legend(loc="upper right")

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("distance")

        ax.set_xlim(self.bounds[0][0] - self.border_sz / 2, self.bounds[1][0] + self.border_sz / 2)
        ax.set_ylim(self.bounds[0][1] - self.border_sz / 2, self.bounds[1][1] + self.border_sz / 2)
        ax.set_aspect('equal', adjustable='box')

        if block:
            plt.show(block=True)

        return ax, im, Z

    ############################################
    # Initialization and environment setup
    def format_obstacles(self, obstacles_list):
        obstacles = []
        if obstacles_list:
            for o in obstacles_list:
                obs_type = o.get('type', 'circle')
                if obs_type == 'circle':
                    obstacles.append({
                        'type': 'circle',
                        'center': np.array(o['center'], dtype=np.float32),
                        'radius': float(o.get('radius', 0.5))
                    })
                elif obs_type == 'rect':
                    obstacles.append({
                        'type': 'rect',
                        'center': np.array(o['center'], dtype=np.float32),
                        'width': float(o['width']),
                        'height': float(o['height'])
                    })
                else:
                    raise ValueError(f"Unsupported obstacle type: {obs_type}")
        return obstacles

    def create_boarder_obstacles(self,obstacles):

        border_sz = 0.5
        self.border_sz = border_sz
        boundH = self.bounds[1][1] - self.bounds[0][1]
        boundW = self.bounds[1][0] - self.bounds[0][0]
        # Left border
        obstacles.append({'type': 'rect',
                               'center': np.array([self.bounds[0][0] - border_sz / 2, self.bounds[1][1] / 2],
                                                  dtype=np.float32),
                               'width': border_sz,
                               'height': boundH + 2 * border_sz
                               })
        # Right border
        obstacles.append({'type': 'rect',
                               'center': np.array([self.bounds[1][0] + border_sz / 2, self.bounds[1][1] / 2],
                                                  dtype=np.float32),
                               'width': border_sz,
                               'height': boundH + 2 * border_sz
                               })
        # Bottom border
        obstacles.append({'type': 'rect',
                               'center': np.array([self.bounds[1][0] / 2, self.bounds[0][1] - border_sz / 2],
                                                  dtype=np.float32),
                               'width': boundW + 2 * border_sz,
                               'height': border_sz
                               })
        # Top border
        obstacles.append({'type': 'rect',
                               'center': np.array([self.bounds[1][0] / 2, self.bounds[1][1] + border_sz / 2],
                                                  dtype=np.float32),
                               'width': boundW + 2 * border_sz,
                               'height': border_sz
                               })
        return obstacles

    def smoothness_score(self, state, prev_state, exp=2):
        if self.reward_smooth == 0:
            return 0

        def normalize(value, bounds):
            return (value - bounds[0]) / (bounds[1] - bounds[0])

        """
        Calculate the smoothness score based on the change in state [0,1]
        """
        dstate = state - prev_state
        # Calculate the smoothness score as the sum of absolute normalized differences
        smoothness = np.mean([
            normalize(dstate[self.nav_state.ix], self.nav_state._state_bounds['x']),
            normalize(dstate[self.nav_state.iy], self.nav_state._state_bounds['y']),
            normalize(dstate[self.nav_state.iv], self.nav_state._state_bounds['v']),
            normalize(dstate[self.nav_state.iθ], self.nav_state._state_bounds['θ'])

        ])
        return smoothness ** exp

    #############################################
    # Properties and accessing states
    @property
    def rem_steps(self):
        return self.max_steps - self.steps

    @property
    def x(self): return self.nav_state.x

    @property
    def y(self): return self.nav_state.y

    @property
    def v(self): return self.nav_state.v

    @property
    def θ(self): return self.nav_state.θ

    @property
    def state(self):
        return self.nav_state.observation

    @property
    def observation_space(self):
        return self.nav_state.observation_space

    @property
    def robot_state(self):
        return self.nav_state.base_state

class ContinuousNavigationEnvVec(ContinuousNavigationEnvBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.terminal_reward = 0
        self.terminal_reason = ''
        self.auto_reset = False
        self.p_rand_state = 0
        self.enable_reset = False

        self.def_info = {'reason': '',
                         'done': False,
                         'rew_step': 0,
                         'rew_dist2goal': 0,
                         'rew_smooth': 0
                         }

    def step(self, action):
        """Gymnasium-style step: returns (obs, reward, terminated, truncated, info). """
        self.steps += 1
        info = self.def_info.copy()

        if self.done:
            info['reason'] = self.terminal_reason
            info['done'] = self.done
            terminated = self.done if self.auto_reset else False
            truncated = False
            return self.state, float(self.terminal_reward), \
                bool(terminated), bool(truncated), info

        self.steps += 1
        prev_state = self.state.copy()
        new_state = self.resolve_action(self.robot_state, action)
        self.done, reason = self.resolve_terminal_state(new_state, info)

        info['reason'] = reason
        info['done'] = self.done

        terminated = False #self.done if self.auto_reset else False
        truncated = False

        # self.done = bool(terminated or truncated)  # keep for render()

        # terminated, truncated, info = self.resolve_terminal_state(new_state, info)
        reward, info = self.resolve_rewards(prev_state, new_state, info, terminated, truncated)

        return new_state, float(reward), bool(terminated), bool(truncated), info

    def resolve_terminal_state(self, state, info):
        done = False
        reason = ''

        if self.check_collision(state):
            done = True
            reason = 'collision'
        elif self.check_goal(state):
            done = True
            reason = 'goal_reached'
        elif self.steps >= self.max_steps:
            done = True
            reason = 'max_steps'


        return done, reason

    def resolve_rewards(self, prev_state, new_state, info, terminated=False, truncated=False):

        """
        Resolve the reward based on the previous and new state.
        """
        # reward, info = super().resolve_rewards(prev_state, new_state, info)
        #
        #
        # if self.done and "collision" in info['reason']:
        #     self.terminal_reward = 0
        #     self.terminal_reason =  info['reason']
        #
        # elif self.done and "goal_reached" in info['reason']:
        #     self.terminal_reward = 0
        #     self.terminal_reason = info['reason']
        #
        #
        # return reward, info
        reward = 0.0

        dGoal = self.nav_state.dGoal
        dist_prog = (1 - (dGoal / self.vgraph.max_dist))  # ** 2  # progress towards goal
        rew_dist2goal = self.reward_dist2goal * dist_prog  # progress towards goal reward
        rew_smooth = self.reward_smooth * self.smoothness_score(prev_state, new_state)  # smoothness reward

        reward += rew_dist2goal  # progress towards goal reward
        reward += self.reward_step  # time cost
        reward += rew_smooth

        info['rew_step'] = self.reward_step
        info['rew_dist2goal'] = rew_dist2goal
        info['rew_smooth'] = rew_smooth

        if self.done and "collision" in info['reason']:
            reward += self.reward_collide
            # self.terminal_reward =  self.reward_step
            # self.terminal_reward = reward
            self.terminal_reward = 0
            self.terminal_reason = info['reason']

        elif self.done and "goal_reached" in info['reason']:
            reward += self.reward_goal
            # self.terminal_reward = rew_dist2goal
            # self.terminal_reward = reward
            self.terminal_reward = 0
            self.terminal_reason = info['reason']

        return reward, info

    def reset(self, seed=None, options=None):

        """  Reset the environment and return the initial observation. """
        p_rand_state = self.p_rand_state
        enable_reset = self.enable_reset
        if options is not None:
            p_rand_state = options.get('p_rand_state', p_rand_state)
            enable_reset = options.get('enable', enable_reset)
            self.reward_dist2goal = options.get('reward_dist2goal', self.reward_dist2goal)


        if self.auto_reset or enable_reset:
            # print(f'Resetting environment... {self.steps}')
            self.terminal_reward = 0
            self.terminal_reason = ''
            state, _ = super().reset(p_rand_state=p_rand_state)
            return state, self.def_info.copy()

        # elif options is not None:
        #
        #     self.reward_dist2goal = options.get('reward_dist2goal', self.reward_dist2goal)
        #
        #     enable = options.get('enable', True) if options is not None else False
        #     if enable:
        #         self.terminal_reward = 0
        #         self.terminal_reason = ''
        #         return super().reset(p_rand_state=options.get('p_rand_state', 0))

        return self.state, self.def_info.copy()


def _make_env_thunk(layout_name, **overrides):
    """Returns a thunk that builds a single ContinuousNavigationEnv."""

    def _thunk():
        layout = read_layout_dict(layout_name)
        layout.update(overrides)
        env = ContinuousNavigationEnvVec(**layout)
        return env

    return _thunk

def build_sync_vector_env(n_envs=4, layout_name='example1', **overrides):
    """
    Create a SyncVectorEnv of N identical ContinuousNavigationEnv instances.
    Use vec_env.reset(seed=...) to seed all subenvs; use reset_done(mask) to
    selectively reset finished envs.
    """
    thunks = [_make_env_thunk(layout_name, **overrides) for _ in range(n_envs)]
    return SyncVectorEnv(thunks)


class FetchRobotMDP:
    def __init__(self,
                 min_lin_vel = 0.0, #min_lin_vel = -0.5,
                 max_lin_vel = 1.0,
                 max_lin_acc = 0.5, # 1.0
                 max_rot_vel = np.pi/2 #3.0
                 ):

        # Robot dynamics parameters
        self.min_lin_vel = min_lin_vel  # minim linear velocity in m/s (<0 means reversing)
        self.max_lin_vel = max_lin_vel  # maximum linear velocity in m/s
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


def main():
    from utils.joystick import VirtualJoystick
    import time

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    joystick = VirtualJoystick(ax=axs[0], deadzone=0.05, smoothing=0.35, spring=True)

    # layout_dict = read_layout_dict('example0')
    # layout_dict = read_layout_dict('example1')
    layout_dict = read_layout_dict('example2')

    layout_dict['vgraph_resolution'] = (10, 10)  # resolution for visibility graph
    env = ContinuousNavigationEnvBase(**layout_dict)
    obs = env.reset()

    # env.render_reward_heatmap(block=True)

    done = False
    total_reward = 0.0
    axs[-1].set_title("Virtual Joystick Input")

    rewards = []
    while not done:
        tstart = time.time()
        x, y, r, th, active = joystick.get()
        print(f'Input: [{x:.2f}{y:.2f}] '
              f'State: x={env.x:.2f}, y={env.y:.2f}, '
              f'v ={env.v:.2f} θ={env.θ:.2f}, '
              f'')
        action = np.array([x, y], dtype=np.float32)  # Use joystick input as action
        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        if len(rewards) > 2:
            axs[-1].plot(rewards)
            axs[-1].relim()

        total_reward += reward
        env.render(dist_lines=True, ax=axs[1])
        while time.time() - tstart < env.dt:
            pass

        if done:
            obs = env.reset()

    print("Episode finished. Total reward:", total_reward, "Info:", info)
    env.close()
if __name__ == "__main__":
   main()