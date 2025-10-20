import numpy as np
import matplotlib.pyplot as plt
from src.utils.visibility_graph import VisibilityGraph
from src.env.layouts import read_layout_dict
from gymnasium import spaces
from abc import ABC, abstractmethod
from collections import deque
import math
import warnings
from env.mdp import make_belief_fetchrobot_mdp, Compiled_LidarFun, Belief_FetchRobotMDP_Compiled
from utils.file_management import get_project_root
import os

from matplotlib import transforms as mtransforms
import matplotlib.image as mpimg


#################################################################################
# MDP and State Classes #########################################################
#################################################################################

class Delayed_LidarState:
    def __init__(self, X0, f_dist2goal, f_lidar, delay_steps, dt, n_rays=10,
                 bounds = ((-np.inf,-np.inf),(np.inf,np.inf)), dynamics_belief=None,
                 robot_state_validator = None,
                 **kwargs):

        # -------------------------------------------------------------------------------------------
        # Unpack ------------------------------------------------------------------------------------
        self._X0 = np.array(X0, dtype=np.float32)   # initial roboto position
        self.f_dist2goal = f_dist2goal              # vectorized function to compute distance and heading to goal
        self.f_lidar = f_lidar                      # vectorized function to compute lidar features
        self._delay_steps = delay_steps             # number of delay steps to simulate
        self.robot_state_validator = robot_state_validator # function to validate robot state

        if dynamics_belief is None:
            dynamics_belief = {}
            warnings.warn("dynamics_belief not provided to Delayed_LidarState. Using default belief.")

        # self.robot = make_belief_fetchrobot_mdp(**dynamics_belief) # robot dynamics model with uncertainty in parameters
        self.robot = Belief_FetchRobotMDP_Compiled(
            n_samples = kwargs.get('n_samples', 50),
            b_min_lin_vel = dynamics_belief.get('b_min_lin_vel',(0.0, 1e-6) ),
            b_max_lin_vel = dynamics_belief.get('b_max_lin_vel',(1.0, 0.5) ),
            b_max_lin_acc = dynamics_belief.get('b_max_lin_acc',(0.5, 0.2) ),
            b_max_rot_vel = dynamics_belief.get('b_max_rot_vel',(math.pi / 2.0, math.pi / 6.0) )
        )

        self.n_samples = self.robot.n_samples       # number of dynamics samples for belief MDP
        self.dt = dt                                # time step in seconds
        self.bounds = bounds

        #-------------------------------------------------------------------------------------------
        # Configure Features ----------------------------------------------------------------------
        self._feature_bounds = {}
        self._feature_names = []
        self._feature_idxs = {}

        # Standard Robot State Features
        self.robot_state_dim = 4  # x,y,v,theta
        self._add_feature('x', bounds=[bounds[0][0], bounds[1][0]])
        self._add_feature('y', bounds=[bounds[0][1], bounds[1][1]])
        self._add_feature('v', bounds=[self.robot.true_min_lin_vel, self.robot.true_max_lin_vel ])
        self._add_feature('θ', bounds=[-np.pi, np.pi])

        # Previous Action Features
        self.action_state_dim = 2 * delay_steps  # ax,ay history
        self.action_hist_state_dim = n_rays
        for i in range(self._delay_steps):
            self._add_feature(f'jx{i}', bounds=[-1, 1])
            self._add_feature(f'jy{i}', bounds=[-1, 1])

        # Distance Features
        self.dist_state_dim = 2  # dGoal, δGoal
        self._add_feature('dGoal', bounds=[0, np.inf])  # distance to goal
        self._add_feature('δGoal', bounds=[-np.pi, np.pi])  # relative heading to the goal

        # Lidar Features
        self.lidar_state_dim = n_rays
        for i in range(n_rays):
            self._add_feature(f'lidar{i}', bounds=[0, np.inf])
        self.lidar_angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)  # lidar beam angles

        # forms observation space
        self._compute_spaces()

        # ---------------------------------------------------------------------------------------
        # Configure States ----------------------------------------------------------------------
        self._true_robot_state = self._X0.copy()                    # true robot state (not delayed)
        self._true_dist2goal_state = self.f_dist2goal(self._true_robot_state[np.newaxis,:])  # true distance to goal state (not delayed)
        self._true_lidar_state = self.f_lidar(self._true_robot_state[np.newaxis,:])  # true lidar state (not delayed)

        self._action_buffer = deque(maxlen=self._delay_steps)       # buffer of previous action samples [delay steps x 2]
        self._robot_state_buffer = deque(maxlen=self._delay_steps)  # buffer of previous robot state samples [delay steps x N x 4]
        self._cached_observation = None


    ###################################################################################################
    # Initialization Methods ##########################################################################
    def _add_feature(self, name, bounds=(-np.inf, np.inf), init_val=np.nan):
        self._feature_idxs[name] = len(self._feature_idxs)
        self._feature_bounds[name] = bounds
        self._feature_names.append(name)

        # create index property
        def idx_prop(self, _name=name):
            return self._feature_idxs[_name]

        # create getter
        def getter(self, _name=name):
            state = self.get_curr_full_state()
            return state[self._feature_idxs[_name]]

        # create setter
        def setter(self, value, _name=name):
            raise ValueError(f"Cannot set {name} directly. Use update_state() instead.")
            # self._state[self._feature_idxs[_name]] = value

        # dynamically set attributes on the class
        setattr(self.__class__, f"i{name}", property(idx_prop))
        setattr(self.__class__, name, property(getter, setter))

    def _compute_spaces(self):
        low_obs = np.array([bound[0] for bound in self._feature_bounds.values()], dtype=np.float32)
        high_obs = np.array([bound[1] for bound in self._feature_bounds.values()], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

    ###################################################################################################
    # Training Loop Methods ###########################################################################
    def step(self,action):
        assert self.is_full, "Delay buffer not full before step()"
        assert action.shape == (2,), f"action shape {action.shape} does not match expected {(2,)}"

        self._cached_observation = None  # clear cached observation

        # Step belief samples (Xhat_t) under new dynamics samples
        self.robot.resample_dynamics()                                  # resample robot dynamics
        Xdelay, ahist = self.delayed_robot_state, self.action_history   # get delayed state and action history
        Xhat_t = self.infer_curr_robot_state(Xdelay, ahist)             # Recover belief about current robot state (Xhat_t)
        Xhat_tt = self.robot.step(Xhat_t, action, self.dt)              # Take step under dynamics samples
        dGhat_tt, δGhat_tt = self.f_dist2goal(Xhat_tt)                  # Compute distances features
        Lhat_tt = self.f_lidar(Xhat_tt)                                 # compute lidar features
        shat_tt = self.compose_state(X= Xhat_tt,                        # compose new state (ensures correct idxs)
                                     ahist = ahist.flatten(),
                                     goal_dist= dGhat_tt,
                                     goal_heading = δGhat_tt,
                                     lidar_dists = Lhat_tt
                                     )

        # Step true state (X_t)
        X_t = self.get_curr_robot_state()                               # resample robot dynamics
        X_tt = self.robot.step_true(X_t, action, self.dt)               # get delayed state and action history
        dG_tt, δG_tt = self.f_dist2goal(X_tt)                           # Recover belief about current robot state (Xhat_t)
        L_tt = self.f_lidar(X_tt)
        s_tt = self.compose_state(X=X_tt,                               # Take step under dynamics samples
                                  ahist=ahist.flatten(),
                                  goal_dist=dG_tt,
                                  goal_heading=δG_tt,
                                  lidar_dists=L_tt
                                  )


        self._true_robot_state = X_tt.copy().reshape(1,self.robot_state_dim)                 # update true robot state
        self._true_dist2goal_state = np.hstack([dG_tt,δG_tt]).reshape(1,self.dist_state_dim) # update true distance to goal state
        self._true_lidar_state = L_tt.reshape(1,self.lidar_state_dim)                        # update true lidar state
        self.put_buffer(X_t, action)  # put new true state and action in buffer
        return s_tt, shat_tt

    def reset(self, is_rand=False):
        self._cached_observation = None
        self._true_robot_state = self._X0.copy()  # true robot state (not delayed)
        if is_rand:  self._true_robot_state = self._get_rand_robot_state()
        self.clear_buffer()
        self.fill_buffer()

    ###################################################################################################
    # State/delay handler methods #####################################################################
    def decompose_state(self,S):
        X = S[0:4]; i =4;
        ahist = S[i:i+self.action_state_dim];i += self.action_state_dim;
        goal_dist = S[i:i+self.dist_state_dim];i += self.dist_state_dim;
        lidar_dists = S[i:i + self.lidar_state_dim];i += self.lidar_state_dim;
        return X,ahist,goal_dist,lidar_dists

    def compose_state(self,X, ahist, goal_dist, goal_heading, lidar_dists):
        return np.hstack([X,
                          ahist[np.newaxis,:].repeat(X.shape[0],axis=0),
                          goal_dist[:,np.newaxis],
                          goal_heading[:,np.newaxis],
                          lidar_dists])

    def clear_buffer(self):
        """ Clear the delay buffers."""
        self._action_buffer.clear()
        self._robot_state_buffer.clear()

    def fill_buffer(self, action=(0.0, 0.0), robot_state=None):
        """ Fill the delay buffers with uniform values (e.g., waiting at beginning). """
        if robot_state is None:
            robot_state = self._true_robot_state.copy().reshape(1,4)
            # robot_state = robot_state.repeat(self.n_samples, axis=0)
        action = np.array(action, dtype=np.float32)
        for _ in range(self._delay_steps +1):
            self.put_buffer(robot_state, action)
        assert self.is_full, "Delay buffer not full after fill_buffer()"

    def put_buffer(self,robot_state,action):
        assert self.is_valid_robot_state(robot_state,stype='single'), f"robot_state shape {robot_state.shape} does not match expected {(4,)}"
        assert action.shape == (2,), f"action shape {action.shape} does not match expected {(2,)}"
        self._robot_state_buffer.append(robot_state)
        self._action_buffer.append(action)

    def _get_rand_robot_state(self):
        x,y,v,θ = self._true_robot_state

        attempt = 0


        for _ in range(50):
            attempt += 1
            _x = np.random.uniform(self.bounds[0][0], self.bounds[1][0])
            _y = np.random.uniform(self.bounds[0][1], self.bounds[1][1])
            _Xt = np.array([_x,_y,0,0],dtype=np.float32)[np.newaxis,:]

            if self.robot_state_validator is None or \
                    not self.robot_state_validator(_Xt):
                x, y = _x, _y
                break

        if attempt > 49:
            warnings.warn("Failed to find a valid random X Y after 50 attempts. Using last standard.")
        else:

            v = np.random.uniform(0, self.robot.true_max_lin_vel)  # random initial velocity
            _, δGoal = self.f_dist2goal(np.array([x, y, 0, 0], dtype=np.float32)[np.newaxis, :])
            θ = δGoal[0] + np.random.uniform(-np.pi / 4, np.pi / 4)

        return np.array([x, y, v, θ],dtype=np.float32).reshape(1,-1)

    def infer_curr_robot_state(self,robot_state,actions):
        assert self.is_valid_robot_state(robot_state,stype='single'), f"robot_state shape {robot_state.shape} does not match expected {(4,)}"
        Xt = np.repeat(robot_state, self.n_samples, axis=0)
        for action in actions:
            Xt = self.robot.step(Xt, action, self.dt)
        return Xt

    @property
    def robot_probs(self):
        return self.robot.robot_prob.copy()

    @property
    def observation(self):
        if self._cached_observation is not None:
            return self._cached_observation

        Xdelay, ahist = self.delayed_robot_state, self.action_history
        dGhat_t, δGhat_t = self.f_dist2goal(Xdelay)                    # Compute distances features
        Lhat_t = self.f_lidar(Xdelay)                                   # compute lidar features

        self._cached_observation = self.compose_state(
            X= Xdelay,
            ahist = ahist.flatten(),
            goal_dist= dGhat_t,
            goal_heading = δGhat_t,
            lidar_dists = Lhat_t
        ).flatten()
        return self._cached_observation.copy()


    @property
    def delayed_robot_state(self):
        return self._robot_state_buffer[0].copy()
    @property
    def action_history(self):
        return np.array(self._action_buffer)

    def get_curr_robot_state(self):
        return self._true_robot_state.copy().reshape(1, -1)
    def get_curr_lidar_state(self):
        return self._true_lidar_state.copy().reshape(1, -1)
    def get_curr_dist2goal_state(self):
        return self._true_dist2goal_state.copy().reshape(1, -1)

    # def get_curr_full_state(self):
    #     return self.compose_state(X=self.get_curr_robot_state(),
    #                               ahist=self.action_history.flatten()[np.newaxis,:],
    #                               goal_dist=self.get_curr_dist2goal_state()[:,0],
    #                               goal_heading=self.get_curr_dist2goal_state()[:,1],
    #                               lidar_dists=self.get_curr_lidar_state()
    #                               )



    ###################################################################################################
    # Status methods ##################################################################################
    def is_valid_robot_state(self, robot_state, stype = None):
        """Checks shape of arbitrary robot state"""
        if stype is None:
            return robot_state.shape[-1] == 4 and not np.any(np.isnan(robot_state))
        elif stype.lower() == 'sampled':
            return robot_state.shape == (self.n_samples,4) and not np.any(np.isnan(robot_state))
        elif stype.lower() == 'true' or stype.lower() == 'single':
            return robot_state.shape == (1,4) and not np.any(np.isnan(robot_state))
        else:
            raise ValueError(f"stype {stype} not recognized. Use None, 'sampled', or 'true'.")

    @property
    def is_consistant_size(self):
        return len(self._action_buffer) == len(self._robot_state_buffer)

    @property
    def is_full(self):
        return len(self._action_buffer) == self._action_buffer.maxlen and len(
            self._robot_state_buffer) == self._robot_state_buffer.maxlen

    @property
    def is_empty(self):
        return len(self._action_buffer) == 0 or len(self._robot_state_buffer) == 0

    @property
    def is_sampled(self):
        return self.dynamics_belief is not None and self.robot_prob is not None

    def is_valid_observation(self,obs):
        assert len(obs) == self.observation_space.shape[0], f"Observation size {len(obs)} does not match observation_space size {self.observation_space.shape[0]}"
        assert not np.any(np.isnan(obs)), "Obs contains NaN values after reset. Likley missed definition in subclass."
        return True

#################################################################################
# Environment Classes ###########################################################
#################################################################################
class EnvBase(ABC):
    def __init__(self,**kwargs):
        self.fig = None
        self.ax = None
        self.bounds = None
        self.goal = None
        self.obstacles = None
        self.dt = None
        self.max_steps = None
        self.state = None


        self.goal_velocity      = kwargs.get('goal_velocity' , 0.1)  # goal velocity          , if any
        self.goal_radius        = kwargs.get('goal_radius'   , 0.25)  # meters
        self.car_radius         = kwargs.get('car_radius'    , 0.65)  # meters

        self.reward_goal        = kwargs.get('reward_goal'   , 10.0)
        self.reward_collide     = kwargs.get('reward_collide', -20.0)  # penalty for collision
        self.reward_step        = kwargs.get('reward_step'   , -0.1)  # penalty for being slow
        self.reward_dist2goal   = kwargs.get('reward_dist',
                                           0.1)  # 20/max_steps)    # maximum possible reward being close to goal
        assert self.reward_dist2goal <= -1 * self.reward_step, 'Reward for distance to goal should be less than or equal to timecost' \
                                                               ' to make completing task faster aslways more optimal. '

    @abstractmethod
    def step(self, *args, **kwargs):
        pass
    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

    def render(self, ax=None,
               draw_robot=True, draw_obstacles=True,
               draw_goal=True, draw_dist2goal=True,
               draw_lidar=True, pause=None):
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
        if draw_obstacles: self.draw_obstacles(self.ax)
        if draw_goal     : self.draw_goal(self.ax)
        if draw_robot    : self.draw_robot(self.ax)
        if draw_dist2goal: self.draw_dist2goal(self.ax)
        if draw_lidar    : self.draw_lidar(self.ax)

        plt.draw()
        if pause is not None:
            plt.pause(pause)

    def draw_robot(self,ax, image_path = "src/env/assets/fetch_robot.png"):
        """insert image of robot from ./assets/fetch_robot.png"""
        project_root = get_project_root()

        image_path = os.path.join(project_root, image_path)

        image_height = self.car_radius
        arrow_len = 1.3 * self.car_radius
        arrow_width = 0.1 * self.car_radius
        arrow_color = 'b'


        x, y, v, θ = self.state.get_curr_robot_state().flatten()


        # Add arrow to indicate direction
        dx = arrow_len * np.cos(θ)
        dy = arrow_len * np.sin(θ)
        ax.arrow(x, y, dx, dy, head_width=arrow_width, color=arrow_color)

        # add image of fetch ########
        img = mpimg.imread(image_path)

        h_px, w_px = img.shape[0], img.shape[1]
        aspect = w_px / h_px if h_px != 0 else 1.0
        image_width = image_height * aspect

        # Build a rotation-around-center transform in data coords
        t = mtransforms.Affine2D().rotate_around(x, y, θ) + ax.transData

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

    def draw_goal(self, ax, goal_alpha= 1.0):
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

    def draw_dist2goal(self, ax):
        x, y, v, θ = self.state.get_curr_robot_state().flatten()
        dGoal, δGoal = self.state.get_curr_dist2goal_state().flatten()

        # Line to goal
        gx = x + dGoal * np.cos(δGoal + θ)
        gy = y + dGoal * np.sin(δGoal + θ)
        ax.plot([x, gx], [y, gy], 'g--', label='to goal')

    def draw_lidar(self,ax):
        x, y, v, θ = self.state.get_curr_robot_state().flatten()
        L = self.state.get_curr_lidar_state().flatten()
        for i, δbeam in enumerate(self.state.lidar_angles):
            ray_angle = θ + δbeam
            dlidar = L[i]# eval(f'self.state.lidar{i}')
            dx = dlidar * np.cos(ray_angle)
            dy = dlidar * np.sin(ray_angle)

            ax.plot([x, x + dx],
                    [y, y + dy],
                    color='r', linestyle='--',
                    label=f'Dist to lidar{i}')

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

    def create_boarder_obstacles(self, obstacles):

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


class ContinousNavEnv(EnvBase):

    @classmethod
    def from_layout(cls, layout_name, *args, **kwargs):

        layout_dict = read_layout_dict(layout_name)
        kwargs.update(layout_dict)
        return cls(*args, **kwargs)


    def __init__(self,
                 goal              = (5.0 , 5.0),
                 bounds            = ((0.0, 0.0), (10.0, 10.0)),
                 obstacles         = None,
                 start_pos         = (0, 0),
                 start_heading     = 0.0,
                 start_velocity    = 0.0,
                 dt                = 0.1,
                 delay_steps       = 3,
                 max_steps         = 600, # 60 sec
                 vgraph_resolution = 30,#(30, 30),
                 **kwargs
                 ):
        super().__init__(**kwargs) # contains default vals that do not need to be changed

        start_pos = np.array(start_pos, dtype=np.float32)
        start_velocity = np.array(start_velocity, dtype=np.float32)
        start_heading = np.deg2rad(start_heading)  # convert heading to radians
        X0 = np.hstack([start_pos, start_velocity, start_heading])

        # Layout parameters ------------------------------------------------------------------------------------------
        self.layout = kwargs.get('layout')
        self.bounds    = bounds
        self.goal      = tuple(goal)
        self.obstacles = self.format_obstacles(obstacles)
        self.obstacles = self.create_boarder_obstacles(self.obstacles)  # add borders to the environment
        self.dt        = dt
        self.max_steps = max_steps

        # Action parameters ------------------------------------------------------------------------------------------
        self.action_bounds = {}
        self.action_bounds['jx'] = [-1, 1]  # normalized joystick commands (steering)
        self.action_bounds['jy'] = [-1, 1]  # normalized joystick commands (velocity)

        low_act = np.array([bound[0] for bound in self.action_bounds.values()], dtype=np.float32)
        high_act = np.array([bound[1] for bound in self.action_bounds.values()], dtype=np.float32)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

        # State parameters ------------------------------------------------------------------------------------------
        self.steps = 0
        self.done = False  # flag to indicate if the episode is done

        if isinstance(vgraph_resolution, int):
            dx = (self.bounds[1][0] - self.bounds[0][0])
            dy = (self.bounds[1][1] - self.bounds[0][1])
            dmax = max(dx, dy)
            vgraph_resolution = (int(vgraph_resolution * (dx/dmax)),int(vgraph_resolution * (dy/dmax)))

        vgraph = VisibilityGraph(self.goal, self.obstacles, self.bounds, resolution=vgraph_resolution )
        self.max_dist = vgraph.max_dist
        f_dist2goal = vgraph.get_compiled_funs()
        f_lidar = Compiled_LidarFun(self.obstacles, kwargs.get('n_rays', 10))
        self.state = Delayed_LidarState(X0, f_dist2goal, f_lidar, delay_steps, dt, bounds=self.bounds,
                                        robot_state_validator = self._check_collision,
                                        **kwargs)

        print(f"ContinousNavEnv initialized with layout: {self.layout}")
        s = "\nBelief_FetchRobotMDP_Compiled:"
        s += f"\n  n_samples: {self.state.robot.n_samples}"
        s += f"\n  belief_mu: {self.state.robot.belief_mu}"
        s += f"\n  belief_std: {self.state.robot.belief_std}"
        s += f"\n  true parameters:"
        s += f"\n    true_min_lin_vel: {self.state.robot.true_min_lin_vel}"
        s += f"\n    true_max_lin_vel: {self.state.robot.true_max_lin_vel}"
        s += f"\n    true_max_lin_acc: {self.state.robot.true_max_lin_acc}"
        s += f"\n    true_max_rot_vel: {self.state.robot.true_max_rot_vel}"
        s += f"\n"
        print(s)


    def step(self, action, true_done = False):
        self.steps += 1
        master_info = {}

        true_info = {}
        infos = {}

        s_tt,shat_tt = self._resolve_action(action) # State transition

        # Compute sampled transition
        dones, infos = self._resolve_terminal_state(shat_tt, infos)  # check reach goal or collision
        rewards, infos = self._resolve_rewards(shat_tt, infos)  # compute rewards
        master_info['sampled_dones'] = dones
        master_info['sampled_rewards'] = rewards
        master_info['sampled_next_states'] = shat_tt
        master_info['sampled_reasons'] = infos['reason']

        # Compute true transition
        true_done, true_info = self._resolve_terminal_state(s_tt, true_info) # check reach goal or collision
        true_reward, true_info = self._resolve_rewards(s_tt, true_info)  # compute rewards
        self.done = true_done
        master_info['true_done'] = true_done
        master_info['true_reward'] = true_reward
        master_info['true_next_state'] = s_tt
        master_info['true_reason'] = true_info['reason']

        _done = self.done if true_done else dones
        return shat_tt, rewards, _done, master_info

    def reset(self,p_rand_state=0):
        self.steps = 0
        is_rand = (np.random.rand() < p_rand_state)
        self.state.reset(is_rand = is_rand)
        col_dones = self._check_collision(self.state._true_robot_state.reshape(1,-1))
        assert not np.any(col_dones), 'invalid reset state (starts in collision)'

    def _resolve_action(self, action):
        return self.state.step(action) # State transition

    def _resolve_terminal_state(self, states, info):
        n = states.shape[0]
        dones = np.zeros(n)

        _info = {}
        _info['reason'] = ['' for _ in range(n)]

        col_dones  = self._check_collision(states)
        goal_dones = self._check_goal(states)
        time_dones = (self.steps >= self.max_steps) *np.ones_like(dones)

        for i in np.where(time_dones == 1)[0]: _info['reason'][i] = 'max_steps'
        for i in np.where(goal_dones == 1)[0]: _info['reason'][i] = 'goal_reached'
        for i in np.where(col_dones  == 1)[0]: _info['reason'][i] = 'collision'
        dones += col_dones + goal_dones + time_dones
        dones = np.clip(dones, 0, 1)

        if n == 1:
            dones = dones[0]
            _info['reason'] = _info['reason'][0]
        info.update(_info)
        return dones, info

        # if self._check_collision(states):  # Check for collisions
        #     done = True
        #     info['reason'] = 'collision'
        # elif self._check_goal(states):  # Check if goal is reached
        #     done = True
        #     info['reason'] = 'goal_reached'
        # elif self.steps >= self.max_steps:  # Check if max steps reached
        #     done = True
        #     info['reason'] = 'max_steps'

        # done = False
        # if self._check_collision(state):  # Check for collisions
        #     done = True
        #     info['reason'] = 'collision'
        # elif self._check_goal(state):  # Check if goal is reached
        #     done = True
        #     info['reason'] = 'goal_reached'
        # elif self.steps >= self.max_steps:  # Check if max steps reached
        #     done = True
        #     info['reason'] = 'max_steps'
        # return done, info

    def _check_collision(self, states):
        n = states.shape[0]
        xy = np.array([states[:,self.state.ix], states[:,self.state.iy]], dtype=np.float32)
        dones = np.zeros(n)

        for obs in self.obstacles:
            if obs['type'] == 'circle':
                is_inside = np.linalg.norm(xy - obs['center'], axis=0) <= obs['radius'] + self.car_radius
                assert is_inside.shape == dones.shape
                dones += is_inside

                # if np.linalg.norm(xy - obs['center']) <= obs['radius'] + self.car_radius:
                #     done = True
                #     return done
            else:
                cx, cy = obs['center']
                w, h = obs['width'], obs['height']
                x, y = xy
                is_inside = (cx - w / 2 <= x) & (x <= cx + w / 2) & (cy - h / 2 <= y) & (y <= cy + h / 2)
                assert is_inside.shape == dones.shape
                dones += is_inside

                # if (cx - w / 2 <= x <= cx + w / 2) and (cy - h / 2 <= y <= cy + h / 2):
                #     done = True
                #     return done

        # done = False
        dones = np.clip(dones, 0, 1)
        return dones
        # return dones[0] if n == 1 else dones


    def _check_goal(self, states):
        dones = np.zeros(states.shape[0])
        xy = np.array([states[:, self.state.ix], states[:, self.state.iy]], dtype=np.float32)
        v = states[:, self.state.iv]

        dist_to_goal = np.linalg.norm(xy - np.array(self.goal)[:,np.newaxis],axis=0)
        is_inside = (dist_to_goal <= self.goal_radius) + (v < self.goal_velocity) == 2 # and condition
        assert is_inside.shape == dones.shape
        dones += is_inside

        dones = np.clip(dones, 0, 1)
        return dones

    def _resolve_rewards(self, state_samples, info):
        """
        Resolve the reward based on the previous and new state.
        """
        n = state_samples.shape[0]
        rewards = np.zeros(n)

        dGoal = state_samples[:,self.state.idGoal]
        rew_dist2goal = self.reward_dist2goal * (1 - (dGoal / self.max_dist))  # progress towards goal reward
        info['rew_dist2goal'] = rew_dist2goal
        assert rew_dist2goal.shape == rewards.shape

        rewards += rew_dist2goal  # progress towards goal reward
        rewards += self.reward_step  # time cost
        rewards += (info['reason'] == 'collision') * self.reward_collide
        rewards += (info['reason'] == 'goal_reached') * self.reward_goal

        return rewards, info

    @property
    def robot_probs(self):
        return self.state.robot_probs
    @property
    def observation_space(self):
        return self.state.observation_space
    @property
    def delay_steps(self):
        return self.state._delay_steps
    @property
    def n_samples(self):
        return self.state.n_samples
    @property
    def observation(self):
        return self.state.observation

def main_vec():
    from utils.joystick import VirtualJoystick
    import time

    layout =  'example2'

    env = ContinousNavEnv.from_layout(layout)

    #
    # env.step(np.array([0.5, 0.5]))
    # print(env.state.get_curr_lidar_state())
    # env.render()
    # plt.show(block=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    joystick = VirtualJoystick(ax=axs[0], deadzone=0.05, smoothing=0.35, spring=True)
    axs[-1].set_title("Virtual Joystick Input")

    env.reset()
    total_reward = 0.0
    rewards = []
    done = False
    print(f'Beginning Simulation...')
    while not done:
        tstart = time.time()
        x, y, r, th, active = joystick.get()
        action = np.array([x, y], dtype=np.float32)  # Use joystick input as action
        ns_samples, r_samples, done_samples, info = env.step(action)
        reward = info['true_reward']
        rewards.append(reward)
        if len(rewards) > 2:
            axs[-1].plot(rewards)
            axs[-1].relim()

        total_reward += reward
        env.render(draw_dist2goal=True, draw_lidar=True, ax=axs[1],pause=0.001)
        # plt.show(block=False)
        while time.time() - tstart < env.dt:
            pass

        if info['true_done']:
            obs = env.reset()








if __name__ == "__main__":
   main_vec()