import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import transforms as mtransforms
import matplotlib.image as mpimg
from gymnasium import spaces
from abc import ABC, abstractmethod
from collections import deque
import math
import warnings

from src.utils.visibility_graph import VisibilityGraph
from src.env.layouts import read_layout_dict
from env.mdp import Compiled_LidarFun, Belief_FetchRobotMDP_Compiled
from utils.file_management import get_project_root


#################################################################################
# GLOBALS #######################################################################
#################################################################################
## BEST---------------
# REWARD_GOAL = 100
# REWARD_COLLIDE = -500
# # REWARD_STEP =  -0.1
# REWARD_DIST2GOAL_EXP = 1.0
# REWARD_DIST2GOAL_PROG = False
# REWARD_DIST2GOAL = 0.1
# REWARD_STOPPING = 0.001
# REWARD_STEP =  -(REWARD_DIST2GOAL + REWARD_STOPPING)
## BEST---------------

# --- (top)
# REWARD_GOAL = 100
# REWARD_COLLIDE = -500
# REWARD_DIST2GOAL_EXP = 1.0
# REWARD_DIST2GOAL_PROG = False
# REWARD_DIST2GOAL = 0.1
# REWARD_STOPPING = 0.001
# REWARD_STEP =  -(REWARD_DIST2GOAL + REWARD_STOPPING)

# 15:33:15 (Bottom)
REWARD_GOAL = 100
REWARD_COLLIDE = -500
REWARD_DIST2GOAL_EXP = 1.0
REWARD_DIST2GOAL_PROG = True
REWARD_DIST2GOAL = 0.1
REWARD_STOPPING = 0.001
REWARD_STEP =  -(REWARD_DIST2GOAL + REWARD_STOPPING)

#################################################################################
# MDP and State Classes #########################################################
#################################################################################

class Delayed_LidarState:
    def __init__(self, X0, f_dist2goal, f_lidar, delay_steps, dt, n_rays=12,
                 bounds = ((-np.inf,-np.inf),(np.inf,np.inf)),
                 robot_state_validator = None,
                 n_samples = 50,
                 **kwargs):


        # -------------------------------------------------------------------------------------------
        # Unpack ------------------------------------------------------------------------------------
        self._X0 = np.array(X0, dtype=np.float32)   # initial roboto position
        self.f_dist2goal = f_dist2goal              # vectorized function to compute distance and heading to goal
        self.f_lidar = f_lidar                      # vectorized function to compute lidar features
        self._delay_steps = delay_steps             # number of delay steps to simulate
        self.robot_state_validator = robot_state_validator # function to validate robot state

        if 'dynamics_belief' in kwargs.keys() is None:
            warnings.warn("dynamics_belief not provided to Delayed_LidarState. Using default belief.")
        dynamics_belief = kwargs.pop('dynamics_belief', {
            'b_min_lin_vel': (0.0, 1e-6),
            'b_max_lin_vel': (1.0, 0.5),
            'b_max_lin_acc': (0.5, 0.2),
            'b_max_rot_vel': (math.pi / 2.0, math.pi / 6.0)
        })


        # self.robot = make_belief_fetchrobot_mdp(**dynamics_belief) # robot dynamics model with uncertainty in parameters


        small_sigmas = [val[0] < 1 / np.sqrt(2 * np.pi) for val in dynamics_belief.values()]
        if np.all(small_sigmas):
            warnings.warn("All dynamics belief stddevs are very small (< 1/sqrt(2pi)). "
                          "Assuming no variation in dynamics for numerical stability")

        self.robot = Belief_FetchRobotMDP_Compiled(
            n_samples       = n_samples,
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

        self._rand_state_buffer = deque(maxlen=1000)

        self._obs_space_dim =   self.robot_state_dim + \
                                self.action_state_dim + \
                                self.dist_state_dim + \
                                self.lidar_state_dim


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
    def step(self,action, return_state_inference=False):
        assert self.is_full, "Delay buffer not full before step()"
        assert action.shape == (2,), f"action shape {action.shape} does not match expected {(2,)}"

        self._cached_observation = None  # clear cached observation
        info = {}

        # Step belief samples (Xhat_t) under new dynamics samples
        self.robot.resample_dynamics()                                  # resample robot dynamics
        Xdelay_t, ahist_t = self.delayed_robot_state, np.copy(self.action_history)   # get delayed state and action history
        Xinference = self.infer_robot_states(Xdelay_t, ahist_t)  # Recover belief about current robot state (Xhat_t)
        info['X_inference'] = Xinference.copy()


        # Compute next observation samples ---------------------
        ahist_tt = np.vstack([ahist_t[1, :], action])
        Xdelay_tt = Xinference[0] # Belief sample of next observation in buffer
        Xhat_t = Xinference[-1]  # Recover belief about current robot state (Xhat_t)
        Xhat_tt = self.robot.step(Xhat_t, action, self.dt)  # Take step under dynamics samples

        Ldelay_tt = self.f_lidar(Xdelay_tt)

        # Compute reward states -----------------------
        dummy_actions = np.ones_like(ahist_t) * np.nan  # dummy actions for inference
        dummy_lidar = np.ones_like(Ldelay_tt) * np.nan  # dummy actions for inference


        # Batch function calls for vectorized computation
        if return_state_inference:
            batched_X = [Xdelay_tt, Xhat_t, Xhat_tt]
            _dG, _δG = self.f_dist2goal(np.vstack(batched_X))
            dGdelay_tt, dGhat_t, dGhat_tt = np.split(_dG, len(batched_X))
            δGdelay_tt, δGhat_t, δGhat_tt = np.split(_δG, len(batched_X))

            # (Reward state) current inferred state
            info['shat_t'] = self.compose_state(X=Xhat_t,  # compose new state (ensures correct idxs)
                                                ahist=dummy_actions.flatten(),
                                                goal_dist=dGhat_t,
                                                goal_heading=δGhat_t,
                                                lidar_dists=dummy_lidar
                                            )
        else:
            batched_X = [Xdelay_tt, Xhat_tt]
            _dG, _δG = self.f_dist2goal(np.vstack(batched_X))
            dGdelay_tt, dGhat_tt = np.split(_dG, len(batched_X))
            δGdelay_tt, δGhat_tt = np.split(_δG, len(batched_X))

            info['shat_t'] = None



        # (MDP state) Next delayed observation
        sdelay_tt = self.compose_state(X=Xdelay_tt,  # compose new state (ensures correct idxs)
                                       ahist=ahist_tt.flatten(),
                                       goal_dist=dGdelay_tt,
                                       goal_heading=δGdelay_tt,
                                       lidar_dists=Ldelay_tt
                                       )
        # (Reward state) next inferred state
        info['shat_tt'] = self.compose_state(X= Xhat_tt,                        # compose new state (ensures correct idxs)
                                     ahist = dummy_actions.flatten(),
                                     goal_dist= dGhat_tt,
                                     goal_heading = δGhat_tt,
                                     lidar_dists = dummy_lidar #Lhat_tt
                                     )


        # Step true state (X_t)
        X_t = self.get_curr_robot_state()                               # resample robot dynamics
        X_tt = self.robot.step_true(X_t, action, self.dt)               # get delayed state and action history
        dG_tt, δG_tt = self.f_dist2goal(X_tt)                           # Recover belief about current robot state (Xhat_t)
        L_tt = self.f_lidar(X_tt)

        # (True state) actual state of robot
        info['s_tt'] = self.compose_state(X=X_tt,                               # Take step under dynamics samples
                                  ahist=dummy_actions.flatten(),
                                  goal_dist=dG_tt,
                                  goal_heading=δG_tt,
                                  lidar_dists= L_tt # not really used but have anyway
                                  )

        assert info['s_tt'].shape[1] == self._obs_space_dim, 'Incorrect state dim'

        self._true_robot_state = X_tt.copy().reshape(1,self.robot_state_dim)                 # update true robot state
        self._true_dist2goal_state = np.hstack([dG_tt,δG_tt]).reshape(1,self.dist_state_dim) # update true distance to goal state
        self._true_lidar_state = L_tt.reshape(1,self.lidar_state_dim)                        # update true lidar state
        self.put_buffer(X_t, action)  # put new true state and action in buffer


        return sdelay_tt, info

    def reset(self, is_rand=False, rand_buffer=False, rand_action_gen=None):
        self._cached_observation = None
        self._true_robot_state = None

        if is_rand:
            # get previous terminal state from buffer
            if rand_buffer and len(self._rand_state_buffer) > 0:
                robot_state, rand_actions = self._get_rand_buffer_state()

            # Get delayed state with random ou actions
            elif rand_action_gen is not None:
                robot_state, rand_actions = self._get_rand_action_state(rand_action_gen)

            else: # sample completely new random state with wait actions
                robot_state = self._get_rand_wait_state()
                rand_actions = (0,0)

        # Start in starting state with buffer full of wait actions
        else:
            robot_state = self._X0.copy()  # true robot state (not delayed)
            rand_actions = (0, 0)

        robot_state = robot_state.reshape(1,4)
        self.clear_buffer()
        self.fill_buffer(action=rand_actions, robot_state=robot_state)
        self._true_robot_state = robot_state
        assert self.is_full, "Delay buffer not full after reset()"
        assert self._true_robot_state is not None, "True robot state not set after reset()"

    def _get_rand_buffer_state(self):
        raise NotImplementedError("Random buffer state retrieval is deprecated.")
        # i = np.random.randint(0, len(self._rand_state_buffer))
        # data = self._rand_state_buffer[i]
        # observation = data['observation']
        # del self._rand_state_buffer[i]
        #
        # self._true_robot_state = None
        # robot_state = observation[:, :4]
        # actions = [(observation[:, getattr(self, f'ijx{n}')],
        #             observation[:, getattr(self, f'ijy{n}')])
        #            for n in range(self._delay_steps)
        #            ]
        # return robot_state, actions

    def _get_rand_action_state(self, rand_action_gen):
        rand_actions = []
        for _ in range(self._delay_steps):
            a = rand_action_gen(advance = False)
            if isinstance(a, torch.Tensor):
                a = a.cpu().numpy()
            rand_actions.append(a)

        robot_state = self._get_rand_wait_state()
        return robot_state, rand_actions

        # self.clear_buffer()
        # true_state = self.fill_buffer(action=rand_actions, robot_state=robot_state)
        # self._true_robot_state = true_state

    def _get_rand_wait_state(self, heading_noise= np.pi / 8):
        # sample new random state
        # x,y,v,θ = self._true_robot_state

        attempt = 0
        for _ in range(50):
            attempt += 1
            _x = np.random.uniform(self.bounds[0][0], self.bounds[1][0])
            _y = np.random.uniform(self.bounds[0][1], self.bounds[1][1])
            _Xt = np.array([_x, _y, 0, 0], dtype=np.float32)[np.newaxis, :]

            if self.robot_state_validator is None or \
                    not self.robot_state_validator(_Xt):
                x, y = _x, _y
                break

        if attempt > 49:
            warnings.warn("Failed to find a valid random X Y after 50 attempts. Using last standard.")
            x, y, v, θ = self._X0.copy()
        else:

            v = np.random.uniform(0, self.robot.true_max_lin_vel)  # random initial velocity
            _, δGoal = self.f_dist2goal(np.array([x, y, 0, 0], dtype=np.float32)[np.newaxis, :])
            θ = δGoal[0] + np.random.uniform(-heading_noise, heading_noise)

        return np.array([x, y, v, θ], dtype=np.float32).reshape(1, -1)

    ###################################################################################################
    # State/delay handler methods #####################################################################
    def decompose_state(self,S, as_dict=False):
        X = S[0:4]; i =4
        ahist = S[i:i+self.action_state_dim];i += self.action_state_dim
        goal_dist,goal_heading = S[i:i+self.dist_state_dim];i += self.dist_state_dim
        lidar_dists = S[i:i + self.lidar_state_dim];i += self.lidar_state_dim

        if as_dict:
            return {
                'X': X,
                'robot_state': X,
                'ahist': ahist,
                'dist2goal': np.array([goal_dist, goal_heading]),
                'goal_dist': goal_dist,
                'gaol_heading': goal_heading,
                'lidar_dists': lidar_dists
            }

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


        action = np.array(action, dtype=np.float32).reshape(-1,2)
        n_action = action.shape[0]

        if n_action ==1: # single action given (e.g., wait)
            for _ in range(self._delay_steps +1):
                self.put_buffer(robot_state, action[0])

        elif n_action == self._delay_steps: # series of actions given
            Xt = robot_state.copy().reshape(1,4)
            for a in action:
                # print(f'Xt:{Xt} \t A:{a}')
                self.put_buffer(Xt, a)
                Xt = self.robot.step_true(Xt, a, self.dt).reshape(1,4)
            return Xt # returns true robot state
        else:
            raise ValueError(f"action shape {action.shape} not compatible with delay steps {self._delay_steps}")

        assert self.is_full, "Delay buffer not full after fill_buffer()"
        return None

    def put_buffer(self,robot_state,action):
        assert self.is_valid_robot_state(robot_state,stype='single'), f"robot_state shape {robot_state.shape} does not match expected (nx4)"
        assert action.shape == (2,), f"action shape {action.shape} does not match expected {(2,)}"
        self._robot_state_buffer.append(robot_state)
        self._action_buffer.append(action)


    def add_random_robot_state(self,observation):
        # assert true_state.shape[0] == 1, f"Only single state (1xn) can be added to random buffer. {true_state.shape[0]} was given"
        assert observation.shape[0] == 1, f"Only single observation (1xn) can be added to random buffer. {observation.shape[0]} was given"

        data = {
            # 'true_state': true_state.copy(),
            'observation': observation.copy()
                }
        self._rand_state_buffer.append(data)


    def infer_robot_states(self,robot_state,actions):
        assert self.is_valid_robot_state(robot_state,stype='single'), \
            f"robot_state shape {robot_state.shape} does not match expected (n,4)"
        Xt = np.repeat(robot_state, self.n_samples, axis=0)
        Xhist = []

        for action in actions:
            # Xt = self.robot.step(Xt, action, self.dt)
            # Xt = self.robot.step_noisy(Xt, action, self.dt)
            Xt = self.robot.step_biased(Xt, action, self.dt)
            Xhist.append(Xt.copy())
        return Xhist

    def infer_curr_robot_state(self,robot_state,actions):
        assert self.is_valid_robot_state(robot_state,stype='single'), \
            f"robot_state shape {robot_state.shape} does not match expected (n,4)"
        Xt = np.repeat(robot_state, self.n_samples, axis=0)
        action_bias = np.random.normal(0,self.action_noise_std,[self.n_samples,2])

        for action in actions:
            # Xt = self.robot.step(Xt, action, self.dt)
            # Xt = self.robot.step_noisy(Xt, action, self.dt)
            Xt = self.robot.step_biased(Xt, action, self.dt, action_bias)
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
        return self._robot_state_buffer[0].copy() if self._delay_steps >0 else self._true_robot_state.copy()
    @property
    def action_history(self):
        return np.array(self._action_buffer) if self._delay_steps >0 else np.array([])

    def get_curr_robot_state(self):
        return self._true_robot_state.copy().reshape(1, -1)
    def get_curr_lidar_state(self):
        return self._true_lidar_state.copy().reshape(1, -1)
    def get_curr_dist2goal_state(self):
        return self._true_dist2goal_state.copy().reshape(1, -1)


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
    @classmethod
    def from_layout(cls, layout_name, *args, **kwargs):

        layout_dict = read_layout_dict(layout_name)
        kwargs.update(layout_dict)
        kwargs['layout'] = layout_name
        return cls(*args, **kwargs)

    def __init__(self,max_steps,dt, **kwargs):
        self.dt = dt
        self.max_steps = max_steps

        self.fig = None
        self.ax = None
        self.bounds = None
        self.goal = None
        self.obstacles = None
        self.state = None
        self.layout = kwargs.get('layout','unknown')


        self.goal_velocity      = kwargs.get('goal_velocity' , 0.2)  # max goal velocity   (m/s)  , if any
        self.goal_radius        = kwargs.get('goal_radius'   , 0.5)  # meters
        self.car_radius         = kwargs.get('car_radius'    , 0.65)  # meters



        # self.reward_goal = kwargs.get('reward_goal', 100.0)
        # self.reward_collide = kwargs.get('reward_collide', -200.0)  # penalty for collision
        # self.reward_step = kwargs.get('reward_step', 0.00)  # penalty for being slow
        #
        # self.rshape = 1
        # self.reward_dist2goal_exponent = 1.0  # exponent for distance to goal reward shaping
        # # self.reward_dist2goal = kwargs.get('reward_dist', 0.005)  # maximum possible reward being close to goal
        # self.reward_dist2goal = kwargs.get('reward_dist', 0.01)  # maximum possible reward being close to goal
        # self.reward_stopping = kwargs.get('reward_stopping',  0.00)  # maximum possible reward being close to stopping to goal

        self.reward_goal = REWARD_GOAL
        self.reward_collide = REWARD_COLLIDE  # penalty for collision
        self.reward_step = REWARD_STEP  # penalty for being slow

        self.rshape = 1
        self.reward_dist2goal_prog = REWARD_DIST2GOAL_PROG
        self.reward_dist2goal_exponent = REWARD_DIST2GOAL_EXP # exponent for distance to goal reward shaping
        self.reward_dist2goal = REWARD_DIST2GOAL  # maximum possible reward being close to goal
        self.reward_stopping = REWARD_STOPPING  # maximum possible reward being close to stopping to goal

        self.reward_max_step = self.reward_goal + self.reward_dist2goal + self.reward_stopping +self.reward_step
        self.reward_min_step = self.reward_collide + self.reward_step

        self.reward_max_run = self.reward_goal + 0 if (self.reward_dist2goal + self.reward_stopping  <= self.reward_step)\
            else (self.reward_dist2goal + self.reward_stopping +self.reward_step)* self.max_steps
        self.reward_min_run = self.reward_collide + (self.reward_step) * self.max_steps
        # Reward Consistency Checks
        gamma = 0.99
        inf_sum = 1 / (1 - gamma)
        exp_cum_reward_dist2goal    = self.reward_dist2goal*inf_sum
        exp_cum_reward_step         = self.reward_step *inf_sum

        cum_reward_dist2goal = self.max_steps*self.reward_dist2goal
        cum_reward_step =  self.max_steps*self.reward_step



        assert self.reward_collide < cum_reward_step, f'REWARD VIOLATION: Crashing < Slow'
        assert self.reward_collide < exp_cum_reward_step, f'REWARD VIOLATION: Crashing < Slow'
        if not self.reward_dist2goal_prog:
            assert self.reward_goal > cum_reward_dist2goal + cum_reward_step, f'REWARD VIOLATION: Goal Always Better than Close:'
            assert self.reward_goal > exp_cum_reward_dist2goal + exp_cum_reward_step, f'REWARD VIOLATION: Goal Always Better than Close:'

        # assert self.reward_collide
        #
        # buff = 1
        # # assert abs(exp_cum_reward_step) < abs(exp_cum_reward_dist2goal), f'REWARD VIOLATION: Avoiding Slow < Getting Closer: '
        # assert buff * abs(exp_cum_reward_step)< abs(self.reward_collide), f'REWARD VIOLATION: Crashing < Slow'
        # assert abs(exp_cum_reward_step) < abs(self.reward_collide), f'REWARD VIOLATION: Crashing < Waiting'
        # assert abs(exp_cum_reward_dist2goal) < abs(self.reward_collide), f'REWARD VIOLATION: Crashing < Close:'
        # # assert abs(buff * exp_cum_reward_dist2goal + exp_cum_reward_step) < abs(self.reward_goal), f'REWARD VIOLATION: Goal Always Better than Close:'
        #
        # assert abs(cum_reward_dist2goal + cum_reward_step) < abs(self.reward_goal), f'REWARD VIOLATION: Goal Always Better than Close:'
        # assert abs(cum_reward_step) < abs(self.reward_collide), f'REWARD VIOLATION: Crashing < Waiting'
        # assert abs(cum_reward_dist2goal) < abs(self.reward_collide), f'REWARD VIOLATION: Crashing < Close:'

    @abstractmethod
    def step(self, *args, **kwargs):
        pass
    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

    def render(self, ax=None,
               draw_robot=True, draw_robot_asimg=True,
               draw_obstacles=True, draw_collision_box=True,
               draw_goal=True, draw_dist2goal=True,
               draw_lidar=True, pause=None,
               draw_delayed=False):
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
        if draw_collision_box: self.draw_collision_box(self.ax)
        if draw_obstacles: self.draw_obstacles(self.ax)
        if draw_goal     : self.draw_goal(self.ax)
        if draw_robot    : self.draw_robot(self.ax, draw_delayed= draw_delayed,draw_robot_asimg=draw_robot_asimg)
        if draw_dist2goal: self.draw_dist2goal(self.ax, draw_delayed= draw_delayed)
        if draw_lidar    : self.draw_lidar(self.ax, draw_delayed= draw_delayed)

        plt.draw()
        if pause is not None:
            plt.pause(pause)


    def draw_collision_box(self,ax,  alpha=0.05, c = 'r'):
        import shapely
        import shapely.geometry as sg
        import shapely.ops as so
        import matplotlib.pyplot as plt

        polys = []
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                # circle = plt.Circle(tuple(obs['center']), obs['radius']+ self.car_radius, color=c, alpha=alpha)
                center_point = sg.Point(*obs['center'])
                circular_polygon = center_point.buffer( obs['radius']+ self.car_radius)
                polys.append(circular_polygon)

            else:
                cx, cy = obs['center']
                w, h = obs['width'], obs['height']
                rect = sg.box(  cx - w / 2 - self.car_radius,
                                cy - h / 2 - self.car_radius,
                                cx + w / 2 + self.car_radius,
                                cy + h / 2 + self.car_radius)
                polys.append(rect)


        # Plot singular union
        for i in range(len(polys)):
            _poly = polys[i]
            for j in range(i+1, len(polys)):
                _poly = _poly.difference(polys[j])
            xs, ys = _poly.exterior.xy
            ax.fill(xs, ys, alpha=alpha, fc=c, ec='none')

    def draw_robot(self,ax, draw_robot_asimg=True,
                   draw_delayed=False, alpha=1.0,
                   image_path = "src/env/assets/fetch_robot.png"):
        """insert image of robot from ./assets/fetch_robot.png"""
        project_root = get_project_root()

        image_path = os.path.join(project_root, image_path)

        image_height = self.car_radius
        arrow_len = 1.3 * self.car_radius
        arrow_width = 0.1 * self.car_radius
        arrow_color = 'b'


        x, y, v, θ = self.state.get_curr_robot_state().flatten() if not draw_delayed \
            else self.state.delayed_robot_state.flatten()



        if not draw_robot_asimg:
            # draw circle with radius self.car_radius
            circle = plt.Circle((x, y), self.car_radius, color='b', alpha=alpha)
            ax.add_patch(circle)
            # ax.scatter(x, y, color='k', alpha=alpha,zorder=99)
            return

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
            alpha=alpha
        )



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

    def draw_dist2goal(self, ax, draw_delayed=False):
        x, y, v, θ = self.state.get_curr_robot_state().flatten() if not draw_delayed \
            else self.state.delayed_robot_state.flatten()
        dGoal, δGoal = self.state.get_curr_dist2goal_state().flatten() if not draw_delayed \
            else np.array(self.state.f_dist2goal(self.state.delayed_robot_state)).flatten()

        # Line to goal
        gx = x + dGoal * np.cos(δGoal + θ)
        gy = y + dGoal * np.sin(δGoal + θ)
        ax.plot([x, gx], [y, gy], 'g--', label='to goal')

    def draw_lidar(self,ax, draw_delayed=False):
        x, y, v, θ = self.state.get_curr_robot_state().flatten() if not draw_delayed \
            else self.state.delayed_robot_state.flatten()
        L = self.state.get_curr_lidar_state().flatten() if not draw_delayed \
            else np.array(self.state.f_lidar(self.state.delayed_robot_state)).flatten()

        for i, δbeam in enumerate(self.state.lidar_angles):
            ray_angle = θ + δbeam
            dlidar = L[i]# eval(f'self.state.lidar{i}')
            dx = dlidar * np.cos(ray_angle)
            dy = dlidar * np.sin(ray_angle)

            ax.plot([x, x + dx],
                    [y, y + dy],
                    color='r', linestyle='--',
                    label=f'Dist to lidar{i}')

    def render_reward_heatmap(self, ax=None, cmap="coolwarm",
                              r_dist2goal=True, r_collision=True, r_goal=True,
                              vmin=None, vmax=None, show_colorbar=True,
                              draw_obstacles=False, draw_goal=False, draw_robot=False,
                              block=True):

        # grid = self.vgraph.grid
        # H, W, _ = grid.shape
        resolution = (50,50)
        X = np.linspace(self.bounds[0][0], self.bounds[1][0], resolution[0])
        Y = np.linspace(self.bounds[0][1], self.bounds[1][1], resolution[1])

        # Compute Z by querying dist at each grid point
        Z = np.empty(resolution, dtype=float)



        for c, x in enumerate(X):
            for r, y in enumerate(Y):
                X = np.array([[x, y, 0.0, 0.0]], dtype=np.float32)
                dGoal, _ = self.state.f_dist2goal(X)
                # dGoal, δGoal = self.nav_state.get_dist2goal_sphere(x, y, 0)  # distances to goal and obstacles
                rew = 0
                if r_dist2goal:
                    rew_dist2goal = self.reward_dist2goal * (1 - (dGoal / self.max_dist))**self.reward_dist2goal_exponent  # progress towards goal reward
                    rew +=  rew_dist2goal
                if r_collision:
                    rew_collision = self._check_collision(X) * self.reward_collide
                    rew = rew_collision
                if r_goal:
                    rew_goal = self._check_goal(X) * self.reward_goal
                    rew += rew_goal

                Z[r, c] = float(rew)

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

        if draw_obstacles: self.draw_obstacles(ax)
        if draw_goal: self.draw_goal(ax)
        if draw_robot: self.draw_robot(ax)

        # Mark goal (if available)
        # if hasattr(self, "goal") and self.goal is not None:
        #     gx, gy = self.goal
        #     ax.plot(gx, gy, marker="*", markersize=12, linewidth=0, label="goal")
        #     ax.legend(loc="upper right")

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
                 vgraph_resolution = 10,#(30, 30),
                 verbose = True,
                 time_is_terminal = False,
                 goal_is_terminal = True,
                 **kwargs
                 ):
        super().__init__(max_steps, dt, **kwargs) # contains default vals that do not need to be changed

        start_pos = np.array(start_pos, dtype=np.float32)
        start_velocity = np.array(start_velocity, dtype=np.float32)
        start_heading = np.deg2rad(start_heading)  # convert heading to radians
        X0 = np.hstack([start_pos, start_velocity, start_heading])

        # Layout parameters ------------------------------------------------------------------------------------------
        self.layout    = kwargs.pop('layout')
        self.bounds    = bounds
        self.goal      = tuple(goal)
        self.obstacles = self.format_obstacles(obstacles)
        self.obstacles = self.create_boarder_obstacles(self.obstacles)  # add borders to the environment
        self.verbose = verbose

        # Action parameters ------------------------------------------------------------------------------------------
        self.action_bounds = {}
        self.action_bounds['jx'] = kwargs.pop('action_bounds_x',[-1, 1])  # normalized joystick commands (steering)
        self.action_bounds['jy'] = kwargs.pop('action_bounds_y',[-1, 1])  # normalized joystick commands (velocity)

        low_act = np.array([bound[0] for bound in self.action_bounds.values()], dtype=np.float32)
        high_act = np.array([bound[1] for bound in self.action_bounds.values()], dtype=np.float32)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

        # State parameters ------------------------------------------------------------------------------------------
        self.steps = 0
        self.done = False  # flag to indicate if the episode is done
        self.time_is_terminal = time_is_terminal # if True, reaching max_steps triggers done
        self.goal_is_terminal = goal_is_terminal # if True, reaching goal triggers done

        self.vgraph_resolution = vgraph_resolution
        vgraph = VisibilityGraph(self.goal, self.obstacles, self.bounds, resolution=vgraph_resolution, verbose=self.verbose)
        self.max_dist = vgraph.max_dist
        f_dist2goal = vgraph.get_compiled_funs()
        f_lidar = Compiled_LidarFun(self.obstacles, kwargs.get('n_rays', 12))
        self.state = Delayed_LidarState(X0, f_dist2goal, f_lidar, delay_steps, dt,
                                        bounds=self.bounds,
                                        robot_state_validator = self._check_collision,
                                        dynamics_belief = kwargs.pop('dynamics_belief', {}),
                                        n_samples = kwargs.pop('n_samples', 50),
                                        **kwargs)

        self.set_global_overrides(**kwargs) # set remaining uncaught params


    def set_global_overrides(self, **kwargs):
        # Parse remaining kwargs and set attributes
        obj_list = [self, self.state, self.state.robot]
        for key, val in kwargs.items():

            is_found = False
            for _obj in obj_list:
                if hasattr(_obj, key):
                    # if _obj == self.state.robot
                    setattr(_obj, key, val)
                    is_found = True
            if not is_found:
                raise ValueError(f"Unknown kwarg {key} in ContinousNavEnv.")

    def step(self, action, true_done = False):
        self.steps += 1
        master_info = {}

        true_info = {}
        infos = {}

        # s_tt,shat_tt = self._resolve_action(action) # State transition
        sdelay_tt, info = self._resolve_action(action, return_state_inference=self.reward_dist2goal_prog) # State transition
        shat_tt = info['shat_tt']
        shat_t = info['shat_t']
        s_tt = info['s_tt']
        master_info['X_inference'] = info['X_inference']

        # Compute sampled transition
        dones, infos = self._resolve_terminal_state(shat_tt, infos)  # check reach goal or collision
        rewards, infos = self._resolve_rewards(shat_tt, infos, shat_t)  # compute rewards

        master_info['sampled_dones'] = dones
        master_info['sampled_rewards'] = rewards
        master_info['sampled_next_states'] = shat_tt
        master_info['sampled_reasons'] = infos['reason']
        master_info['rew_dist2goal'] = infos['rew_dist2goal']

        # Compute true transition
        true_done, true_info = self._resolve_terminal_state(s_tt, true_info) # check reach goal or collision
        true_reward, true_info = self._resolve_rewards(s_tt, true_info)  # compute rewards
        self.done = true_done
        master_info['true_done'] = true_done
        master_info['true_reward'] = true_reward
        master_info['true_next_state'] = s_tt
        master_info['true_reason'] = true_info['reason']

        _done = self.done if true_done else dones

        return sdelay_tt, rewards, _done, master_info

    def reset(self,p_rand_state=0, **kwargs):
        info = {}
        self.steps = 0
        is_rand = (np.random.rand() < p_rand_state)
        for attempt in range(kwargs.get('rand_attempts',50)):
            self.state.reset(is_rand = is_rand, **kwargs)
            col_dones = self._check_collision(self.state._true_robot_state.reshape(1,-1))
            if not np.any(col_dones):
                break

        if np.any(col_dones):
            self.state.reset()
            col_dones = self._check_collision(self.state._true_robot_state.reshape(1, -1))

        assert not np.any(col_dones), 'invalid reset state (starts in collision)'
        info['is_rand'] = is_rand
        return info
    def _resolve_action(self, action, **kwargs):
        return self.state.step(action,**kwargs) # State transition

    def _resolve_terminal_state(self, states, info):
        assert states.ndim == 2, f"states should be 2D array, got shape {states.shape}"
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

        dones += col_dones
        dones += goal_dones if self.goal_is_terminal else 0
        dones += time_dones if self.time_is_terminal else 0
        dones = np.clip(dones, 0, 1)

        if n == 1:
            dones = dones[0]
            _info['reason'] = _info['reason'][0]
        info.update(_info)
        return dones, info

    def _check_collision(self, states):
        """Checks if the robot has collided with any obstacles"""
        assert states.ndim == 2, f"states should be 2D array, got shape {states.shape}"
        n = states.shape[0]
        xy = np.array([states[:,self.state.ix], states[:,self.state.iy]], dtype=np.float32)
        dones = np.zeros(n)

        for obs in self.obstacles:
            if obs['type'] == 'circle':
                is_inside = np.linalg.norm(xy - obs['center'].reshape(2,1), axis=0) <= obs['radius'] + self.car_radius
                assert is_inside.shape == dones.shape, f"{is_inside.shape} vs {dones.shape}"
                dones += is_inside
            else:
                cx, cy = obs['center']
                w, h = obs['width'], obs['height']
                x, y = xy
                # is_inside = (cx - w / 2 <= x) & (x <= cx + w / 2) & (cy - h / 2 <= y) & (y <= cy + h / 2)
                is_inside = (cx - w / 2 - self.car_radius <= x) &\
                            (x <= cx + w / 2 + self.car_radius) & \
                            (cy - h / 2 - self.car_radius <= y) & \
                            (y <= cy + h / 2 + self.car_radius)
                assert is_inside.shape == dones.shape,  f"{is_inside.shape} vs {dones.shape}"
                dones += is_inside

        dones = np.clip(dones, 0, 1)
        return dones

    def render(self,*args,ns_prospects=None,**kwargs):
        pause = kwargs.pop('pause', None)
        super().render(*args, **kwargs)

        if ns_prospects is not None:
            x, y = ns_prospects[:,0], ns_prospects[:,1]
            if ns_prospects.shape[1] ==3:
                p = ns_prospects[:,2]
                # make p range from [0.1, 1]
                p = (p - np.min(p)) / (np.max(p) - np.min(p) + 1e-6)
                alpha = np.clip(p,0.1,1.0)
            self.ax.scatter(x, y, color='orange', s=3, alpha= alpha)

        plt.draw()
        if pause is not None:
            plt.pause(pause)

    def _check_goal(self, states):
        """Checks if the robot has reached the goal"""
        assert states.ndim == 2, f"states should be 2D array, got shape {states.shape}"
        n_samples = states.shape[0]
        dones = np.zeros(n_samples)

        # Perform checks
        # dist_to_goal = np.linalg.norm(xy - goal_xy, axis=1).reshape(n_samples,1)
        # is_inside = np.array(dist_to_goal <= goal_rad, dtype=int)


        goal_v = np.array([self.goal_velocity]).reshape(1, 1)
        v = np.abs(states[:, self.state.iv].reshape(n_samples, 1))
        is_stopped = np.array(np.abs(v) <= goal_v,dtype=int)
        is_inside = self._check_inside_goal(states)
        assert is_inside.shape == is_stopped.shape

        # Combine checks and return
        dones += (is_inside + is_stopped == 2).flatten()
        dones = np.clip(dones, 0, 1)
        return dones


    def _check_inside_goal(self, states):
        n_samples = states.shape[0]

        # Get relevant vars and reshape
        goal_rad = np.array([self.goal_radius]).reshape(1, 1)
        goal_xy = np.array(self.goal, dtype=np.float32).reshape(1, 2)
        xy = states[:, :2]

        # Perform checks
        dist_to_goal = np.linalg.norm(xy - goal_xy, axis=1).reshape(n_samples, 1)
        is_inside = np.array(dist_to_goal <= goal_rad, dtype=int)
        return is_inside

    def _resolve_rewards(self, state_samples, info, prev_state_samples = None):
        """
        Resolve the reward based on the previous and new state.
        """
        n = state_samples.shape[0]
        rewards = np.zeros(n)

        # Distance to goal shaped reward
        dGoal = state_samples[:,self.state.idGoal]
        rew_scale = np.power(1 - (dGoal / self.max_dist), self.reward_dist2goal_exponent)
        rew_scale = np.clip(rew_scale, 0.0, 1.0)
        # assert np.all(rew_scale <= 1.0) and np.all(rew_scale >= 0.0), f"rew_scale out of bounds: {rew_scale}"
        rew_dist2goal = self.reward_dist2goal * rew_scale
        info['rew_dist2goal'] = rew_dist2goal

        if prev_state_samples is not None:
            dGoal_prev = prev_state_samples[:,self.state.idGoal]
            rew_scale = np.power(1 - (dGoal_prev / self.max_dist), self.reward_dist2goal_exponent)
            rew_dist2goal_prev = self.reward_dist2goal * rew_scale
            rew_dist2goal = np.power(rew_dist2goal - rew_dist2goal_prev,1)  # reward is the progress towards goal
            # rew_dist2goal = np.clip(rew_dist2goal, self.reward_dist2goal, self.reward_dist2goal)
            info['rew_dist2goal'] = rew_dist2goal

        # Stopping in goal shaped reward
        if self.reward_stopping > 0:
            is_inside = self._check_inside_goal(state_samples)
            goal_v = np.array([self.goal_velocity]).reshape(1, 1)
            v = state_samples[:, self.state.iv].reshape(n, 1)
            dv = np.max(np.hstack([np.abs(v) - goal_v, np.zeros_like(v)]), axis=1)
            max_vel = self.state.robot.true_max_lin_vel
            rew_stopping = is_inside.flatten() * self.reward_stopping * (1 - (dv / max_vel))**2
        else:
            rew_stopping = np.zeros(n)

        assert rew_stopping.shape == rewards.shape
        assert rew_dist2goal.shape == rewards.shape

        rewards += rew_dist2goal * self.rshape  # progress towards goal reward
        rewards += rew_stopping * self.rshape
        rewards += self.reward_step #* self.rshape  # time cost
        rewards += (np.array(info['reason']) == 'collision')      * self.reward_collide
        rewards += (np.array(info['reason']) == 'goal_reached')   * self.reward_goal

        # Sanity check
        # assert np.all(rewards <= self.reward_max_step), f"Rewards out of max bounds: {rewards[rewards >= self.reward_min_step]}"
        # assert np.all(rewards >= self.reward_min_step), f"Rewards out of min bounds: {rewards[rewards <= self.reward_min_step]}"
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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        # print(f"ContinousNavEnv initialized with layout: {self.layout}")

        s = ''
        s += f"\nContinousNavEnv:"
        s += f"\n\t| layout: {self.layout}"
        s += f"\n\t| goal: {self.goal}"
        s += f"\n\t| dt: {self.dt}"
        s += f"\n\t| max_steps: {self.max_steps}"
        s += f"\n\t| action_bounds: {self.action_bounds}"
        s += f"\n\t| delay_steps: {self.delay_steps}"
        s += f"\n\t| n_samples: {self.n_samples}"
        s += f"\n\t| vgraph_resolution: {self.vgraph_resolution}"


        s += "\n\nBelief_FetchRobotMDP_Compiled:"
        s += f"\n\t| n_samples: {self.state.robot.n_samples}"
        # s += f"\n\t| belief_mu: {self.state.robot.belief_mu}"
        # s += f"\n\t| belief_std: {self.state.robot.belief_std}"
        s += f"\n\t| belief:"
        s += f"\n\t|\t| min_lin_vel: {self.state.robot.true_min_lin_vel} + {self.state.robot.belief_std[0]}"
        s += f"\n\t|\t| max_lin_vel: {self.state.robot.true_max_lin_vel} + {self.state.robot.belief_std[1]}"
        s += f"\n\t|\t| max_lin_acc: {self.state.robot.true_max_lin_acc} + {self.state.robot.belief_std[2]}"
        s += f"\n\t|\t| max_rot_vel: {self.state.robot.true_max_rot_vel} + {self.state.robot.belief_std[3]}"
        s += f"\n"
        return s


def preview_action_samples(ax, env, action, block = True):
    ns_samples, r_samples, done_samples, info = env.step(action)
    p_samples = np.array(env.robot_probs)
    p_samples = np.array(p_samples).reshape(ns_samples.shape[0], 1)
    # ns_prospects = np.hstack([ns_samples[:,0:2], p_samples])  # for rendering
    ns_prospects = np.hstack([info['sampled_next_states'][:, 0:2], p_samples])  # for rendering
    env.render(ns_prospects=ns_prospects,
               draw_delayed=True,
               draw_robot=True,
               draw_robot_asimg=False,
               draw_dist2goal=False,
               draw_lidar=False,
               ax=ax,
               pause=0.001)

    x,y = info['true_next_state'][0,0:2]
    ax.scatter(x, y, color='red', s=5, marker='x', label='True Next State')

    # if block:
    #     plt.ioff()
    #     plt.show()


def main_preview():
    """Usefull for tuning dynamics"""
    layout = 'spath'
    vgraph_resolution = 50
    P_RAND_STATE = 0
    actions = [[0, 1.0],
               [-0.5, 1.0],
               [-1.0, 1.0]]
    delay_steps = 10
    V0 = 1.0

    dynamics_belief = {
        'b_min_lin_vel': (0.0, 1e-6),
        'b_max_lin_vel': (1.5, 0.5),
        'b_max_lin_acc': (0.5, 0.2),
        'b_max_rot_vel': (math.pi / 2.0, math.pi / 6.0)
        # 'b_min_lin_vel': (0.0, 1e-6),
        # 'b_max_lin_vel': (1.5, 1e-6),
        # 'b_max_lin_acc': (0.5, 1e-6),
        # 'b_max_rot_vel': (math.pi / 2.0, 1e-6)
    }
    env = ContinousNavEnv.from_layout(layout,
                                      dynamics_belief=dynamics_belief,
                                      delay_steps=delay_steps,
                                      vgraph_resolution=vgraph_resolution)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    env.state._X0[env.state.iv] = V0  # set to max vel initially

    for i, action in enumerate(actions):
        env.reset(p_rand_state=P_RAND_STATE)
        # env._true_robot_state[2] = 1.5

        # Fill buffer -----------------
        xy_hist = np.empty((0,2), dtype=np.float32)

        for _ in range(delay_steps + 1):
            action = np.array(action, dtype=np.float32)  # Use joystick input as action
            ns_samples, r_samples, done_samples, info = env.step(action)
            p_samples = np.array(env.robot_probs)
            p_samples = np.array(p_samples).reshape(ns_samples.shape[0], 1)
            # true_state_hist.append(env.state._true_robot_state[0:2].copy())

            xy_hist = np.vstack([xy_hist, env.state._true_robot_state[0,0:2].reshape(1,2)])


        x,y = xy_hist[:,0], xy_hist[:,1]
        preview_action_samples(ax[i], env, action)
        ax[i].scatter(x,y, color='red', s=5, marker='x', label='True Next State')
        ax[i].set_title(f'Action: {action}')
        print(f'_true_robot_state.shape = {env.state._true_robot_state}')


    plt.ioff()
    plt.show()


def main_simulate():
    from utils.joystick import VirtualJoystick
    import time

    layout =  'spath'
    vgraph_resolution = 50
    P_RAND_STATE = 1

    dynamics_belief =  {
        # 'b_min_lin_vel': (-0.1, 1e-6),
        # 'b_max_lin_vel': (1.0, 1e-6),
        # 'b_max_lin_acc': (0.5, 1e-6),
        # 'b_max_rot_vel': (np.pi / 4, 1e-6)
        'b_min_lin_vel': (0.0, 1e-6),
        'b_max_lin_vel': (1.5, 0.5),
        'b_max_lin_acc': (0.5, 0.2),
        'b_max_rot_vel': (math.pi / 2.0, math.pi / 6.0)
        # 'b_min_lin_vel': (-0.1, 1e-6),
        # 'b_max_lin_vel': (1.0, 0.25),
        # 'b_max_lin_acc': (0.5, 0.1),
        # 'b_max_rot_vel': (np.pi / 4, 0.1 * np.pi)
    }
    delay_steps = 10
    env = ContinousNavEnv.from_layout(layout,dynamics_belief=dynamics_belief,delay_steps=delay_steps,vgraph_resolution=vgraph_resolution)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    env.render_reward_heatmap(ax=axs[0], r_collision=False, r_dist2goal=True ,r_goal=False,draw_obstacles=True,show_colorbar=True, block=False)
    # env.render_reward_heatmap(ax=axs[1], r_collision=False, r_dist2goal=True ,r_goal=True,draw_obstacles=True,show_colorbar=True, block=False)
    # env.render_reward_heatmap(ax=axs[2], r_collision=True, r_dist2goal=True ,r_goal=True,draw_obstacles=False,show_colorbar=True, block=True)


    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    joystick = VirtualJoystick(ax=axs[0], deadzone=0.05, smoothing=0.35, spring=True)
    axs[-1].set_title("Virtual Joystick Input")

    env.reset(p_rand_state=P_RAND_STATE)


    total_reward = 0.0
    rewards = []
    done = False
    rew_line = None
    print(f'Beginning Simulation...')
    loop_durs = deque(maxlen=10)
    while not done:
        tstart = time.time()
        x, y, r, th, active = joystick.get()
        action = np.array([x, y], dtype=np.float32)  # Use joystick input as action
        ns_samples, r_samples, done_samples, info = env.step(action)
        p_samples =  np.array(env.robot_probs)
        p_samples = np.array(p_samples).reshape(ns_samples.shape[0],1)
        # ns_prospects = np.hstack([ns_samples[:,0:2], p_samples])  # for rendering
        ns_prospects = np.hstack([info['sampled_next_states'][:,0:2], p_samples])  # for rendering



        reward = info['true_reward']
        rewards.append(reward)
        if len(rewards) > 2:
            if rew_line is None:
                rew_line, = axs[-1].plot(rewards)
            else:
                rew_line.set_ydata(rewards)
                rew_line.set_xdata(np.arange(len(rewards)))
            axs[-1].relim()

        total_reward += reward

        env.render(ns_prospects= ns_prospects,
                   draw_delayed=True,
                   draw_robot=True,
                   draw_robot_asimg=False,
                   draw_dist2goal=False,
                   draw_lidar=False,
                   ax=axs[1],
                   pause=0.001)
        # plt.show(block=False)

        loop_dur =time.time() - tstart
        while loop_dur < env.dt:
            loop_dur =time.time() - tstart
        loop_durs.append(loop_dur)
        print(f'')
        print(reward)
        print(f'Loop dur: {np.mean(loop_durs).round(2)}, rew_dist2goal: {np.mean(info["rew_dist2goal"])}')
        if info['true_done']:
            obs = env.reset(p_rand_state=P_RAND_STATE)








if __name__ == "__main__":
   # main_simulate()
   main_preview()