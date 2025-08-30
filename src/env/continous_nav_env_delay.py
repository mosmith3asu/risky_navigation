import numpy as np
import matplotlib.pyplot as plt
from src.utils.visibility_graph import VisibilityGraph
from src.env.robots import Fetch_Lidar,FetchBase,Fetch_DistanceHeading

from src.env.layouts import read_layout_dict
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv

import warnings


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



        # Action parameters ------------------------------------------------------------------------------------------
        self.action_bounds = {}
        self.action_bounds['jx'] = [-1, 1]  # normalized joystick commands (steering)
        self.action_bounds['jy'] = [-1, 1]  # normalized joystick commands (velocity)

        low_act = np.array([bound[0] for bound in self.action_bounds.values()], dtype=np.float32)
        high_act = np.array([bound[1] for bound in self.action_bounds.values()], dtype=np.float32)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

        self.steps = 0
        self.done = False  # flag to indicate if the episode is done

        robot_dynamics = kwargs.get('robot_dynamics', {})

        # self.robot = State_DistanceHeading(start_pos, start_velocity, start_heading, self, base_state_validator=self.check_collision)
        self.robot = Fetch_Lidar(start_pos, start_velocity, start_heading, self,
                                     base_state_validator=self.check_collision, **robot_dynamics)

    def reset(self,*args,**kwargs):
        self.steps = 0
        self.done = False  # flag to indicate if the episode is done

        is_random_state = kwargs.get('p_rand_state', 0) > np.random.rand()
        state = self.robot.reset(random_start=is_random_state)
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
            dGoal = new_state[self.robot.idGoal]
        except:
            warnings.warn("State does not have dGoal feature. Computing manually.")
            x, y, θ = new_state[self.robot.ix], new_state[self.robot.iy], new_state[self.robot.iθ]
            dGoal, _ = self.robot.get_dist2goal_sphere(x, y, θ)

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
        new_state = self.robot.update_state(x, y, v, θ)  # Update the navigation state
        return new_state

    def check_collision(self, state):
        xy = np.array([state[self.robot.ix], state[self.robot.iy]], dtype=np.float32)

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
        xy = np.array([state[self.robot.ix], state[self.robot.iy]], dtype=np.float32)
        # xy = np.array([state[self.sidx['x']], state[self.sidx['y']]], dtype=np.float32)
        v = state[self.robot.iv]

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
            self.robot.draw_features(self.ax)
            # self.draw_dist_lines(self.ax)

        plt.draw()
        plt.pause(0.0001)

    def draw_robot(self,ax, image_path = "src/env/assets/fetch_robot.png"):
        """insert image of robot from ./assets/fetch_robot.png"""
        import matplotlib.image as mpimg
        from matplotlib import transforms as mtransforms
        import os
        project_root = os.getcwd().split('src')[0]
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
                dGoal, δGoal = self.robot.get_dist2goal_sphere(x, y, 0)  # distances to goal and obstacles
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
            normalize(dstate[self.robot.ix], self.robot._state_bounds['x']),
            normalize(dstate[self.robot.iy], self.robot._state_bounds['y']),
            normalize(dstate[self.robot.iv], self.robot._state_bounds['v']),
            normalize(dstate[self.robot.iθ], self.robot._state_bounds['θ'])

        ])
        return smoothness ** exp

    #############################################
    # Properties and accessing states
    @property
    def rem_steps(self):
        return self.max_steps - self.steps

    @property
    def x(self): return self.robot.x

    @property
    def y(self): return self.robot.y

    @property
    def v(self): return self.robot.v

    @property
    def θ(self): return self.robot.θ

    @property
    def state(self):
        return self.robot.observation

    @property
    def observation_space(self):
        return self.robot.observation_space

    @property
    def robot_state(self):
        return self.robot.base_state

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

        dGoal = self.robot.dGoal
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


def main():
    from utils.joystick import VirtualJoystick
    import time

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    joystick = VirtualJoystick(ax=axs[0], deadzone=0.05, smoothing=0.35, spring=True)

    # layout_dict = read_layout_dict('example0')
    # layout_dict = read_layout_dict('example1')
    layout_dict = read_layout_dict('example2')

    layout_dict['vgraph_resolution'] = (25, 25)  # resolution for visibility graph
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