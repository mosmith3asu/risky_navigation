# import gym
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from src.env.layouts import read_layout_dict

class ContinuousNavigationEnv(gym.Env):
    """
    A continuous navigation environment where an agent is an RC car that controls throttle and steering angle
    to move in a 2D plane towards a goal while avoiding circular and rectangular obstacles.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        goal=(5.0, 5.0),
        start  = (1.0, 1.0, 45),
        obstacles=None,
        bounds=((0.0, 0.0), (10.0, 10.0)),
        max_steps=1000,
        goal_radius=0.5,
        max_throttle=2.0,
        max_steering=np.pi / 3,
        car_radius=0.2,
        drift_coeff=0.1,  # Coefficient for drift effect, if applicable
        dt=0.1,
        goal_velocity=0.1  # Velocity threshold to consider goal reached
    ):
        super().__init__()


        self.goal = np.array(goal, dtype=np.float32)
        self.start = np.array(start, dtype=np.float32)
        self.start[2] = np.deg2rad(self.start[2])  # Convert heading from degrees to radians

        self.bounds = np.array(bounds, dtype=np.float32)
        self.max_steps = max_steps
        self.goal_radius = goal_radius
        self.max_throttle = max_throttle
        self.max_steering = max_steering
        self.car_radius = car_radius
        self.dt = dt
        self.drift_coeff = drift_coeff
        self.velocity = 0.0  # Initial velocity
        self.goal_velocity = goal_velocity  # Velocity threshold to consider goal reached
        self.graph = None  # Placeholder for visibility graph, if used

        # reward settings
        self.max_dist = np.linalg.norm(self.bounds[1] - self.bounds[0])
        self.dist_reward = 0.05 # maximum reward proportional to remaining dist to goal
        self.dist_pow = 1.0           # power to scale distance reward
        self.goal_reward = 20.0        # Reward for reaching the goal
        self.collision_reward = -50.0  # Reward for collision with obstacles
        self.timestep_reward = -10/max_steps # cost to encourage speed

        # Obstacles
        self.add_obstacles(obstacles)

        # Observation: {x, y, heading, velocity, x_dist2goal, y_dist2goal, x_dist2obstacle, y_dist2obstacle}
        heading_bounds = [-np.pi, np.pi]
        velocity_bounds = [0, self.max_throttle]
        dist2goal_bounds = [[0, 0],[self.max_dist, self.max_dist]]
        dist2obstacle_bounds = [[0, 0],[self.max_dist, self.max_dist]]
        low_obs = np.array([*self.bounds[0], heading_bounds[0], velocity_bounds[0], *dist2goal_bounds[0],*dist2obstacle_bounds[0]], dtype=np.float32)
        high_obs = np.array([*self.bounds[1], heading_bounds[1], velocity_bounds[1], *dist2goal_bounds[0],*dist2obstacle_bounds[0]], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Action: throttle, steering angle
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_steering/2], dtype=np.float32),
            high=np.array([self.max_throttle, self.max_steering/2], dtype=np.float32),
            dtype=np.float32
            #!!!!! IF MODIFYING, need to change scaling of tanh in Actor network
        )

        self.state = None  # [x, y, theta]
        self.steps = 0
        self.fig = None
        self.ax = None

        self.reset()


    def add_obstacles(self,obstacles):
        self.obstacles = []
        if obstacles:
            for o in obstacles:
                obs_type = o.get('type', 'circle')
                if obs_type == 'circle':
                    self.obstacles.append({
                        'type': 'circle',
                        'center': np.array(o['center'], dtype=np.float32),
                        'radius': float(o.get('radius', 0.5))
                    })
                elif obs_type == 'rect':
                    self.obstacles.append({
                        'type': 'rect',
                        'center': np.array(o['center'], dtype=np.float32),
                        'width': float(o['width']),
                        'height': float(o['height'])
                    })
                else:
                    raise ValueError(f"Unsupported obstacle type: {obs_type}")

        border_sz = 0.5
        self.border_sz = border_sz
        boundH = self.bounds[1][1] - self.bounds[0][1]
        boundW = self.bounds[1][0] - self.bounds[0][0]
        # Left border
        self.obstacles.append({'type': 'rect',
                               'center': np.array([self.bounds[0][0] - border_sz / 2, self.bounds[1][1] / 2],
                                                  dtype=np.float32),
                               'width': border_sz,
                               'height': boundH + 2 * border_sz
                               })
        # Right border
        self.obstacles.append({'type': 'rect',
                               'center': np.array([self.bounds[1][0] + border_sz / 2, self.bounds[1][1] / 2],
                                                  dtype=np.float32),
                               'width': border_sz,
                               'height': boundH + 2 * border_sz
                               })
        # Bottom border
        self.obstacles.append({'type': 'rect',
                               'center': np.array([self.bounds[1][0] / 2, self.bounds[0][1] - border_sz / 2],
                                                  dtype=np.float32),
                               'width': boundW + 2 * border_sz,
                               'height': border_sz
                               })
        # Top border
        self.obstacles.append({'type': 'rect',
                               'center': np.array([self.bounds[1][0] / 2, self.bounds[1][1] + border_sz / 2],
                                                  dtype=np.float32),
                               'width': boundW + 2 * border_sz,
                               'height': border_sz
                               })

    def reset(self,random_state = False,heading_noise = None):
        # pos = np.random.uniform(self.bounds[0], self.bounds[1])
        # heading = np.random.uniform(-np.pi, np.pi)
        # self.state = np.array([*pos, heading], dtype=np.float32)
        if random_state:
            pos = np.random.uniform(self.bounds[0], self.bounds[1])
            heading = np.random.uniform(-np.pi, np.pi)
            self.state = np.array([*pos, heading], dtype=np.float32)
        else:
            self.state = self.start.copy()
            if heading_noise is not None:
                self.state[2]+= np.random.uniform(-heading_noise/2, heading_noise/2)

        self.steps = 0
        self.state = self.augment_state(self.state)
        return self.state

    def step(self, action):
        throttle, steering = np.clip(action, self.action_space.low, self.action_space.high)
        reward = 0.0
        done = False
        info = {}

        x, y, theta = self.state[:3]
        v_step = min(abs(self.velocity - throttle), self.drift_coeff)
        v_step *= -1 if throttle < self.velocity else 1
        self.velocity += v_step

        theta += steering * self.dt
        dx = self.velocity * np.cos(theta) * self.dt
        dy = self.velocity * np.sin(theta) * self.dt
        x += dx
        y += dy

        x = np.clip(x, self.bounds[0][0], self.bounds[1][0])
        y = np.clip(y, self.bounds[0][1], self.bounds[1][1])
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  # keep in [-pi, pi]
        self.state = np.array([x, y, theta], dtype=np.float32)
        self.state = self.augment_state(self.state)

        # Timestep reward to encourage speed
        reward += self.timestep_reward

        # Check for collisions
        did_collide, new_info = self.check_collision()
        if did_collide:
            reward += self.collision_reward
            done = True
            # prevents colliding to avoid timecost & premotes staying alive longer
            reward += 1.2*self.timestep_reward * self.rem_steps
        info.update(new_info)

        # Calculate distance to goal reward
        if not done:
            reward +=  self.dist2goal_reward_fn(self.state,self.goal)

        # Check if goal is reached
        if not done and self.check_goal():
            reward += self.goal_reward
            reward += self.dist_reward * self.rem_steps # prevents moving slow on purpose
            done = True
            info['reason'] = 'goal_reached'

        # Check if max steps reached
        self.steps += 1
        if not done and self.steps >= self.max_steps:
            done = True
            info['reason'] = 'max_steps'

        return self.state, reward, done, info

    def check_collision(self,state=None):
        if state is None:
            state = self.state

        for obs in self.obstacles:
            if obs['type'] == 'circle':
                if np.linalg.norm(state[:2] - obs['center']) <= obs['radius'] + self.car_radius:
                    done = True
                    info = {'reason': 'collision_circle'}
                    return done, info
            else:
                cx, cy = obs['center']
                w, h = obs['width'], obs['height']
                x, y = state[:2]
                if (cx - w / 2 <= x <= cx + w / 2) and (cy - h / 2 <= y <= cy + h / 2):
                    done = True
                    info = {'reason': 'collision_rect'}
                    return done, info

        done = False
        info = {}
        return done,info

    def check_goal(self, state=None):
        if state is None:
            state = self.state
        dist_to_goal = np.linalg.norm(state[:2] - self.goal)
        if dist_to_goal <= self.goal_radius and self.velocity < self.goal_velocity:
            return True
        return False


    def augment_state(self, state):
        state = np.hstack([state, self.velocity],dtype=np.float32)
        state = np.hstack([state, self.xy_dist2goal(state)],dtype=np.float32)
        state = np.hstack([state, self.xy_dist2obstacle(state)],dtype=np.float32)
        return state


    def dist2goal_reward_fn(self, state, goal):
        "can overwrite this with a custom reward function if needed. Default is distance to goal."
        # dist_to_goal = np.linalg.norm(state[:2] - goal)
        dist_to_goal = np.linalg.norm(np.array([state[0] - goal[0], state[1] - goal[1]]))
        reward = self.dist_reward * (1 - (dist_to_goal / self.max_dist)) ** self.dist_pow
        return reward

    def xy_dist2goal(self,state):
        """can be overwrittin in favor of visability graph"""
        return np.array([self.goal[0]-state[0], self.goal[1]-state[1] ])

    def xy_dist2obstacle(self,state):
        """
        Computes the distance to the nearest obstacle.
        """
        min_dist = float('inf')
        min_x = None
        min_y = None
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                r = obs['radius']
                dx,dy  = state[:2] - obs['center']
                dist_to_center = np.linalg.norm(state[:2] - obs['center'])# - obs['radius']
                dist = dist_to_center - r
                scale = (dist) / dist_to_center

                # get distance to perimeter of circle in x,y
                # Directional components
                dist_x = dx * scale
                dist_y = dy * scale
            else:
                x,y = state[:2]
                xmin = obs['center'][0] - obs['width'] / 2
                xmax = obs['center'][0] + obs['width'] / 2
                ymin = obs['center'][1] - obs['height'] / 2
                ymax = obs['center'][1] + obs['height'] / 2

                # xedges = np.array([xmin, xmax])
                # yedges = np.array([ymin, ymax])
                corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
                edges = []
                if x < xmin and ymin <= y <= ymax:  edges.append([xmin, y]) # left edge
                if x > xmax and ymin <= y <= ymax: edges.append([xmax, y]) # right edge
                if y < ymin and xmin <= x <= xmax:  edges.append([x, ymin]) # bottom edge
                if y > ymax and xmin <= x <= xmax:  edges.append([x, ymax])


                if len(edges) == 0: checks = corners
                else: checks = np.vstack([corners, np.array(edges)])
                dist_x,dist_y,dist = float('inf'),float('inf'),float('inf')
                for check in checks:
                    _dx, _dy = state[:2] - check
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
        return -1*np.array([min_x,min_y], dtype=np.float32)

    @property
    def rem_steps(self):
        return self.max_steps - self.steps

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

        # Draw obstacles
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                circle = plt.Circle(tuple(obs['center']), obs['radius'], color='k', alpha=obs_alpha)
                self.ax.add_patch(circle)
            else:
                cx, cy = obs['center']
                w, h = obs['width'], obs['height']
                rect = plt.Rectangle((cx - w/2, cy - h/2), w, h, color='k', alpha=obs_alpha)
                self.ax.add_patch(rect)

        # Draw goal
        goal_circle = plt.Circle(tuple(self.goal), self.goal_radius, color='green', alpha=goal_alpha)
        self.ax.add_patch(goal_circle)

        # Draw car
        x, y, theta = self.state[:3]
        car = plt.Circle((x, y), self.car_radius, color='blue')
        dx = 0.3 * np.cos(theta)
        dy = 0.3 * np.sin(theta)
        self.ax.arrow(x, y, dx, dy, head_width=0.1, color='black')
        self.ax.add_patch(car)

        # draw distance line features
        if dist_lines:

            dist2goal = self.xy_dist2goal(self.state)
            dist2obstacle = self.xy_dist2obstacle(self.state)

            self.ax.plot([x, x + dist2goal[0]], [y, y + dist2goal[1]], color='g', linestyle='--', label='Dist to Goal')
            self.ax.plot([x, x + dist2obstacle[0]], [y, y + dist2obstacle[1]], color='r', linestyle='--', label='Dist to Obstacle')

            if self.graph is not None:
                length, path = self.graph(self.state[:2])
                path = np.array(path)
                self.ax.plot(path[:, 0], path[:, 1], color='y', linestyle='--', label='Dist to Goal')


        self.ax.set_aspect('equal', adjustable='box')
        plt.draw()
        plt.pause(0.0001)
        # self.fig.canvas.flush_events()

    def close(self):
        if self.fig:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None



if __name__ == "__main__":
    layout_dict = read_layout_dict('example0')
    # layout_dict = read_layout_dict('example1')
    env = ContinuousNavigationEnv(**layout_dict)
    obs = env.reset()

    done = False
    total_reward = 0.0
    while not done:
        # action = env.action_space.sample()
        action = np.array([0.5, 0.0])  # Example action: throttle=1.0, steering=0.1
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render(dist_lines=True)
    print("Episode finished. Total reward:", total_reward, "Info:", info)
    env.close()