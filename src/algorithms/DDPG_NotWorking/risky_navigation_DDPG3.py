import gym
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Set seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


# ==========================
# Networks
# ==========================

def weights_init(m):
    if isinstance(m, nn.Linear):  # For Linear layers
        init.xavier_uniform_(m.weight,gain= nn.init.calculate_gain('leaky_relu', 0.2) )  # Xavier uniform initialization
        if m.bias is not None:
            init.zeros_(m.bias)  # Initialize bias to zeros

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_action,
                 size_hidden_layers = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, size_hidden_layers), nn.LeakyReLU(),
            nn.Linear(size_hidden_layers, size_hidden_layers), nn.LeakyReLU(),
            nn.Linear(size_hidden_layers, size_hidden_layers), nn.LeakyReLU(),
            nn.Linear(size_hidden_layers, size_hidden_layers), nn.LeakyReLU(),
            nn.Linear(size_hidden_layers, action_dim), nn.Tanh()
        )


        self.max_action = torch.from_numpy(max_action)
        self.min_action = torch.from_numpy(min_action)
        self.offset = torch.FloatTensor([1, 0])  # Offset to shift action space
        self.rescale = torch.FloatTensor([0.5, 1])  # Offset to shift action space
        self.net.apply(weights_init)  # Apply weight initialization

    def forward(self, state):
        x = self.net(state)  # output in [-1, 1]
        x = (x + self.offset) * self.max_action * self.rescale  # scale [-1, 1] to [min_action, max_action]
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,
                 size_hidden_layers = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, size_hidden_layers), nn.LeakyReLU(),
            nn.Linear(size_hidden_layers, size_hidden_layers), nn.LeakyReLU(),
            nn.Linear(size_hidden_layers, size_hidden_layers), nn.LeakyReLU(),
            nn.Linear(size_hidden_layers, size_hidden_layers), nn.LeakyReLU(),
            nn.Linear(size_hidden_layers, 1)
        )
        self.net.apply(weights_init)  # Apply weight initialization

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


# ==========================
# Replay Buffer
# ==========================
class ReplayBuffer:
    def __init__(self, capacity=500_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.from_numpy(np.array(state)),
            torch.from_numpy(np.array(action)),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.from_numpy(np.array(next_state)),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

# Ornstein-Uhlenbeck process for exploration noise\

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed=42, mu=0.0, theta=0.1, sigma=.5, sigma_min = 0.05, sigma_decay=.999,scale=None):
        # theta=0.1
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.scale = np.ones(size) if scale is None else np.array(scale)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu.copy()
        """Resduce  sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state * self.scale
# class OUNoise(object):
#     def __init__(self, action_dimension, mu=0.0, theta=0.1, sigma=0.1):
#         """
#         Ornstein-Uhlenbeck process for generating exploration noise.
#         :param action_dimension:
#         :param mu: mean of what the signal should be
#         :param theta: restoration coefficient (how fast noise returns to mu when deviation is large)  0.25
#         :param sigma: diffusion coefficient (how different noise will make policies) 0.25
#         """
#         self.mu = mu * np.ones(action_dimension)
#         self.theta = theta
#         self.sigma = sigma
#         self.state = self.mu.copy()
#
#     def reset(self):
#         self.state = self.mu.copy()
#
#     def sample(self):
#         dx = self.theta * (self.mu - self.state) + self.sigma * np.random.normal(size=self.mu.shape) #np.random.randn(len(self.state))
#         # dx = self.theta * (self.mu - self.state) + np.random.normal(size=self.mu.shape,scale=self.sigma)
#
#         self.state += dx
#         return self.state
# ==========================
# DDPG Agent
# ==========================
class DDPGAgent:
    # def __init__(self, env, gamma=0.99, tau=0.005, actor_lr=None, critic_lr=1e-3):
    def __init__(self, env, gamma=0.99, tau=0.01, actor_lr=5e-4, critic_lr=1e-3):
        self.env = env
        self.gamma = gamma
        self.tau = tau

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        # max_action = float(env.action_space.high[0])
        max_action = env.action_space.high
        min_action = env.action_space.low

        # Networks
        self.actor = Actor(obs_dim, act_dim, max_action,min_action)
        self.actor_target = Actor(obs_dim, act_dim, max_action,min_action)
        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        if actor_lr is None: actor_lr = critic_lr * 0.1
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer()
        self.max_action = max_action
        self.min_action = min_action

        # mu = [(self.env.action_space.high[i] + self.env.action_space.low[i]) / 2 for i in
        #  range(len(self.env.action_space.high))]
        # mu = np.array([0.05*self.max_action[0],0]) # encourage a little throttle in exploration
        # sigma = np.array((self.max_action-self.min_action)/6)
        # self.noise = OUNoise(act_dim,mu=mu,sigma=sigma)
        # self.noise = OUNoise(act_dim, sigma=sigma)
        # self.noise = OUNoise(act_dim,scale=np.array((self.max_action-self.min_action)/6))
        self.noise = OUNoise(act_dim)


    def select_action(self, state, noise=True, noise_gain=1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy().flatten()
        if noise:
            action += noise_gain * self.noise.sample()
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)
    def train(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = reward + self.gamma * (1 - done) * self.critic_target(next_state, next_action)

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def reset(self):
        self.noise.reset()

class Logger:
    def __init__(self,env,history_len=10):
        self.env = env

        # history
        self.filt_window = 10
        self.reward_history = []  # deque(maxlen=history_len)
        self.filt_reward_history = np.empty([0, 2])  # deque(maxlen=history_len)
        self.traj_history = deque(maxlen=history_len)
        self.terminal_history = deque(maxlen=history_len)
        self.fig_assets = {}

        self.test_reward_history = []  # deque(maxlen=history_len)
        self.test_filt_reward_history = np.empty([0, 2])  # deque(maxlen=history_len)
        self.test_traj_history = deque(maxlen=history_len)
        self.test_terminal_history = deque(maxlen=history_len)
        self.test_fig_assets = {}

        self.fig = None
        self.axes = None


    def log_train(self,ep_reward,state_seq,terminal_cause):

        self.reward_history.append(ep_reward)
        self.traj_history.append(np.array(state_seq))
        self.terminal_history.append(terminal_cause)

        mu = np.mean(self.reward_history[-min(len(self.reward_history), self.filt_window):])
        std = np.std(self.reward_history[-min(len(self.reward_history), self.filt_window):])
        self.filt_reward_history = np.vstack((self.filt_reward_history, np.array([[mu, std]])))
    def log_test(self,ep_reward,state_seq,terminal_cause):

        self.test_reward_history.append(ep_reward)
        self.test_traj_history.append(np.array(state_seq))
        self.test_terminal_history.append(terminal_cause)

        mu = np.mean(self.test_reward_history[-min(len(self.test_reward_history), self.filt_window):])
        std = np.std(self.test_reward_history[-min(len(self.test_reward_history), self.filt_window):])
        self.test_filt_reward_history = np.vstack((self.test_filt_reward_history, np.array([[mu, std]])))


    def _plot_history(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
            self.train_axes = self.axes[0,:].flatten()
            self.test_axes = self.axes[1, :].flatten()
            self.fig.suptitle('DDPG Training Progress')

        if len(self.reward_history) < self.filt_window:
            print("LOGGER: Not enough data to plot. Waiting for more episodes...")
            return

        self.draw(type='Train')
        self.draw(type='Test')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)


    def draw(self, type='Train'):


        axes = self.train_axes if type.lower() == 'train' else self.test_axes
        fig_assets = self.fig_assets if type.lower() == 'train' else self.test_fig_assets
        filt_reward_history = self.filt_reward_history if type.lower() == 'train' else self.test_filt_reward_history
        traj_history = self.traj_history if type.lower() == 'train' else self.test_traj_history
        terminal_history = self.terminal_history if type.lower() == 'train' else self.test_terminal_history

        x = np.arange(len(filt_reward_history))
        mean, std = filt_reward_history[:, 0], filt_reward_history[:, 1]

        # Reward plot
        if 'reward_line' not in fig_assets.keys():
            # self.fig_assets['reward_line'] = self.axes[0].plot(list(self.reward_history),lw=1,color='b')[0]
            fig_assets['reward_line'] = axes[0].plot(x, mean, lw=1, color='b')[0]
            # axes[0].set_title(f'{type} Reward (eps)')
            if type.lower() != 'train': axes[0].set_xlabel('Episode')
            axes[0].set_ylabel(f'{type} Reward')

            # plot 1 std deviation as shaded region
            fig_assets['reward_patch'] = axes[0].fill_between(x, mean - std, mean + std, color='b', alpha=0.2
                                                                        )


        else:
            # x = np.arange(len(self.reward_history))
            # self.fig_assets['reward_line'].set_data(x,list(self.reward_history))
            fig_assets['reward_line'].set_data(x, mean)

            fig_assets['reward_patch'].remove()
            # self.fig_assets['reward_patch'] = add_patch(self.reward_history)
            fig_assets['reward_patch'] = axes[0].fill_between(
                x,  # np.arange(len(mean)),
                mean - std,
                mean + std,
                color='b',
                alpha=0.2

            )
            axes[0].relim()
            axes[0].autoscale_view()

        # # Trajectories

        if 'traj_lines' not in fig_assets.keys():
            self.env.reset()
            self.env.render(ax=axes[1])

            fig_assets['traj_lines'] = []
            for (traj, term) in zip(traj_history, terminal_history):
                if term == 'goal_reached':
                    plt_params = {'color': 'green', 'alpha': 0.7}
                elif 'collision' in term:
                    plt_params = {'color': 'red', 'alpha': 0.7}
                elif term == 'max_steps':
                    plt_params = {'color': 'gray', 'alpha': 0.7}
                else:
                    raise ValueError(f"Unknown terminal cause: {term}")
                fig_assets['traj_lines'].append(axes[1].plot(traj[:, 0], traj[:, 1], **plt_params)[0])

            if type.lower() == 'train':
                axes[1].set_title('Trajectories (last {} eps)'.format(len(traj_history)))
            # axes[1].set_xlabel('X')
            # axes[1].set_ylabel('Y')
        else:
            for i, traj, term in zip(np.arange(len(traj_history)), traj_history, terminal_history):
                if term == 'goal_reached':
                    plt_params = {'color': 'green', 'alpha': 0.7}
                elif 'collision' in term:
                    plt_params = {'color': 'red', 'alpha': 0.7}
                elif term == 'max_steps':
                    plt_params = {'color': 'gray', 'alpha': 0.7}
                else:
                    raise ValueError(f"Unknown terminal cause: {term}")
                fig_assets['traj_lines'][i].set_data(traj[:, 0], traj[:, 1])
                fig_assets['traj_lines'][i].set_color(plt_params['color'])
                fig_assets['traj_lines'][i].set_alpha(plt_params['alpha'])

    def spin(self):
        if self.fig is not None:
            self.fig.canvas.flush_events()


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        """
        Normalizes the actions to be in between action_space.high and action_space.low.
        If action_space.low == -action_space.high, this is equals to action_space.high*action.

        :param action:
        :return: normalized action
        """
        # action = (action + 1) / 2  # [-1, 1] => [0, 1]
        # action *= (self.action_space.high - self.action_space.low)
        # action += self.action_space.low
        # return action

        action += np.array([1, 0])
        action *= np.array([0.5, 1.0])
        action *= self.action_space.high
        return action

    def reverse_action(self, action):
        """
        Reverts the normalization

        :param action:
        :return:
        """
        # action -= self.action_space.low
        # action /= (self.action_space.high - self.action_space.low)
        # action = action * 2 - 1
        action /= self.action_space.high
        action /= np.array([0.5, 1.0])
        action -= np.array([1, 0])
        return action

# ==========================
# Training Loop
# ==========================
def train_ddpg(env_name="Pendulum-v1", max_episodes=1000, max_steps=200, batch_size=64):
    from continuous_nav_env import ContinuousNavigationEnv  # assumes env code is in this module

    # env = gym.make(env_name)
    # env = ContinuousNavigationEnv(
    #     goal=(8, 8),
    #     max_steps=500,
    #     obstacles=[{'type': 'circle', 'center': (5, 5), 'radius': 1.0}]
    # )
    env = ContinuousNavigationEnv(
        goal=(8, 8),
        start = (2, 8, -np.pi/2),  # start position and heading
        max_steps=500,
        obstacles=[ {'type': 'rect', 'center': (5, 7.5), 'width': 1.0, 'height': 8}]
    )
    # env = NormalizedActions(env)

    max_steps = env.max_steps
    agent = DDPGAgent(env)
    logger = Logger(env)

    for episode in range(max_episodes):
        logger.spin()
        actions = []
        trajectories = []
        state, ep_reward = env.reset(heading_noise=np.pi/6), 0
        agent.reset()
        trajectories.append(state[:2])  # Store only position for trajectory
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            actions.append(action)

            agent.train(batch_size)

            state = next_state
            trajectories.append(state[:2])  # Store only position for trajectory

            ep_reward += reward
            if done:
                terminal_cause = info['reason']
                break
        logger.log_train(ep_reward, trajectories, terminal_cause)

        actions = np.array(actions)
        Amax = np.max(actions,axis=0).round(2)
        Amin = np.min(actions,axis=0).round(2)
        Amean = np.mean(actions,axis=0).round(2)
        print(f"Episode {episode}, Reward: {ep_reward:.2f} "
              f"\tThrottle: [max: {Amax[0]:.2f} min: {Amin[0]:.2f} mean: {Amean[0]:.2f}]"
              f"\tHeading: [max: {Amax[1]:.2f} min: {Amin[1]:.2f} mean: {Amean[1]:.2f}]"
              f"\tTerminal Cause: {terminal_cause}"
              f"\t Sigma: {agent.noise.sigma:.3f} "
              )

        if episode % 3 == 0 and episode != 0:
            print(f"Logging episode {episode} to logger plot...")
            logger._plot_history()

    plt.ioff()
    plt.show()
    env.close()


if __name__ == "__main__":
    train_ddpg()
