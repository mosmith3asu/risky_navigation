import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Actor network: Gaussian policy
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,
                 num_hidden_layers =2,
                 size_hidden_layers= 128,
                 activation='ReLU',
                 max_action = None,
                 min_action=None,
                 offset_action=None):
        # TODO: scale output to range with tanh func
        super(Actor, self).__init__()

        # Simple 2-layer MLP for policy mean
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, action_dim)
        self.activation = nn.ReLU()

        # Action scaling
        self.max_action = max_action if max_action is None else torch.from_numpy(max_action)
        self.min_action = min_action if min_action is None else torch.from_numpy(min_action)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable log std for Gaussian

    def forward(self, x):
        # Forward pass for actor: outputs mean and std of Gaussian policy
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = F.tanh(self.mu(x))
        x = x + torch.tensor([1, 0])
        x = x * torch.tensor([0.5, 1.0])
        x = x * self.max_action

        std = torch.exp(self.log_std)
        return x, std  # mean and std

# Critic network: state-value
class Critic(nn.Module):
    def __init__(self, state_dim,
                 num_hidden_layers=2,
                 size_hidden_layers=128,
                 activation='ReLU'):
        super(Critic, self).__init__()

        # Flexible MLP for value function
        self.input_dim = state_dim
        self.output_dim = 1
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.activation = getattr(nn, activation)

        layers = [nn.Linear(state_dim, self.size_hidden_layers), self.activation()]
        for i in range(1, self.num_hidden_layers - 1):
            layers.extend([nn.Linear(self.size_hidden_layers, self.size_hidden_layers), self.activation()])
        layers.extend([nn.Linear(self.size_hidden_layers, self.output_dim)])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass for critic: outputs value estimate
        for module in self.layers:
            x = module(x)
        return x

class A2CAgent:
    def __init__(self, env, gamma=0.99, lr=5e-4,entropy_reg=0.001, history_len=10,scale_actions=True):
        """
        env: environment to interact with
        gamma: discount factor
        lr: learning rate
        entropy_reg: entropy regularization coefficient
        history_len: number of episodes to keep in history for plotting
        scale_actions: whether to scale actions to env action space
        """
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # For action scaling
        offset_action = np.mean([env.action_space.high, env.action_space.low],axis=0)
        centered_max_action = env.action_space.high - offset_action

        actor_kwargs = {}
        if scale_actions:
            actor_kwargs['max_action'] = env.action_space.high  # assuming action space is symmetric
            actor_kwargs['min_action'] = env.action_space.low
        self.actor = Actor(state_dim, action_dim,**actor_kwargs)
        self.critic = Critic(state_dim)
        self.gamma = gamma
        self.entropy_reg = entropy_reg
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=0.5*lr)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=lr)

        # For plotting and history
        self.reward_history = [] # total reward per episode
        self.filt_reward_history = np.empty([0,2])  # filtered mean/std for plotting
        self.filt_window = 10
        self.traj_history = deque(maxlen=history_len)  # stores last N trajectories
        self.terminal_history = deque(maxlen=history_len)  # stores last N terminal causes

        self.fig = None
        self.axes = None
        self.fig_assets = {}
        print("A2CAgent initialized.")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Gamma: {gamma}, Learning rate: {lr}, Entropy reg: {entropy_reg}")

    def select_action(self, state):
        """
        Given the current state, use the actor (policy) to sample an action.
        Returns:
            action_clipped: action clipped to env action space
            log_prob: log probability of the action (for policy gradient)
            entropy: entropy of the action distribution (for exploration)
        """
        state_t = torch.FloatTensor(state)
        mu, std = self.actor(state_t)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action_clipped = action.clamp(
            torch.FloatTensor(self.env.action_space.low),
            torch.FloatTensor(self.env.action_space.high)
        )
        log_prob = dist.log_prob(action).sum()
        entropy = dist.entropy()
        return action_clipped.detach().numpy(), log_prob,entropy

    def update(self, trajectory):
        """
        After an episode, update actor and critic using the collected trajectory.
        trajectory: list of (state, action, reward, next_state, done, logp, ent)
        """
        # Convert trajectory to tensors
        states = torch.from_numpy(np.array([t[0] for t in trajectory]))
        actions = torch.from_numpy(np.array([t[1] for t in trajectory]))
        rewards = [t[2] for t in trajectory]
        dones = [t[4] for t in trajectory]
        log_probs = torch.stack([t[5] for t in trajectory])
        entropies = torch.stack([t[6] for t in trajectory])

        # Compute discounted returns (reward-to-go)
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + (0 if done else self.gamma * R)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)

        # Critic evaluates value of each state
        values = self.critic(states).squeeze()
        advantages = returns - values.detach()  # advantage = actual - predicted

        # Actor loss: policy gradient + entropy regularization
        entropy_loss = -self.entropy_reg * entropies.mean()  # encourage exploration
        actor_loss = -(log_probs * advantages).mean() + entropy_loss
        # Critic loss: mean squared error between predicted and actual returns
        critic_loss = F.mse_loss(values, returns)

        # Update actor (player) network
        self.optimizerA.zero_grad()
        actor_loss.backward()
        self.optimizerA.step()
        # Update critic (coach) network
        self.optimizerC.zero_grad()
        critic_loss.backward()
        self.optimizerC.step()

    def train(self, max_episodes=1000,blocking=True):
        """
        Main training loop.
        For each episode:
            - Reset environment
            - Collect trajectory by interacting with env
            - Update actor and critic after episode
            - Record and plot history
        """
        for ep in range(1, max_episodes+1):
            state = self.env.reset(heading_noise=np.pi/4)
            trajectory = []
            ep_reward = 0.0
            states_seq = []
            actions_seq = []
            while True:
                # Actor (player) chooses action
                action, logp,ent = self.select_action(state)
                # Environment (game world) responds
                next_state, reward, done, info = self.env.step(action)
                # Store transition for learning
                trajectory.append((state, action, reward, next_state, done, logp,ent))
                actions_seq.append(action)
                states_seq.append(state[:2])
                state = next_state
                ep_reward += reward
                if done:
                    terminal_cause = info['reason']
                    break

            # After episode, update actor and critic (player and coach review the match)
            self.update(trajectory)

            # Record history for plotting and analysis
            self.reward_history.append(ep_reward)
            self.traj_history.append(np.array(states_seq))
            self.terminal_history.append(terminal_cause)

            # Compute filtered mean/std for reward plot
            mu = np.mean(self.reward_history[-min(len(self.reward_history),self.filt_window):])
            std = np.std(self.reward_history[-min(len(self.reward_history),self.filt_window):])
            self.filt_reward_history = np.vstack((self.filt_reward_history,
                                                  np.array([[mu,std]])))

            # Every history_len episodes, plot progress
            if ep % self.traj_history.maxlen == 0:
                self._plot_history()

            # Print progress every 10 episodes
            if ep % 10 == 0:
                mean_acts = np.array(actions_seq).mean(axis=0).round(2)
                print(f"Episode {ep}, Reward: {ep_reward:.2f} "
                      f"Mean Throttle {mean_acts[0]:.2f} "
                      f"Mean Steering {mean_acts[1]:.2f} ")

        # Show plot at end if blocking
        if blocking:
            plt.ioff()
            plt.show()

    def _plot_history(self):
        """
        Plot reward and trajectory history for visualization.
        """
        if self.fig is None:
            plt.ion()
            self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 4))

        x = np.arange(len(self.filt_reward_history))
        mean, std = self.filt_reward_history[:,0],self.filt_reward_history[:,1]

        # Reward plot
        if 'reward_line' not in self.fig_assets.keys():
            self.fig_assets['reward_line'] = self.axes[0].plot(x,mean, lw=1, color='b')[0]
            self.axes[0].set_title('Reward (eps)')
            self.axes[0].set_xlabel('Episode')
            self.axes[0].set_ylabel('Total Reward')
            # plot 1 std deviation as shaded region
            self.fig_assets['reward_patch'] = self.axes[0].fill_between( x, mean - std, mean + std,
                color='b',  alpha=0.2)
        else:
            self.fig_assets['reward_line'].set_data(x, mean)
            self.fig_assets['reward_patch'].remove()
            self.fig_assets['reward_patch'] = self.axes[0].fill_between(
                x,
                mean - std,
                mean + std,
                color='b',
                alpha=0.2
            )
            self.axes[0].relim()
            self.axes[0].autoscale_view()

        # Trajectories plot
        if 'traj_lines' not in self.fig_assets.keys():
            self.env.reset()
            self.env.render(ax=self.axes[1])

            self.fig_assets['traj_lines'] = []
            for (traj,term) in zip(self.traj_history,self.terminal_history):
                if term == 'goal_reached': plt_params = {'color':'green', 'alpha': 0.7}
                elif 'collision' in term: plt_params = {'color':'red', 'alpha': 0.7}
                elif term == 'max_steps': plt_params = {'color':'gray', 'alpha': 0.7}
                else: raise ValueError(f"Unknown terminal cause: {term}")
                self.fig_assets['traj_lines'].append(self.axes[1].plot(traj[:,0], traj[:,1],**plt_params)[0])

            self.axes[1].set_title('Trajectories (last {} eps)'.format(len(self.traj_history)))
            self.axes[1].set_xlabel('X'); self.axes[1].set_ylabel('Y')
        else:
            for i, traj, term in zip(np.arange(len(self.traj_history)),self.traj_history, self.terminal_history):
                if term == 'goal_reached':  plt_params = {'color': 'green', 'alpha': 0.7}
                elif 'collision' in term: plt_params = {'color': 'red', 'alpha': 0.7}
                elif term == 'max_steps':  plt_params = {'color': 'gray', 'alpha': 0.7}
                else: raise ValueError(f"Unknown terminal cause: {term}")
                self.fig_assets['traj_lines'][i].set_data(traj[:, 0], traj[:, 1])
                self.fig_assets['traj_lines'][i].set_color(plt_params['color'])
                self.fig_assets['traj_lines'][i].set_alpha(plt_params['alpha'])

        plt.tight_layout()
        plt.draw()