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
                 num_hidden_layers = 4,
                 size_hidden_layers= 128,
                 activation='ReLU'
                 ):
        # TODO: scale output to range with tanh func
        super(Actor, self).__init__()

        self.output_dim = action_dim
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.activation = getattr(nn, activation)




        ##########################
        layers = [nn.Linear(state_dim, self.size_hidden_layers), self.activation()]
        for i in range(self.num_hidden_layers):
            layers.extend([nn.Linear(self.size_hidden_layers, self.size_hidden_layers), self.activation()])
        layers.extend([nn.Linear(self.size_hidden_layers, self.output_dim)])
        self.layers = nn.Sequential(*layers)

        ##########################
        # self.fc1 = nn.Linear(state_dim, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.mu = nn.Linear(128, action_dim)
        # self.activation = nn.ReLU()

        self.log_std = nn.Parameter(torch.zeros(action_dim))



    def forward(self, x):
        for module in self.layers:
            x = module(x)
        x = F.tanh(x)

        # x = self.activation(self.fc1(x))
        # x = self.activation(self.fc2(x))
        # x = F.tanh(self.mu(x))
        std = torch.exp(self.log_std)


        return x, std


# Critic network: state-value
class Critic(nn.Module):
    def __init__(self, state_dim,
                 num_hidden_layers=4,
                 size_hidden_layers=128,
                 activation='ReLU'):
        super(Critic, self).__init__()

        # self.fc1 = nn.Linear(state_dim, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.v = nn.Linear(128, 1)

        self.input_dim = state_dim
        self.output_dim = 1
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.activation = getattr(nn, activation)

        layers = [nn.Linear(state_dim, self.size_hidden_layers), self.activation()]
        for i in range(self.num_hidden_layers):
            layers.extend([nn.Linear(self.size_hidden_layers, self.size_hidden_layers), self.activation()])
        layers.extend([nn.Linear(self.size_hidden_layers, self.output_dim)])
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # return self.v(x)
        for module in self.layers:
            x = module(x)
        return x

class A2CAgent:
    def __init__(self, env, gamma=0.99, lr=1e-4,entropy_reg=1e4, history_len=10,scale_actions=True):
        """

        :param env:
        :param gamma:
        :param lr: 3e-4
        :param history_len:
        """
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        offset_action = np.mean([env.action_space.high, env.action_space.low],axis=0)
        centered_max_action = env.action_space.high - offset_action

        actor_kwargs = {}
        # if scale_actions:
        #     actor_kwargs['action_bounds'] = torch.tensor((env.action_space.low, env.action_space.high),
        #                                                  dtype=torch.float32)
        #     actor_kwargs['max_action'] = env.action_space.high  # assuming action space is symmetric
        #     actor_kwargs['min_action'] = env.action_space.low
        #     # actor_kwargs['max_action'] = centered_max_action  # assuming action space is symmetric
        #     # actor_kwargs['offset_action'] = offset_action
        self.actor = Actor(state_dim, action_dim,**actor_kwargs)
        self.critic = Critic(state_dim)
        self.gamma = gamma
        self.entropy_reg = entropy_reg
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=0.5*lr)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=lr)

        # history
        self.reward_history = [] #deque(maxlen=history_len)
        self.filt_reward_history = np.empty([0,2])  # deque(maxlen=history_len)
        self.filt_window = 100
        self.traj_history = deque(maxlen=history_len)
        self.terminal_history = deque(maxlen=history_len)

        self.fig = None
        self.axes = None
        self.fig_assets = {}

    def select_action(self, state):
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
        stats_arr = np.array([t[0] for t in trajectory])
        states = torch.from_numpy(stats_arr)
        actions = torch.from_numpy(np.array([t[1] for t in trajectory]))
        # states = torch.FloatTensor([t[0] for t in trajectory])
        # actions = torch.FloatTensor([t[1] for t in trajectory])
        rewards = [t[2] for t in trajectory]
        dones = [t[4] for t in trajectory]
        log_probs = torch.stack([t[5] for t in trajectory])
        entropies = torch.stack([t[6] for t in trajectory])

        # compute returns
        returns = self.compute_returns(rewards, dones,)
        # returns = []
        # R = 0
        # for r, done in zip(reversed(rewards), reversed(dones)):
        #     R = r + (0 if done else self.gamma * R)
        #     returns.insert(0, R)

        returns = torch.FloatTensor(returns)

        values = self.critic(states).squeeze()
        advantages = returns - values.detach()

        # actor loss
        entropy_loss = -self.entropy_reg * entropies.mean()  # encourage exploration
        actor_loss = -(log_probs * advantages).mean() + entropy_loss
        # critic loss
        critic_loss = F.mse_loss(values, returns)

        self.optimizerA.zero_grad(); actor_loss.backward(); self.optimizerA.step()
        self.optimizerC.zero_grad(); critic_loss.backward(); self.optimizerC.step()

    def compute_returns(self, rewards, dones, values, next_values):
        """
        rewards: [T, N]
        dones  : [T, N]  (True if episode ended at step t)
        values : [T, N]
        next_values: [T, N]  (values of s_{t+1}, aligned with t)
        Returns: returns_tn [T, N]
        """
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + (0 if done else self.gamma * R)
            returns.insert(0, R)
        return returns


    def train(self, max_episodes=1000,blocking=True, rand_start_epi = 5000):
        ep_len = []
        for ep in range(1, max_episodes+1):
            # state = self.env.reset(heading_noise=np.pi/6)
            p = (rand_start_epi - ep) / rand_start_epi if ep < rand_start_epi else 0
            state,_ = self.env.reset( random_state= np.random.rand() < p)
            trajectory = []
            ep_reward = 0.0

            states_seq = [state[:2]]
            actions_seq = []
            while True:

                action, logp,ent = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                trajectory.append((state, action, reward, next_state, done, logp,ent))
                actions_seq.append(action)
                states_seq.append(next_state[:2])
                state = next_state
                ep_reward += reward

                if done:
                    terminal_cause = info['reason']
                    ep_len.append(len(trajectory))
                    break
            self.update(trajectory)


            # record history
            self.reward_history.append(ep_reward)
            self.traj_history.append(np.array(states_seq))
            self.terminal_history.append(terminal_cause)
            mu = np.mean(self.reward_history[-min(len(self.reward_history),self.filt_window):])
            std = np.std(self.reward_history[-min(len(self.reward_history),self.filt_window):])
            self.filt_reward_history = np.vstack((self.filt_reward_history,  np.array([[mu,std]])))


            # every history_len episodes, plot
            if ep % self.traj_history.maxlen == 0:
                self._plot_history()

            if ep % 10 == 0:
                mean_acts = np.array(actions_seq).mean(axis=0).round(2)
                print(f"Episode {ep} [len={np.mean(ep_len)}], Reward: {ep_reward:.2f} "
                      f"Mean Joy [{mean_acts[0]:.2f} {mean_acts[1]:.2f} ]")
                ep_len = []
        if blocking:
            plt.ioff()
            plt.show()

    def _plot_history(self,lw = 0.1):
        if self.fig is None:
            plt.ion()
            self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 4))

        x = np.arange(len(self.filt_reward_history))
        mean, std = self.filt_reward_history[:,0],self.filt_reward_history[:,1]

        # Reward plot
        if 'reward_line' not in self.fig_assets.keys():
            # self.fig_assets['reward_line'] = self.axes[0].plot(list(self.reward_history),lw=1,color='b')[0]
            self.fig_assets['reward_line'] = self.axes[0].plot(x,mean, lw=lw, color='b')[0]
            self.axes[0].set_title('Reward (eps)')
            self.axes[0].set_xlabel('Episode')
            self.axes[0].set_ylabel('Total Reward')

            # plot 1 std deviation as shaded region
            self.fig_assets['reward_patch'] = self.axes[0].fill_between( x, mean - std, mean + std,
                color='b',  alpha=0.2
                                                                         )


        else:
            # x = np.arange(len(self.reward_history))
            # self.fig_assets['reward_line'].set_data(x,list(self.reward_history))
            self.fig_assets['reward_line'].set_data(x, mean)

            self.fig_assets['reward_patch'].remove()
            # self.fig_assets['reward_patch'] = add_patch(self.reward_history)
            self.fig_assets['reward_patch'] = self.axes[0].fill_between(
                x,#np.arange(len(mean)),
                mean - std,
                mean + std,
                color='b',
                alpha=0.2

            )
            self.axes[0].relim()
            self.axes[0].autoscale_view()


        # # Trajectories
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
        plt.pause(0.001)
