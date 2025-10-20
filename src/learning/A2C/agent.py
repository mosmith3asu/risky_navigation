import copy

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
                 num_hidden_layers = 3,
                 size_hidden_layers= 128,
                 activation='LeakyReLU'
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
                 num_hidden_layers=3,
                 size_hidden_layers=128,
                 activation='LeakyReLU'):
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
    def __init__(self, env,
                 gamma=0.99, lr=5e-4,
                 entropy_reg=0.000,
                 grad_clip = 0.75):
        """

        :param env:
        :param gamma:
        :param lr: 3e-4
        :param history_len:
        """
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.entropy_reg = entropy_reg
        self.grad_clip = grad_clip

        # Initialize actor and critic networks
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=0.5*self.lr)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.reset_params = {'options': {'enable': True,
                                         'p_rand_state': 0.0,
                                         'reward_dist2goal': self.env.get_attr('reward_dist2goal')[0]
                                         }}

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
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().mean(-1)
        return action_clipped.detach().numpy(), log_prob, entropy

    def update(self, traj_batch):
        states      = torch.from_numpy(traj_batch['states'])
        actions     = traj_batch['actions']
        rewards     = traj_batch['rewards'].flatten()
        next_states = traj_batch['mext_states']
        dones       = torch.from_numpy(traj_batch['dones'].flatten())
        log_probs   = traj_batch['log_probs']
        entropies   = traj_batch['entropies']


        if states.shape[0] <= 1:
            # print("Not enough data to update likely due to random start state. Skipping.")
            return

        values = self.critic(states).squeeze()
        returns = torch.FloatTensor(self.compute_returns(rewards, dones,terminal_return = values[-1]))
        advantages = returns - values.detach()

        # actor loss
        actor_loss = -(log_probs * advantages).mean()
        if entropies is not None:
            actor_loss += -self.entropy_reg * entropies.mean()  # encourage exploration

        # critic loss
        critic_loss = F.mse_loss(values, returns)

        self.optimizerA.zero_grad(); actor_loss.backward()
        self.optimizerC.zero_grad(); critic_loss.backward()

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)

        self.optimizerA.step()
        self.optimizerC.step()

    def compute_returns(self, rewards, dones, terminal_return= 0):
        """ computes discounted cumulative returns from ordered trajector of rewards"""

        returns = np.nan * np.ones_like(rewards)
        R = terminal_return.detach().numpy() \
            if isinstance(terminal_return, torch.Tensor) else terminal_return

        if R.size > 1: R = R[:,np.newaxis]

        for i in reversed(range(len(rewards))):
            R = rewards[i] + (1-dones[i]) * self.gamma * R
            returns[i] = R
        return returns

    def train(self, max_episodes=1000,blocking=True, rand_start_epi = 5000):
        history = {
            'ep_len': deque(maxlen=10),
            'ep_reward': deque(maxlen=10),
            'action': deque(maxlen=10),
            'xy': deque(maxlen=10),
            'terminal': deque(maxlen=10),
            'timeseries_filt_rewards': np.zeros([0,2]),
        }

        for ep in range(1, max_episodes+1):
            # state = self.env.reset(heading_noise=np.pi/6)
            p = (rand_start_epi - ep) / rand_start_epi if ep < rand_start_epi else 0
            state,_ = self.env.reset( random_state= np.random.rand() < p)

            history['ep_len'].append(0)
            history['ep_reward'].append(0.0)
            history['action'].append(np.array([0,0]))
            history['xy'].append(state[np.newaxis, :2])  # initial state for xy tracking
            history['terminal'].append('')

            ns = state.shape[-1] # number of states
            traj_batch = {
                'states'     : np.empty([0, ns], dtype = np.float32),
                'actions'    : np.empty([0, 2] , dtype = np.float32),
                'rewards'    : np.empty([0, 1] , dtype = np.float32),
                'mext_states': np.empty([0, ns], dtype = np.float32),
                'dones'      : np.empty([0, 1] , dtype = np.float32),
                'log_probs'  : torch.zeros([0, 1] , dtype = torch.float32),
                'entropies'  : torch.zeros([0, 1] , dtype = torch.float32)
            }




            while True:
                action, logp,ent = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                traj_batch['states']    = np.vstack([traj_batch['states'], state[np.newaxis, :]])
                traj_batch['actions']   = np.vstack([traj_batch['actions'], action[np.newaxis, :]])
                traj_batch['rewards']   = np.vstack([traj_batch['rewards'], reward[np.newaxis, np.newaxis]])
                traj_batch['mext_states'] = np.vstack([traj_batch['mext_states'], next_state[np.newaxis, :]])
                traj_batch['dones']     = np.vstack([traj_batch['dones'], np.array([[done]])])
                traj_batch['log_probs'] = torch.vstack([traj_batch['log_probs'], logp.unsqueeze(0)])
                traj_batch['entropies'] = torch.vstack([traj_batch['entropies'], ent.unsqueeze(0)])

                history['ep_len'][-1] += 1
                history['ep_reward'][-1] += reward
                history['xy'][-1] = np.vstack([history['xy'][-1] , next_state[np.newaxis, :2]])  # track xy positions

                state = next_state

                if done:
                    terminal_cause = info['reason']
                    history['terminal'][-1] = terminal_cause
                    history['action'][-1] = np.mean(traj_batch['actions'],axis=0)
                    break

            self.update(traj_batch)

            # record history
            mu, std = np.mean(history['ep_reward']), np.std(history['ep_reward'])
            history['timeseries_filt_rewards'] =  np.vstack((history['timeseries_filt_rewards'], np.array([[mu, std]])))

            if ep % history['xy'].maxlen == 0:

                self._plot_history(traj_history=history['xy'],
                                   terminal_history=history['terminal'],
                                   filt_reward_history=history['timeseries_filt_rewards'],)

            if ep % 10 == 0:
                mean_epi_reward = np.mean(history['ep_reward'])
                mean_epilen = np.mean(history['ep_len'])
                mean_acts = np.mean(np.array(history['action']), axis=0)

                print(f"Episode {ep} [len={mean_epilen}], "
                      f"Reward: {mean_epi_reward:.2f} "
                      f"Mean Joy [{mean_acts[0]:.2f} {mean_acts[1]:.2f} ]")

        if blocking:
            plt.ioff()
            plt.show()

    def _plot_history(self,
                      traj_history, terminal_history,
                      filt_reward_history,
                      env = None,
                      lw = 0.5):

        env = env or self.env

        if self.fig is None:
            plt.ion()
            self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 4))

        x = np.arange(len(filt_reward_history))
        mean, std = filt_reward_history[:,0],filt_reward_history[:,1]

        # Reward plot
        if 'reward_line' not in self.fig_assets.keys():
            # self.fig_assets['reward_line'] = self.axes[0].plot(list(self.reward_history),lw=1,color='b')[0]
            self.fig_assets['reward_line'] = self.axes[0].plot(x,mean, lw=lw, color='k')[0]
            self.axes[0].set_title('Reward (eps)')
            self.axes[0].set_xlabel('Episode')
            self.axes[0].set_ylabel('Total Reward')

            # plot 1 std deviation as shaded region
            self.fig_assets['reward_patch'] = self.axes[0].fill_between( x, mean - std, mean + std,
                color='b',  alpha=0.2
                                                                         )


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


        # # Trajectories
        if 'traj_lines' not in self.fig_assets.keys():
            env.reset(**self.reset_params)
            if hasattr(env, 'num_envs'):
                env.call('render', ax=self.axes[1]) # retrieve vec render
            else:
                env.render(ax=self.axes[1]) # single env render

            self.fig_assets['traj_lines'] = []
            for (traj,term) in zip(traj_history,terminal_history):
                if term == 'goal_reached': plt_params = {'color':'green', 'alpha': 0.7,  'lw':2*lw}
                elif 'collision' in term: plt_params = {'color':'red', 'alpha': 0.7,  'lw':lw}
                elif term == 'max_steps': plt_params = {'color':'gray', 'alpha': 0.7,  'lw':lw}
                else: raise ValueError(f"Unknown terminal cause: {term}")
                self.fig_assets['traj_lines'].append(self.axes[1].plot(traj[:,0], traj[:,1], **plt_params)[0])

            self.axes[1].set_title('Trajectories (last {} eps)'.format(len(traj_history)))
            self.axes[1].set_xlabel('X'); self.axes[1].set_ylabel('Y')
        else:
            for i, traj, term in zip(np.arange(len(traj_history)),traj_history, terminal_history):
                if term == 'goal_reached':  plt_params = {'color': 'green', 'alpha': 0.7,'lw':2*lw}
                elif 'collision' in term: plt_params = {'color': 'red', 'alpha': 0.7,'lw':lw}
                elif term == 'max_steps':  plt_params = {'color': 'gray', 'alpha': 0.7,'lw':lw}
                else: raise ValueError(f"Unknown terminal cause: {term}")
                self.fig_assets['traj_lines'][i].set_data(traj[:, 0], traj[:, 1])
                self.fig_assets['traj_lines'][i].set_color(plt_params['color'])
                self.fig_assets['traj_lines'][i].set_alpha(plt_params['alpha'])

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)



class A2CAgentVec(A2CAgent):
    def __init__(self, envs, **kwargs):
        super().__init__(envs, **kwargs)

        self.num_envs = self.env.num_envs
        self.action_low = torch.tensor(self.env.single_action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(self.env.single_action_space.high, dtype=torch.float32)

        # Re-initialize actor and critic networks
        state_dim = self.env.single_observation_space.shape[0]
        action_dim = self.env.single_action_space.shape[0]
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=0.5 * self.lr)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=self.lr)



    def train(self, max_episodes=20_000, blocking=True,
              # rand_start_epi= 500, # 1
              rand_start_epi=100,  # 1
              rshape_scale = 1, # 20
              rshape_epi = 1, #500
              report_interval = 5):


        reset_params = copy.deepcopy(self.reset_params)
        init_reward_dist2goal = self.env.get_attr('reward_dist2goal')[0]


        history = {
            'ep_len': deque(maxlen=report_interval),
            'ep_reward': deque(maxlen=report_interval),
            'action': deque(maxlen=report_interval),
            'xy': deque(maxlen= report_interval *  self.num_envs),
            'terminal': deque(maxlen= report_interval *  self.num_envs),
            'timeseries_filt_rewards': np.zeros([0, 2]),

            'rew_dist2goal': deque(maxlen=report_interval),
            'rew_smooth': deque(maxlen=report_interval),
            'rew_step': deque(maxlen=report_interval)
        }

        for ep in range(1, max_episodes + 1):

            # Schedules
            _rshape_scale = rshape_scale * (rshape_epi - ep) / rshape_epi if ep < rshape_epi else 1
            _p_rand_state = (rand_start_epi - ep) / rand_start_epi if ep < rand_start_epi else 0
            reset_params['options']['p_rand_state'] = _p_rand_state
            reset_params['options']['reward_dist2goal'] = _rshape_scale*init_reward_dist2goal

            states, _ = self.env.reset(**reset_params)

            history['ep_len'].append(0)
            history['ep_reward'].append(0.0)
            history['action'].append(np.array([0, 0]))

            history['rew_dist2goal'].append(0)
            history['rew_smooth'].append(0)
            history['rew_step'].append(0)


            ns = states.shape[-1]  # number of states
            traj_batch = {
                'states': np.empty([0, self.num_envs, ns], dtype=np.float32),
                'actions': np.empty([0, self.num_envs, 2], dtype=np.float32),
                'rewards': np.empty([0, self.num_envs, 1], dtype=np.float32),
                'mext_states': np.empty([0, self.num_envs, ns], dtype=np.float32),
                'dones': np.empty([0, self.num_envs, 1], dtype=np.float32),
                'log_probs': torch.zeros([0, self.num_envs, 1], dtype=torch.float32),
                'entropies': torch.zeros([0, self.num_envs, 1], dtype=torch.float32)

            }

            dones = np.zeros(self.num_envs, dtype=bool)

            while not np.all(dones):
                states_pt = torch.as_tensor(self._to_batch(states), dtype=torch.float32)
                actions, logps, ents = self.select_action(states_pt)  # select action
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                dones = infos['done'] # np.logical_or(terminated, truncated)

                traj_batch['states']    = np.vstack([traj_batch['states'], states[np.newaxis, :, :]])
                traj_batch['actions']   = np.vstack([traj_batch['actions'], actions[np.newaxis, :, :]])
                traj_batch['rewards']   = np.vstack([traj_batch['rewards'], rewards[np.newaxis, :, np.newaxis]])
                traj_batch['dones']     = np.vstack([traj_batch['dones'], dones[np.newaxis, :, np.newaxis]])
                traj_batch['log_probs'] = torch.vstack([traj_batch['log_probs'], logps.unsqueeze(0)])
                traj_batch['entropies'] = torch.vstack([traj_batch['entropies'], ents.unsqueeze(0)])

                history['ep_len'][-1] +=  np.mean(1-dones)
                history['ep_reward'][-1] += np.mean(rewards)

                history['rew_dist2goal'][-1] += np.mean(infos['rew_dist2goal'])
                history['rew_smooth'][-1] += np.mean(infos['rew_smooth'])
                history['rew_step'][-1] += np.mean(infos['rew_step'])

                states = next_states


            # Update model
            self.update(traj_batch)


            # update histories
            for ienv in range(self.num_envs):
                history['xy'].append(traj_batch['states'][:, ienv, :2])
                history['terminal'].append(infos['reason'][ienv])
            # history['terminal'][-1] = infos['reason']
            history['action'][-1] = np.mean(traj_batch['actions'], axis=0)
            mu, std = np.mean(history['ep_reward']), np.std(history['ep_reward'])
            history['timeseries_filt_rewards'] = np.vstack((history['timeseries_filt_rewards'],
                                                            np.array([[mu, std]])))


            # Rollout Report #############################################

            if ep % report_interval == 0 or ep == max_episodes:

                # Graph reporting
                self._plot_history(traj_history=history['xy'],
                                   terminal_history=history['terminal'],
                                   filt_reward_history=history['timeseries_filt_rewards'])

                # Console report
                mean_epi_reward = np.mean(history['ep_reward'])
                mean_epilen = np.mean(history['ep_len'])
                mean_acts = np.mean(np.array(history['action']), axis=0)
                mean_acts = np.mean(mean_acts, axis=0)
                print(f"Episode {ep} [len={mean_epilen:.1f}], "
                      f"Reward: {mean_epi_reward:.2f} "
                      f"Mean Joy [{mean_acts[0]:.2f} {mean_acts[1]:.2f} ]"
                      f"\t | Shaped Rewards: \t"
                          f"Step: {np.mean(history['rew_step']):.2f} "
                          f"Smooth: {np.mean(history['rew_smooth']):.2f} "
                          f"Dist2goal: {np.mean(history['rew_dist2goal']):.2f} "
                      f"\t | Params:"
                      f"P(randS) = {reset_params['options']['p_rand_state']:.2f}"
                      # f"Dist2goal Rew = {reset_params['options']['reward_dist2goal']:.2f}"
                      f"Reward Dist2goal: {self.env.get_attr('reward_dist2goal')[0]:.2f}"
                      f"")

        if blocking:
            plt.ioff()
            plt.show()


    def select_action(self, state):
        state = self._to_batch(state)
        action, log_prob, entropy = super().select_action(state)
        return action, log_prob.unsqueeze(-1), entropy.unsqueeze(-1)

    def update(self, traj_batch):
        states      = traj_batch['states']
        actions     = traj_batch['actions']
        rewards     = traj_batch['rewards']
        next_states = traj_batch['mext_states']
        dones       = traj_batch['dones']
        log_probs   = traj_batch['log_probs'].squeeze(-1)
        entropies   = traj_batch['entropies'].squeeze(-1)

        values = self.critic(torch.from_numpy(states)).squeeze()
        returns = self.compute_returns(rewards, dones,terminal_return = values[-1])
        returns = torch.FloatTensor(returns).squeeze(-1)

        # Trim values that are after terminal done state and flatten
        first_done = np.argmax(dones, axis=0).flatten()
        values    = torch.concatenate([values[: first_done[i], i] for i in range(self.num_envs)])
        returns   = torch.concatenate([returns[: first_done[i], i] for i in range(self.num_envs)])
        log_probs = torch.concatenate([log_probs[: first_done[i], i] for i in range(self.num_envs)])
        entropies = torch.concatenate([entropies[: first_done[i], i] for i in range(self.num_envs)])

        advantages = returns - values.detach()

        # actor loss
        actor_loss = (-(log_probs * advantages).mean(0))
        actor_loss += - self.entropy_reg * entropies.mean(0)  # entropy loss encourage exploration
        actor_loss = actor_loss.mean() # mean across all parallel environments

        # critic loss
        critic_loss = F.mse_loss(values, returns)

        self.optimizerA.zero_grad(); actor_loss.backward(); self.optimizerA.step()
        self.optimizerC.zero_grad(); critic_loss.backward(); self.optimizerC.step()

    def _to_batch(self, x):
        """Ensure x has shape [B, ...]."""
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, ...]
        return x

    def _clip_to_space(self, actions_t):
        # actions_t: [B, A]
        low = self.action_low.to(actions_t.device)
        high = self.action_high.to(actions_t.device)
        return torch.max(torch.min(actions_t, high), low)