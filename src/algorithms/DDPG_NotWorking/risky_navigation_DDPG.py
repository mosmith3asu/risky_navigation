import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from continuous_nav_env import ContinuousNavigationEnv  # assumes env code is in this module
import matplotlib.pyplot as plt
import multiprocessing

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Actor network\
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, size_hidden_layers = 128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, size_hidden_layers)
        self.fc2 = nn.Linear(size_hidden_layers, size_hidden_layers)
        self.fc3 = nn.Linear(size_hidden_layers, action_dim)
        self.max_action = max_action


        # if isinstance(max_action, dict):
        #     self.max_action = torch.from_numpy( max_action['max_action'])
        #     self.offset_action = torch.from_numpy(max_action['offset_action'])
        # else:
        #     self.max_action = torch.from_numpy(max_action)

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     # x = torch.tanh(self.fc3(x))
    #     # Scale output to action bounds
    #     x = x.clamp(*self.action_bounds)
    #     return x #* self.max_action +self.offset_action
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = (x + torch.tensor([1,0],device=DEVICE)) * torch.tensor([0.5,1.0],device=DEVICE) * self.max_action[1]
        # Scale output to action bounds
        # x = x.clamp(*self.action_bounds)
        return x  # * self.max_action +self.offset_action

# Critic network\
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,size_hidden_layers = 128):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, size_hidden_layers)
        self.fc2 = nn.Linear(size_hidden_layers, size_hidden_layers)
        self.fc3 = nn.Linear(size_hidden_layers, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x = F.relu(self.fc1(xu))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Ornstein-Uhlenbeck process for exploration noise\
class OUNoise(object):
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.1):
        """
        Ornstein-Uhlenbeck process for generating exploration noise.
        :param action_dimension:
        :param mu: mean of what the signal should be
        :param theta: restoration coefficient (how fast noise returns to mu when deviation is large)
        :param sigma: diffusion coefficient (how different noise will make policies)
        """
        self.mu = mu * np.ones(action_dimension)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

# Replay buffer\
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayBuffer(object):
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))
        return batch

    def __len__(self):
        return len(self.buffer)

# Soft update function\
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# DDPG agent\
class DDPGAgent(object):
    def __init__(self, state_dim, action_dim, max_action,min_action, device,
                 lr = 1e-3,tau= 1e-3,gamma=0.97,batch_size=128):
        self.device = device
        self.max_action = max_action
        self.min_action = min_action

        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.1*lr) # 1e-4

        # Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr) # 1e-3

        # Replay buffer and noise
        self.replay_buffer = ReplayBuffer()
        self.noise = OUNoise(action_dim)

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma #0.99
        self.tau = tau#1e-3

    def select_action(self, state, noise=True,noise_gain=1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().cpu().numpy().flatten()
        if noise:
            action += noise_gain*self.noise.sample()
        # return action #np.clip(action, 0.0, max_action['max_action'] + max_action['offset_action'])
        return np.clip(action, self.min_action, self.max_action)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # Sample a batch
        batch = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action = torch.FloatTensor(np.array(batch.action)).to(self.device)
        reward = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            q_target_next = self.critic_target(next_state, next_action)
            q_target = reward + (1 - done) * self.gamma * q_target_next
        q_current = self.critic(state, action)
        critic_loss = F.mse_loss(q_current, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (maximize expected Q)
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def test_rollout(self, q, env, episodes):
        """
        Run deterministic policy for a number of episodes and return rewards.
        """
        episodes = 10
        rewards = []
        trajectories = []
        infos = []
        for _ in range(episodes):
            # self.noise.reset()
            state = env.reset()
            ep_r = 0
            traj = [env.state[:2]]  # Store only position for trajectory
            for t in range(env.max_steps+1):
                # action = self.actor_target(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                # state, r, done, info = env.step(action.cpu().data.numpy().flatten())
                action = self.select_action(state,noise=False) # noise=False
                state, reward, done, info = env.step(action)
                traj.append(state[:2])  # Store only position for trajectory
                ep_r += reward

                if done:
                    terminal_cause = info['reason']
                    break

            rewards.append(ep_r)
            trajectories.append(traj)
            infos.append(terminal_cause)

        q.put((rewards, trajectories, infos))  # Send results to queue

        # return rewards, trajectories, infos

    def async_test(self, env, episodes):
        """
        Launch non-blocking test rollouts in a separate process.
        Returns (process, queue) where queue will receive the rewards list.
        """
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            # target=lambda q, e, num: q.put(self.test_rollout(e, num)),
            target=self.test_rollout,
            args=(queue, env, episodes)
        )
        proc.daemon = True
        proc.start()
        return proc, queue


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

# Training loop
def train(env, test_env,agent, episodes=2500, max_steps=500,test_interval=10,test_episodes=10):
    # rand_start_dur = 2#episodes
    # rew_shape_dur = int(episodes/4)
    rand_start = False

    logger = Logger(env)

    # shaped_reward_sched = np.power(np.linspace(1, 0.1, rew_shape_dur), 1)
    # shaped_reward_sched = 0.1 * shaped_reward_sched

    pending = []  # list of (proc, queue, episode)

    for ep in range(episodes):
        # update schedules
        # rand_start_prog = max(1 - (ep / rand_start_dur), 0.0)
        # rand_start = np.random.choice([True, False], p=[rand_start_prog, 1 - rand_start_prog])
        # env.dist_reward = shaped_reward_sched[min(ep, len(shaped_reward_sched) - 1)]

        # resets

        state = env.reset(random_state=rand_start)  # reset with random state
        agent.noise.reset()

        # -------------------------------------------------
        # Training rollout --------------------------------
        # -------------------------------------------------
        ep_reward = 0
        trajectories = []
        for t in range(max_steps+1):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            agent.update()
            trajectories.append(state[:2])  # Store only position for trajectory
            state = next_state
            ep_reward += reward
            if done:
                terminal_cause = info['reason']
                break

        logger.log_train(ep_reward, trajectories, terminal_cause)

        # -------------------------------------------------
        # Asynchronous testing ----------------------------
        # -------------------------------------------------

        # launch tests
        if ep % test_interval == 0:
            proc, queue = agent.async_test(test_env, test_episodes)
            pending.append((proc, queue, ep))
            print(f"Launched async test at episode {ep}")

        # check pending tests
        for proc, queue, start_ep in pending[:]:
            if not proc.is_alive() or not queue.empty():
                results = queue.get() if not queue.empty() else []
                pending.remove((proc, queue, start_ep))
                print(f'Finished async test for episode {start_ep}')
                if len(results)> 0:
                    for i in range(len(results[0])):
                        rew, tr, terminal_cause = results[0][i], results[1][i], results[2][i]
                        logger.log_test(rew, tr, terminal_cause)



        # mean_acts = np.array(actions_seq).mean(axis=0)
        print(f"Episode {ep}, Reward: {ep_reward:.2f} "
              # f"Mean Throttle {mean_acts[0]:.2f} "
              # f"Mean Steering {mean_acts[1]:.2f} "
              )

        if ep % 5 == 0 and ep != 0:
            logger._plot_history()
    plt.ioff()
    plt.show()
    return None


# def train(env, agent, episodes=500, max_steps=1000):
#     rand_start_dur = 500
#     rew_shape_dur = 500
#
#     logger = Logger(env)
#
#     shaped_reward_sched = np.power(np.linspace(1, 0.1, rew_shape_dur),2)
#     shaped_reward_sched = 3* shaped_reward_sched
#
#     pending = []  # list of (proc, queue, episode)
#     scores = []
#     for ep in range(episodes):
#         # schedules
#         rand_start_prog = max(1-(ep/rand_start_dur)**2,0.0)
#         rand_start = np.random.choice([True,False],p=[rand_start_prog, 1-rand_start_prog])
#         env.dist_reward = shaped_reward_sched[min(ep, len(shaped_reward_sched) - 1)]
#
#         # resets
#         state = env.reset(random_state=rand_start)  # reset with random state
#         agent.noise.reset()
#
#
#         actions_seq = []
#         states_seq = []
#         ep_reward = 0
#         for t in range(max_steps):
#             action = agent.select_action(state)
#             next_state, reward, done, info = env.step(action)
#             agent.replay_buffer.push(state, action, reward, next_state, float(done))
#             agent.update()
#
#             states_seq.append(state[:2])
#             actions_seq.append(action)
#             state = next_state
#             ep_reward += reward
#             if done:
#                 terminal_cause = info['reason']
#                 break
#         scores.append(ep_reward)
#
#         # record history
#         logger.log(ep_reward, states_seq, terminal_cause)
#
#         mean_acts = np.array(actions_seq).mean(axis=0)
#         print(f"Episode {ep}, Reward: {ep_reward:.2f} "
#               f"Mean Throttle {mean_acts[0]:.2f} "
#               f"Mean Steering {mean_acts[1]:.2f} ")
#
#         if ep % 5 == 0 and ep !=0:
#             logger._plot_history()
#
#     return scores
def main():
    # Initialize environment\ nimport continuous_nav_env
    env = ContinuousNavigationEnv(
        goal=(8, 8),
        max_steps=500,
        obstacles=[{'type': 'circle', 'center': (5, 5), 'radius': 1.0}]
    )
    test_env = copy.deepcopy(env)
    test_env.dist_reward = 0


    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_action = env.action_space.high
    min_action = env.action_space.low
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = DDPGAgent(state_dim, action_dim, max_action=max_action,min_action=min_action, device=device)
    scores = train(env,test_env, agent, episodes=5000, max_steps=env.max_steps)
    env.close()

if __name__ == '__main__':
    main()

