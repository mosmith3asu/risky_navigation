import gc
import logging
import os
import torch

import torch.nn.functional as F
from torch.optim import Adam

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import argparse
import logging
import os
import random
import time

import gym
import numpy as np
# import roboschool
import torch
# from torch.utils.tensorboard import SummaryWriter

import random
from collections import namedtuple
import numpy as np
from math import sqrt
from continuous_nav_env import ContinuousNavigationEnv  # assumes env code is in this module
from collections import deque
import matplotlib.pyplot as plt

class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        """
        Normalizes the actions to be in between action_space.high and action_space.low.
        If action_space.low == -action_space.high, this is equals to action_space.high*action.

        :param action:
        :return: normalized action
        """
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def reverse_action(self, action):
        """
        Reverts the normalization

        :param action:
        :return:
        """
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action
# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
# and adapted to be synchronous with https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OUNoise:
    def __init__(self, action_dimension, dt=0.01, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.dt = dt
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)


def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist
# Taken from
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward')
                        )


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output Layer
        self.mu = nn.Linear(hidden_size[1], num_outputs)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.mu.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.mu.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        mu = torch.tanh(self.mu(x))
        return mu


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        V = self.V(x)
        return V
logger = logging.getLogger('ddpg')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):

    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, checkpoint_dir=None):
        """
        Deep Deterministic Policy Gradient
        Read the detail about it here:
        https://arxiv.org/abs/1509.02971

        Arguments:
            gamma:          Discount factor
            tau:            Update factor for the actor and the critic
            hidden_size:    Number of units in the hidden layers of the actor and critic. Must be of length 2.
            num_inputs:     Size of the input states
            action_space:   The action space of the used environment. Used to clip the actions and 
                            to distinguish the number of outputs
            checkpoint_dir: Path as String to the directory to save the networks. 
                            If None then "./saved_models/" will be used
        """

        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        # Define the actor
        self.actor = Actor(hidden_size, num_inputs, self.action_space).to(device)
        self.actor_target = Actor(hidden_size, num_inputs, self.action_space).to(device)

        # Define the critic
        self.critic = Critic(hidden_size, num_inputs, self.action_space).to(device)
        self.critic_target = Critic(hidden_size, num_inputs, self.action_space).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=1e-4)  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-2
                                     )  # optimizer for the critic network

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Set the directory to save the models
        if checkpoint_dir is None:
            self.checkpoint_dir = "./saved_models/"
        else:
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info('Saving all checkpoints to {}'.format(self.checkpoint_dir))

    def calc_action(self, state, action_noise=None):
        """
        Evaluates the action to perform in a given state

        Arguments:
            state:          State to perform the action on in the env. 
                            Used to evaluate the action.
            action_noise:   If not None, the noise to apply on the evaluated action
        """
        x = state.to(device)

        # Get the continous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(x)
        self.actor.train()  # Sets the actor in training mode
        mu = mu.data

        # During training we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(device)
            mu += noise

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.action_space.low[0], self.action_space.high[0])

        return mu

    def update_params(self, batch):
        """
        Updates the parameters/networks of the agent according to the given batch.
        This means we ...
            1. Compute the targets
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update

        Arguments:
            batch:  Batch to perform the training of the parameters
        """
        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.cat(batch.done).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # TODO: Clipping the expected values here?
        # expected_value = torch.clamp(expected_value, min_value, max_value)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_checkpoint(self, last_timestep, replay_buffer):
        """
        Saving the networks and all parameters to a file in 'checkpoint_dir'

        Arguments:
            last_timestep:  Last timestep in training before saving
            replay_buffer:  Current replay buffer
        """
        checkpoint_name = self.checkpoint_dir + '/ep_{}.pth.tar'.format(last_timestep)
        logger.info('Saving checkpoint...')
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': replay_buffer,
        }
        logger.info('Saving model at timestep {}...'.format(last_timestep))
        torch.save(checkpoint, checkpoint_name)
        gc.collect()
        logger.info('Saved model at timestep {} to {}'.format(last_timestep, self.checkpoint_dir))

    def get_path_of_latest_file(self):
        """
        Returns the latest created file in 'checkpoint_dir'
        """
        files = [file for file in os.listdir(self.checkpoint_dir) if (file.endswith(".pt") or file.endswith(".tar"))]
        filepaths = [os.path.join(self.checkpoint_dir, file) for file in files]
        last_file = max(filepaths, key=os.path.getctime)
        return os.path.abspath(last_file)

    def load_checkpoint(self, checkpoint_path=None):
        """
        Saving the networks and all parameters from a given path. If the given path is None
        then the latest saved file in 'checkpoint_dir' will be used.

        Arguments:
            checkpoint_path:    File to load the model from

        """

        if checkpoint_path is None:
            checkpoint_path = self.get_path_of_latest_file()

        if os.path.isfile(checkpoint_path):
            logger.info("Loading checkpoint...({})".format(checkpoint_path))
            key = 'cuda' if torch.cuda.is_available() else 'cpu'

            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            replay_buffer = checkpoint['replay_buffer']

            gc.collect()
            logger.info('Loaded model at timestep {} from {}'.format(start_timestep, checkpoint_path))
            return start_timestep, replay_buffer
        else:
            raise OSError('Checkpoint not found')

    def set_eval(self):
        """
        Sets the model in evaluation mode

        """
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        """
        Sets the model in training mode

        """
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def get_network(self, name):
        if name == 'Actor':
            return self.actor
        elif name == 'Critic':
            return self.critic
        else:
            raise NameError('name \'{}\' is not defined as a network'.format(name))



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


# Create logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

# Libdom raises an error if this is not set to true on Mac OSX
# see https://github.com/openai/spinningup/issues/16 for more information
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parse given arguments
# gamma, tau, hidden_size, replay_size, batch_size, hidden_size are taken from the original paper
parser = argparse.ArgumentParser()
# parser.add_argument("--env", default="RoboschoolInvertedPendulumSwingup-v1",
#                     help="the environment on which the agent should be trained "
#                          "(Default: RoboschoolInvertedPendulumSwingup-v1)")
parser.add_argument("--render_train", default=False, type=bool,
                    help="Render the training steps (default: False)")
parser.add_argument("--render_eval", default=False, type=bool,
                    help="Render the evaluation steps (default: False)")
parser.add_argument("--load_model", default=False, type=bool,
                    help="Load a pretrained model (default: False)")
parser.add_argument("--save_dir", default="./saved_models/",
                    help="Dir. path to save and load a model (default: ./saved_models/)")
parser.add_argument("--seed", default=0, type=int,
                    help="Random seed (default: 0)")
parser.add_argument("--timesteps", default=1e6, type=int,
                    help="Num. of total timesteps of training (default: 1e6)")
parser.add_argument("--batch_size", default=64, type=int,
                    help="Batch size (default: 64; OpenAI: 128)")
parser.add_argument("--replay_size", default=1e6, type=int,
                    help="Size of the replay buffer (default: 1e6; OpenAI: 1e5)")
parser.add_argument("--gamma", default=0.99,
                    help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.001,
                    help="Update factor for the soft update of the target networks (default: 0.001)")
parser.add_argument("--noise_stddev", default=0.2, type=int,
                    help="Standard deviation of the OU-Noise (default: 0.2)")
parser.add_argument("--hidden_size", nargs=2, default=[400, 300], type=tuple,
                    help="Num. of units of the hidden layers (default: [400, 300]; OpenAI: [64, 64])")
parser.add_argument("--n_test_cycles", default=10, type=int,
                    help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")
args = parser.parse_args()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))

if __name__ == "__main__":

    # Define the directory where to save and load models
    # checkpoint_dir = args.save_dir + args.env
    # writer = SummaryWriter('runs/run_1')

    # Create the env
    # kwargs = dict()
    # if args.env == 'RoboschoolInvertedPendulumSwingup-v1':
    #     # 'swingup=True' must be passed as an argument
    #     # See pull request 'https://github.com/openai/roboschool/pull/192'
    #     kwargs['swingup'] = True
    # env = gym.make(args.env, **kwargs)

    num_episodes = 1000
    env = ContinuousNavigationEnv(
        goal=(8, 8),
        max_steps=500,
        obstacles=[{'type': 'circle', 'center': (5, 5), 'radius': 1.0}]
    )
    env = NormalizedActions(env)
    LOGGER = Logger(env)
    # Define the reward threshold when the task is solved (if existing) for model saving
    # reward_threshold = gym.spec(args.env).reward_threshold if gym.spec(
    #     args.env).reward_threshold is not None else np.inf
    reward_threshold = 100000
    enable_render = False

    # Set random seed for all used libraries where possible
    # env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define and build DDPG agent
    hidden_size = tuple(args.hidden_size)
    agent = DDPG(args.gamma,
                 args.tau,
                 hidden_size,
                 env.observation_space.shape[0],
                 env.action_space,
                 # checkpoint_dir=checkpoint_dir
                 )

    # Initialize replay memory
    memory = ReplayMemory(int(args.replay_size))

    # Initialize OU-Noise
    nb_actions = env.action_space.shape[-1]
    ou_noise = OUNoise(nb_actions) #OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
               #                             sigma=float(args.noise_stddev) * np.ones(nb_actions))

    # Define counters and other variables
    start_step = 0
    # timestep = start_step
    if args.load_model:
        # Load agent if necessary
        start_step, memory = agent.load_checkpoint()
    timestep = start_step // 10000 + 1
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0
    time_last_checkpoint = time.time()

    # Start training
    # logger.info('Train agent on {} env'.format({env.unwrapped.spec.id}))
    # logger.info('Doing {} timesteps'.format(args.timesteps))
    # logger.info('Start at timestep {0} with t = {1}'.format(timestep, t))
    # logger.info('Start training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

    for epoch in range(num_episodes): #args.timesteps:
        ou_noise.reset()
        epoch_return = 0

        state = torch.Tensor([env.reset()]).to(device)
        trajectories = []
        trajectories.append(state[:2].cpu().numpy().flatten())  # Store only position for trajectory
        t= 0
        timestep = 1


        while True:
            if enable_render:#args.render_train:
                env.render()

            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, info = env.step(action.cpu().numpy()[0])
            timestep += 1
            epoch_return += reward

            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)

            memory.push(state, action, mask, next_state, reward)

            state = next_state
            trajectories.append(state[:2].cpu().numpy().flatten())

            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > args.batch_size:
                transitions = memory.sample(args.batch_size)
                # Transpose the batch
                # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
                batch = Transition(*zip(*transitions))

                # Update actor and critic according to the batch
                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

            if done:
                terminal_cause = info['reason']
                break

        print(f'Epi{epoch} R= {epoch_return:.2f}')
        LOGGER.log_train(epoch_return, trajectories, terminal_cause)

        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)
        # writer.add_scalar('epoch/return', epoch_return, epoch)



        # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
        if epoch%10 == 0:
            t += 1

            test_rewards = []
            for _ in range(args.n_test_cycles):
                state = torch.Tensor([env.reset()]).to(device)
                test_reward = 0
                test_traj = []
                test_traj.append(state[:2].cpu().numpy().flatten())
                while True:
                    if args.render_eval:
                        env.render()

                    action = agent.calc_action(state)  # Selection without noise

                    next_state, reward, done, info = env.step(action.cpu().numpy()[0])
                    test_reward += reward

                    next_state = torch.Tensor([next_state]).to(device)

                    state = next_state
                    if done:
                        terminal_cause = info['reason']
                        break
                LOGGER.log_test(test_reward, test_traj, terminal_cause)
                test_rewards.append(test_reward)

            mean_test_rewards.append(np.mean(test_rewards))

            # for name, param in agent.actor.named_parameters():
            #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            # for name, param in agent.critic.named_parameters():
            #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            #
            # writer.add_scalar('test/mean_test_return', mean_test_rewards[-1], epoch)
            logger.info("Epoch: {}, current timestep: {}, last reward: {}, "
                        "mean reward: {}, mean test reward {}".format(epoch,
                                                                      timestep,
                                                                      rewards[-1],
                                                                      np.mean(rewards[-10:]),
                                                                      np.mean(test_rewards)))

            # Save if the mean of the last three averaged rewards while testing
            # is greater than the specified reward threshold
            # TODO: Option if no reward threshold is given
            if np.mean(mean_test_rewards[-3:]) >= reward_threshold:
                agent.save_checkpoint(timestep, memory)
                time_last_checkpoint = time.time()
                logger.info('Saved model at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

        if epoch % 10 == 0 and epoch != 0:
            LOGGER._plot_history()

        # epoch += 1

    agent.save_checkpoint(timestep, memory)
    logger.info('Saved model at endtime {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    logger.info('Stopping training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    env.close()