# ---------------------------
# DDPG: Deep Deterministic Policy Gradient
# ---------------------------
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


from utils.file_management import save_pickle, load_pickle, load_latest_pickle, get_algorithm_dir
from learning.utils import OUNoise,OUNoiseBounded, ReplayBuffer, ProspectReplayBuffer, Schedule, \
    pdf_expectation_torch_batch, quantile_expectation

# ===== Networks =====

class ActorDeterministic(nn.Module):
    """
    Deterministic policy: a = tanh(f_theta(s)) scaled to action bounds at call site.
    """
    def __init__(self, state_dim, action_dim,
                 num_hidden_layers=3, size_hidden_layers=128,
                 activation='LeakyReLU'):
        super().__init__()
        Act = getattr(nn, activation)
        layers = [nn.Linear(state_dim, size_hidden_layers), Act()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(size_hidden_layers, size_hidden_layers), Act()]
        layers += [nn.Linear(size_hidden_layers, action_dim)]
        self.net = nn.Sequential(*layers)
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers

    def forward(self, s):
        # output in [-1,1]; caller rescales to env bounds
        return torch.tanh(self.net(s))

    def __repr__(self):
        return f"ActorDeterministic(in:{self.net[0].in_features} -> {self.size_hidden_layers}X{self.num_hidden_layers} -> {self.net[-1].out_features}: out)"


class CriticQ(nn.Module):
    """
    Q(s,a): concatenates [s,a] then regresses a scalar.
    """
    def __init__(self, state_dim, action_dim,
                 num_hidden_layers=3, size_hidden_layers=128,
                 activation='LeakyReLU'):
        super().__init__()
        Act = getattr(nn, activation)
        inp = state_dim + action_dim
        layers = [nn.Linear(inp, size_hidden_layers), Act()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(size_hidden_layers, size_hidden_layers), Act()]
        layers += [nn.Linear(size_hidden_layers, 1)]
        self.net = nn.Sequential(*layers)
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)

    def __repr__(self):
        return f"CriticQ([in:{self.net[0].in_features} -> {self.size_hidden_layers}X{self.num_hidden_layers} -> 1:out)"


# ===== Utilities =====


def soft_update(target, source, tau):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


# ===== Agent (single-env) =====

class DDPGAgent:
    """
    DDPG for continuous control.
    Swap in for A2CAgent. Matches your plotting-friendly train loop structure.
    """
    def __init__(self, env,
                 gamma=0.99,
                 tau=0.005,

                 actor_lr=1e-4,
                 actor_num_hidden_layers = 3,
                 actor_size_hidden_layers = 128,

                 critic_lr=1e-3,
                 critic_num_hidden_layers=3,
                 critic_size_hidden_layers=128,

                 buffer_size=int(1e5),
                 batch_size=256,

                 rollouts_per_epi = 10,
                 warmup_epis=100,
                 random_start_epis = 10,

                 ou_theta=0.15,
                 ou_sigma=0.5,
                 ou_sigma_decay = 0.99,
                 grad_clip=None,
                 device=None,
                 **kwargs):

        self.enable_print_report = True
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.rollouts_per_epi = rollouts_per_epi
        self.warmup_epis = warmup_epis
        self.random_start_epis = random_start_epis
        self.start_time_str = datetime.now().strftime("%Y-%m-%d TIME: %H:%M:%S")
        self.fig_title = kwargs.pop('fig_title', f'DDPG Agent ({self.start_time_str}')


        # dims
        self.state_dim  = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.a_low      = torch.tensor(env.action_space.low , dtype = torch.float32, device = self.device)
        self.a_high     = torch.tensor(env.action_space.high, dtype = torch.float32, device = self.device)

        # nets
        self.actor      = ActorDeterministic(self.state_dim, self.action_dim,
                                              num_hidden_layers=actor_num_hidden_layers,
                                              size_hidden_layers=actor_size_hidden_layers
                                              ).to(self.device)
        self.actor_targ = ActorDeterministic(self.state_dim, self.action_dim,
                                            num_hidden_layers = actor_num_hidden_layers,
                                            size_hidden_layers = actor_size_hidden_layers
                                            ).to(self.device)
        self.actor_targ.load_state_dict(self.actor.state_dict())


        self.critic      = CriticQ(self.state_dim, self.action_dim,
                                   num_hidden_layers=critic_num_hidden_layers,
                                   size_hidden_layers=critic_size_hidden_layers
                                   ).to(self.device)
        self.critic_targ = CriticQ(self.state_dim, self.action_dim,
                                   num_hidden_layers=critic_num_hidden_layers,
                                   size_hidden_layers=critic_size_hidden_layers
                                   ).to(self.device)
        self.critic_targ.load_state_dict(self.critic.state_dict())

        self.optimA = optim.Adam(self.actor.parameters() , lr = actor_lr)
        self.optimC = optim.Adam(self.critic.parameters(), lr = critic_lr)

        self.replay = ReplayBuffer(self.state_dim, self.action_dim, size=buffer_size, device=self.device)
        self.ou = OUNoise(self.action_dim, theta=ou_theta, sigma=ou_sigma,
                          sigma_decay=ou_sigma_decay, n_envs=1, device=self.device)


        # Logging and Schedules
        self.max_episodes = None
        self.total_rollouts = 0
        self.total_steps = 0

        self.history = {
            'ep_len': deque(maxlen=self.rollouts_per_epi),
            'ep_reward': deque(maxlen=self.rollouts_per_epi),
            'action': deque(maxlen=self.rollouts_per_epi),
            'xy': deque(maxlen=self.rollouts_per_epi),
            'terminal': deque(maxlen=self.rollouts_per_epi),
            'timeseries_filt_rewards': np.zeros([0, 2]),
        }

        self.fig,self.axes = None,None
        self.fig_assets = {}
        self.reset_params = {}


    # ----- action helpers -----

    def _scale_to_bounds(self, act_t):
        # act_t in [-1,1] -> env bounds
        return self.a_low + (act_t + 1.0) * 0.5 * (self.a_high - self.a_low)

    def _act(self, s_t, explore=True):
        with torch.no_grad():
            a = self.actor(s_t)
            a = self._scale_to_bounds(a)
            if explore:
                # OU noise + small Gaussian
                a = a + self.ou.sample()
            a = torch.max(torch.min(a, self.a_high), self.a_low)
        return a

    # ----- optimization -----

    def _update_sgd(self, updates=1):
        mean_critic_loss = 0.0
        mean_actor_loss  = 0.0

        for _ in range(updates):
            if len(self.replay) < self.batch_size:
                return
            s, a, r, s2, d = self.replay.sample(self.batch_size)

            with torch.no_grad():
                a2 = self._scale_to_bounds(self.actor_targ(s2))
                q_targ = self.critic_targ(s2, a2)
                y = r + (1.0 - d) * self.gamma * q_targ

            # Critic
            q = self.critic(s, a)
            critic_loss = F.mse_loss(q, y)
            self.optimC.zero_grad()
            critic_loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.optimC.step()

            # Actor (maximize Q(s, pi(s)) -> minimize -Q)
            a_pred = self._scale_to_bounds(self.actor(s))
            actor_loss = -self.critic(s, a_pred).mean()
            self.optimA.zero_grad()
            actor_loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_((self.actor.parameters()), self.grad_clip)
            self.optimA.step()

            # Targets
            soft_update(self.actor_targ, self.actor, self.tau)
            soft_update(self.critic_targ, self.critic, self.tau)

            mean_critic_loss += critic_loss.item()
            mean_actor_loss += actor_loss.item()

        return mean_critic_loss / updates, mean_actor_loss / updates

    # ----- training loop -----

    def train(self, max_episodes=1_000):
        raise NotImplementedError('Needs to be updated')
        self.max_episodes = max_episodes

        ep_returns = deque(maxlen=self.rollouts_per_epi)
        ep_lens = deque(maxlen=self.rollouts_per_epi)

        total_steps = 0
        for ep in range(1, max_episodes + 1):

            s, _ = self.env.reset()

            self.ou.reset()
            self.history['ep_len'].append(0)
            self.history['ep_reward'].append(0.0)
            self.history['action'].append(np.array([0, 0]))
            self.history['xy'].append(s[np.newaxis, :2])  # initial state for xy tracking
            self.history['terminal'].append('')


            ep_ret = 0.0
            ep_len = 0
            done = False
            max_ep_len = self.env.get_attr('max_steps')[0]

            while not done:
                total_steps += 1
                ep_len += 1


                s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                a = self._act(s_t, explore=True)
                a_np = a.cpu().numpy()

                # s2, r, terminated, truncated, info = self.env.step(a_np.squeeze(0))
                # done = bool(terminated or truncated or info.get('done', False))

                s2, r, done, info = self.env.step(a_np.squeeze(0))
                self.replay.add(s_t.cpu().numpy(), a_np, np.array([r], dtype=np.float32),
                                np.array(s2, dtype=np.float32), np.array([done], dtype=np.float32))

                self.history['ep_len'][-1] += 1
                self.history['ep_reward'][-1] += r
                self.history['xy'][-1] = np.vstack([self.history['xy'][-1],
                                                    s2[np.newaxis, :2]])  # track xy positions

                s = s2
                ep_ret += r

                if total_steps >= self.update_after and total_steps % self.update_every == 0:
                    self._update_sgd(updates=self.update_every)

                if done:
                    terminal_cause = info['reason']
                    self.history['terminal'][-1] = terminal_cause
                    break



            ep_returns.append(ep_ret)
            ep_lens.append(ep_len)

            mu, std = np.mean(self.history['ep_reward']), np.std(self.history['ep_reward'])
            self.history['timeseries_filt_rewards'] = np.vstack((self.history['timeseries_filt_rewards'], np.array([[mu, std]])))

            if ep % self.rollouts_per_epi == 0:
                print(f"[DDPG] Episode {ep} | AvgRet: {np.mean(ep_returns):.2f} | "
                      f"AvgLen: {np.mean(ep_lens):.1f} | Steps: {total_steps}")

                self._plot_history(traj_history=self.history['xy'],
                                   terminal_history=self.history['terminal'],
                                   filt_reward_history=self.history['timeseries_filt_rewards'], )

        return {"avg_return": np.mean(ep_returns), "avg_len": np.mean(ep_lens)}

    def _spawn_figure(self,max_title_chars = 100):
        from textwrap import wrap

        plt.ion()

        # Create a figure
        self.fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        self.fig.suptitle(
            '\n'.join(wrap(self.fig_title + f' ({self.start_time_str})', max_title_chars))
            # self.fig_title + f' ({self.start_time_str})'
        )

        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 0.75], figure=self.fig)

        # Left column: 3 stacked axes
        self.rew_ax    = self.fig.add_subplot(gs[0, 0])
        self.loss_ax = self.fig.add_subplot(gs[1, 0])
        self.sched_ax   = self.fig.add_subplot(gs[2, 0])

        # format last axis differently
        for ax in [self.rew_ax, self.loss_ax]:
            ax.set_xticks([])
            ax.set_xticklabels([])
        self.sched_ax.set_xlabel('Episode')

        # Right column: single axis spanning all rows
        self.render_ax = self.fig.add_subplot(gs[:, 1])
        self.render_ax.set_xticks([])
        self.render_ax.set_xticklabels([])
        self.render_ax.set_yticks([])
        self.render_ax.set_yticklabels([])

        # Adjust layout
        # plt.tight_layout()
        plt.show()

    def _plot_history(self,
                      traj_history, terminal_history,
                      filt_reward_history,
                      env = None,
                      lw = 0.5,
                      tpause=0.1,
                      log_interval=1):

        env = env or self.env

        if self.fig is None:
            self._spawn_figure()

        x = np.arange(len(filt_reward_history)) * log_interval
        mean, std = filt_reward_history[:,0],filt_reward_history[:,1]

        # Reward plot
        if 'reward_line' not in self.fig_assets.keys():
            # self.fig_assets['reward_line'] = self.axes[0].plot(list(self.reward_history),lw=1,color='b')[0]
            self.fig_assets['reward_line'] = self.rew_ax.plot(x,mean, lw=lw, color='k')[0]
            # self.rew_ax.set_title('Reward (eps)')
            # self.rew_ax.set_xlabel('Episode')
            self.rew_ax.set_ylabel('Total Reward')

            # plot 1 std deviation as shaded region
            self.fig_assets['reward_patch'] = self.rew_ax.fill_between( x, mean - std, mean + std,
                color='b',  alpha=0.2
                                                                         )


        else:
            self.fig_assets['reward_line'].set_data(x, mean)
            self.fig_assets['reward_patch'].remove()
            self.fig_assets['reward_patch'] = self.rew_ax.fill_between(
                x,
                mean - std,
                mean + std,
                color='b',
                alpha=0.2

            )
            self.rew_ax.relim()
            self.rew_ax.autoscale_view()


        # # Trajectories
        if 'traj_lines' not in self.fig_assets.keys():
            env.reset(**self.reset_params)
            if hasattr(env, 'num_envs'):
                env.call('render', ax=self.render_ax) # retrieve vec render
            else:

                env.render(ax=self.render_ax,draw_dist2goal=False, draw_lidar=False) # single env render

            self.fig_assets['traj_lines'] = []
            for (traj,term) in zip(traj_history,terminal_history):
                if term == 'goal_reached': plt_params = {'color':'green', 'alpha': 0.7,  'lw':2*lw}
                elif 'collision' in term: plt_params = {'color':'red', 'alpha': 0.7,  'lw':lw}
                elif term == 'max_steps': plt_params = {'color':'gray', 'alpha': 0.7,  'lw':lw}
                else: raise ValueError(f"Unknown terminal cause: {term}")
                self.fig_assets['traj_lines'].append(self.render_ax.plot(traj[:,0], traj[:,1], **plt_params)[0])

            self.render_ax.set_title('Trajectories (last {} eps)'.format(len(traj_history)))
            # self.render_ax.set_xlabel('X'); self.render_ax.set_ylabel('Y')
        else:
            for i, traj, term in zip(np.arange(len(traj_history)),traj_history, terminal_history):
                if term == 'goal_reached':  plt_params = {'color': 'green', 'alpha': 0.7,'lw':2*lw}
                elif 'collision' in term: plt_params = {'color': 'red', 'alpha': 0.7,'lw':lw}
                elif term == 'max_steps':  plt_params = {'color': 'gray', 'alpha': 0.7,'lw':lw}
                else: raise ValueError(f"Unknown terminal cause: {term}")

                self.fig_assets['traj_lines'][i].set_data(traj[:, 0], traj[:, 1])
                self.fig_assets['traj_lines'][i].set_color(plt_params['color'])
                self.fig_assets['traj_lines'][i].set_alpha(plt_params['alpha'])

        # plt.tight_layout()
        plt.draw()
        plt.pause(tpause)

    def _plot_losses(self, actor_loss_history, critic_loss_history, tpause=0.1, log_interval=1,lw=0.5):

        if self.fig is None:
            self._spawn_figure()

        # Plot a dual axis loss plot
        x = np.arange(len(actor_loss_history)) * log_interval
        if 'actor_loss_line' not in self.fig_assets.keys():
            self.fig_assets['actor_loss_line'], = self.loss_ax.plot(x, actor_loss_history, color='b', label='Actor Loss', lw=lw)
            # self.loss_ax.set_xlabel('Episode')
            self.loss_ax.set_ylabel('Actor Loss', color='b')
            self.loss_ax.tick_params(axis='y', labelcolor='b')

            self.loss_ax2 = self.loss_ax.twinx()
            self.fig_assets['critic_loss_line'], = self.loss_ax2.plot(x, critic_loss_history, color='r', label='Critic Loss', lw=lw)
            self.loss_ax2.set_ylabel('Critic Loss', color='r')
            self.loss_ax2.tick_params(axis='y', labelcolor='r')

            # self.loss_ax.set_title('Losses (eps)')
        else:
            self.fig_assets['actor_loss_line'].set_data(x, actor_loss_history)
            self.fig_assets['critic_loss_line'].set_data(x, critic_loss_history)
            self.loss_ax.relim()
            self.loss_ax.autoscale_view()
            self.loss_ax2.relim()
            self.loss_ax2.autoscale_view()


        # plt.tight_layout()
        plt.draw()
        plt.pause(tpause)

    def _plot_schedules(self, tpause=0.1, log_interval=1,lw=0.5,  **kwargs):

        assert len(kwargs) <= 2, "Can only plot up to 2 schedules at once."
        max_val = 0
        min_val = 0
        for key, val in kwargs.items():
            x = np.arange(len(val)) * log_interval
            max_val = max(np.max(val), max_val)
            min_val = min(np.min(val), min_val)
            if key not in self.fig_assets.keys():
                self.fig_assets[key], = self.sched_ax.plot(x, val, label=key,lw=lw)
                self.sched_ax.legend(loc ='upper right')
                self.sched_ax.set_ylabel('Schedules')
            else:
                self.fig_assets[key].set_data(x, val)

        self.sched_ax.set_xlim(0, x[-1])
        self.sched_ax.set_ylim(min_val*0.9, max_val*1.1)
        # plt.tight_layout()
        plt.draw()
        plt.pause(tpause)

    def _plot_spin(self):
        """Call frequently to unfreeze plot """
        if self.fig is not None:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _get_fpath(self, fname=None, prefix='', suffix= '', save_dir = './models/', with_dir = True, with_tstamp=True):
        assert not (fname is not None and suffix == ''), "Provide either a whole fname or a suffix, not both."
        # Format filename

        if fname is None:
            layout =  self.env.layout if hasattr(self.env, 'layout') else self.env.get_attr('layout')[0]
            tstamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if with_tstamp else ''
            fname = f'{tstamp}_{self.__class__.__name__}_{layout}'

        fname = f'{prefix}_{fname}' if prefix != '' else fname
        fname = f'{fname}_{suffix}' if suffix != '' else fname
        return save_dir + fname

    def _get_fdata(self):
        return {
            'actor': copy.deepcopy(self.actor.state_dict()),
            'critic': copy.deepcopy(self.critic.state_dict()),
            'actor_targ': copy.deepcopy(self.actor_targ.state_dict()),
            'critic_targ': copy.deepcopy(self.critic_targ.state_dict()),
        }

    def save(self, prefix='',suffix='' ):
        data = self._get_fdata()
        fpath = self._get_fpath(prefix=prefix,suffix=suffix)
        save_pickle(data, fpath)

        # save fig as well
        if self.fig is not None:
            fig_fpath = fpath.replace('.pkl', '.png')
            self.fig.savefig(fig_fpath)

        print(f"\nObject saved to {fpath}\n")

    def load(self, fpath):
        if fpath.lower() == 'latest':
            fname = self._get_fpath(save_dir = '', with_tstamp=False)
            models_dir = get_algorithm_dir() + 'models/'
            load_dict, fpath = load_latest_pickle(models_dir,base_fname=fname)
        else:
            load_dict = load_pickle(fpath)

        self.actor.load_state_dict(load_dict['actor'])
        self.critic.load_state_dict(load_dict['critic'])
        self.actor_targ.load_state_dict(load_dict['actor_targ'])
        self.critic_targ.load_state_dict(load_dict['critic_targ'])
        print(f"\nObject loaded from {fpath}\n")

    def __repr__(self):
        s = f"{self.__class__.__name__}({self.start_time_str})"
        s += f"\n\t| Device: {self.device}"
        s += f"\n\t| Actor: {self.actor}"
        s += f"\n\t| Critic: {self.critic}"
        s += f"\n\t| dt: {self.env.dt}"
        return s


# ===== Agent (vectorized env) =====


class DDPGAgent_EUT(DDPGAgent):
    """
    Rational (Expected unitlity) agent that computes the expected utility over a distribution of rewards from dynamics models.
    """
    def __init__(self, env, abstract_overrides = {}, **kwargs):
        super().__init__(env, **kwargs)
        del self.rollouts_per_epi
        del self.history



        self.fig_title = kwargs.pop('fig_title', f'DDPG-EUT Agent ({self.start_time_str})')
        self.delay_steps = self.env.delay_steps
        self.update_interval = kwargs.pop('update_interval', 1)


        self.replay = ProspectReplayBuffer(self.state_dim,
                                           self.action_dim,
                                           n_samples=self.env.n_samples,
                                           size=self.replay.size, device=self.device)
        ou_noise_params = {
            'shape': self.action_dim,
            'mu': 0.0,
            'theta': 0.15,
            'sigma': 0.5,
            'sigma_decay': 0.99,
            'sigma_min': 0.05,
            'dt': self.env.dt,
            'low': (-0.5, -0.5),
            'high': (0.5, 0.75), # bias to drive forward
            # 'low': -1.0,
            # 'high': 1.0,
            'mode': "squash",
            'device': self.device,
            'decay_freq': None # (decay on reset)

        }
        ou_noise_params.update(kwargs.pop('ou_noise', {}))
        # self.ou = OUNoiseBounded(**ou_noise_params)
        self.ou = OUNoise(**ou_noise_params)

        # self.ou.sigma_decay = 0.995
        # self.ou.simulate()

        def_rand_start_sched_params = {
            'xstart': 1,
            'xend': 0,
            'horizon': self.random_start_epis,
            'mode': 'linear',
            # 'mode': 'exponential',
            # 'exp_gain': 2, # higher gain=faster decay
        }
        rand_start_sched_params = kwargs.pop('rand_start_sched', def_rand_start_sched_params)
        rand_start_sched_params.update({'name':'rand_start_sched'})
        self.rand_start_sched = Schedule(**rand_start_sched_params)

        self.log_interval = kwargs.pop('log_interval', 10)
        self.history = {
            'ep_len': deque(maxlen=self.log_interval),
            'ep_reward': deque(maxlen=self.log_interval),
            'action': deque(maxlen=self.log_interval),
            'xy': deque(maxlen=self.log_interval),
            'terminal': deque(maxlen=self.log_interval),
            'timeseries_filt_rewards': np.zeros([0, 2]),
            'critic_loss': [],
            'actor_loss': [],
            'ou_sigma': [],
            'rand_start': []
        }


        self.reset_params = {
            'rand_action_gen': self.rand_action_gen
        }


        self.set_global_overrides(**abstract_overrides) # set any top level overrides


    def set_global_overrides(self, **kwargs):
        # Parse remaining kwargs and set attributes
        obj_list = [self, self.env,self.env.state, self.env.state.robot, self.ou]
        for key, val in kwargs.items():

            is_found = False
            for _obj in obj_list:
                # _k = key.split('.')[0] if '.' in key else key


                if hasattr(_obj, key):
                    setattr(_obj, key, val)
                    is_found = True


            if not is_found:
                raise ValueError(f"Unknown kwarg {key} in ContinousNavEnv.")


    def rand_action_gen(self,*args, **kwargs):
        self.ou.reset()
        a = self.ou.sample(*args, **kwargs).cpu().numpy()
        self.ou.reset()
        return a

    def warmup(self):
        _spin = ['-','|','\\','-','/']

        for ep in range(self.warmup_epis):
            print(f'\r [{_spin[ep%len(_spin)]}] Running Warmup... [Prog: {int(ep/self.warmup_epis*100)}%]',end='',flush=True)

            # RESETS
            p = self.rand_start_sched._x0 if self.random_start_epis>0 else 0
            self.env.reset(p_rand_state = p, **self.reset_params)
            self.ou.reset()


            i=0
            while i <= self.env.max_steps:
                o1 = self.env.observation

                # Step envs ############################################
                a_pt = self.ou.sample()
                # a = np.random.uniform(self.a_low.cpu().numpy(), self.a_high.cpu().numpy(),size=(self.action_dim)).astype(np.float32)
                ns_samples, r_samples, done_samples, info = self.env.step(a_pt.cpu().numpy().flatten())

                o2 = self.env.observation

                # Create transition prospects and store in mem ################################
                r_probs = np.array(self.env.robot_probs)
                r_prospects = np.vstack([r_samples, r_probs])

                self.replay.add(o1, a_pt, r_prospects, o2, done_samples)

                i += 1
                if info['true_done']:
                    break
        print('\n')
        self.ou.reset_sigma()

    def train(self, max_episodes=1_000):
        self.max_episodes = max_episodes
        self.warmup()

        running_rewards = 0
        running_xys = deque(maxlen=self.env.max_steps)
        running_acts = deque(maxlen=self.env.max_steps)
        running_actor_loss = deque(maxlen=self.env.max_steps)
        running_critic_loss = deque(maxlen=self.env.max_steps)

        for ep in range(1, max_episodes + 1):
            self._plot_spin()
            p_rand_state = self.rand_start_sched.sample(at=ep)
            # p_rand_state = max(0, (1 - ep / self.random_start_epis) if self.random_start_epis>0 else 0)
            self.env.reset(p_rand_state = p_rand_state,**self.reset_params)
            self.ou.reset()

            # ----------------------------------------------------------------------------
            for i in range(self.env.max_steps):
                # Observe and act ############################################
                o1 = self.env.observation
                o1_pt = torch.tensor(o1, dtype=torch.float32, device=self.device)

                a1_pt = self._act(o1_pt, explore=True)
                a1 = a1_pt.cpu().numpy().flatten()

                # Step env ############################################
                ns_samples, r_samples, done_samples, info = self.env.step(a1)
                o2 = self.env.observation
                o2_pt = torch.tensor(o2, dtype=torch.float32, device=self.device)

                # Create transition prospects, store in mem, and update ################################
                r_probs = np.array(self.env.robot_probs)
                r_prospects = np.vstack([r_samples, r_probs])
                self.replay.add(o1_pt, a1_pt, r_prospects, o2_pt, done_samples)

                # self._update_sgd()
                if i % self.update_interval == 0:
                    critic_loss,actor_loss = self._update_sgd()
                    running_actor_loss.append(actor_loss)
                    running_critic_loss.append(critic_loss)


                # Log for reporting #########################################################
                running_rewards += info['true_reward']
                running_xys.append(info['true_next_state'][0,:2])
                running_acts.append(a1)

                if info['true_done']:
                    self.history['terminal'].append(info['true_reason'])
                    self.history['ep_len'].append(i)
                    self.history['ep_reward'].append(running_rewards);  running_rewards = 0
                    self.history['xy'].append(np.array(running_xys));   running_xys.clear()
                    self.history['action'].append(np.mean(np.array(running_acts),axis=0)); running_acts.clear()
                    self.history['actor_loss'].append(np.mean(np.array(running_actor_loss)) if len(running_actor_loss)>0 else 0.0)
                    self.history['critic_loss'].append(np.mean(np.array(running_critic_loss)) if len(running_critic_loss)>0 else 0.0)
                    self.history['ou_sigma'].append(np.mean(self.ou.sigma))
                    self.history['rand_start'].append(p_rand_state)

                    running_actor_loss.clear()
                    running_critic_loss.clear()


                    if info['true_reason'] == 'max_steps':
                        self.env.state.add_random_robot_state(
                            observation = info['true_next_state']
                        )
                    break

                # o1    = o2.copy()
                # o1_pt = o2_pt

                # end timestep ---------------------------------------------------------------


            # LOG EPISODE #################################################################
            if self.enable_print_report:
                print(f"[DDPG] Episode {ep} ({i} it; {i * self.env.dt:.1f}sec)"
                      f"| AvgRet: {np.mean(self.history['ep_reward']):.2f} "
                      f"| MemLen: {len(self.replay)}"
                      f"| P(rand state):{p_rand_state:.2f}"
                      f"| OU(sigma): {np.mean(self.ou.sigma): .3f}"
                      f"| Terminal: {self.history['terminal'][-1]}"
                      f"| Mean Act: {np.round(self.history['action'][-1],2)}"
                      )

            if ep % self.log_interval == 0:
                if self.enable_print_report: print("")

                mu, std = np.mean(self.history['ep_reward']), np.std(self.history['ep_reward'])
                self.history['timeseries_filt_rewards'] = np.vstack((
                    self.history['timeseries_filt_rewards'],
                    np.array([[mu, std]]))
                )

                self._plot_history(traj_history=self.history['xy'],
                                   terminal_history=self.history['terminal'],
                                   filt_reward_history=self.history['timeseries_filt_rewards'],
                                   log_interval=self.log_interval)
                self._plot_losses(
                    actor_loss_history=self.history['actor_loss'],
                    critic_loss_history=self.history['critic_loss'],
                    log_interval=1
                )
                self._plot_schedules( ou_sigma = self.history['ou_sigma'],
                                    rand_start = self.history['rand_start'],
                                     log_interval=1
                                   )

                self.history['terminal'].clear()
                self.history['ep_len'].clear()
                self.history['ep_reward'].clear()
                self.history['xy'].clear()


        # return {"avg_return": np.mean(ep_returns), "avg_len": np.mean(ep_lens)}

        # ----- optimization -----

    def risk_measure(self,vals,probs):
        """Computes rational expectation from vals and pdf (probs)."""
        # return pdf_expectation_torch_batch(vals, probs)
        expectations= quantile_expectation(vals)
        return torch.tensor(expectations, dtype=torch.float32, device=self.device)


    def _update_sgd(self, updates=1):
        mean_critic_loss = 0.0
        mean_actor_loss = 0.0

        for _ in range(updates):
            if len(self.replay) < self.batch_size:
                return 0,0

            o1, a1, r_prospects, o2, d_prospects = self.replay.sample(self.batch_size)

            with torch.no_grad():
                a2 = self._scale_to_bounds(self.actor_targ(o2))
                q_targ = self.critic_targ(o2, a2)

                r_vals,r_probs = [r_prospects[:,i,:] for i in range(2)]
                # assert torch.all(r_probs <= 1), 'Reward probabilities should be <= 1 (not necessarily sum to 1).'

                td_targets = r_vals + (1.0 - d_prospects.squeeze(1)) * self.gamma * q_targ
                td_expectation = self.risk_measure(td_targets, r_probs)

                assert td_expectation.shape == (self.batch_size,)
                assert torch.all(torch.isfinite(td_expectation)), f'Non-finite td_expectation'
                # assert np.all(np.isfinite(td_expectation.item())), f'Non-finite critic loss: {td_expectation.item()}'


                # y = r + (1.0 - d) * self.gamma * q_targ

            # Critic ##############################################################
            q = self.critic(o1, a1).squeeze()

            critic_loss = F.mse_loss(q, td_expectation)
            assert np.isfinite(critic_loss.item()), f'Non-finite critic loss: {critic_loss.item()}'

            self.optimC.zero_grad()
            critic_loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.optimC.step()

            # Actor (maximize Q(s, pi(s)) -> minimize -Q) ###############################
            a_pred = self._scale_to_bounds(self.actor(o1))
            actor_loss = -self.critic(o1, a_pred).mean()
            assert np.isfinite(actor_loss.item()), f'Non-finite actor loss: {actor_loss.item()}'

            self.optimA.zero_grad()
            actor_loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_((self.actor.parameters()), self.grad_clip)
            self.optimA.step()

            # Targets
            soft_update(self.actor_targ, self.actor, self.tau)
            soft_update(self.critic_targ, self.critic, self.tau)

            mean_critic_loss += critic_loss.item()
            mean_actor_loss += actor_loss.item()

        return mean_critic_loss / updates, mean_actor_loss / updates

    def __str__(self):
        s = super().__str__()
        s += str(self.ou) + f'\n'
        s += str(self.rand_start_sched) + f'\n'
        s += str(self.env) + f'\n'
        return s

class DDPGAgentVec(DDPGAgent):
    """
    Vectorized DDPG. Mirrors your A2CAgentVec patterns (env is Sync/AsyncVectorEnv).
    NOT DELAYED!!
    """
    def __init__(self, envs, **kwargs):
        super().__init__(envs, **kwargs)


        self.num_envs = self.env.num_envs
        self.single_state_dim = self.env.single_observation_space.shape[0]
        self.single_action_dim = self.env.single_action_space.shape[0]
        self.a_low = torch.tensor(self.env.single_action_space.low, dtype=torch.float32, device=self.device)
        self.a_high = torch.tensor(self.env.single_action_space.high, dtype=torch.float32, device=self.device)


        # reinit networks with single dims
        self.state_dim = self.single_state_dim
        self.action_dim = self.single_action_dim
        self.actor = ActorDeterministic(self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticQ(self.state_dim, self.action_dim).to(self.device)
        self.actor_targ = ActorDeterministic(self.state_dim, self.action_dim).to(self.device)
        self.critic_targ = CriticQ(self.state_dim, self.action_dim).to(self.device)
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())
        self.optimA = optim.Adam(self.actor.parameters(), lr=self.optimA.param_groups[0]['lr'])
        self.optimC = optim.Adam(self.critic.parameters(), lr=self.optimC.param_groups[0]['lr'])
        self.replay = ReplayBuffer(self.state_dim, self.action_dim, size=self.replay.size, device=self.device)
        self.ou = OUNoise(self.action_dim,
                          theta=self.ou.theta, sigma=self.ou.sigma,
                          sigma_decay=self.ou.sigma_decay,
                          n_envs=self.num_envs, device=self.device)



    def warmup(self):
        _spin = ['-','|','\\','-','/']
        warmup_steps = 0
        warmup_rollouts = 0

        self.env.set_attr('p_rand_state', 0 if self.random_start_epis == 0 else 1)
        self.env.set_attr('enable_reset', True)
        states, _ = self.env.reset()
        self.env.set_attr('enable_reset', False)

        for ep in range(self.warmup_epis):
            print(f'\r [{_spin[ep%len(_spin)]}] Running Warmup... [Prog: {int(ep/self.warmup_epis*100)}%]',end='',flush=True)
            this_epi_rollouts = 0

            while this_epi_rollouts <= self.rollouts_per_epi:

                warmup_steps += 1

                # Transition
                actions = np.random.uniform(self.a_low.cpu().numpy(), self.a_high.cpu().numpy(),
                                                size=(1, self.num_envs, self.action_dim)).astype(np.float32)
                next_states, rewards, terminated, truncated, info = self.env.step(actions.squeeze(0))
                dones = info['done']

                # add each transition
                self.replay.add(states, actions, rewards.reshape(-1, 1), next_states, dones.reshape(-1, 1))

                # Check dones
                for i in range(self.num_envs):
                    if dones[i]:
                        warmup_rollouts += 1
                        this_epi_rollouts += 1

                self.env.set_attr('enable_reset', list(dones))
                next_states, _ = self.env.reset()
                self.env.set_attr('enable_reset', False)
        print('\n')
        self.env.set_attr('p_rand_state', 0)

    def train(self, max_episodes=1_000):
        self.max_episodes = max_episodes
        self.warmup()

        self.env.set_attr('enable_reset', True)
        states, _ = self.env.reset()
        self.env.set_attr('enable_reset', False)
        self.ou.reset()


        max_steps = self.env.get_attr('max_steps')[0]
        running_lens = np.zeros(self.num_envs)
        running_rewards = np.zeros(self.num_envs)
        running_xys = [deque(maxlen=max_steps) for _ in range(self.num_envs)]

        for ep in range(1, max_episodes + 1):
            this_epi_rollouts = 0
            p_rand_start = max(0, (1 - ep / self.random_start_epis) if self.random_start_epis>0 else 0)
            self.env.set_attr('p_rand_state',p_rand_start)

            # ----------------------------------------------------------------------------
            for _ in range(self.rollouts_per_epi * max_steps):
                self.total_steps += 1

                # Transition
                states_pt = torch.tensor(states, dtype=torch.float32, device=self.device).unsqueeze(0)
                actions = self._act(states_pt, explore=True).cpu().numpy()
                next_states, rewards, terminated, truncated, info = self.env.step(actions.squeeze(0))
                dones = info['done']

                # add each transition
                self.replay.add(states, actions, rewards.reshape(-1, 1), next_states, dones.reshape(-1, 1))

                # Log for reporting
                running_lens += 1
                running_rewards += rewards
                [running_xys[i].append(next_states[i,:2]) for i in range(self.num_envs)]

                for i in range(self.num_envs):
                    if dones[i]:
                        self.total_rollouts += 1
                        this_epi_rollouts += 1

                        # Store in history and reset buffers
                        self.history['terminal'].append(info['reason'][i])
                        self.history['ep_len'].append(running_lens[i]);         running_lens[i] = 0
                        self.history['ep_reward'].append(running_rewards[i]);   running_rewards[i] = 0
                        self.history['xy'].append(np.array(running_xys[i]));    running_xys[i].clear()
                        self.ou.reset(i=i)

                # Update
                self._update_sgd()

                # Reset terminal envs
                self.env.set_attr('enable_reset', list(dones))
                next_states, _ = self.env.reset()
                self.env.set_attr('enable_reset', False)

                # End timestep
                states = next_states

                # Break sync env loop if we have enough values to report on
                if this_epi_rollouts >= self.rollouts_per_epi:
                    break
                # ----------------------------------------------------------------------------

            mu, std = np.mean(self.history['ep_reward']), np.std(self.history['ep_reward'])
            self.history['timeseries_filt_rewards'] = np.vstack((
                self.history['timeseries_filt_rewards'],
                np.array([[mu, std]]))
            )
            print(f"[DDPG] Episode {ep} "
                  f"| AvgRet: {np.mean(self.history['ep_reward']):.2f} "
                  f"| AvgLen: {np.mean(self.history['ep_len']):.1f}"
                  # f"| Steps: {self.total_steps}"
                  # f"| Rollouts: {self.total_rollouts}"
                  f"| MemLen: {len(self.replay)}"
                  f"| P(rand state):{ self.env.get_attr('p_rand_state')[0]:.2f}"
                  f"| OU(sigma): {self.ou.sigma: .3f}"
                  )

            self.env.set_attr('p_rand_state', 0)
            self.env.set_attr('enable_reset', True)
            self._plot_history(traj_history=self.history['xy'],
                               terminal_history=self.history['terminal'],
                               filt_reward_history=self.history['timeseries_filt_rewards'], )
            self.env.set_attr('enable_reset', False)

            self.history['terminal'].clear()
            self.history['ep_len'].clear()
            self.history['ep_reward'].clear()
            self.history['xy'].clear()


        # return {"avg_return": np.mean(ep_returns), "avg_len": np.mean(ep_lens)}


############################################################################################
# Deprecated: Vectorized DDPG Agent ########################################################
############################################################################################

# class DDPGAgent_EUT(DDPGAgent):
#     """
#     Rational (Expected unitlity) agent that computes the expected utility over a distribution of rewards from dynamics models.
#     """
#     def __init__(self, base_env, envs, **kwargs):
#         super().__init__(envs, **kwargs)
#         self.base_env = base_env
#         self.base_env.auto_reset = True
#
#         self.num_envs = self.env.num_envs
#         self.delay_steps = self.env.get_attr('delay_steps')[0]
#
#         self.single_state_dim = self.env.single_observation_space.shape[0]
#         self.single_action_dim = self.env.single_action_space.shape[0]
#
#         self.a_low = torch.tensor(self.env.single_action_space.low, dtype=torch.float32, device=self.device)
#         self.a_high = torch.tensor(self.env.single_action_space.high, dtype=torch.float32, device=self.device)
#         # reinit networks with single dims
#         self.state_dim = self.single_state_dim
#         self.action_dim = self.single_action_dim
#         self.actor = ActorDeterministic(self.state_dim, self.action_dim).to(self.device)
#         self.critic = CriticQ(self.state_dim, self.action_dim).to(self.device)
#         self.actor_targ = ActorDeterministic(self.state_dim, self.action_dim).to(self.device)
#         self.critic_targ = CriticQ(self.state_dim, self.action_dim).to(self.device)
#         self.actor_targ.load_state_dict(self.actor.state_dict())
#         self.critic_targ.load_state_dict(self.critic.state_dict())
#         self.optimA = optim.Adam(self.actor.parameters(), lr=self.optimA.param_groups[0]['lr'])
#         self.optimC = optim.Adam(self.critic.parameters(), lr=self.optimC.param_groups[0]['lr'])
#         self.replay = ProspectReplayBuffer(self.state_dim, self.action_dim, n_samples=self.num_envs, size=self.replay.size, device=self.device)
#
#
#         self.ou = OUNoise(self.action_dim,
#                           theta=self.ou.theta, sigma=self.ou.sigma,
#                           sigma_decay=self.ou.sigma_decay,
#                           # n_envs=self.num_envs, # DO NOT USE SINCE WE ONLY TAKE 1 ACTION FOR ALL ENVS
#                           device=self.device)
#
#         # _db = {
#         #     # 'min_lin_vel': (0.0, 0.1),
#         #     # 'max_lin_vel': (1.0, 0.1),
#         #     # 'max_lin_acc': (0.5, 0.1),
#         #     # 'max_rot_vel': (np.pi/2, 0.1)
#         #
#         #     'max_lin_vel': (1.0, 0.5),
#         #     'max_lin_acc': (0.5, 0.5),
#         #     'max_rot_vel': (np.pi / 2, 0.5*np.pi)
#         #
#         #     # 'max_lin_vel': (1.0, 4),
#         #     # 'max_lin_acc': (0.5, 4),
#         #     # 'max_rot_vel': (np.pi / 2, 4 * np.pi)
#         # }
#         # dynamics_belief = kwargs.get('dynamics_belief', _db)
#         # self.env.set_attr('dynamics_belief', dynamics_belief)
#
#         # Fill up delay buffer
#         # for _ in range(100):
#         #     self.env.call('update_delay_buffers',
#         #              current_robot_state=base_env.state[0:4],
#         #              action=np.array([0.0, 0.0], dtype=np.float32))
#
#     def reset_envs(self):
#         self.base_env.enable_reset =  True
#         self.env.set_attr('enable_reset', True)
#         self.base_env.reset()
#         self.env.reset()
#         self.env.set_attr('enable_reset', False)
#         self.base_env.enable_reset =  False
#
#     def warmup(self):
#         _spin = ['-','|','\\','-','/']
#
#         for ep in range(self.warmup_epis):
#             print(f'\r [{_spin[ep%len(_spin)]}] Running Warmup... [Prog: {int(ep/self.warmup_epis*100)}%]',end='',flush=True)
#
#             # RESETS
#             self.env.set_attr('p_rand_state', 0 if self.random_start_epis == 0 else 1)
#             self.reset_envs()
#
#             i=0
#             while i <= self.env.get_attr('max_steps')[0]:
#                 self.env.call('resample_robot')
#
#                 # Get starting observation/state ############################################
#                 prev_state = self.base_env.state.copy()
#                 o1 = self.env.call('observe')[0]  # Uniform starting observation for all possibilities
#
#                 # Step envs ############################################
#                 a = np.random.uniform(self.a_low.cpu().numpy(), self.a_high.cpu().numpy(),size=(self.action_dim)).astype(np.float32)
#                 ns_true, _, done, _ = self.base_env.step(a)  # step the base env for rendering reference
#                 future_state_samples, r_samples, terminated, truncated, info = self.env.step(np.repeat(a[np.newaxis, :], self.num_envs, axis=0))
#
#                 # Ending observation/state ############################################
#                 self.env.call('put_delay_buffers', current_robot_state=prev_state[:4].copy(), action=a)
#                 o2 = self.env.call('observe')[0]  # Consistent across all possibilities give buffer > 1
#
#                 # Create transition prospects and store in mem ################################
#                 r_probs = np.array(self.env.get_attr('robot_prob'))
#                 r_prospects = np.vstack([r_samples, r_probs])
#
#                 self.replay.add(o1, a, r_prospects, o2, done)
#                 # self.replay.add(o1, a, r_prospects, o2_latest, done)
#
#                 i += 1
#                 if done:
#                     break
#
#         print('\n')
#         self.env.set_attr('p_rand_state', 0)
#
#     def train(self, max_episodes=1_000):
#         self.warmup()
#
#         self.reset_envs()
#         self.ou.reset()
#
#
#         max_steps = self.env.get_attr('max_steps')[0]
#         running_rewards = 0
#         running_xys = deque(maxlen=max_steps)
#
#         for ep in range(1, max_episodes + 1):
#             this_epi_rollouts = 0
#             p_rand_start = max(0, (1 - ep / self.random_start_epis) if self.random_start_epis>0 else 0)
#             self.env.set_attr('p_rand_state',p_rand_start)
#             self.reset_envs()
#             self.ou.reset()
#
#             # ----------------------------------------------------------------------------
#             i=0
#             for _ in range(self.rollouts_per_epi * max_steps):
#                 self.env.call('resample_robot')
#
#                 # Get starting observation/state ############################################
#                 prev_state = self.base_env.state.copy()
#                 o1 = self.env.call('observe')[0]  # Uniform starting observation for all possibilities
#                 o1_pt = torch.tensor(o1, dtype=torch.float32, device=self.device)
#
#                 # Step envs ############################################
#                 a_pt = self._act(o1_pt, explore=True)
#                 a = a_pt.cpu().numpy().flatten()
#                 ns_true, r_true, done, info_true = self.base_env.step(a)  # step the base env for rendering reference
#                 future_state_samples, r_samples, terminated, truncated, infos = self.env.step(
#                     np.repeat(a[np.newaxis, :], self.num_envs, axis=0))
#
#                 # Ending observation/state ############################################
#                 self.env.call('put_delay_buffers', current_robot_state=prev_state[:4].copy(), action=a)
#                 o2 = self.env.call('observe')[0]  # Consistent across all possibilities give buffer > 1
#
#                 # Create transition prospects and store in mem ################################
#                 r_probs = np.array(self.env.get_attr('robot_prob'))
#                 r_prospects = np.vstack([r_samples, r_probs])
#
#                 self.replay.add(o1_pt, a_pt, r_prospects, o2, done)
#                 running_rewards += r_true
#                 running_xys.append(ns_true[:2])
#
#                 # Update ###################################################################
#                 self._update_sgd()
#
#                 if done:
#                     # Logging #################################################################
#                     self.history['terminal'].append(info_true['reason'])
#                     self.history['ep_len'].append(i)
#                     self.history['ep_reward'].append(running_rewards);  running_rewards = 0
#                     self.history['xy'].append(np.array(running_xys));   running_xys.clear()
#                     break
#                 i += 1
#
#                 # end timestep ---------------------------------------------------------------
#
#
#             # LOG EPISODE #################################################################
#             print(f"[DDPG] Episode {ep} ({i} it; {i * self.base_env.dt:.1f}sec)"
#                   f"| AvgRet: {np.mean(self.history['ep_reward']):.2f} "
#                   f"| AvgLen: {np.mean(self.history['ep_len']):.1f}"
#                   # f"| Steps: {self.total_steps}"
#                   # f"| Rollouts: {self.total_rollouts}"
#                   f"| MemLen: {len(self.replay)}"
#                   f"| P(rand state):{self.env.get_attr('p_rand_state')[0]:.2f}"
#                   f"| OU(sigma): {self.ou.sigma: .3f}"
#                   f"| Terminal: {self.history['terminal'][-1]}"
#                   )
#
#             if ep % self.rollouts_per_epi == 0:
#                 mu, std = np.mean(self.history['ep_reward']), np.std(self.history['ep_reward'])
#                 self.history['timeseries_filt_rewards'] = np.vstack((
#                     self.history['timeseries_filt_rewards'],
#                     np.array([[mu, std]]))
#                 )
#
#
#                 self.env.set_attr('p_rand_state', 0)
#                 self.env.set_attr('enable_reset', True)
#                 self._plot_history(traj_history=self.history['xy'],
#                                    terminal_history=self.history['terminal'],
#                                    filt_reward_history=self.history['timeseries_filt_rewards'], )
#                 self.env.set_attr('enable_reset', False)
#
#                 self.history['terminal'].clear()
#                 self.history['ep_len'].clear()
#                 self.history['ep_reward'].clear()
#                 self.history['xy'].clear()
#
#
#         # return {"avg_return": np.mean(ep_returns), "avg_len": np.mean(ep_lens)}
#
#         # ----- optimization -----
#
#     def risk_measure(self,vals,probs):
#         return torch.sum(vals*probs, dim=1).squeeze() # rational expectation
#
#     def _update_sgd(self, updates=1):
#         for _ in range(updates):
#             if len(self.replay) < self.batch_size:
#                 return
#             o1, a1, r_prospects, o2, d = self.replay.sample(self.batch_size)
#
#             with torch.no_grad():
#                 a2 = self._scale_to_bounds(self.actor_targ(o2))
#                 q_targ = self.critic_targ(o2, a2)
#
#                 r_vals,r_probs = [r_prospects[:,i,:] for i in range(2)]
#                 td_targets = r_vals + (1.0 - d) * self.gamma * q_targ
#                 td_expectation = self.risk_measure(td_targets, r_probs)
#
#                 # y = r + (1.0 - d) * self.gamma * q_targ
#
#             # Critic
#             q = self.critic(o1, a1).squeeze()
#             critic_loss = F.mse_loss(q, td_expectation)
#             self.optimC.zero_grad()
#             critic_loss.backward()
#             if self.grad_clip:
#                 nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
#             self.optimC.step()
#
#             # Actor (maximize Q(s, pi(s)) -> minimize -Q)
#             a_pred = self._scale_to_bounds(self.actor(o1))
#             actor_loss = -self.critic(o1, a_pred).mean()
#             self.optimA.zero_grad()
#             actor_loss.backward()
#             if self.grad_clip:
#                 nn.utils.clip_grad_norm_((self.actor.parameters()), self.grad_clip)
#             self.optimA.step()
#
#             # Targets
#             soft_update(self.actor_targ, self.actor, self.tau)
#             soft_update(self.critic_targ, self.critic, self.tau)
#
#
