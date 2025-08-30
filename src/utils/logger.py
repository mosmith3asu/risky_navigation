import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class Logger:
    def __init__(self,env, history_len=10):
        self.env = env
        # self.history = {}
        self.filt_reward_history = np.empty([0,2])  # deque(maxlen=history_len)
        self.filt_window = history_len
        self.history_len = history_len

        self.reward_history = []  # deque(maxlen=history_len)
        self.traj_history = deque(maxlen=history_len)
        self.terminal_history = deque(maxlen=history_len)

        self.fig = None
        self.axes = None
        self.fig_assets = {}

    def log(self,ep_reward,states_seq, **kwargs):

        self.reward_history.append(ep_reward)
        self.traj_history.append(np.array(states_seq))
        # self.terminal_history.append(terminal_cause)

        mu = np.mean(self.reward_history[-min(len(self.reward_history), self.filt_window):])
        std = np.std(self.reward_history[-min(len(self.reward_history), self.filt_window):])
        self.filt_reward_history = np.vstack((self.filt_reward_history, np.array([[mu, std]])))

        #
        # for key, value in kwargs.items():
        #     if key not in self.history:
        #         self.history[key] = []
        #     self.history[key].append(value)


    def draw(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 4))

        x = np.arange(len(self.filt_reward_history))
        mean, std = self.filt_reward_history[:, 0], self.filt_reward_history[:, 1]

        # Reward plot
        if 'reward_line' not in self.fig_assets.keys():
            # self.fig_assets['reward_line'] = self.axes[0].plot(list(self.reward_history),lw=1,color='b')[0]
            self.fig_assets['reward_line'] = self.axes[0].plot(x, mean, lw=1, color='b')[0]
            self.axes[0].set_title('Reward (eps)')
            self.axes[0].set_xlabel('Episode')
            self.axes[0].set_ylabel('Total Reward')

            # plot 1 std deviation as shaded region
            self.fig_assets['reward_patch'] = self.axes[0].fill_between(x, mean - std, mean + std,
                                                                        color='b', alpha=0.2
                                                                        )


        else:
            # x = np.arange(len(self.reward_history))
            # self.fig_assets['reward_line'].set_data(x,list(self.reward_history))
            self.fig_assets['reward_line'].set_data(x, mean)

            self.fig_assets['reward_patch'].remove()
            # self.fig_assets['reward_patch'] = add_patch(self.reward_history)
            self.fig_assets['reward_patch'] = self.axes[0].fill_between(
                x,  # np.arange(len(mean)),
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
            for (traj, term) in zip(self.traj_history, self.terminal_history):
                if term == 'goal_reached':
                    plt_params = {'color': 'green', 'alpha': 0.7}
                elif 'collision' in term:
                    plt_params = {'color': 'red', 'alpha': 0.7}
                elif term == 'max_steps':
                    plt_params = {'color': 'gray', 'alpha': 0.7}
                else:
                    raise ValueError(f"Unknown terminal cause: {term}")
                self.fig_assets['traj_lines'].append(self.axes[1].plot(traj[:, 0], traj[:, 1], **plt_params)[0])

            self.axes[1].set_title('Trajectories (last {} eps)'.format(len(self.traj_history)))
            self.axes[1].set_xlabel('X');
            self.axes[1].set_ylabel('Y')
        else:
            for i, traj, term in zip(np.arange(len(self.traj_history)), self.traj_history, self.terminal_history):
                if term == 'goal_reached':
                    plt_params = {'color': 'green', 'alpha': 0.7}
                elif 'collision' in term:
                    plt_params = {'color': 'red', 'alpha': 0.7}
                elif term == 'max_steps':
                    plt_params = {'color': 'gray', 'alpha': 0.7}
                else:
                    raise ValueError(f"Unknown terminal cause: {term}")
                self.fig_assets['traj_lines'][i].set_data(traj[:, 0], traj[:, 1])
                self.fig_assets['traj_lines'][i].set_color(plt_params['color'])
                self.fig_assets['traj_lines'][i].set_alpha(plt_params['alpha'])

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

