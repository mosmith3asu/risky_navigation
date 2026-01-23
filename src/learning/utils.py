import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
from matplotlib.table import Table
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

class Logger:
    def __init__(self, env, fig_title='', start_time_str='',reset_params={}):
        self.env = env
        self.fig_title = fig_title
        self.fig, self.axes = None, None
        self.fig_num = None
        self.fig_assets = {}
        self.start_time_str = start_time_str
        self.reset_params = reset_params

    def _spawn_figure(self, max_title_chars=100):
        from textwrap import wrap

        plt.ion()

        # Create a figure
        self.fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        self.fig_num = self.fig.number
        self.fig.suptitle(
            '\n'.join(wrap(self.fig_title + self.start_time_str, max_title_chars))
            # self.fig_title + f' ({self.start_time_str})'
        )


        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 0.75], figure=self.fig)

        # Left column: 3 stacked axes
        self.rew_ax = self.fig.add_subplot(gs[0, 0])
        self.loss_ax = self.fig.add_subplot(gs[1, 0])
        self.sched_ax = self.fig.add_subplot(gs[2, 0])

        # format last axis differently
        # for ax in [self.rew_ax, self.loss_ax]:
        #     ax.set_xticks([])
        #     ax.set_xticklabels([])
        self.sched_ax.set_xlabel('Episode')

        # Right column: single axis spanning all rows
        self.render_ax = self.fig.add_subplot(gs[:-1, 1])
        self.render_ax.set_xticks([])
        self.render_ax.set_xticklabels([])
        self.render_ax.set_yticks([])
        self.render_ax.set_yticklabels([])

        self.status_ax = self.fig.add_subplot(gs[-1, 1])

        # Adjust layout
        # plt.tight_layout()
        plt.show()

    def _plot_history(self,
                      traj_history,
                      terminal_history,
                      filt_reward_history,
                      ep_len_history=None,
                      env=None,
                      lw=0.5,
                      tpause=0.1,
                      log_interval=1):

        env = env or self.env

        if self.fig is None:
            self._spawn_figure()

        x = np.arange(len(filt_reward_history)) * log_interval
        mean, std = filt_reward_history[:, 0], filt_reward_history[:, 1]

        # Reward plot
        if 'reward_line' not in self.fig_assets.keys():
            self.fig_assets['reward_line'] = self.rew_ax.plot(x, mean, lw=lw, color='k')[0]
            self.rew_ax.set_ylabel('Total Reward')

            # plot 1 std deviation as shaded region
            self.fig_assets['reward_patch'] = self.rew_ax.fill_between(x, mean - std, mean + std,
                                                                       color='b', alpha=0.2)

            if ep_len_history is not None:
                self.rew_ax2 = self.rew_ax.twinx()
                self.fig_assets['ep_len_line'], = self.rew_ax2.plot(x, ep_len_history, color='g',
                                                                    label='Episode Length', lw=lw)
                self.rew_ax2.set_ylabel('Episode Length', color='g')
                self.rew_ax2.tick_params(axis='y', labelcolor='g')



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

            if ep_len_history is not None:
                self.fig_assets['ep_len_line'].set_data(x, ep_len_history)
                self.rew_ax2.relim()
                self.rew_ax2.autoscale_view()

        # # Trajectories
        if 'traj_lines' not in self.fig_assets.keys():

            # render environment once for background
            env.reset(**self.reset_params)
            if hasattr(env, 'num_envs'):  env.call('render', ax=self.render_ax)  # retrieve vec render
            else: env.render(ax=self.render_ax, draw_dist2goal=False, draw_lidar=False)  # single env render

            # plot trajectories and save objects for later updating
            self.fig_assets['traj_lines'] = []
            self.fig_assets['traj_starts'] = []
            for (traj, term) in zip(traj_history, terminal_history):
                if term == 'goal_reached': plt_params = {'color': 'green', 'alpha': 0.7, 'lw': 2 * lw}
                elif 'collision' in term: plt_params = {'color': 'red', 'alpha': 0.7, 'lw': lw}
                elif term == 'max_steps': plt_params = {'color': 'gray', 'alpha': 0.7, 'lw': lw}
                else:
                    raise ValueError(f"Unknown terminal cause: {term}")
                self.fig_assets['traj_lines'].append(self.render_ax.plot(traj[:, 0], traj[:, 1], **plt_params)[0])
                self.fig_assets['traj_starts'].append(self.render_ax.plot(traj[0, 0], traj[0, 1], marker='o', markersize=3, **plt_params)[0])

            self.render_ax.set_title('Trajectories (last {} eps)'.format(len(traj_history)))

        else:
            # update existing trajectory lines
            for i, traj, term in zip(np.arange(len(traj_history)), traj_history, terminal_history):
                if term == 'goal_reached': plt_params = {'color': 'green', 'alpha': 0.7, 'lw': 2 * lw}
                elif 'collision' in term: plt_params = {'color': 'red', 'alpha': 0.7, 'lw': lw}
                elif term == 'max_steps': plt_params = {'color': 'gray', 'alpha': 0.7, 'lw': lw}
                else: raise ValueError(f"Unknown terminal cause: {term}")

                self.fig_assets['traj_lines'][i].set_data(traj[:, 0], traj[:, 1])
                self.fig_assets['traj_lines'][i].set_color(plt_params['color'])
                self.fig_assets['traj_lines'][i].set_alpha(plt_params['alpha'])

                self.fig_assets['traj_starts'][i].set_data(traj[0, 0], traj[0, 1])
                self.fig_assets['traj_starts'][i].set_color(plt_params['color'])
                self.fig_assets['traj_starts'][i].set_alpha(plt_params['alpha'])

        # plt.draw()
        # plt.pause(tpause)

    def _plot_losses(self, actor_loss_history, critic_loss_history, tpause=0.1, log_interval=1, lw=0.5):

        if self.fig is None:
            self._spawn_figure()

        # Plot a dual axis loss plot
        x = np.arange(len(actor_loss_history)) * log_interval
        if 'actor_loss_line' not in self.fig_assets.keys():
            self.fig_assets['actor_loss_line'], = self.loss_ax.plot(x, actor_loss_history, color='b',
                                                                    label='Actor Loss', lw=lw)
            # self.loss_ax.set_xlabel('Episode')
            self.loss_ax.set_ylabel('Actor Loss', color='b')
            self.loss_ax.tick_params(axis='y', labelcolor='b')

            self.loss_ax2 = self.loss_ax.twinx()
            self.fig_assets['critic_loss_line'], = self.loss_ax2.plot(x, critic_loss_history, color='r',
                                                                      label='Critic Loss', lw=lw)
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


    def _plot_schedules(self, tpause=0.01, log_interval=1, lw=0.5, **kwargs):
        max_val = 0
        min_val = 0
        for key, val in kwargs.items():
            x = np.arange(len(val)) * log_interval
            max_val = max(np.max(val), max_val)
            min_val = min(np.min(val), min_val)
            if key not in self.fig_assets.keys():
                self.fig_assets[key], = self.sched_ax.plot(x, val, label=key, lw=lw)
                self.sched_ax.legend(loc='upper right')
                self.sched_ax.set_ylabel('Schedules')
            else:
                self.fig_assets[key].set_data(x, val)

        self.sched_ax.relim()
        self.sched_ax.autoscale_view()
        self.sched_ax.set_ylim(min_val * 0.9, max_val * 1.1)


    def _plot_spin(self):
        """Call frequently to unfreeze plot """
        if self.fig is not None:
            self.fig.canvas.flush_events()

    def _compute_status_fontsize(self, rows, cols):
        """Pick a fontsize (in points) to fit current axis size and cell count."""
        ax = self.status_ax
        fig = ax.figure
        # axis bbox in inches
        bbox_in = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        w_px = bbox_in.width * fig.dpi
        h_px = bbox_in.height * fig.dpi
        # conservative fit: leave room for padding and variable string lengths
        cell_w = max(1.0, w_px / max(1, cols))
        cell_h = max(1.0, h_px / max(1, rows))
        # Heuristics: characters roughly square-ish in many UI fonts; prefer height
        fs_from_h = cell_h * 0.45
        fs_from_w = cell_w * 0.22  # allow ~4–5 chars per cell-width at this scale
        fs = min(fs_from_h, fs_from_w)
        return max(6, min(24, fs))  # clamp to a reasonable range

    def _ensure_resize_hook(self):
        """Connect a resize hook once; it just re-tunes fonts next draw."""
        if getattr(self, "_status_resize_cid", None) is None and self.status_ax.figure.canvas is not None:
            def _on_resize(event):
                if getattr(self, "_status_table", None) is not None:
                    rows, cols = self._status_shape
                    fs = self._compute_status_fontsize(rows, cols)
                    for (r, c), cell in self._status_cells.items():
                        cell.get_text().set_fontsize(fs)
                    self.status_ax.figure.canvas.draw_idle()

            self._status_resize_cid = self.status_ax.figure.canvas.mpl_connect("resize_event", _on_resize)

    def _plot_status(self, n_cols=1, add_key=False, cell_facecolor="#f5f5f7", edgecolor="#dddddd", **kwargs):
        """
        Draw/update a values-only table of kwargs on self.status_ax.

        Parameters
        ----------
        n_cols : int
            Number of table columns. Rows are computed automatically.
        cell_facecolor : color
            Background color for cells.
        edgecolor : color
            Edge color for grid lines.
        **kwargs :
            Key/value pairs to render. Only the *values* are shown in cells.

        Behavior
        --------
        - Rebuilds the grid if shape/column count changes.
        - Otherwise updates cell texts in-place for speed.
        - Font size auto-adjusts to axis size on each call and on window resize.
        """
        if self.fig is None:
            self._spawn_figure()

        ax = self.status_ax
        if add_key:
            for k,val in kwargs.items():
                # kwargs[k] = f"{k}: {val:.4f}" if isinstance(val, float) else f'{k}: {val}'
                kwargs[k] = f'{k}: {val}'


        items = list(kwargs.items())
        n_items = len(items)
        n_cols = max(1, int(n_cols))
        n_rows = max(1, math.ceil(n_items / n_cols))

        # Prepare axis
        ax.set_axis_off()

        # If no table exists yet, or geometry changed, rebuild
        needs_rebuild = (
                getattr(self, "_status_table", None) is None
                or getattr(self, "_status_shape", None) != (n_rows, n_cols)
        )

        if needs_rebuild:
            ax.clear()
            ax.set_axis_off()

            tbl = Table(ax, bbox=[0, 0, 1, 1])
            ax.add_table(tbl)

            # Even column widths, row heights
            col_w = 1.0 / n_cols
            row_h = 1.0 / n_rows

            cells = {}
            # Build all cells (values only)
            for r in range(n_rows):
                for c in range(n_cols):
                    # Table cells are positioned by (row, col)
                    cell = tbl.add_cell(row=r, col=c,
                                        width=col_w, height=row_h,
                                        text="",
                                        loc="center",
                                        facecolor=cell_facecolor,
                                        edgecolor=edgecolor)
                    # Slightly tighter text box so long values don't clip too early
                    cell.PAD = 0.02
                    cells[(r, c)] = cell

            self._status_table = tbl
            self._status_cells = cells
            self._status_shape = (n_rows, n_cols)
            self._status_keys = []  # will set below
            self._ensure_resize_hook()

        # Update texts and remember current key order -> cell mapping
        fs = self._compute_status_fontsize(n_rows, n_cols)
        keys_in_order = []
        for idx, (k, v) in enumerate(items):
            r = idx // n_cols
            c = idx % n_cols
            if (r, c) in self._status_cells:
                cell = self._status_cells[(r, c)]
                cell.get_text().set_text("" if v is None else str(v))
                cell.get_text().set_fontsize(fs)
                cell.get_text().set_color("black")
            keys_in_order.append(k)

        # Clear any extra cells (e.g., last row partially filled)
        for idx in range(n_items, n_rows * n_cols):
            r = idx // n_cols
            c = idx % n_cols
            if (r, c) in self._status_cells:
                self._status_cells[(r, c)].get_text().set_text("")

        self._status_keys = keys_in_order

        # Redraw efficiently
        if self.status_ax.figure.canvas is not None:
            self.status_ax.figure.canvas.draw_idle()

    def spin(self):
        self._plot_spin()

    def draw(self, tpause=0.1):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # plt.draw()
        # plt.pause(tpause)


    def checkpoint(self, ep):
        """
        Mark a training checkpoint at episode `ep` by drawing a vertical line (x = ep)
        on the reward/loss/schedule axes. If called again, the existing checkpoint
        lines are moved to the new episode.

        Notes
        -----
        - Although the docstring says "horizontal", episode is on the x-axis in this logger,
          so the checkpoint marker is a *vertical* line at x = ep.
        """
        if self.fig is None:
            self._spawn_figure()

        try:
            ep = float(ep)
        except (TypeError, ValueError):
            raise ValueError(f"`ep` must be numeric; got {ep!r}")

        # Style for the checkpoint marker
        line_kwargs = dict(color="purple", linestyle="--", linewidth=1.0, alpha=0.8, zorder=10)

        def _set_or_update_vline(ax, key):
            # Create or update the line object
            if key not in self.fig_assets:
                self.fig_assets[key] = ax.axvline(ep, **line_kwargs)
            else:
                ln = self.fig_assets[key]
                # axvline returns a Line2D; update its x-position
                ln.set_xdata([ep, ep])

            # Ensure the marker is visible even if ep extends beyond current limits
            xmin, xmax = ax.get_xlim()
            if ep < xmin or ep > xmax:
                pad = 0.02 * max(1.0, abs(xmax - xmin))
                ax.set_xlim(min(xmin, ep) - pad, max(xmax, ep) + pad)

        _set_or_update_vline(self.rew_ax, "ckpt_vline_rew")
        _set_or_update_vline(self.loss_ax, "ckpt_vline_loss")
        _set_or_update_vline(self.sched_ax, "ckpt_vline_sched")

        # Efficient redraw
        if self.fig is not None and self.fig.canvas is not None:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def is_closed(self):
        """Returns True if the logger figure has been closed."""
        if self.fig is None:
            return False
        else:
            return not plt.fignum_exists(self.fig.number)


####################################################################
### LEARNING TOOLS #################################################
####################################################################
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=int(1e6), device='cpu', seed=None):
        self.device = device
        self.size = size
        self.ptr = 0
        self.full = False
        self.s = torch.zeros((size, obs_dim),   dtype=torch.float32, device=device)
        self.a = torch.zeros((size, act_dim),   dtype=torch.float32, device=device)
        self.r = torch.zeros((size, 1),         dtype=torch.float32, device=device)
        self.s2 = torch.zeros((size, obs_dim),  dtype=torch.float32, device=device)
        self.d = torch.zeros((size, 1),         dtype=torch.float32, device=device)
        if seed is not None:
            torch.manual_seed(seed)

    def add(self, s, a, r, s2, d):
        n = s.shape[0] if s.ndim == 2 else 1
        # ensure batched
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).view(-1, 1)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).view(-1, 1)

        idxs = (self.ptr + torch.arange(n, device=self.device)) % self.size
        self.s[idxs] = s
        self.a[idxs] = a
        self.r[idxs] = r
        self.s2[idxs] = s2
        self.d[idxs] = d

        self.ptr = (self.ptr + n) % self.size
        if self.ptr == 0:
            self.full = True

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch_size):
        max_idx = self.size if self.full else self.ptr
        idxs = torch.randint(0, max_idx, (batch_size,), device=self.device)
        return self.s[idxs], self.a[idxs], self.r[idxs], self.s2[idxs], self.d[idxs]

class ProspectReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, n_samples, size=int(1e6), device='cpu',seed=None):
        super().__init__(obs_dim, act_dim, size, device, seed)
        prospect_dim = 2 # [reward, prob]
        self.r = torch.zeros((size, prospect_dim, n_samples), dtype=torch.float32, device=device)
        self.d = torch.zeros((size, 1, n_samples), dtype=torch.float32, device=device)

    def add(self, s, a, r_prospects, s2, d_prospects):
        n = s.shape[0] if s.ndim == 2 else 1
        # ensure batched
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r_prospects = torch.as_tensor(r_prospects, dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d_prospects = torch.as_tensor(d_prospects, dtype=torch.float32, device=self.device).view(-1, 1)


        idxs = (self.ptr + torch.arange(n, device=self.device)) % self.size
        self.s[idxs] = s
        self.a[idxs] = a
        self.r[idxs] = r_prospects
        self.s2[idxs] = s2
        self.d[idxs] = d_prospects.reshape(1,-1)

        self.ptr = (self.ptr + n) % self.size
        if self.ptr == 0:
            self.full = True

    def save(self, filepath):
        """Saves the replay buffer to a file."""
        data = {
            's': self.s.cpu(),
            'a': self.a.cpu(),
            'r': self.r.cpu(),
            's2': self.s2.cpu(),
            'd': self.d.cpu(),
            'ptr': self.ptr,
            'full': self.full
        }
        torch.save(data, filepath)
        print(f"Replay buffer saved to {filepath}")

    def load(self, filepath):
        """Loads the replay buffer from a file."""
        data = torch.load(filepath, map_location=self.device)
        self.s = data['s'].to(self.device)
        self.a = data['a'].to(self.device)
        self.r = data['r'].to(self.device)
        self.s2 = data['s2'].to(self.device)
        self.d = data['d'].to(self.device)
        self.ptr = data['ptr']
        self.full = data['full']
        print(f"Replay buffer loaded from {filepath}")


class Schedule:
    """
    RL-friendly schedule that evolves a scalar from xstart -> xend over a fixed horizon.

    Modes
    -----
    - 'linear':        linear interpolation from xstart to xend.
    - 'exponential':   geometric interpolation (handles signs and zeros robustly).
    - 'decay':         hyperbolic decay toward xend: x_t = xend + (xstart-xend)/(1+decay_rate*t)

    Step Events
    -----------
    - 'reset':  advance one schedule step when reset() is called (e.g., per-episode schedule).
    - 'step':   advance one schedule step when step() is called (e.g., per-environment step).
    - 'sample': advance one schedule step when sample() is called (e.g., per-optimizer step).
    - None:     schedule does not auto-advance; you can manually call advance(n).
    """

    def __init__(self, xstart, xend, horizon,
                 step_event='sample', mode='exponential',
                 decay_rate=None,exp_gain=1.0, name=''):
        assert step_event in ['reset', 'sample', None], f"Unknown step_event [{step_event}] must be 'reset', 'sample', or None"
        assert mode in ['exponential', 'linear', 'decay','manual']
        if mode == 'decay':
            assert decay_rate is not None, "decay_rate must be specified for 'decay' mode"

        assert horizon >= 0, "horizon must be >= 0"
        self.is_trivial = horizon == 0

        if decay_rate is not None and mode=='decay' and decay_rate >= 0.5:
            warnings.warn(f"High (possibly inverse) decay_rate={decay_rate} detected. "
                          f"Try values decay_rate<= 0.01 for smoother decay. ")

        self.name = name
        self._step_event = step_event
        self._mode = mode
        self.horizon = int(horizon) if not self.is_trivial else 100

        self._x0 = float(xstart) if not self.is_trivial else float(xend)
        self._xf = float(xend)
        self.state = float(xstart)

        self._decay_rate = decay_rate
        self._exp_gain = exp_gain
        self._schedule_step = 0
        self._schedule = self._precomputed_schedule()

    # ---------- internals ----------
    def _precomputed_schedule(self):
        """Precompute and return a numpy array of length `horizon`."""

        T = self.horizon
        if T == 1:
            return np.array([self._x0], dtype=float)
        elif self.is_trivial:
            return self._xf * np.ones(T)

        if self._mode == 'linear':
            # inclusive endpoints across T points
            sched = np.linspace(self._x0, self._xf, T, dtype=float)

        elif self._mode == 'exponential':
            # Geometric interpolation that is robust to zeros and sign changes.
            # If xstart and xend have the same sign and both non-zero:
            if self._x0 != 0.0 and self._xf != 0.0 and np.sign(self._x0) == np.sign(self._xf):
                # ratio goes 0..1 across T points
                r = np.linspace(0.0, 1.0, T, dtype=float)
                r = np.power(r, 1/self._exp_gain)
                sched = self._x0 * ((self._xf / self._x0) ** r)
            else:
                # Fall back to smooth exponential easing via exp(-k * t)
                # x_t = x_f + (x_0 - x_f) * exp(-k * t), t in [0,1]
                k = 5.0
                tau = np.linspace(0.0, 1.0, T, dtype=float)
                sched = self._xf + (self._x0 - self._xf) * np.exp(-k * tau)

        elif self._mode == 'decay':
            # Hyperbolic decay toward x_f. At t=0 => x0; as t grows, -> x_f.
            # t = np.arange(T, dtype=float)
            t = np.array([1], dtype=float)
            sched = self._xf + (self._x0 - self._xf) / (1.0 + self._decay_rate * t)
            # continue decay until self.xf is reached


        else:
            raise ValueError(f"Unknown mode {self._mode}")

        return sched.astype(float)

    # ---------- helpers ----------
    def _clamp_step(self):
        if self._schedule_step < 0:
            self._schedule_step = 0

        if self._mode == 'decay':
            # allow unbounded growth of step for decay mode
            pass
        elif self._schedule_step >= self.horizon:
            self._schedule_step = self.horizon - 1
        # else:
        #     raise ValueError(f"Unknown clamp mode for {self._mode}")


    def _apply_state_from_step(self):
        self._clamp_step()

        if self._mode == 'decay':
            t = float(self._schedule_step)
            self.state = self._xf + (self._x0 - self._xf) / (1.0 + self._decay_rate * t)
        else:
            self.state = float(self._schedule[self._schedule_step])

    def advance(self, n=1):
        """Manually advance the schedule by n steps (can be negative)."""
        self._schedule_step = int(self._schedule_step + n)
        self._apply_state_from_step()
        return self.state

    # ---------- API ----------
    def reset(self,at=None):
        """Resets the schedule to initial state; optionally advances on reset-event."""
        self._schedule_step = 0
        self._apply_state_from_step()
        if at is not None:
            self.state = self.at(at)
        elif self._step_event == 'reset':
            # Advance one step so next episode uses the next scheduled value
            self.advance(1)
        return self.state

    def sample(self,at = None, advance = True):
        """Return current value; advance if step_event == 'sample'."""
        if at is not None:
            return self.at(at)

        val = self.state
        if self._step_event == 'sample' and advance:
            self.advance(1)
        return val

    def value(self):
        """Current scheduled value without advancing."""
        return self.state

    def at(self, idx):
        """Peek the schedule value at absolute index idx (clamped)."""
        if self._mode == 'decay':
            t = float(idx)
            return self._xf + (self._x0 - self._xf) / (1.0 + self._decay_rate * t)

        i = int(np.clip(idx, 0, self.horizon - 1))
        return float(self._schedule[i])

    def plot(self, ax= None, T=None, show=True, label=None,
             with_title = False):
        """Plots the schedule over T steps with a vertical line at current schedule step."""
        label = self._mode if label is None else label

        T = self.horizon if T is None else int(T)
        T = max(1, T)
        # Extend/trim displayed schedule to length T (repeat last value if T > horizon)
        if self._mode == 'decay':
            t = np.arange(T, dtype=float)
            y = self._xf + (self._x0 - self._xf) / (1.0 + self._decay_rate * t)
        elif T <= self.horizon:
            y = self._schedule[:T]
        else:
            pad = np.full(T - self.horizon, self._schedule[-1], dtype=float)
            y = np.concatenate([self._schedule, pad], axis=0)

        x = np.arange(T)
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(x, y, linewidth=2, label=label)
        # vertical line at current step (clamped to T-1)
        v = int(np.clip(self._schedule_step, 0, T - 1))
        ax.axvline(v, linestyle='--')
        if with_title:
            ax.set_title(f"Schedule ({self._mode}, step_event={self._step_event})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")


        # ax.tight_layout()
        if show:
            plt.show()
        # plt.figure()
        # plt.plot(x, y, linewidth=2)
        # # vertical line at current step (clamped to T-1)
        # v = int(np.clip(self._schedule_step, 0, T - 1))
        # plt.axvline(v, linestyle='--')
        # plt.title(f"Schedule ({self._mode}, step_event={self._step_event})")
        # plt.xlabel("Step")
        # plt.ylabel("Value")
        # plt.tight_layout()
        # plt.show()

    # ---------- dunder ----------mode={self._mode}
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = f"Schedule(name = {self.name}) \n" \
            f"\t| step_event={self._step_event} \n" \
            f"\t| horizon={0 if self.is_trivial else self.horizon}\n" \
            f"\t| x0={self._x0} --> xf={self._xf}\n" \
            f"\t| mode={self._mode} "

        if self._mode == 'decay':
            s += f" - decay_rate={self._decay_rate}"
        elif self._mode == 'exponential':
            s += f"- exp_gain={self._exp_gain}"
        return s

class OUNoise: # (Schedule): # Soft Inherit Schedule
    """
    Ornstein–Uhlenbeck noise for exploration.
    Handles both single env and vectorized envs by broadcasting on shape.
    - theta: rate of mean reversion (returns noise to mu)
    - sigma: scale of noise
    """
    def __init__(self, shape, mode = 'exponential',
                 mu=0.0, theta=0.15, sigma=0.5,  sigma_min = 0.05,
                 n_envs=1, device='cpu',
                 **kwargs):

        self.sched = Schedule(xstart= sigma,
                         xend = sigma_min,
                         step_event=kwargs.pop('step_event',None),
                         mode= mode,
                         exp_gain=kwargs.pop('exp_gain', 2),
                         horizon=kwargs.pop('horizon',3000),
                         decay_rate= kwargs.pop('sigma_decay', None),
                         name='OUNoise')

        assert len(kwargs.keys()) == 0, f"Unknown kwargs passed to OUNoise: {kwargs.keys()}"

        self.action_dim = shape
        self.shape = shape
        self.mu = mu
        self.theta = theta

        self._sigma0 = float(sigma)
        self._sigma = float(sigma)
        # self.sigma = float(sigma)
        self.sigma_min = sigma_min

        self.n_envs = n_envs
        self.device = device
        self.size = (self.n_envs, self.action_dim)
        self.reset()

    def reset(self,i=None,at=None, reset_sigma= False):
        if i is None:
            self.state = self.mu * torch.zeros(self.n_envs, self.action_dim, device=self.device)
        else:
            self.state[i] = 0
        if reset_sigma:
            self.sigma = self._sigma0
            self.sched.reset()
        if at is not None:
            self.sigma = self.sched.at(at)

    def sample(self, at = None, advance = True):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0, 1, self.size, device=self.device)
        self.state = x + dx

        # Decay sigma exploration (on sample)
        self.sigma = self.sched.sample(at=at, advance=advance) #* torch.ones_like(self.sigma)


        return self.state

    def plot(self,*args,**kwargs):
        self.sched.plot(*args, **kwargs)

    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, value):
        self._sigma = float(value)

    def __repr__(self):
        return self.__str__()
    def __str__(self):

        s = f"\nOUNoise(shape={self.shape})\n" \
                f"\t| mu={self.mu}\n " \
                f"\t| theta={self.theta}\n" \
                f"\t| sigma={self.sigma}\n, " \
                f"\t| sigma_min={self.sigma_min}\n"
        s += '\t|' + self.sched.__str__().replace('\t','\t\t')
        return s

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



def main():
    # test_risk_sensitivity()
    fig,axs = plt.subplots(1,2, figsize=(12,9))

    sched = OUNoise(sigma=0.5, sigma_min=0.05, shape=2, horizon=0.9*3000, mode='exponential',  exp_gain=1,step_event=None)
    sched.plot(axs[0], show=False, T=3500, label='OUNoise', )
    #
    # sched = Schedule(xstart=1.0, xend=0.1, horizon=3000, mode='exponential',exp_gain=2.0)
    # sched.plot(axs[0], show =False, T=3500, label='exponential0', )
    # print(sched)

    # sched = Schedule(xstart=1.0, xend=0.1, horizon=600, mode='exponential',exp_gain=2.0)
    # sched.plot(axs[0], show=False, T=800, label='exponential1')
    #
    # sched = Schedule(xstart=1.0, xend=0.1, horizon=600, mode='linear')
    # sched.plot(axs[0], show =False, T=800)
    #
    sched = OUNoise(sigma=0.5, sigma_min=0.05, shape=2, horizon=0.9*3000, mode='exponential',  exp_gain=2,step_event=None)
    sched.plot(axs[0], show =False, T=3500,label='Ounoise (exp gain=2)')
    axs[0].legend()
    plt.show()
    # # pass

if __name__ == '__main__':
    main()

# class OUNoise:
#     """
#     Ornstein–Uhlenbeck noise for exploration.
#     Handles both single env and vectorized envs by broadcasting on shape.
#     - theta: rate of mean reversion (returns noise to mu)
#     - sigma: scale of noise
#     """
#     def __init__(self, shape, mu=0.0, theta=0.15, sigma=0.5, n_envs=1,
#                  sigma_min = 0.05, sigma_decay=.995, device='cpu', decay_freq= None,
#                  precompute = True, precompute_steps=600,
#                  precompute_low=None, precompute_high=None,
#                  **kwargs):
#
#
#         self.action_dim = shape
#         self.shape = shape
#         self.mu = mu
#         self.theta = theta
#
#         self._sigma0 = sigma
#         self.sigma = sigma
#         self.sigma_min = sigma_min
#         self.sigma_decay = sigma_decay
#         self.decay_freq = decay_freq # if none, decay on reset
#         self.decay_count = 1
#
#
#         # self.dt = dt
#         self.n_envs = n_envs
#         self.device = device
#         self.size = (self.n_envs, self.action_dim)
#         self.state = self.mu * torch.zeros(self.n_envs, self.action_dim, device=self.device)
#
#         # Precomputing and scaling
#         self.precompute = precompute
#         low = precompute_low
#         high = precompute_high
#         self.low = np.broadcast_to(np.asarray(low, dtype=float), self.shape) if low is not None else None
#         self.high = np.broadcast_to(np.asarray(high, dtype=float), self.shape) if high is not None else None
#         self._precompute_steps = precompute_steps
#
#         self._precomputed_noise = None
#         self._precompute_step = 0
#         if precompute:
#             self._precomputed_noise = self._precompute_noise()
#
#
#         # Reset
#
#         self.reset()
#
#
#
#     def _precompute_noise(self):
#         assert self.precompute, "Precompute flag is false. Should not be calling this method."
#         precomputed_noise = torch.zeros((self._precompute_steps, self.n_envs, self.action_dim), device=self.device)
#         for t in range(self._precompute_steps):
#             noise = self.sample()
#             precomputed_noise[t] = noise
#         # Scale precomputed noise into bounds if specified
#         if self.low is not None and self.high is not None:
#             dim_max = precomputed_noise.max()
#             dim_min = precomputed_noise.min()
#             # Scale each dimension separately
#             for d in range(self.action_dim):
#                 # dim_max = self.precomputed_noise[:,:,d].max()
#                 # dim_min = self.precomputed_noise[:,:,d].min()
#                 scale = (self.high[d] - self.low[d]) / (dim_max - dim_min + 1e-12)
#                 precomputed_noise[:,:,d] = self.low[d] + (precomputed_noise[:,:,d] - dim_min) * scale
#         return precomputed_noise
#
#     def reset(self,i=None):
#         if i is None:
#             self.state = self.mu * torch.zeros(self.n_envs, self.action_dim, device=self.device)
#         else:
#             self.state[i] = 0
#
#         # Decay sigma exploration (on reset)
#         if self.decay_freq is None:
#             dsig = - self.sigma * ( 1-self.sigma_decay)
#             sigma = self.sigma + (dsig)/self.n_envs
#             self.sigma = max(self.sigma_min, sigma)
#
#         # Handle precomputation
#
#         self._precomputed_noise = None
#         self._precompute_step = 0
#         if self.precompute:
#             self._precomputed_noise = self._precompute_noise()
#
#     def sample(self):
#         if self.precompute and self._precomputed_noise is not None:
#             assert self._precompute_step <= self._precompute_steps, "Precomputed steps exceeded."
#             noise = self._precomputed_noise[self._precompute_step]
#             self._precompute_step += 1
#             return noise
#
#
#
#         x = self.state
#         # dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0,1,self.size,device=self.device)
#         dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0, 1, self.size, device=self.device)
#         self.state = x + dx
#
#         # Decay sigma exploration (on sample)
#         if self.decay_freq is not None:
#             if self.decay_count % self.decay_freq == 0:
#                 dsig = - self.sigma * (1 - self.sigma_decay)
#                 sigma = self.sigma + (dsig)
#                 self.sigma = np.max([self.sigma_min * np.ones_like(sigma), sigma], axis=0)
#                 self.decay_count = 1
#             else:
#                 self.decay_count += 1
#
#         return self.state
#
#     def simulate(self,T=600):
#         n_trials = 3
#         self.reset()
#         # self.theta = 1
#         # self.sigma = 0.05
#         fig,axs = plt.subplots(self.action_dim,n_trials)
#
#         for trial in range(n_trials):
#             self.reset()
#             data = np.zeros((T, self.action_dim))
#             # xy = np.array([[0.0,0.0]])
#
#             for t in range(T):
#                 xy = np.array([[0.0, 0.0]])
#                 xy += self.sample().cpu().numpy()
#                 data[t,:] = xy
#             # plot data
#             for d in range(self.action_dim):
#                 axs[d,trial].plot(data[:,d])
#                 axs[d,trial].set_title(f'OU Noise Dimension {["x","y"][d]}')
#                 axs[d,trial].hlines(0, xmin=0, xmax=T, colors='k', linestyles='dashed', alpha=0.5)
#                 axs[d,trial].set_xlabel('Timestep')
#
#
#         plt.show()
#
#     def reset_sigma(self):
#         self.sigma = self._sigma0
#         self.decay_count = 1
#     def __repr__(self):
#         return self.__str__()
#     def __str__(self):
#
#         s = f"OUNoise(shape={self.shape})\n" \
#                 f"\t| mu={self.mu}\n " \
#                 f"\t| theta={self.theta}\n" \
#                 f"\t| sigma={self.sigma}\n, " \
#                 f"\t| sigma_decay={self.sigma_decay}\n" \
#                 f"\t| decay_freq:{self.decay_freq }\n" \
#                 f"\t| sigma_min={self.sigma_min}\n"\
#                 f"\t| precompute={self.precompute}\n"
#         return s


# class OUNoise(Schedule):
#     """
#     Ornstein–Uhlenbeck noise for exploration.
#     Handles both single env and vectorized envs by broadcasting on shape.
#     - theta: rate of mean reversion (returns noise to mu)
#     - sigma: scale of noise
#     """
#
#     def __init__(self, shape, mu=0.0, theta=0.15, sigma=0.5, n_envs=1,
#                  sigma_min=0.05, device='cpu', sigma_decay=.995,
#                  decay_freq=None,
#                  sigma_sched=None,
#                  **kwargs):
#
#         super().__init__(xstart, xend,
#                          **kwargs)
#
#         # xstart, xend, horizon,
#         # step_event = 'sample', mode = 'exponential',
#         # decay_rate = None, exp_gain = 1.0, name = ''
#
#         self.action_dim = shape
#         self.shape = shape
#         self.mu = mu
#         self.theta = theta
#
#         self._sigma0 = sigma
#         self.sigma = sigma
#         self.sigma_min = sigma_min
#         self.sigma_decay = sigma_decay
#         self.decay_freq = decay_freq  # if none, decay on reset
#         self.decay_count = 1
#
#         self.sigma_sched = sigma_sched
#         if self.sigma_sched is not None:
#             self.sigma_sched['xstart'] = self.sigma
#             self.sigma_sched['xend'] = self.sigma_min
#             self.sigma_sched = Schedule(**self.sigma_sched)
#             # self.sigma_sched = Schedule(xstart=sigma,
#             #                             xend=sigma_min,
#             #                             horizon=1_000,
#             #                             step_event='sample',
#             #                             mode='exponential',
#             #                             exp_gain=2.0
#             #                             )
#
#         # self.dt = dt
#         self.n_envs = n_envs
#         self.device = device
#         self.size = (self.n_envs, self.action_dim)
#         self.reset()
#
#     def reset(self, i=None):
#         if i is None:
#             self.state = self.mu * torch.zeros(self.n_envs, self.action_dim, device=self.device)
#         else:
#             self.state[i] = 0
#
#         # Decay sigma exploration (on reset)
#         if self.decay_freq is None:
#             self.decay_sigma()
#
#     def sample(self, increment_decay=True):
#         x = self.state
#         # dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0,1,self.size,device=self.device)
#         dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0, 1, self.size, device=self.device)
#         self.state = x + dx
#
#         # Decay sigma exploration (on sample)
#         if self.decay_freq is not None and increment_decay:
#             if self.decay_count % self.decay_freq == 0:
#                 self.decay_sigma()
#                 self.decay_count = 1
#             else:
#                 self.decay_count += 1
#
#         return self.state
#
#     def simulate(self, T=600):
#         n_trials = 3
#         self.reset()
#         # self.theta = 1
#         # self.sigma = 0.05
#         fig, axs = plt.subplots(self.action_dim, n_trials)
#
#         for trial in range(n_trials):
#             data = np.zeros((T, self.action_dim))
#             # xy = np.array([[0.0,0.0]])
#
#             for t in range(T):
#                 xy = np.array([[0.0, 0.0]])
#                 xy += self.sample().cpu().numpy()
#                 data[t, :] = xy
#             # plot data
#             for d in range(self.action_dim):
#                 axs[d, trial].plot(data[:, d])
#                 axs[d, trial].set_title(f'OU Noise Dimension {["x", "y"][d]}')
#                 axs[d, trial].hlines(0, xmin=0, xmax=T, colors='k', linestyles='dashed', alpha=0.5)
#                 axs[d, trial].set_xlabel('Timestep')
#
#         plt.show()
#
#     def reset_sigma(self):
#         self.sigma = self._sigma0
#         self.decay_count = 1
#
#     def decay_sigma(self):
#         if self.sigma_sched is not None:
#             self.sigma = self.sigma_sched.sample(advance=True)
#             return
#         dsig = - self.sigma * (1 - self.sigma_decay)
#         sigma = self.sigma + (dsig)
#         self.sigma = np.max([self.sigma_min * np.ones_like(sigma), sigma], axis=0)
#
#     def __repr__(self):
#         return self.__str__()
#
#     def __str__(self):
#
#         s = f"\nOUNoise(shape={self.shape})\n" \
#             f"\t| mu={self.mu}\n " \
#             f"\t| theta={self.theta}\n" \
#             f"\t| sigma={self.sigma}\n, " \
#             f"\t| sigma_decay={self.sigma_decay}\n" \
#             f"\t| decay_freq:{self.decay_freq}\n" \
#             f"\t| sigma_min={self.sigma_min}\n"
#         return s
#

