# sac_agent.py
"""
Soft Actor-Critic (SAC) re-implementation of the provided DDPG-style ACAgent.

Key differences vs the original:
- Stochastic tanh-squashed Gaussian policy (no OU noise)
- Twin Q-networks (Q1, Q2) + target Q-networks
- Entropy-regularized targets and actor objective, optional automatic temperature tuning
- Preserves the *prospect-based reward samples* + your risk-measure hook via `_risk_measure(vals, probs)`

This file is designed to be a drop-in sibling to `ac_agent.py` and to work with the same
`ProspectReplayBuffer` and environment API used there.
"""

from __future__ import annotations
import copy
import warnings

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from datetime import datetime
from typing import Tuple, Optional
from learning.SAC.sac_model import QNetwork, TanhGaussianPolicy
from learning.utils import ProspectReplayBuffer, Schedule, soft_update, hard_update, Logger
from learning.risk_measures import CumulativeProspectTheory
from learning.spsa import SPSA
from utils.file_management import save_pickle, load_pickle, load_latest_pickle, get_algorithm_dir

# -----------------------------
# SAC Agent (prospect + risk-measure)
# -----------------------------
class SACAgent_BASE:
    def __init__(
        self,
        env,
        batch_size: int = 256,
        replay_sz: int = 30_000,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_lr: float = 1e-4, # 3e-4,
        q_lr: float =  5e-4,
        alpha_init: float = 0.2,
        alpha_range: Tuple[float, float] = (0.01, 1.0),
        alpha_lr: float = 3e-4, #  1e-4,
        automatic_entropy_tuning: bool = True,
        grad_clip: Optional[float] = None,

        # target_entropy: Optional[float] = None,
        target_entropy_scale: float = 0.98,
        randstart_sched = (1,0.25,250),
        rshape_sched = (1,1,500),
        # rshape_epis: int = 1000,
        # random_start_epis: int = 1000,
        warmup_steps: int = 2_000,
        updates_per_step: int = 1,
        num_hidden_layers: int = 5,
        size_hidden_layers: int = 256,
            normailize_reward = False,
        loads: Optional[str] = None,
        note: str = '',
    ):
        self.env = env
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.break_epi_on_terminal = True
        self.init_params = self._parse_params(locals())
        self.init_params['R'] = f'[G:{self.env.reward_goal}  C:{self.env.reward_collide} dt:{self.env.reward_step}]'
        self.init_params['Rs'] = f'[dist:{self.env.reward_dist2goal}-prog {self.env.reward_dist2goal_prog}  stop:{self.env.reward_stopping} ]'
        self.init_params['Term'] = f'[T: {self.env.time_is_terminal} G: {self.env.goal_is_terminal} ]'
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = int(warmup_steps)
        self.updates_per_step = int(updates_per_step)
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.grad_clip = grad_clip
        self._last_terminal_scale = 1 # accounts for cumulated rewards not terminating in sampled states
        self.normailize_reward = normailize_reward
        if self.normailize_reward:
            warnings.warn("Reward normalization is enabled for SACAgent_BASE. Make sure this is intended.")

        # dims & bounds
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.a_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        self.a_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=self.device)

        # schedules / replay
        rand_start_sched_params =  {"xstart": randstart_sched[0], "xend": randstart_sched[1], "horizon": randstart_sched[2], "mode": "linear"}
        rshape_sched_params = {"xstart": rshape_sched[0], "xend": rshape_sched[1], "horizon": rshape_sched[2], "mode": "linear"}

        self.rand_start_sched = Schedule(**rand_start_sched_params)
        self.rshape_sched = Schedule(**rshape_sched_params)

        self.replay = ProspectReplayBuffer(
            self.state_dim,
            self.action_dim,
            n_samples=self.env.n_samples,
            size=replay_sz,
            device=self.device,
        )

        # networks
        self.policy = TanhGaussianPolicy(
            self.state_dim,
            self.action_dim,
            action_low  = self.a_low,
            action_high = self.a_high,
            hidden_dim  = size_hidden_layers,
            n_hidden    = num_hidden_layers,
        ).to(self.device)

        self.q_loss_fn = F.smooth_l1_loss
        # self.q_loss_fn = F.mse_loss; warnings.warn("Using MSE loss for Q-function updates.")

        self.q1      = QNetwork(self.state_dim, self.action_dim, hidden_dim = size_hidden_layers, n_hidden = num_hidden_layers).to(self.device)
        self.q2      = QNetwork(self.state_dim, self.action_dim, hidden_dim = size_hidden_layers, n_hidden = num_hidden_layers).to(self.device)
        self.q1_targ = QNetwork(self.state_dim, self.action_dim, hidden_dim = size_hidden_layers, n_hidden = num_hidden_layers).to(self.device)
        self.q2_targ = QNetwork(self.state_dim, self.action_dim, hidden_dim = size_hidden_layers, n_hidden = num_hidden_layers).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        # optimizers
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=q_lr)

        # entropy temperature
        self.automatic_entropy_tuning = automatic_entropy_tuning
        # self.target_entropy = - target_entropy_scale * torch.log(1 / torch.tensor(self.action_dim, dtype=torch.float32,device=self.device))
        self.target_entropy = -target_entropy_scale *(self.action_dim)


        self.alpha_range = alpha_range
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32, device=self.device, requires_grad=False)
            self.alpha_opt = None

        # logging
        self.enable_print_report = True
        self.log_interval = 10
        self.max_episodes = None

        self.history = {
            "ep_dur"   : deque(maxlen = self.log_interval),
            "ep_len"   : deque(maxlen = self.log_interval),
            "ep_reward": deque(maxlen = 3*self.log_interval),
            "action"   : deque(maxlen = self.log_interval),
            "xy"       : deque(maxlen = self.log_interval),
            "terminal" : deque(maxlen = self.log_interval),
            "timeseries_filt_rewards": np.zeros([0, 2]),
            "timeseries_ep_len": [],
            "critic_loss": [],
            "actor_loss": [],
            "alpha": [],
            "rand_start": [],
            "rshape": [],
        }

        self.q_stats = {'max': deque(maxlen=self.env.max_steps),
                        'min': deque(maxlen=self.env.max_steps)
                        }

        self.emojis = {
            'goal_reached': '✅',
            'collision': '💥',
            'max_steps': '⌛︎',
        }

        # -----------------------------
        # Checkpointing (best model by filtered return)
        # -----------------------------
        self._best_ckpt_score = -np.inf
        self._checkpoint = None  # {'score': float, 'episode': int|None, 'created_at': str, 'data': state_dicts}
        self.start_time_str = datetime.now().strftime("%Y-%m-%d TIME: %H:%M:%S")
        self.reset_params = {}
        self.logger = Logger(self.env, fig_title=f"BASE-SAC Agent ({self.start_time_str})")

        if loads is not None:
            self.load(loads)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def _act(self, state: torch.Tensor, explore: bool = True) -> torch.Tensor:
        """
        Returns an env-scaled action tensor on self.device.
        SAC exploration comes from sampling the stochastic policy.
        """
        if explore:
            a, _, _ = self.policy.sample(state)
            return a
        else:
            _, _, mean_a = self.policy.sample(state)
            return mean_a

    def _warmup(self) -> None:
        """
        Populate replay with random actions (in env bounds) before learning.
        Uses the same prospect reward / done signal produced by your env.
        """
        if self.warmup_steps <= 0:
            return

        self.env.reset(p_rand_state=self.rand_start_sched.sample(at=0), **self.reset_params)
        steps = 0
        while steps < self.warmup_steps:
            o1 = self.env.observation
            a_np = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high, size=self.action_dim)
            _, r_samples, done_samples, info = self.env.step(a_np)
            if self.normailize_reward:
                r_samples = self.env._normalize_reward(r_samples)

            o2 = self.env.observation

            o1_pt = torch.tensor(o1, dtype=torch.float32, device=self.device)
            o2_pt = torch.tensor(o2, dtype=torch.float32, device=self.device)
            a1_pt = torch.tensor(a_np, dtype=torch.float32, device=self.device)

            r_probs = np.array(self.env.robot_probs)
            r_prospects = np.vstack([r_samples, r_probs])
            self.replay.add(o1_pt, a1_pt, r_prospects, o2_pt, done_samples)

            steps += 1
            if info.get("true_done", False):
                self.env.reset(p_rand_state=self.rand_start_sched.sample(at=0), **self.reset_params)

    def train(self, max_episodes: int = 1_000) -> None:
        # Init training params
        self.logger._plot_status(n_cols = 3,add_key=True, **self.init_params)
        self.max_episodes = max_episodes
        self._warmup()

        # init episode buffers
        running_rewards = 0.0
        running_xys = deque(maxlen=self.env.max_steps)
        running_acts = deque(maxlen=self.env.max_steps)
        running_actor_loss = deque(maxlen=self.env.max_steps)
        running_critic_loss = deque(maxlen=self.env.max_steps)

        # print(f'Step Reward bounds {self.env.reward_min_step} to {self.env.reward_max_step}')
        # print(f'Run Reward bounds {self.env.reward_min_run} to {self.env.reward_max_run}')

        # Training Loop
        for ep in range(1, max_episodes + 1):
            tstart = datetime.now()
            # Update schedules
            self.logger.spin()
            if self.logger.is_closed():
                break

            self.env.rshape = self.rshape_sched.sample(at=ep)
            p_rand_state = self.rand_start_sched.sample(at=ep)

            # Reset env and buffers
            reset_info = self.env.reset(p_rand_state=p_rand_state, **self.reset_params)
            self._last_terminal_scale = 1
            running_rewards = 0.0
            # running_rewards2 = 0.0

            running_acts.clear()
            running_xys.clear()
            running_actor_loss.clear()
            running_critic_loss.clear()


            # Rollout -----------------------------------------------
            for i in range(self.env.max_steps):
                # Observe / act
                o1 = self.env.observation
                o1_pt = torch.tensor(o1, dtype=torch.float32, device=self.device)

                a1_pt = self._act(o1_pt, explore=True)
                a1 = a1_pt.detach().cpu().numpy().flatten()

                # Step
                _, r_samples, done_samples, info = self.env.step(a1)
                if self.normailize_reward:
                    r_samples = self.env._normalize_reward(r_samples)

                o2 = self.env.observation
                o2_pt = torch.tensor(o2, dtype=torch.float32, device=self.device)

                # Store prospects
                r_probs = np.array(self.env.robot_probs)
                r_prospects = np.vstack([r_samples, r_probs])

                self.replay.add(o1_pt, a1_pt, r_prospects, o2_pt, done_samples)

                # Update
                if len(self.replay) >= self.batch_size:
                    for _ in range(self.updates_per_step):
                        critic_loss, actor_loss, alpha_val = self._update()
                        running_actor_loss.append(actor_loss)
                        running_critic_loss.append(critic_loss)
            # ------------------------------------------------------

                # Log for reporting
                # running_rewards += float(info["true_reward"])
                info['reward_samples'] = r_samples # TODO: Optimize this for a vectorized call at end of episode
                # running_rewards2 += np.mean(r_samples)
                running_rewards += self._reward_eval(info)

                running_xys.append(info["true_next_state"][0, :2])
                running_acts.append(a1)

                if (info["true_done"] or info['true_reason']=='goal_reached') and self.break_epi_on_terminal:
                        break

            # Episode end handling
            self.history["terminal"].append(info["true_reason"])
            self.history["ep_reward"].append(running_rewards)
            self.history["xy"].append(np.array(running_xys))
            self.history["action"].append(np.mean(np.array(running_acts), axis=0))

            self.history["ep_dur"].append((self.env.max_steps/(i+1)) * (datetime.now() - tstart).total_seconds())
            self.history["actor_loss"].append(float(np.mean(np.array(running_actor_loss))) if len(running_actor_loss) > 0 else 0.0)
            self.history["critic_loss"].append(float(np.mean(np.array(running_critic_loss))) if len(running_critic_loss) > 0 else 0.0)
            self.history["rand_start"].append(p_rand_state) #
            self.history["rshape"].append(self.env.rshape)
            self.history["alpha"].append(float(self.alpha.detach().cpu().item()))

            # only track non-random start lengths
            if not reset_info['is_rand']:
                self.history["ep_len"].append(i)
            # self.history["ep_len"].append(i)

            if self.enable_print_report:
                alpha_scalar = float(self.alpha.detach().cpu().item())
                min_q = np.mean(self.q_stats['min']) if len(self.q_stats['min']) > 0 else 0.0
                max_q = np.mean(self.q_stats['max']) if len(self.q_stats['max']) > 0 else 0.0
                print(
                    f"[{np.mean(self.history['ep_dur']):.2f} sec/ep]"
                    f"[{self.__class__.__name__}] "
                    f"ep {ep} "
                    f"| T: {i} it" #{i * self.env.dt:.1f}sec)"
                    f"| ∑r: {self.history['ep_reward'][-1]:.2f}"# ({running_rewards2:.2f})"
                    # f"| MemLen: {len(self.replay)}"
                    f"| P(rand):{p_rand_state:.2f}"
                    f"| α: {alpha_scalar:.3f}"
                    f"| Rshape: {self.env.rshape:.2f}"
                    f"| {self.emojis[self.history['terminal'][-1]]}: {self.history['terminal'][-1]}"
                    F"| Qrange: [{min_q:.2f}, {max_q:.2f}]"
                    # f"| Mean Act: {np.round(self.history['action'][-1], 2)}"
                )

            ############### REPORTING ###############
            if ep % self.log_interval == 0:
                mu, std = np.mean(self.history["ep_reward"]), np.std(self.history["ep_reward"])
                bu_len = self.history["timeseries_ep_len"][-1] if len(self.history["timeseries_ep_len"]) > 0 else 0
                mu_len = np.mean(self.history["ep_len"]) if len(self.history["ep_len"]) > 0 else bu_len
                if self.history["timeseries_filt_rewards"].shape[0] == 0: # append filler value
                    self.history["timeseries_filt_rewards"] = np.vstack((self.history["timeseries_filt_rewards"], np.array([[mu, std]])))
                    self.history["timeseries_ep_len"].append(mu_len)
                self.history["timeseries_filt_rewards"] = np.vstack((self.history["timeseries_filt_rewards"], np.array([[mu, std]])))
                self.history["timeseries_ep_len"].append(mu_len)

                # handle checkpointing
                did_cp = self.maybe_checkpoint(ep=ep)
                if did_cp:
                    self.logger.checkpoint(ep=ep)
                    if self.enable_print_report:
                        print("*** New checkpoint created ***")
                if self.enable_print_report:
                    print("")

                self.logger._plot_history(
                    traj_history=self.history["xy"],
                    terminal_history=self.history["terminal"],
                    filt_reward_history=self.history["timeseries_filt_rewards"],
                    ep_len_history=self.history["timeseries_ep_len"],
                    log_interval=self.log_interval,
                )
                self.logger._plot_losses(
                    actor_loss_history=self.history["actor_loss"],
                    critic_loss_history=self.history["critic_loss"],
                    log_interval=1,
                )


                self.logger._plot_schedules(alpha=self.history["alpha"],
                                     rand_start=self.history["rand_start"],
                                        rshape=self.history["rshape"],
                                            log_interval=1)

                self.logger.draw(tpause=0.1)

                self.history["terminal"].clear()
                # self.history["ep_len"].clear()
                # self.history["ep_reward"].clear()
                self.history["xy"].clear()


    def test(self,max_episodes=10):
        test_info = {
            "terminal" : [],
            "xy"       : [],
            "ep_len"  : [],
            "ep_reward": [],
        }
        running_xys = deque(maxlen=self.env.max_steps)
        running_acts = deque(maxlen=self.env.max_steps)
        running_actor_loss = deque(maxlen=self.env.max_steps)
        running_critic_loss = deque(maxlen=self.env.max_steps)

        for ep in range(1, max_episodes + 1):
            # Update schedules
            self.logger.spin()
            self.env.rshape = 0
            p_rand_state = 0

            # Reset env and buffers
            self.env.reset(p_rand_state=p_rand_state, **self.reset_params)
            running_rewards = 0.0
            running_acts.clear()
            running_xys.clear()
            running_actor_loss.clear()
            running_critic_loss.clear()

            # Rollout
            for i in range(self.env.max_steps):
                # Observe / act
                o1 = self.env.observation
                o1_pt = torch.tensor(o1, dtype=torch.float32, device=self.device)

                a1_pt = self._act(o1_pt, explore=True)
                a1 = a1_pt.detach().cpu().numpy().flatten()

                # Step
                _, r_samples, done_samples, info = self.env.step(a1)
                if self.normailize_reward:
                    r_samples = self.env._normalize_reward(r_samples)


                o2 = self.env.observation
                o2_pt = torch.tensor(o2, dtype=torch.float32, device=self.device)

                # Store prospects
                r_probs = np.array(self.env.robot_probs)
                r_prospects = np.vstack([r_samples, r_probs])
                self.replay.add(o1_pt, a1_pt, r_prospects, o2_pt, done_samples)

                # Log for reporting
                running_rewards += float(info["true_reward"])
                running_xys.append(info["true_next_state"][0, :2])
                running_acts.append(a1)

                if (info["true_done"] or info['true_reason']=='goal_reached') and self.break_epi_on_terminal:
                        break
                    # self.env.reset(p_rand_state=p_rand_state, **self.reset_params)


            # Episode end handling
            test_info["terminal"].append(info["true_reason"])
            test_info["ep_len"].append(i)
            test_info["ep_reward"].append(running_rewards)
            test_info["xy"].append(np.array(running_xys))
            # self.history["action"].append(np.mean(np.array(running_acts), axis=0))

        return test_info

    def eval(self):
        self.policy.eval()
        self.q1.eval()
        self.q2.eval()
        self.q1_targ.eval()
        self.q2_targ.eval()


    def _update(self) -> Tuple[float, float, float]:
        o1, a1, r_prospects, o2, d_prospects = self.replay.sample(self.batch_size)

        # unpack prospects: r_vals, r_probs both [B, N]
        r_vals, r_probs = [r_prospects[:, i, :] for i in range(2)]

        # robust done shape handling
        d = d_prospects
        if d.dim() == 3 and d.shape[1] == 1:
            d = d.squeeze(1)
        if d.dim() == 2 and d.shape[1] == 1:
            d = d.expand(-1, r_vals.shape[1])

        # ---------------------- target ----------------------
        y = self._get_td_target_expectations(o1,a1, o2, r_vals, r_probs, d)  # [B]
        assert torch.isfinite(y).all(), "Non-finite td-target values!"

        # ---------------------- critic update ----------------------
        if isinstance(self.q_opt, SPSA):
            # NOTE: Do NOT call backward() here.

            def closure():
                q1 = self.q1(o1, a1).squeeze(-1)  # [B]
                q2 = self.q2(o1, a1).squeeze(-1)  # [B]
                # assert torch.isfinite(q1).all(), "Non-finite Q1 values!"
                # assert torch.isfinite(q2).all(), "Non-finite Q2 values!"
                loss = self.q_loss_fn(q1, y) + self.q_loss_fn(q2, y)
                return loss

            loss_q = self.q_opt.step(closure)
            assert torch.isfinite(loss_q).all(), "Non-finite critic loss!"


        else:

            q1 = self.q1(o1, a1).squeeze(-1)  # [B]
            q2 = self.q2(o1, a1).squeeze(-1)  # [B]
            assert torch.isfinite(q1).all(), "Non-finite Q1 values!"
            assert torch.isfinite(q2).all(), "Non-finite Q2 values!"
            loss_q = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)


            self.q_opt.zero_grad(set_to_none=True)
            loss_q.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.q1.parameters()) + list(self.q2.parameters()),
                                               max_norm = self.grad_clip
                )


            self.q_opt.step()

        # ---------------------- actor update ----------------------
        a_new, logp_new, _ = self.policy.sample(o1)
        q1_new = self.q1(o1, a_new)
        q2_new = self.q2(o1, a_new)
        min_q_new = torch.min(q1_new, q2_new)
        self.q_stats['max'].append(float(torch.max(min_q_new).detach().cpu().item()))
        self.q_stats['min'].append(float(torch.min(min_q_new).detach().cpu().item()))

        # actor minimizes: E[ alpha*log_pi - Q ]
        loss_pi = (self.alpha.detach() * logp_new - min_q_new).mean()

        if not torch.isfinite(loss_pi):
            raise FloatingPointError(f"Non-finite actor loss: {loss_pi.item()}")

        self.policy_opt.zero_grad(set_to_none=True)
        loss_pi.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                           max_norm = self.grad_clip)
        self.policy_opt.step()

        # ---------------------- temperature update ----------------------
        if self.automatic_entropy_tuning:
            # alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean() # [generated]
            # alpha_loss = (-self.log_alpha.exp() * (logp_new + self.target_entropy).detach()).mean()
            alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
            # https: // docs.cleanrl.dev / rl - algorithms / sac /  # implementation-details_1
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            self.log_alpha.clamp(np.log(self.alpha_range[0]), np.log(self.alpha_range[1]))



        # ---------------------- target networks ----------------------
        soft_update(self.q1_targ, self.q1, self.tau)
        soft_update(self.q2_targ, self.q2, self.tau)

        return float(loss_q.item()), float(loss_pi.item()), float(self.alpha.detach().cpu().item())

    def _get_td_target_expectations(self,o1,a1,o2,r_vals,r_probs,d):
        with torch.no_grad():
            a2, logp2, _ = self.policy.sample(o2)  # a2 [B,A], logp2 [B,1]
            q1_t = self.q1_targ(o2, a2)
            q2_t = self.q2_targ(o2, a2)
            min_q_t = torch.min(q1_t, q2_t).squeeze(-1)  # [B]
            ent_term = (self.alpha.detach() * logp2).squeeze(-1)  # [B]
            target_v = min_q_t - ent_term  # [B]
            td_targets = r_vals + (1.0 - d) * self.gamma * target_v.unsqueeze(1)  # [B,N]
            assert torch.sum(r_probs, dim=1).allclose(
                torch.ones(r_probs.shape[0], device=self.device)), "Reward probs do not sum to 1!"
            y = self._risk_measure(td_targets, r_probs)  # [B]
        return y

    # ---------------------- risk measure hook ----------------------
    def _risk_measure(self, vals: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        vals:  [B, N]  (TD targets per reward sample)
        probs: [B, N]  (pdf / weights for those reward samples)

        Return: [B] scalar risk-adjusted target for each transition.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def _reward_eval(self, info: dict) -> float:
        """
        Helper to compute the reward used for logging / reporting.
        Objective is different depending on which agent is being used
        """
        return float(info["true_reward"])

    def _parse_params(self, params: dict) -> dict:
        """Helper to capture init params for logging."""

        _ = params.pop('self')
        _ = params.pop('env')
        _ = params.pop('updates_per_step')
        params['alpha'] = f'[init: {params.pop("alpha_init")} lr: {params.pop("alpha_lr")} tune:{params.pop("automatic_entropy_tuning")}]'
        params['NN Shape'] = f"{params.pop('size_hidden_layers')} x {params.pop('num_hidden_layers')}"
        for key, val in params.items():
            if 'lr' in key:
                params[key] = f"{val:.1e}"
        return params


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
            'policy': copy.deepcopy(self.policy.state_dict()),
            'q1': copy.deepcopy(self.q1.state_dict()),
            'q2': copy.deepcopy(self.q2.state_dict()),
            'q1_targ': copy.deepcopy(self.q1_targ.state_dict()),
            'q2_targ': copy.deepcopy(self.q2_targ.state_dict()),
        }

    # -----------------------------
    # Checkpointing
    # -----------------------------
    def _checkpoint_score(self) -> float:
        """
        Compute a checkpoint score from `self.history["timeseries_filt_rewards"]`.

        `timeseries_filt_rewards` is accumulated every `self.log_interval` episodes as an
        (K, 2) array where each row is [mu, std], with:
          - mu  = mean episodic return over the last log interval
          - std = std. dev. over the last log interval

        By default we checkpoint on the mean of the filtered means (mu), i.e., mean over column 0.
        """
        ts = self.history.get("timeseries_filt_rewards", None)
        if ts is None:
            return float("-inf")

        ts = np.asarray(ts)
        if ts.size == 0:
            return float("-inf")

        if ts.ndim == 2 and ts.shape[1] >= 1:
            return float(np.mean(ts[-1:, 0]))  # last score only
            # i = min(self.log_interval, ts.shape[0]-1)
            # return float(np.mean(ts[-i:, 0]))

        # Fallback for unexpected shapes
        return float(np.mean(ts))

    def maybe_checkpoint(self,ep, score: Optional[float] = None, min_episodes=300) -> bool:
        """
        Cache a deep-copied in-memory checkpoint (self._checkpoint) if `score` improves.

        Args:
            score: Optional pre-computed score. If None, uses `_checkpoint_score()`.
            ep:    Optional episode index to store in checkpoint metadata.

        Returns:
            True if a new best checkpoint was created, else False.
        """
        if score is None:
            score = self._checkpoint_score()

        if not np.isfinite(score):
            return False

        if ep < min_episodes:
            return False

        if score > self._best_ckpt_score:
            self._best_ckpt_score = float(score)
            self._checkpoint = {
                "score": float(score),
                "episode": int(ep) if ep is not None else None,
                "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "data": self._get_fdata(),
            }
            return True

        return False

    def has_checkpoint(self) -> bool:
        return isinstance(self._checkpoint, dict) and ("data" in self._checkpoint)

    def save(self, prefix='',suffix='' ,use_checkpoint: bool = True):

        if use_checkpoint and self.has_checkpoint():
            data = copy.deepcopy(self._checkpoint["data"])
            # Attach lightweight metadata (no tensors) for traceability.
            data["_checkpoint"] = {k: v for k, v in self._checkpoint.items() if k != "data"}
            data["_checkpoint"]["best_score"] = float(self._best_ckpt_score)
        else:
            data = self._get_fdata()
            data["_checkpoint"] = None
        # data = self._get_fdata()
        fpath = self._get_fpath(prefix=prefix,suffix=suffix)
        save_pickle(data, fpath)

        # save fig as well
        if self.logger.fig is not None:
            self.policy.load_state_dict(self._checkpoint['data']['policy'])
            test_info = self.test(max_episodes=self.log_interval)
            self.logger._plot_history(
                traj_history=test_info["xy"],
                terminal_history=test_info["terminal"],
                filt_reward_history=self.history["timeseries_filt_rewards"],
                log_interval=self.log_interval,
            )

            fig_fpath = fpath.replace('.pkl', '.png')
            self.logger.fig.savefig(fig_fpath)

        print(f"\nObject saved to {fpath}\n")


    def _load_dict(self, fpath) -> dict:
        if fpath.lower() == 'latest':
            fname = self._get_fpath(save_dir = '', with_tstamp=False)
            models_dir = get_algorithm_dir() + 'models/'
            load_dict, fpath = load_latest_pickle(models_dir,base_fname=fname)
        else:
            try:
                models_dir = get_algorithm_dir() + 'models/'
                load_dict = load_pickle(models_dir + fpath)
            except:
                load_dict = load_pickle(fpath)
        return load_dict

    def load(self, fpath, only=None, verbose=True):
        load_dict = self._load_dict(fpath)

        if only is not None:
            assert isinstance(only, (str, list)), "only must be a string or list of strings."

            if isinstance(only, str):
                only = [only]
            for model_name  in only:
                assert model_name in load_dict, f"Model name '{model_name}' not found in loaded data."
                self.__dict__[model_name].load_state_dict(load_dict[model_name])
                if verbose: print(f"\nObject '{model_name}' loaded from {fpath}\n")

        else:
            self.policy.load_state_dict(load_dict['policy'])
            self.q1.load_state_dict(load_dict['q1'])
            self.q2.load_state_dict(load_dict['q2'])
            self.q1_targ.load_state_dict(load_dict['q1_targ'])
            self.q2_targ.load_state_dict(load_dict['q2_targ'])
            if verbose: print(f"\nObject loaded from {fpath}\n")


class SACAgent_AVE(SACAgent_BASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.fig_title = f"AVE-SAC Agent ({self.start_time_str})"
        self.logger.fig_title = f"AVE-SAC Agent ({self.start_time_str})"


    def _risk_measure(self, vals: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return torch.mean(vals, dim=1)

    # def _reward_eval(self, info: dict) -> float:
    #     """
    #     Helper to compute the reward used for logging / reporting.
    #     Objective is different depending on which agent is being used
    #     """
    #     # use cpt expectation reward for reporting
    #     ave_rew = self._last_terminal_scale * np.mean(info["reward_samples"])
    #     # self._last_terminal_scale = info["sampled_dones"].mean()
    #     #
    #     self._last_terminal_scale = self._last_terminal_scale * (1-info["sampled_dones"].mean())
    #     return ave_rew
    #

class SACAgent_EUT(SACAgent_BASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.fig_title = f"EUT-SAC Agent ({self.start_time_str})"

    def _risk_measure(self, vals: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        assert torch.sum(probs, dim=1).allclose(torch.ones(probs.shape[0], device=self.device)), "Probs do not sum to 1!"
        return torch.sum(vals*probs, dim=1)



class SACAgent_CPT(SACAgent_BASE):
    def __init__(self, cpt_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.fig_title = f"CPT-SAC Agent ({self.start_time_str})"
        self.logger.fig_title = f"CPT-SAC Agent ({self.start_time_str})"

        self.cpt = CumulativeProspectTheory(**cpt_params)
        self.init_params[''] = str(self.cpt)

        # self.q_opt = SPSA(list(self.q1.parameters()) + list(self.q2.parameters()))
        # warnings.warn("Using SPSA optimizer for CPT-SAC critic!")

        # n_freeze = 0
        # self.q_opt = SPSA(list(self.q1.parameters())[n_freeze*2:] +
        #                   list(self.q2.parameters())[n_freeze*2:])


        # self.q_opt = optim.RMSprop(list(self.q1.parameters()) + list(self.q2.parameters()),
        #                            lr=float(self.init_params['q_lr']))

        # self.q_opt = optim.RMSprop(list(self.q1.parameters())[n_freeze*2] +
        #                            list(self.q2.parameters())[n_freeze*2])

    def _risk_measure(self, vals: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return self.cpt.sample_expectation_batch(vals)

    def _reward_eval(self, info: dict) -> float:
        """
        Helper to compute the reward used for logging / reporting.
        Objective is different depending on which agent is being used
        """
        # use cpt expectation reward for reporting
        r_samples = info["reward_samples"].reshape(1, -1)
        assert r_samples.shape == (1,self.env.n_samples), f"Expected reward samples shape [1, N], got {r_samples.shape}"
        cpt_rew = self.cpt.sample_expectation_batch(r_samples)
        assert cpt_rew.shape == (1,), f"Expected CPT reward shape [1,], got {cpt_rew.shape}"

        # # Sanity check numpy against pytorch ------------------------
        # pt_r_samples = torch.tensor(r_samples, dtype=torch.float32, device=self.device)
        # pt_cpt_reward =  self.cpt.sample_expectation_batch(pt_r_samples)[0]
        # assert np.isclose(cpt_rew, pt_cpt_reward.detach().cpu().item()), f"CPT reward mismatch: numpy {cpt_rew} vs torch {pt_cpt_reward.detach().cpu().item()}"


        return cpt_rew[0]

class SACAgent_CPT_Ref(SACAgent_CPT):
    def __init__(self, reference_fname, cpt_params, *args, **kwargs):
        self.ref_level = 0.9 # percent of optimal reference policy


        super().__init__(cpt_params, *args, **kwargs)
        # self.fig_title = f"CPT-SAC Agent ({self.start_time_str})"
        self.logger.fig_title = f"CPT-SAC-REF Agent ({self.start_time_str})"

        self.reference_fname = reference_fname
        _load_dict = self._load_dict(reference_fname)
        assert self.cpt.b == 0, "SACAgent_CPT_Ref requires zero reference point."

        self.init_params[''] = str(self.cpt)
        self.init_params['Ref'] = f"{reference_fname.split('/')[-1]} | {self.ref_level:.2f}"

        self.q1_ref = QNetwork(self.state_dim, self.action_dim,
                           hidden_dim=self.size_hidden_layers,
                           n_hidden=self.num_hidden_layers).to(self.device)
        self.q2_ref = QNetwork(self.state_dim, self.action_dim,
                               hidden_dim=self.size_hidden_layers,
                               n_hidden=self.num_hidden_layers).to(self.device)

        self.q1_ref.load_state_dict(_load_dict['q1'])
        self.q2_ref.load_state_dict(_load_dict['q2'])
        self.q1_ref.eval()
        self.q2_ref.eval()


        # Re-check reward structure under new CPT perceptions

        gamma = self.gamma
        inf_sum = 1 / (1 - gamma)
        exp_cum_reward_dist2goal = self.env.reward_dist2goal * inf_sum
        exp_cum_reward_step = self.env.reward_step * inf_sum

        cum_reward_dist2goal = self.env.max_steps * self.env.reward_dist2goal
        cum_reward_step = self.env.max_steps * self.env.reward_step

        assert - self.cpt.lam * abs(self.env.reward_collide)**self.cpt.eta_n \
               < - self.cpt.lam * abs(cum_reward_step) ** self.cpt.eta_n\
            , f'REWARD VIOLATION: Crashing < Slow'

        assert - self.cpt.lam * abs(self.env.reward_collide) ** self.cpt.eta_n \
               < - self.cpt.lam * abs(exp_cum_reward_step) ** self.cpt.eta_n \
            , f'REWARD VIOLATION: Crashing < Slow'

        print("Reference Q-networks loaded and reward structure verified.")
        # if not self.env.reward_dist2goal_prog:
        #     assert self.reward_goal > cum_reward_dist2goal + cum_reward_step, f'REWARD VIOLATION: Goal Always Better than Close:'
        #     assert self.reward_goal > exp_cum_reward_dist2goal + exp_cum_reward_step, f'REWARD VIOLATION: Goal Always Better than Close:'

    def _get_td_target_expectations(self, o1,a1, o2, r_vals, r_probs, d):
        with torch.no_grad():
            # ---------------------- reference Q-values ----------------------
            q1_ref_vals = self.q1_ref(o1, a1).squeeze(-1)
            q2_ref_vals = self.q2_ref(o1, a1).squeeze(-1)
            q_ref_vals = torch.min(q1_ref_vals, q2_ref_vals)  # [B]

            # ---------------------- target ----------------------
            a2, logp2, _ = self.policy.sample(o2)  # a2 [B,A], logp2 [B,1]
            q1_t = self.q1_targ(o2, a2)
            q2_t = self.q2_targ(o2, a2)
            min_q_t = torch.min(q1_t, q2_t).squeeze(-1)  # [B]
            ent_term = (self.alpha.detach() * logp2).squeeze(-1)  # [B]
            target_v = min_q_t - ent_term  # [B]
            td_targets = r_vals + (1.0 - d) * self.gamma * target_v.unsqueeze(1)  # [B,N]

            # Artificial reference value for b
            td_targets = td_targets  # [B,N]

            assert torch.sum(r_probs, dim=1).allclose(
                torch.ones(r_probs.shape[0], device=self.device)), "Reward probs do not sum to 1!"

            _b =  q_ref_vals.unsqueeze(-1) * self.ref_level  # [B,1]
            y = self._risk_measure(td_targets - _b, r_probs)  # [B]
        return y



