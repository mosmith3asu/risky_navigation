from .ddpg_agent import DDPGAgent



class EUT_DDPGAgent(DDPGAgent):
    """Applies expected utility theory (EUT) to the DDPG algorithm for risk-sensitive reinforcement learning."""
    def __init__(self, env, **kwargs):
        super(EUT_DDPGAgent, self).__init__(env, **kwargs)
        self.risk_sensitive = False  # Ensure risk sensitivity is disabled for EUT

    def train(self, max_episodes=1_000):
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
            p_rand_start = max(0, (1 - ep / self.random_start_epis) if self.random_start_epis > 0 else 0)
            self.env.set_attr('p_rand_state', p_rand_start)

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
                [running_xys[i].append(next_states[i, :2]) for i in range(self.num_envs)]

                for i in range(self.num_envs):
                    if dones[i]:
                        self.total_rollouts += 1
                        this_epi_rollouts += 1

                        # Store in history and reset buffers
                        self.history['terminal'].append(info['reason'][i])
                        self.history['ep_len'].append(running_lens[i]);
                        running_lens[i] = 0
                        self.history['ep_reward'].append(running_rewards[i]);
                        running_rewards[i] = 0
                        self.history['xy'].append(np.array(running_xys[i]));
                        running_xys[i].clear()
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
                  f"| P(rand state):{self.env.get_attr('p_rand_state')[0]:.2f}"
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
    def compute_target_q(self, rewards, next_q_values, dones):
        """
        Compute the target Q values using the Expected Utility Theory (EUT) approach.
        """
        # Standard Q-learning target
        target_q = rewards + self.gamma * next_q_values * (1 - dones)
        return target_q


if __name__ == "__main__":
    main()
