import os
import numpy as np
import matplotlib.pyplot as plt
from src.learning.SAC.sac_agent import SACAgent_AVE,SACAgent_EUT, SACAgent_CPT, SACAgent_CPT_Ref
from src.env.continous_nav_env_delay_comp import ContinousNavEnv
import torch


class SACAgentSampler:
    """
    Class to load and sample SAC agents trained on a single layouts
    Features:
    - automatic loading of all agents in specified directory for given layout
    - resampling of agents
    - getting actions from current agent given observation
    """
    def __init__(self, env, agents_dir= './final_agents/'):
        self.env = env
        self.layout = env.layout
        self._verbose = True

        self._agents_dir = agents_dir
        self._agent_fnames = self._get_fnames(self._agents_dir)
        self._agents = self._load_agents(self._agent_fnames)
        self.num_agents = len(self._agent_fnames)

        self._current_agent = None
        self._current_agent_index = None
        self._current_agent_fname = None


    def set_agent(self, index):
        """
        sets the current agent to the agent at specified index
        Useful for evaluating specific agents or debugging if necessary
        """
        assert 0 <= index < self.num_agents, f"Index {index} out of bounds for number of agents {self.num_agents}."
        self._current_agent_index = index
        self._current_agent_fname = self._agent_fnames[index]
        self._current_agent = self._agents[index]
        return self._current_agent_index, self._current_agent_fname

    def resample_agent(self):
        """
        resamples the current agent randomly from the loaded agents
        returns the index and fname of the newly sampled agent
        """
        assert len(self._agents) > 0, "No agents loaded to sample from."


        self._current_agent_index = np.random.randint(0, self.num_agents)
        self._current_agent_fname = self._agent_fnames[self._current_agent_index]
        self._current_agent = self._agents[self._current_agent_index]
        return self._current_agent_index, self._current_agent_fname

    def act(self, observation, explore=True):
        """
        returns action from current agent given observation
        LEAVE EXPLORATION = TRUE as this models stochasticity in human behavior
        Use  o1 = self.env.observation to get observation from environment

        """
        assert self._current_agent is not None, "No agent is currently set. Please set or resample an agent before calling act()."
        assert isinstance(observation, (np.ndarray, torch.Tensor)), "Observation must be a numpy array or a pytorch tensor."
        assert observation.shape == self.env.observation_space.shape, f"Observation shape {observation.shape} does not match environment observation space shape {self.env.observation_space.shape}."

        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=self._current_agent.device)
        a1_pt = self._current_agent._act(observation, explore=explore)
        a1 = a1_pt.cpu().numpy()
        return  a1

    def _get_fnames(self,agents_dir):
        """gets all fnames in specified directory"""
        fnames = []
        for fname in os.listdir(agents_dir):
            if fname.endswith('.pickle') and self.layout in fname:
                fnames.append(os.path.join(agents_dir, fname))

        if self._verbose:
            print(f"Found {len(fnames)} agent files in {agents_dir}")
        assert len(fnames) > 0, f"No agent files found in {agents_dir} for layout {self.layout}"
        return fnames

    def _load_agents(self,agent_fnames):
        agents = []
        for fname in agent_fnames:
            # layout = fname.split('_')[-1].split('.')[0]
            fname_nodate = fname.split("/")[-1][20:]
            class_name = fname_nodate.split('_' + self.layout)[0]

            if class_name == 'SACAgent_AVE':
                _agent = SACAgent_AVE(self.env)
            elif class_name == 'SACAgent_EUT':
                _agent = SACAgent_EUT(self.env)
            elif class_name == 'SACAgent_CPT':
                _agent = SACAgent_CPT(self.env)
            else:
                raise ValueError(f"Unknown agent class name: {class_name}")
            _agent.load(fname, verbose=self._verbose)
            _agent.eval()
            agents.append(_agent)
        return agents






def main():
    """
    Example usage of SACAgentSampler to generate dataset from randomly sampled agents
    Notes and terminology:
    - state: true underlying state of the MDP (x,y,θ,v + dist2goal + lidar readings) used for reward calculations [!!! not sufficient for agent input !!!]
    - observation: full featerization of mdp information and input to agents/nets (delayed state + past actions)
    """


    layout = 'spath'
    env_config = {}
    env_config['dt'] = 0.1
    env_config['delay_steps'] = 10  # (two-way delay)  delay_time = delay_steps * dt
    env_config['n_samples'] = 500  # 50
    env_config['vgraph_resolution'] = 50
    env_config['max_steps'] = 600

    env = ContinousNavEnv.from_layout(layout, **env_config)
    sampler = SACAgentSampler(env)


    ################################################################################
    # GENERATE DATASET #############################################################
    # Generate a dataset of slices from randomly sampled agents
    # can do this once and save the big dataset for training your own model later
    # this will allow more easily to do a train/test set split

    NUM_EPISODES = 5
    for episode in range(NUM_EPISODES):
        # Resample agent -------------------------------------------------------------
        sampler.resample_agent()        # randomly sample new agent for each episode
        print("Episode:", episode+1,
              f"Using agent [{sampler._current_agent_index}]: "
              f"{sampler._current_agent_fname}")

        # Episode rollout ---------------------------------------------------------------
        # perform full episode rollout with current sampled agent, then slice up trajectory after episode
        env.reset()
        ep_observations = []
        ep_actions = []
        for i in range(env.max_steps):
            obs = env.observation # [delayed_state, series of past actions]
            action = sampler.act(obs)
            _, _, _, info = env.step(action) # uncertain dynamics mdp step (should only need info)

            true_reward = info['true_reward']       # whats happening in non-delayed reality (you dont need but here if want)
            true_state = info['true_next_state']    # whats happening in non-delayed reality (you dont need but here if want)
            true_done = info['true_done']           # whats happening in non-delayed reality

            ep_observations.append(obs) # append this timestep's state-action
            ep_actions.append(action)   # append this timestep's state-action

            if true_done:
                break


        # TODO: slice these buffers up into necessary components and add to your training buffer
        """
        # EXAMPLE/TEMPLATE IMPLEMENTATION: -----------------------------------------------------------
        n_timseries = 5 # want to use last 5 state-action time steps to predict next action
        for t in range(len(ep_observations) - n_timseries):
        
            obs_slice = ep_observations[t:t+n_timseries] # model input
            action_slice = ep_actions[t:t+n_timseries]   # model input
            # !! not sure what format input to networks are so you have to do some formatting here !!
                       
            next_action = ep_actions[t+n_timseries]      # model output
            
            dataset.add(obs_slice, action_slice, next_action) # add to training buffer
        # -----------------------------------------------------------------------------------
    
        """

    # END GENERATE DATASET #########################################################
    ################################################################################
    """ 
    dataset.save() # save dataset to disk so we do not have to keep doing this 
    # you can then perform train/test split on saved dataset later
    """





if __name__ == "__main__":
    main()
