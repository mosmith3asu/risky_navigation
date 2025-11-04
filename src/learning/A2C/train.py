from learning.A2C import DEFAULT_CONFIG
from learning.A2C.agent import A2CAgent
# from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.env.continous_nav_env_V2 import ContinuousNavigationEnv

from src.env.layouts import read_layout_dict


if __name__ == '__main__':
    # layout_dict = read_layout_dict('example0')
    layout_dict = read_layout_dict('example1')
    # layout_dict = read_layout_dict('example2')
    # layout_dict = read_layout_dict('example1_BU')
    env = ContinuousNavigationEnv(**layout_dict,**DEFAULT_CONFIG )
    agent = A2CAgent(env)
    agent.train(max_episodes=20_000,rand_start_epi=1000)

