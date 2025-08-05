from src.algorithms.A2C import DEFAULT_CONFIG
from src.algorithms.A2C.agent import A2CAgent
from src.env.continuous_nav_env import ContinuousNavigationEnv
from src.env.layouts import read_layout_dict

if __name__ == '__main__':
    layout_dict = read_layout_dict('example0')
    env = ContinuousNavigationEnv(**layout_dict,**DEFAULT_CONFIG )
    agent = A2CAgent(env)
    agent.train(max_episodes=5000)

