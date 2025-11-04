from learning.A2C.agent import A2CAgentVec
from src.env.continous_nav_env import build_sync_vector_env, ContinuousNavigationEnvVec
from src.env.layouts import read_layout_dict

if __name__ == '__main__':
    # layout_name = 'example0'
    # layout_name = 'example1'
    layout_name = 'example2'
    layout_dict = read_layout_dict(layout_name)
    dummy_env = ContinuousNavigationEnvVec(vgraph_resolution=(20, 20),**layout_dict)

    vec_env = build_sync_vector_env(
        n_envs=5,
        layout_name=layout_name,
        vgraph = dummy_env.vgraph
    )
    agent = A2CAgentVec(vec_env)
    agent.train()

