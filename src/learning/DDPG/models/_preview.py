from learning.DDPG.ddpg_agent import DDPGAgent,DDPGAgentVec
from src.env.continous_nav_env import build_sync_vector_env, ContinuousNavigationEnvBase
from src.env.layouts import read_layout_dict
import time

if __name__ == '__main__':
    VECTORIZE = True
    PARALLEL_ENVS = 5
    LAYOUT = 'example2'
    agent_config = {}


    layout_dict = read_layout_dict(LAYOUT)
    env = ContinuousNavigationEnvBase(vgraph_resolution=(20, 20),**layout_dict)
    if VECTORIZE:
        vec_env = build_sync_vector_env(
            n_envs=PARALLEL_ENVS,
            layout_name=LAYOUT,
            vgraph = env.vgraph
        )
        agent = DDPGAgentVec(vec_env, **agent_config)
    else:
        agent = DDPGAgent(env,**agent_config) #TODO: Base class needs updating

    agent.load('latest')
    agent.random_start_epis = 0
    agent.warmup_epis = 0
    agent.ou.sigma = 0 # No noise
    agent.train()
    while True:
        time.sleep(0.1)

    for t in (epi):
        state = env.state
        action = agent.act(state)

