from src.algorithms.DDPG.ddpg_agent import DDPGAgentRationalProspects
from src.env.continous_nav_env_delay import build_sync_vector_env, DelayedContinuousNavigationEnv
from src.env.layouts import read_layout_dict

if __name__ == '__main__':
    VECTORIZE = True
    PARALLEL_ENVS = 5
    LAYOUT = 'example2'

    layout_dict = read_layout_dict(LAYOUT)
    base_env = DelayedContinuousNavigationEnv(vgraph_resolution=(10, 10),**layout_dict)
    sampled_env = build_sync_vector_env(
        n_envs=PARALLEL_ENVS,
        layout_name=LAYOUT,
        vgraph = base_env.vgraph
    )

    agent_config = {
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'buffer_size': 100_000,
        'batch_size': 256,
        'grad_clip': 0.75,

        'rollouts_per_epi': 10,
        'warmup_epis': 10,
        'random_start_epis': 30,
    }
    agent = DDPGAgentRationalProspects(base_env, sampled_env, **agent_config)


    agent.train(max_episodes=300)
    agent.save()
    agent.load('latest')
    agent.random_start_epis = 0

    agent.train()



