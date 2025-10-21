from learning.DDPG.ddpg_agent import DDPGAgent_EUT
from src.env.continous_nav_env_delay_comp import ContinousNavEnv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    layout = 'example1'

    env_config = {}
    env_config['dynamics_belief'] = {
        # 'b_max_lin_vel': (1.0, 0.5),
        # 'b_max_lin_acc': (0.5, 0.2),
        # 'b_max_rot_vel': (np.pi / 2, 0.2 * np.pi)
        'b_max_lin_vel': (1.0, 1e-6),
        'b_max_lin_acc': (0.5, 1e-6),
        'b_max_rot_vel': (np.pi / 4, 1e-6)
    }
    env_config['n_samples'] = 2
    env_config['vgraph_resolution'] = 50


    agent_config = {
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'buffer_size': 100_000,
        # 'buffer_size': 50_000,
        # 'buffer_size': 20_000,
        'batch_size': 256,
        'grad_clip': 0.75,

        'warmup_epis': 10,
        'random_start_epis': 30,

        'ou_noise': {
                'theta': 0.15,
                'sigma': 0.4,
                'sigma_decay': 0.995,
                'sigma_min': 0.05,
                'mode': "reflect"
        }

    }

    env = ContinousNavEnv.from_layout(layout, **env_config)
    agent = DDPGAgent_EUT(env, **agent_config)
    agent.train(max_episodes=10_000)
    agent.save()
    plt.show()



    agent.load('latest')
    agent.random_start_epis = 0

    agent.train()




# from src.algorithms.DDPG.ddpg_agent import DDPGAgent_EUT
# from src.env.continous_nav_env_delay import DelayedContinuousNavigationEnv,ContinuousNavigationEnvBase
# from src.env.layouts import read_layout_dict
# import numpy as np
#
# if __name__ == '__main__':
#     VECTORIZE = True
#     PARALLEL_ENVS = 15
#     LAYOUT = 'example2'
#
#     dynamics_belief = {
#         # 'min_lin_vel': (0.0, 0.1),
#         # 'max_lin_vel': (1.0, 0.1),
#         # 'max_lin_acc': (0.5, 0.1),
#         # 'max_rot_vel': (np.pi/2, 0.1)
#
#         'max_lin_vel': (1.0, 0.5),
#         'max_lin_acc': (0.5, 0.5),
#         'max_rot_vel': (np.pi / 2, 0.5 * np.pi)
#
#         # 'max_lin_vel': (1.0, 4),
#         # 'max_lin_acc': (0.5, 4),
#         # 'max_rot_vel': (np.pi / 2, 4 * np.pi)
#     }
#
#     layout_dict = read_layout_dict(LAYOUT)
#     base_env = ContinuousNavigationEnvBase(vgraph_resolution=(20, 20), **layout_dict)
#     sampled_env = DelayedContinuousNavigationEnv.build_sync_vector_env(
#         n_envs=PARALLEL_ENVS,
#         layout_name=LAYOUT,
#         vgraph=base_env.vgraph,
#         dynamics_belief=dynamics_belief
#     )
#
#     agent_config = {
#         'actor_lr': 1e-4,
#         'critic_lr': 1e-3,
#         'gamma': 0.99,
#         'tau': 0.005,
#         'buffer_size': 100_000,
#         'batch_size': 256,
#         'grad_clip': 0.75,
#
#         'rollouts_per_epi': 10,
#         'warmup_epis': 10,
#         'random_start_epis': 30,
#     }
#     agent = DDPGAgent_EUT(base_env, sampled_env, **agent_config)
#
#
#     agent.train(max_episodes=300)
#     agent.save()
#     agent.load('latest')
#     agent.random_start_epis = 0
#
#     agent.train()
#
#
#
