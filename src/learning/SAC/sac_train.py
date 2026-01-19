from src.learning.SAC.sac_agent import SACAgent_AVE,SACAgent_EUT, SACAgent_CPT, SACAgent_CPT_Ref
from src.env.continous_nav_env_delay_comp import ContinousNavEnv
import math
import matplotlib.pyplot as plt

def main(layout, MAX_EPIS = 3_000):



    env_config = {}
    env_config['dynamics_belief'] = {
        'b_min_lin_vel': (0.0, 1e-6),
        'b_max_lin_vel': (1.5, 0.5),
        'b_max_lin_acc': (0.5, 0.2),
        'b_max_rot_vel': (math.pi / 2.0, math.pi / 6.0)
        # 'b_min_lin_vel': (0.0, 1e-6),
        # 'b_max_lin_vel': (1.0, 1e-6),
        # 'b_max_lin_acc': (0.5, 1e-6),
        # 'b_max_rot_vel': (np.pi / 4, 1e-6)
    }

    env_config['dt'] = 0.1
    env_config['delay_steps'] = 10 # (two-way delay)  delay_time = delay_steps * dt
    env_config['n_samples'] = 500 #50
    env_config['vgraph_resolution'] = 50
    env_config['max_steps'] = 600


    # OG Config
    agent_config = {  }


    env = ContinousNavEnv.from_layout(layout, **env_config)


    ############ AVE AGENT ##############
    # agent_config['randstart_sched'] = (0, 0, 250)
    # agent_config['grad_clip'] = 1.0
    # agent = SACAgent_AVE(env, **agent_config)
    # # agent.load(r'D:\PycharmProjects\risky_navigation\src\learning\SAC\models\2026-01-15_02-38-46_SACAgent_AVE_spath')


    ############ CPT AGENT ##############
    agent_config['alpha_init'] = 0.02
    agent_config['randstart_sched'] = (0.9,0.25,250)
    # agent_config['randstart_sched'] = (0,0,250)
    agent_config['warmup_steps'] = 10_000 # fill replay memory
    agent_config['policy_lr'] = 1e-4/2
    agent_config['q_lr'] = 5e-4/2
    # agent_config['grad_clip'] = 1.0
    agent = SACAgent_CPT(env, **agent_config)

    agent.load(r"D:\PycharmProjects\risky_navigation\src\learning\SAC\models\2026-01-19_01-11-42_SACAgent_AVE_spath")
    # agent.load(r'D:\PycharmProjects\risky_navigation\src\learning\SAC\models\2026-01-15_02-38-46_SACAgent_AVE_spath')

    ########### REFERENCE CPT AGENT ##############
    # agent_config['alpha_init'] = 0.02
    # agent_config['randstart_sched'] = (0, 0, 250)
    # agent_config['warmup_steps'] = 5_000  # fill replay memory
    # agent_config['policy_lr'] = 1e-4 / 2
    # agent_config['q_lr'] = 5e-4 / 2
    # agent_config['grad_clip'] = 1.0
    #
    # ref_model = r'D:\PycharmProjects\risky_navigation\src\learning\SAC\models\2026-01-15_02-38-46_SACAgent_AVE_spath'
    # agent = SACAgent_CPT_Ref(ref_model,env, **agent_config)
    # agent.load(ref_model, only='policy')
    # # agent.ou.plot(T=MAX_EPIS*1.1)

    agent.train(max_episodes=MAX_EPIS)
    agent.save()
    plt.show()

if __name__ == '__main__':
    main('spath')
