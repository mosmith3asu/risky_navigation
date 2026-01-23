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
    env_config['n_samples'] = 1000 #50
    env_config['vgraph_resolution'] = 50
    env_config['max_steps'] = 600


    # OG Config
    agent_config = {  }


    env = ContinousNavEnv.from_layout(layout, **env_config)


    ############ AVE AGENT ##############
    # agent_config['grad_clip'] = 1.0
    # agent_config['note'] = 'reduced rshape'
    # agent_config['randstart_sched'] = (1.0, 0.75, 250)
    # agent_config['rshape_sched'] = (1.0, 0.0, 1000)
    # # # agent_config['normailize_reward'] = True
    # agent = SACAgent_AVE(env, **agent_config)
    # # agent = SACAgent_EUT(env, **agent_config)
    # # # agent.load(r'D:\PycharmProjects\risky_navigation\src\learning\SAC\models\2026-01-15_02-38-46_SACAgent_AVE_spath')
    # agent.load(r"D:\PycharmProjects\risky_navigation\src\learning\SAC\models\2026-01-19_01-11-42_SACAgent_AVE_spath")

    ############ CPT AGENT ##############
    agent_config['note'] = 'reduced rshape'

    agent_config['alpha_init'] = 0.05
    agent_config['randstart_sched'] = (1.0,0.75,250)
    agent_config['rshape_sched'] = (1.0, 0, 1000)
    # agent_config['warmup_steps'] = 5_000 # fill replay memory
    # agent_config['warmup_steps'] = 10_000 # fill replay memory

    # agent_config['warmup_steps'] = 10_000 # fill replay memory
    # agent_config['policy_lr'] = 1e-4/2
    # agent_config['q_lr'] = 5e-4/2
    # agent_config['automatic_entropy_tuning'] = False
    # agent_config['grad_clip'] = 1.0
    agent_config['loads'] = r"D:\PycharmProjects\risky_navigation\src\learning\SAC\models\2026-01-21_18-29-20_SACAgent_AVE_spath"
    agent_config['normailize_reward'] = True

    cpt_params = {
        "b": 0,
        "lam": 1,
        # "lam": 1/2.25,
        "eta_p": 0.88,
        "eta_n": 1,
        "delta_p": 1,
        "delta_n": 1,
        "offset_ref": False
    }
    # cpt_params = {
    #     "b": 0,
    #     "lam": 1,
    #     # "lam": 1/2.25,
    #     "eta_p": 1.0,
    #     "eta_n": 0.88,
    #     "delta_p": 1,
    #     "delta_n": 1,
    #     "offset_ref": False
    # }
    #
    # # }
    # # cpt_params = {
    # #     "b": 0,
    # #     "lam": 1,
    # #     "eta_p": 1,
    # #     "eta_n": 1,
    # #     "delta_p": 1,
    # #     "delta_n": 1,
    # #     "offset_ref": False
    # # }
    #
    agent = SACAgent_CPT(cpt_params, env, **agent_config)
    # # progress dist2goal
    # agent.load(r"D:\PycharmProjects\risky_navigation\src\learning\SAC\models\2026-01-19_01-11-42_SACAgent_AVE_spath")
    # # # standard dist2goal
    # # agent.load(r'D:\PycharmProjects\risky_navigation\src\learning\SAC\models\2026-01-15_02-38-46_SACAgent_AVE_spath')

    ########### REFERENCE CPT AGENT ##############
    # cpt_params = {
    #     "b": 0,
    #     "lam": 1,
    #     # "lam": 1/2.25,
    #     "eta_p": 0.7,
    #     "eta_n": 0.95,
    #     "delta_p": 1,
    #     "delta_n": 1,
    #     "offset_ref": False
    # }
    #
    # # ref_model = r'D:\PycharmProjects\risky_navigation\src\learning\SAC\models\2026-01-15_02-38-46_SACAgent_AVE_spath'
    # ref_model = r"D:\PycharmProjects\risky_navigation\src\learning\SAC\models\2026-01-19_01-11-42_SACAgent_AVE_spath"
    # agent = SACAgent_CPT_Ref(ref_model,cpt_params,env, **agent_config)
    # # agent.load(ref_model, only='policy')
    # #



    agent.train(max_episodes=MAX_EPIS)
    agent.save()
    plt.show()

if __name__ == '__main__':
    main('spath')
    # main('spath_narrow')