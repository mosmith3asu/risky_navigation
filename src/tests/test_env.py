import numpy as np
from env.continous_nav_env_delay_comp import ContinousNavEnv
from learning.utils import OUNoise


def test_terminal_goal(env):
    print("Testing Terminal Goal States...", end='')

    ##############################################################
    # Test Goal ##################################################
    env.reset()
    n_samples = 10
    max_vel = env.state.robot.true_max_lin_vel
    goal_loc = np.array(env.goal).reshape(1,2)

    # EXACTLY at goal (done=True)
    true_state          = env.state._true_robot_state.copy().reshape(1,4)
    true_state[:,:2]    = goal_loc
    true_state[:, env.state.iv] = 0
    sample_states       = np.tile(true_state, (n_samples,1))
    assert np.all(env._check_goal(true_state)), "Goal check failed when robot is EXACTLY at the goal (true state)."
    assert np.all(env._check_goal(sample_states)), "Goal check failed when robot is EXACTLY at the goal (sampled state)."

    # Within goal radius (done=True)  #######################################
    mag = env.goal_radius * np.random.rand()
    dxy = (2 * np.random.rand(2) - 1)
    dxy = mag * dxy / np.linalg.norm(dxy)
    assert np.linalg.norm(dxy) <= env.goal_radius
    true_state[:,:2] +=  dxy
    true_state[:, env.state.iv] = 0
    assert np.all(env._check_goal(true_state)), "Goal check failed when robot is WITHIN the goal (true state)."

    mag = env.goal_radius * np.random.rand(n_samples).reshape(n_samples, 1)
    dxy = (2 * np.random.rand(n_samples*2) - 1).reshape(n_samples,2)
    dxy = mag * (dxy / np.linalg.norm(dxy, axis=1)[:, np.newaxis])
    assert np.all(np.linalg.norm(dxy,axis=1) <= env.goal_radius)
    sample_states[:,:2] += dxy
    sample_states[:, env.state.iv]  = 0
    assert np.all(env._check_goal(sample_states)), "Goal check failed when robot is WITHIN the goal (sampled state)."

    # Within goal radius but not stopped (done=False)  #######################################
    true_state[:, env.state.iv]     += np.random.rand() * max_vel + env.goal_velocity
    assert not np.any(env._check_goal(true_state)), "Goal check incorrectly passed when robot is WITHIN + NOT STOPPED the goal (true state)."
    sample_states[:, env.state.iv]  += np.random.rand(n_samples)* max_vel  + env.goal_velocity
    assert not np.any(env._check_goal(sample_states)), "Goal check incorrectly passed when robot is WITHIN + NOT STOPPED the goal (sampled state)."

    # Outside of goal radius and not stopped (done=False) #######################################
    mag =  env.goal_radius * (1 + np.random.rand())
    dxy = (2 * np.random.rand(2) - 1)
    dxy = mag * dxy / np.linalg.norm(dxy)
    true_state[:, :2] = goal_loc
    true_state[:,:2]    += dxy
    assert not np.any(env._check_goal(true_state)), "Goal check incorrectly passed when robot is OUTSIDE + NOT STOPPED the goal (true state)."

    mag = env.goal_radius * (1 + np.random.rand(n_samples).reshape(n_samples, 1))
    dxy = (2 * np.random.rand(n_samples * 2) - 1).reshape(n_samples, 2)
    dxy = mag * (dxy / np.linalg.norm(dxy, axis=1)[:, np.newaxis])
    sample_states[:, :2]   = goal_loc
    sample_states[:, :2]    += dxy
    assert not np.any(env._check_goal(sample_states)), "Goal check incorrectly passed when robot is OUTSIDE + NOT STOPPED the goal (sampled state)."

    # Outside of goal radius but stopped (done=False)  #######################################
    true_state[:, env.state.iv] = 0
    assert not np.any(env._check_goal(true_state)), "Goal check incorrectly passed when robot is OUTSIDE + STOPPED the goal (true state)."
    sample_states[:, env.state.iv] = 0
    assert not np.any(env._check_goal(sample_states)), "Goal check incorrectly passed when robot is OUTSIDE + STOPPED the goal (sampled state)."

    print(" [passed]")

    return True

def test_transition(env):
    print("Testing Transition...", end='')
    ou = OUNoise(env.action_space.shape[0], mu=0.0, theta=0.15, sigma=0.2)

    env.reset()
    ou.reset()

    # Wait action ##########################################
    o1 = env.observation
    a1 = np.array([0.0, 0.0])
    ns_samples, r_samples, done_samples, info = env.step(a1)
    o2 = env.observation
    o1_dict = env.state.decompose_state(o1,as_dict=True)
    o2_dict = env.state.decompose_state(o2,as_dict=True)

    assert np.all(o1_dict['robot_state'] == o2_dict['robot_state']), "Robot position changed on wait action."
    assert np.all(o1_dict['ahist'] == o2_dict['ahist']), "Robot velocity changed on wait action."
    assert np.all(o1_dict['dist2goal'] == o2_dict['dist2goal']), "dist2goal  changed on wait action."
    assert np.all(o1_dict['lidar_dists'] == o2_dict['lidar_dists']), "lidars changed on wait action."

    # Y-action ##########################################
    env.reset()
    abase = np.array([0.0, 0.1])
    for i in range(env.delay_steps):
        o1 = env.observation
        a1 = abase * (i+1)
        ns_samples, r_samples, done_samples, info = env.step(a1)
        o2 = env.observation
        o1_dict = env.state.decompose_state(o1,as_dict=True)
        o2_dict = env.state.decompose_state(o2,as_dict=True)

        assert not np.all(o1_dict['ahist'] == o2_dict['ahist']), "Robot ahist did not change on action."
        assert np.all(o2_dict['ahist'][-2:] == a1), 'New observation did not contain latest action'

        # Stays same while buffer is filling with new action
        assert np.all(o1_dict['robot_state'] == o2_dict['robot_state']), "Robot position changed before expected"
        assert np.all(o1_dict['dist2goal'] == o2_dict['dist2goal']), "dist2goal changed before expected"
        assert np.all(o1_dict['lidar_dists'] == o2_dict['lidar_dists']), "lidars changed before expected"

    for i in range(5):
        o1 = env.observation
        a1 = abase * (i+1 + env.delay_steps)
        ns_samples, r_samples, done_samples, info = env.step(a1)
        o2 = env.observation
        o1_dict = env.state.decompose_state(o1,as_dict=True)
        o2_dict = env.state.decompose_state(o2,as_dict=True)

        assert not np.all(o1_dict['ahist'] == o2_dict['ahist']), "Robot ahist did not change on action."
        assert np.all(o2_dict['ahist'][-2:] == a1), 'New observation did not contain latest action'
        assert not np.all(o1_dict['robot_state'] == o2_dict['robot_state']), "Robot position did not change on action."
        assert not np.all(o1_dict['dist2goal'] == o2_dict['dist2goal']), "dist2goal  changed on wait action."
        assert not np.all(o1_dict['lidar_dists'] == o2_dict['lidar_dists']), "lidars changed on wait action."
        assert not np.all(o1_dict['robot_state'] == env.state._true_robot_state), "Observation should not match true robot state."
        assert not np.all(o2_dict['robot_state'] == env.state._true_robot_state), "Observation should not match true robot state."

    # X-action ##########################################
    env.reset()
    abase = np.array([0.1, 0.0])
    for i in range(env.delay_steps):
        o1 = env.observation
        a1 = abase * (i+1)
        ns_samples, r_samples, done_samples, info = env.step(a1)
        o2 = env.observation
        o1_dict = env.state.decompose_state(o1,as_dict=True)
        o2_dict = env.state.decompose_state(o2,as_dict=True)

        assert not np.all(o1_dict['ahist'] == o2_dict['ahist']), "Robot ahist did not change on action."
        assert np.all(o2_dict['ahist'][-2:] == a1), 'New observation did not contain latest action'

        # Stays same while buffer is filling with new action
        assert np.all(o1_dict['robot_state'] == o2_dict['robot_state']), "Robot position changed before expected"
        assert np.all(o1_dict['dist2goal'] == o2_dict['dist2goal']), "dist2goal changed before expected"
        assert np.all(o1_dict['lidar_dists'] == o2_dict['lidar_dists']), "lidars changed before expected"

    for i in range(5):
        o1 = env.observation
        a1 = abase * (i+1 + env.delay_steps)
        ns_samples, r_samples, done_samples, info = env.step(a1)
        o2 = env.observation
        o1_dict = env.state.decompose_state(o1,as_dict=True)
        o2_dict = env.state.decompose_state(o2,as_dict=True)

        assert not np.all(o1_dict['ahist'] == o2_dict['ahist']), "Robot ahist did not change on action."
        assert np.all(o2_dict['ahist'][-2:] == a1), 'New observation did not contain latest action'
        assert not np.all(o1_dict['robot_state'] == o2_dict['robot_state']), "Robot position did not change on action."
        assert not np.all(o1_dict['dist2goal'] == o2_dict['dist2goal']), "dist2goal  changed on wait action."
        assert not np.all(o1_dict['lidar_dists'] == o2_dict['lidar_dists']), "lidars changed on wait action."
        assert not np.all(o1_dict['robot_state'] == env.state._true_robot_state), "Observation should not match true robot state."
        assert not np.all(o2_dict['robot_state'] == env.state._true_robot_state), "Observation should not match true robot state."

    # XY-action ##########################################
    env.reset()
    abase = np.array([0.1, 0.1])
    for i in range(env.delay_steps):
        o1 = env.observation
        a1 = abase * (i + 1)
        ns_samples, r_samples, done_samples, info = env.step(a1)
        o2 = env.observation
        o1_dict = env.state.decompose_state(o1, as_dict=True)
        o2_dict = env.state.decompose_state(o2, as_dict=True)

        assert not np.all(o1_dict['ahist'] == o2_dict['ahist']), "Robot ahist did not change on action."
        assert np.all(o2_dict['ahist'][-2:] == a1), 'New observation did not contain latest action'

        # Stays same while buffer is filling with new action
        assert np.all(o1_dict['robot_state'] == o2_dict['robot_state']), "Robot position changed before expected"
        assert np.all(o1_dict['dist2goal'] == o2_dict['dist2goal']), "dist2goal changed before expected"
        assert np.all(o1_dict['lidar_dists'] == o2_dict['lidar_dists']), "lidars changed before expected"

    for i in range(5):
        o1 = env.observation
        a1 = abase * (i + 1 + env.delay_steps)
        ns_samples, r_samples, done_samples, info = env.step(a1)
        o2 = env.observation
        o1_dict = env.state.decompose_state(o1, as_dict=True)
        o2_dict = env.state.decompose_state(o2, as_dict=True)

        assert not np.all(o1_dict['ahist'] == o2_dict['ahist']), "Robot ahist did not change on action."
        assert np.all(o2_dict['ahist'][-2:] == a1), 'New observation did not contain latest action'
        assert not np.all(o1_dict['robot_state'] == o2_dict['robot_state']), "Robot position did not change on action."
        assert not np.all(o1_dict['dist2goal'] == o2_dict['dist2goal']), "dist2goal  changed on wait action."
        assert not np.all(o1_dict['lidar_dists'] == o2_dict['lidar_dists']), "lidars changed on wait action."
        assert not np.all(o1_dict['robot_state'] == env.state._true_robot_state), "Observation should not match true robot state."
        assert not np.all(o2_dict['robot_state'] == env.state._true_robot_state), "Observation should not match true robot state."

    print(" [passed]")

def test_state_inference(env):
    print("Testing Current State Inference...", end='')

    if not np.all(env.state.robot.belief_std <= 1e-3):
        print(" [skipped] (dynamics belief needs to be near certain).")
        return True

    env.reset()

    # Y-action ##########################################
    env.reset()
    abase = np.array([0.0, 0.1])

    # Fill beffer till fully new state
    for i in range(env.delay_steps):
        o1 = env.observation
        a1 = abase * (i + 1)
        ns_samples, r_samples, done_samples, info = env.step(a1)
        o2 = env.observation

    o2_dict = env.state.decompose_state(o2, as_dict=True)

    delayed_state = o2_dict['robot_state'].copy().reshape(1, 4)
    actions = np.array([o2_dict['ahist'][2*i:2*i+2] for i in range(env.delay_steps)])
    assert actions.shape == (env.delay_steps, 2), "Action history shape incorrect for inference test."
    assert np.all(actions == env.state.action_history), "Action history does not match for inference test."
    assert not np.all(delayed_state == env.state._true_robot_state), "Delayed state should not match true robot state before inference."


    inferred_robot_states = env.state.infer_curr_robot_state(
        robot_state = delayed_state,
        actions =actions
    )

    true_robot_state = env.state._true_robot_state
    checks = [np.allclose(shat, true_robot_state, atol=1e-4) for shat in inferred_robot_states]
    assert np.all(checks), 'Did not recover current state in inference'

    print(" [passed]")

def test_terminal_collision(env):
    print("Testing Terminal Obstacle Collision States...", end='')

    obstacles = env.obstacles
    for obs in obstacles:
        # print(f"  - Testing obstacle: {obs['type']}")
        if obs['type'] == 'rect':
            _test_rect_obstacle(env, obs, n_samples=20, mode='within')
            # _test_rect_obstacle(env, obs, n_samples=20, mode='outside') # Could hit other obstacle if more than one exists

        elif obs['type'] == 'circle':
            _test_circle_obstacle(env, obs, n_samples=20, mode='within')
            # _test_circle_obstacle(env, obs, n_samples=20, mode='outside') # Could hit other obstacle if more than one exists

        else:
            raise ValueError(f"Unknown obstacle type: {obs['type']}")

    print(" [passed]")
    return True

def _test_rect_obstacle(env, obstacle, n_samples=10, mode='within'):
    assert mode in ['within', 'outside'], "Mode must be either 'within' or 'outside'"
    cx, cy = obstacle['center']
    w = obstacle['width']
    h = obstacle['height']

    if mode == 'outside':
        xdir = np.array([np.random.choice([-1,1]) for _ in range(n_samples)])
        ydir = np.array([np.random.choice([-1,1]) for _ in range(n_samples)])

        dx = xdir* (w/2) + xdir * (np.random.rand(n_samples)) * (w)
        dy = ydir* (h/2) + ydir * (np.random.rand(n_samples)) * (h)

    elif mode == 'within':
        dx = (2 * np.random.rand(n_samples) - 1) * (w / 2)
        dy = (2 * np.random.rand(n_samples) - 1) * (h / 2)
    else:
        raise ValueError("Invalid mode specified.")

    x = cx + dx.reshape(n_samples, 1)
    y = cy + dy.reshape(n_samples,1)
    sample_states = np.hstack([x, y, np.zeros_like(x), np.zeros_like(x)])

    if mode == 'outside':
        assert not np.any(env._check_collision(sample_states)), \
            "Collision check failed for points outside rectangle obstacle."

    elif mode == 'within':
        assert np.all(env._check_collision(sample_states)), \
            "Collision check passed for points within rectangle obstacle."


    # return env._check_collision(sample_states)

def _test_circle_obstacle(env, obstacle, n_samples=10,mode='within'):
    raise NotImplementedError


def main():
    layout = 'example2'

    env_config = {}
    env_config['dynamics_belief'] = {
        'b_min_lin_vel': (0.0, 1e-6),
        'b_max_lin_vel': (1.0, 0.5),
        'b_max_lin_acc': (0.5, 0.2),
        'b_max_rot_vel': (np.pi / 2, 0.2 * np.pi)
        # 'b_min_lin_vel': (0.0, 1e-6),
        # 'b_max_lin_vel': (1.0, 1e-6),
        # 'b_max_lin_acc': (0.5, 1e-6),
        # 'b_max_rot_vel': (np.pi / 4, 1e-6)
    }
    env_config['delay_steps'] = 6


    env = ContinousNavEnv.from_layout(layout, **env_config)

    for i in range(1):
        print(f'\n\n--- Test Iteration {i} ---')
        test_state_inference(env)
        test_terminal_goal(env)
        test_transition(env)
        test_terminal_collision(env)

if __name__ == '__main__':
    main()
