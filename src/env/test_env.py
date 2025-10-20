import numpy as np

from env.continous_nav_env_delay_comp import ContinousNavEnv


def test_rewards(env):
    print("Testing Rewards...")

    ##############################################################
    # Test Goal ##################################################
    env.reset()
    n_samples = 5
    max_vel = env.state.robot.true_max_lin_vel
    goal_loc = np.array(env.goal).reshape(1,2)

    # EXACTLY at goal (done=True)
    true_state          = env.state._true_robot_state.copy().reshape(1,4)
    true_state[:,:2]    = goal_loc
    sample_states          = np.tile(true_state, (n_samples,1))
    assert np.all(env._check_goal(true_state)), "Goal check failed when robot is EXACTLY at the goal (true state)."
    assert np.all(env._check_goal(sample_states)), "Goal check failed when robot is EXACTLY at the goal (sampled state)."

    # Within goal radius (done=True)
    true_state[:,:2]    += (2 * np.random.rand(2) - 1) * env.goal_radius
    sample_states[:,:2] += (2 * np.random.rand(n_samples*2) - 1).reshape(n_samples,2) * env.goal_radius
    assert np.all(env._check_goal(true_state)), "Goal check failed when robot is WITHIN the goal (true state)."
    assert np.all(env._check_goal(sample_states)), "Goal check failed when robot is WITHIN the goal (sampled state)."

    # Within goal radius but not stopped (done=False)
    true_state[:, env.state.iv]     += np.random.rand() * max_vel + env.goal_velocity
    sample_states[:, env.state.iv]  += np.random.rand(n_samples)* max_vel  + env.goal_velocity
    assert not np.any(env._check_goal(true_state)),    "Goal check incorrectly passed when robot is WITHIN + NOT STOPPED the goal (true state)."
    assert not np.any(env._check_goal(sample_states)), "Goal check incorrectly passed when robot is WITHIN + NOT STOPPED the goal (sampled state)."

    # Outside of goal radius and not stopped (done=False)
    true_state[:,:2]    += (np.random.rand(2) + 1.1) * env.goal_radius
    sample_states[:,:2] += (np.random.rand(n_samples*2) + 1.1).reshape(n_samples,2) * env.goal_radius
    assert not np.any(env._check_goal(true_state)), "Goal check incorrectly passed when robot is OUTSIDE + NOT STOPPED the goal (true state)."
    assert not np.any(env._check_goal(sample_states)), "Goal check incorrectly passed when robot is OUTSIDE + NOT STOPPED the goal (sampled state)."

    # Outside of goal radius but stopped (done=False)
    true_state[:, env.state.iv] = 0
    true_state[:, env.state.iv] = 0
    assert not np.any(env._check_goal(true_state)), "Goal check incorrectly passed when robot is OUTSIDE + STOPPED the goal (true state)."
    assert not np.any(env._check_goal(sample_states)), "Goal check incorrectly passed when robot is OUTSIDE + STOPPED the goal (sampled state)."

    print("\t| Reward tests passed...")




    return True
def test_dones(env):
    return True
def test_transition(env):
    return True

def main():
    layout = 'example2'
    env = ContinousNavEnv.from_layout(layout)

    test_rewards(env)
    test_dones(env)
    test_transition(env)
if __name__ == '__main__':
    main()
