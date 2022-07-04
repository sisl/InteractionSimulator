import gym
import numpy as np

def test_noinf():
    env = gym.make('intersim:intersim-v0', disable_env_checker=True)
    env.reset()
    assert not (env._svt.simstate == np.inf).any()
    assert not (env._svt.simstate == -np.inf).any()

def test_noinf_svt():
    from intersim.utils import get_svt
    svt, _ = get_svt()
    assert not (svt.simstate == np.inf).any()
    assert not (svt.simstate == -np.inf).any()

def test_playback():
    env = gym.make('intersim:intersim-v0', disable_env_checker=True)
    env.reset()
    for state in env._svt.simstate[1:]:
        action = env.target_state(state)
        _, _, done, _ = env.step(action)
        if done:
            break

def test_playback_noinf():
    env = gym.make('intersim:intersim-v0', disable_env_checker=True)
    env.reset()
    assert not (env._state == np.inf).any()
    assert not (env.projected_state == np.inf).any()
    assert not (env._state == -np.inf).any()
    assert not (env.projected_state == -np.inf).any()

    for state in env._svt.simstate[1:]:
        action = env.target_state(state)
        _, _, done, _ = env.step(action)

        assert not (env._state == np.inf).any()
        assert not (env.projected_state == np.inf).any()
        assert not (env._state == -np.inf).any()
        assert not (env.projected_state == -np.inf).any()
        
        if done:
            break

def test_playback_collisions():
    env = gym.make(
        'intersim:intersim-v0',
        disable_env_checker=True,
        stop_on_collision=True,
    )
    env.reset()
    for state in env._svt.simstate[1:]:
        action = env.target_state(state)
        _, _, done, _ = env.step(action)
        if done:
            break
