import gym
import numpy as np

def test_nv():
    env = gym.make('intersim:intersim-v0', disable_env_checker=True)
    action_space_nv, _ = env.action_space.shape
    _, simstate_nv, _ = env._svt.simstate.shape
    assert env._nv == env._svt.nv
    assert action_space_nv == env._nv
    assert simstate_nv == env._nv

def test_T():
    env = gym.make('intersim:intersim-v0', disable_env_checker=True)
    assert env._svt.simstate.shape[0] == (env._svt._maxTind - env._svt._minTind)

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

        # vehicle spawning timesteps might be slightly off
        # target_state is not perfectly accurate
        assert np.allclose(env._state - state, 0 * (env._state - state), atol=1.375, equal_nan=True)
        
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

def test_state_relstate_consistent():
    env = gym.make(
        'intersim:intersim-v0',
        disable_env_checker=True,
    )
    env.reset()

    def _wrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    for state in env._svt.simstate[1:]:
        action = env.target_state(state)
        obs, _, _, _ = env.step(action)

        ego = obs['state'][0]
        others = obs['state'][1:]
        others_rel = obs['relative_state'][0, 1:]

        assert ego.isnan().all() or np.allclose(ego[:2] + others_rel[:, :2], others[:, :2], equal_nan=True) # x, y
        assert ego.isnan().all() or np.allclose(ego[4:] + others_rel[:, 5:], others[:, 4:], equal_nan=True) # psidot
        
        print('diff', _wrap(ego[3] + others_rel[:, 4] - others[:, 3]))
        assert ego.isnan().all() or np.allclose(_wrap(ego[3] + others_rel[:, 4] - others[:, 3]), 0 * others[:, 3], atol=5e-7, equal_nan=True) # psi
