from intersim.envs import Intersimple, NRasterized
import numpy as np
import logging
import pytest


def test_init_runs():
    env = Intersimple()
    assert env is not None

def test_reset_runs():
    env = Intersimple()
    obs = env.reset()
    assert obs is not None

def test_reset_obs_shape():
    env = Intersimple(n_obs=5)
    obs = env.reset()
    assert obs.shape == (6, 7)

def test_render():
    env = Intersimple()
    env.reset()
    env.render()

def test_close():
    env = Intersimple()
    env.close()

def test_reset_before_step():
    env = Intersimple()
    with pytest.raises(AssertionError):
        env.step(0)

def test_initial_observation_correct():
    env = Intersimple(n_obs=5) # location 0, track 0
    obs = env.reset()

    expected = np.array([
        [  1000.4850,   1003.6670,  6.9991e+00, -2.5243e+00,  2.8830e-02,
          0.0000e+00,  1.0000e+00],
        [-7.1420e+00, -1.8625e+01,  1.4268e+01,  3.8583e+00,  2.5018e+00,
         -2.0951e-02,  1.0000e+00],
        [ 3.2061e+01,  1.4843e+01,  2.8088e+00,  1.9156e+00,  1.7693e-02,
         -1.4138e-02,  1.0000e+00],
        [          0,            0,          0,           0,           0,
                   0,  0.0000e+00],
        [          0,            0,          0,           0,           0,
                   0,  0.0000e+00],
        [          0,            0,          0,           0,           0,
                   0,  0.0000e+00]
    ])

    print('got', obs)
    print('expected', expected)
    print('diff', obs - expected)

    assert np.allclose(obs, expected, atol=2.6e-1)

def test_step_runs():
    env = Intersimple()
    obs0 = env.reset()
    assert obs0 is not None
    obs, reward, done, info = env.step(0)
    assert obs is not None
    assert reward is not None
    assert done is not None
    assert info is not None

def test_step_reward_done():
    env = Intersimple()
    env.reset()
    obs, reward, done, info = env.step(0)
    assert obs is not None
    assert reward == 0
    assert done == False
    assert info is not None

def test_step_obs_shape():
    env = Intersimple(n_obs=5)
    env.reset()
    obs, _, _, _ = env.step(0)
    assert obs.shape == (6, 7)

def test_step_gt_action_obs():
    env = Intersimple(n_obs=5) # location 0, track 0
    env.reset()
    obs, _, _, _ = env.step(0.1458)

    expected = np.array([
        [   999.9111,   1003.2660,  7.0137e+00, -2.5372e+00,  1.2727e-02,
          0.0000e+00,  1.0000e+00],
        [-5.7120e+00, -1.8245e+01,  1.4332e+01,  3.7669e+00,  2.5117e+00,
         -1.1853e-02,  1.0000e+00],
        [ 3.2344e+01,  1.5031e+01,  2.8352e+00,  1.8466e+00,  2.5231e-02,
          2.1772e-03,  1.0000e+00],
        [          0,           0,           0,           0,           0,
                   0,  0.0000e+00],
        [          0,           0,           0,           0,           0,
                   0,  0.0000e+00],
        [          0,           0,           0,           0,           0,
                   0,  0.0000e+00]
    ])

    print('got', obs)
    print('expected', expected)
    print('diff', obs - expected)

    assert np.allclose(obs, expected, atol=1.4e-1)

def test_obs_xor_done():
    env = Intersimple()
    env.reset()
    for _ in range(1000):
        obs, _, done, _ = env.step(1)
        sobs = obs[:, :-1]
        assert done ^ (not np.allclose(sobs, np.zeros_like(sobs)))
        if done: return
    assert False

def test_ego_valid_xor_done():
    env = Intersimple()
    env.reset()
    for _ in range(1000):
        obs, _, done, _ = env.step(1)
        assert done ^ bool(obs[0, -1] == 1)
        if done: return
    assert False

def test_restart_after_done():
    env = Intersimple()
    assert not env._env._svt.simstate[0][0].isnan().any()
    print('state0', env._env._svt.state0[0])

    env.reset()
    assert not env._env._svt.simstate[0][0].isnan().any()
    print('state0', env._env._svt.state0[0])
    
    while True:
        assert not env._env._svt.simstate[0][0].isnan().any()
        print('state0', env._env._svt.state0[0])
        _, _, done, _ = env.step(1)
        if done:
            break

    print('state0', env._env._svt.state0[0])
    assert not env._env._svt.simstate[0][0].isnan().any()

    env.reset()
    print('state0', env._env._svt.state0[0])
    assert not env._env._svt.simstate[0][0].isnan().any()

def test_obs_after_restart():
    env = Intersimple()
    obs1 = env.reset()

    while True:
        _, _, done, _ = env.step(1)
        if done: break
    
    obs2 = env.reset()
    assert np.allclose(obs1, obs2)

def test_other_agent(caplog):
    caplog.set_level(logging.INFO)
    env = Intersimple()
    env._agent = 17
    obs = env.reset()
    assert obs is not None
    assert 'Skipping to frame 276.' in caplog.text

def test_last_agent():
    env = Intersimple()
    env._agent = env._env._nv - 1
    assert env._agent == 150

    T, _, _ = env._env._svt.simstate.shape
    assert T == 3008

    env.reset()

    while env._env._ind < T - 1:
        next_state = env._env._svt.simstate[env._env._ind + 1]
        gt_action = env._env.target_state(next_state)
        expert_action = gt_action[env._agent].item()
        _, _, done, _ = env.step(expert_action)
        assert not (done ^ env._env._exceeded.all().item())
    
    _, _, done, _ = env.step(1)
    assert done
    assert env._env._exceeded.all()

def test_reset_equals_playback():
    env_skip = Intersimple()
    env_skip._agent = 17
    obs_skip = env_skip.reset()

    env = Intersimple()
    obs = env.reset()
    env._agent = 17
    obs *= 0 # first observation is for agent 0, skip

    while env._env._ind < env_skip._env._ind:
        assert not obs[0].any()
        next_state = env._env._svt.simstate[env._env._ind + 1]
        gt_action = env._env.target_state(next_state)
        expert_action = gt_action[env._agent].item()
        obs, _, done, _ = env.step(expert_action)

    assert not done
    assert np.allclose(obs, obs_skip, atol=0.17)
    assert np.allclose(env._env._state, env_skip._env._state, atol=0.20, equal_nan=True)
    assert env._env._ind == env_skip._env._ind
    assert (env._env._exceeded == env_skip._env._exceeded).all()
    # comparison not implemented for InteractionGraph, skipping ._env.graph
    assert np.allclose(env._env.info['raw_state'], env_skip._env.info['raw_state'], atol=0.20, equal_nan=True)
