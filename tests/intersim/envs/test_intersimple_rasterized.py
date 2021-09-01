from intersim.envs import IntersimpleRasterized, NRasterized
import numpy as np
import os

def test_obs_shape_dtype():
    env = IntersimpleRasterized(height=152, width=491)
    obs = env.reset()
    assert obs.shape == (1, 152, 491)
    assert obs.dtype == np.uint8

def test_agent51_frame0():
    env = IntersimpleRasterized()
    env._agent = 51
    obs = env.reset()
    expected = np.load(os.path.join(os.path.dirname(__file__), 'agent51frame0.npy'))
    assert np.array_equal(obs, expected)

def test_image_observation_rendering(tmp_path):
    env = IntersimpleRasterized()
    env._agent = 51

    obs = env.reset()
    assert not env._observations
    assert np.array_equal(env._last_observation, obs)

    env.render()
    assert len(env._observations) == 1
    assert np.array_equal(env._last_observation, obs)

    obs, _, _, _ = env.step(1)
    assert len(env._observations) == 1
    assert np.array_equal(env._last_observation, obs)

    env.render()
    assert len(env._observations) == 2
    assert np.array_equal(env._last_observation, obs)

    obs, _, _, _ = env.step(1)
    assert len(env._observations) == 2
    assert np.array_equal(env._last_observation, obs)

    env.render()
    assert len(env._observations) == 3
    assert np.array_equal(env._last_observation, obs)

    env.close(filestr=str(tmp_path/'render'))
    assert len(env._observations) == 3
    assert np.array_equal(env._last_observation, obs)
    assert (tmp_path / 'render_observation.mp4').is_file()

def test_rollout(tmp_path):
    env = NRasterized()
    env.reset()
    env.render()

    for _ in range(1000):
        _, _, done, _ = env.step(0.1)
        env.render()
        if done:
            break

    env.close(filestr=str(tmp_path))

def test_n_rasterized_check_env():
    env = NRasterized()

def test_n_rasterized_obs_shape_dtype():
    env = NRasterized(n_frames=10, height=152, width=491)
    obs = env.reset()
    assert obs.shape == (10, 152, 491)
    assert obs.dtype == np.uint8

def test_n_rasterized_agent51_frame0():
    env = IntersimpleRasterized()
    env._agent = 51
    obs = env.reset()
    expected = np.load(os.path.join(os.path.dirname(__file__), 'agent51frame0.npy'))
    assert np.array_equal(obs.sum(0, keepdims=True), expected)

def test_n_rasterized_observation_rendering(tmp_path):
    env = NRasterized()
    env._agent = 51

    obs = env.reset()
    assert not env._observations
    assert np.array_equal(env._last_observation, obs)

    env.render()
    assert len(env._observations) == 1
    assert np.array_equal(env._last_observation, obs)

    obs, _, _, _ = env.step(1)
    assert len(env._observations) == 1
    assert np.array_equal(env._last_observation, obs)

    env.render()
    assert len(env._observations) == 2
    assert np.array_equal(env._last_observation, obs)

    obs, _, _, _ = env.step(1)
    assert len(env._observations) == 2
    assert np.array_equal(env._last_observation, obs)

    env.render()
    assert len(env._observations) == 3
    assert np.array_equal(env._last_observation, obs)

    env.close(filestr=str(tmp_path/'render'))
    assert len(env._observations) == 3
    assert np.array_equal(env._last_observation, obs)
    assert (tmp_path / 'render_observation.mp4').is_file()

def test_n_rasterized_rollout(tmp_path):
    env = NRasterized()
    env.reset()
    env.render()

    for _ in range(1000):
        _, _, done, _ = env.step(0.1)
        env.render()
        if done:
            break

    env.close(filestr=str(tmp_path))

def test_agent0_step10_5frames():
    env = NRasterized(
        agent=17,
        n_frames=5,
        skip_frames=1,
    )
    env.reset()
    env.render()

    for _ in range(10):
        env.step(0.1)
        env.render()

    obs, _, _, _ = env.step(0.1)
    expected = np.load(os.path.join(os.path.dirname(__file__), 'agent17_step10_5frames.npy'))
    assert np.array_equal(obs, expected)
