from intersim.envs import NRasterizedRoute
import numpy as np
import os

def test_obs_shape_dtype():
    env = NRasterizedRoute(n_frames=5, height=152, width=491)
    obs = env.reset()
    assert obs.shape == (6, 152, 491)
    assert obs.dtype == np.uint8

def test_agent76_frame0():
    env = NRasterizedRoute(route_thickness=1)
    env._agent = 76
    obs = env.reset()
    expected = np.load(os.path.join(os.path.dirname(__file__), 'nrroute_agent76_frame0.npy'))
    assert np.array_equal(obs, expected)

def test_rollout(tmp_path):
    env = NRasterizedRoute()
    env.reset()
    env.render()

    for _ in range(1000):
        _, _, done, _ = env.step(0.1)
        env.render()
        if done:
            break

    env.close(filestr=str(tmp_path))
