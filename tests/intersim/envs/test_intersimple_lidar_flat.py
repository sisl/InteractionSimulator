from intersim.envs import IntersimpleLidarFlat
import numpy as np

def test_check_env():
    env = IntersimpleLidarFlat()
    assert env._agent == 0

def test_obs_space_shape():
    env = IntersimpleLidarFlat(n_rays=5)
    assert env.observation_space.shape == (6 * 6,)

def test_obs_shape():
    env = IntersimpleLidarFlat(n_rays=5)
    obs = env.reset()
    assert obs.shape == (6 * 6,)

def test_ego_state():
    env = IntersimpleLidarFlat(n_rays=5)
    obs = env.reset()
    assert np.array_equal(obs[:5], env._env.projected_state[env._agent])
    assert obs[5] == 0

def test_viz(tmp_path):
    env = IntersimpleLidarFlat(n_rays=128)
    obs = env.reset()
    env.render()

    done = False
    while not done:
        _, _, done, _ = env.step(0)
        env.render()

    env.close(filestr=str(tmp_path))
