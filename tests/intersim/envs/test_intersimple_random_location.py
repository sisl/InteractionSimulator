from intersim.envs import Intersimple
from intersim.envs.intersimple import FixedLocation, RandomLocation
import numpy as np

class IntersimpleFixedLocation(FixedLocation, Intersimple):
    pass

class IntersimpleRandomLocation(RandomLocation, Intersimple):
    pass

def test_init_fixed():
    env = IntersimpleFixedLocation(loc=3, track=4, n_obs=2)
    assert env._location == 3
    assert env._track == 4
    assert env._n_obs == 2

def test_init():
    env = IntersimpleRandomLocation(loc=3, track=4, n_obs=2)
    assert env._location == 3
    assert env._track == 4
    assert env._n_obs == 2

def test_reset():
    env = IntersimpleRandomLocation(n_obs=4)
    obs = env.reset()
    print(f'location {env._location}, track {env._track}')
    assert env._n_obs == 4
    assert obs.shape == (5, 7)

def test_reset_observation():
    env = IntersimpleRandomLocation()
    for _ in range(5):
        obs = env.reset()
        print(f'location {env._location}, track {env._track}')
        assert not np.allclose(obs[0, :5], np.zeros_like(obs[0, :5])), f'All-zero initial ego states for agent {env._agent}'
        assert obs[0, 6] == 1, f'Invalid initial ego state for agent {env._agent}'
