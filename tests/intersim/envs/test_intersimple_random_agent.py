from intersim.envs import IntersimpleFlatRandomAgent
import numpy as np

def test_init():
    env = IntersimpleFlatRandomAgent()
    n_agents = env._env._nv
    assert n_agents == 151
    assert 0 <= env._agent < 151

def test_reset():
    env = IntersimpleFlatRandomAgent()
    obs = env.reset()
    assert obs is not None

    assert 0 <= env._agent < 151

def test_reset_observation():
    env = IntersimpleFlatRandomAgent()
    for _ in range(5):
        obs = env.reset()
        
        n_agents = env._env._nv
        assert n_agents == 151
        assert 0 <= env._agent < 151

        assert not np.allclose(obs[:5], np.zeros_like(obs[:5])), f'All-zero initial ego states for agent {env._agent}'
        assert obs[6] == 1, f'Invalid initial ego state for agent {env._agent}'
