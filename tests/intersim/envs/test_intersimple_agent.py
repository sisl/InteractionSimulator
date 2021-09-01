from intersim.envs import IntersimpleFlatAgent
import numpy as np

def test_init():
    env = IntersimpleFlatAgent(agent=43)
    assert env._agent == 43

def test_reset():
    env = IntersimpleFlatAgent(agent=17)
    obs = env.reset()
    assert env._agent == 17
    assert obs[6] == 1
    assert not np.allclose(obs[0], np.zeros_like(obs[0]))

def test_step():
    env = IntersimpleFlatAgent(agent=78)
    env.reset()
    o, r, d, i = env.step(0)
    assert o is not None
    assert r is not None
    assert d is not None
    assert i is not None
