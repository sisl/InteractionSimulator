from intersim.envs import IntersimpleNormalizedActions

def test_min_action():
    env = IntersimpleNormalizedActions()
    env.reset()
    _, _, _, info = env.step(-1)
    assert info['action_taken'][env._agent] == env._env._min_acc

def test_zero_action():
    env = IntersimpleNormalizedActions()
    env.reset()
    _, _, _, info = env.step(0)
    assert info['action_taken'][env._agent] == 0

def test_max_action():
    env = IntersimpleNormalizedActions()
    env.reset()
    _, _, _, info = env.step(1)
    assert info['action_taken'][env._agent] == env._env._max_acc
