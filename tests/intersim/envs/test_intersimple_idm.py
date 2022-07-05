from intersim.default_policies.idm import IDM2
from intersim.envs import IntersimpleLidarFlat

def test_idm(tmp_path):
    env = IntersimpleLidarFlat(agent=42, use_idm=True, idm_policy=IDM2)
    env.reset()
    env.render()

    idm_agents = set()
    for _ in range(1000):
        _, _, done, _ = env.step(-0.01)
        idm_agents = idm_agents | set(agent.item() for agent in env._apply_idm.nonzero())

        env.render()
        if done:
            break

    assert idm_agents == {44, 48}
    env.close(filestr=str(tmp_path))

def test_idm_agent_148(tmp_path):
    env = IntersimpleLidarFlat(agent=148, use_idm=True, idm_policy=IDM2)
    env.reset()
    env.render()

    for _ in range(1000):
        assert not env._apply_idm.any()
        assert not env._env._graph._neighbor_dict
        _, _, done, _ = env.step(-0.01)

        env.render()
        if done:
            break

    env.close(filestr=str(tmp_path))

def test_idm_penultimate_agent(tmp_path):
    env = IntersimpleLidarFlat(agent=149, use_idm=True, idm_policy=IDM2)
    env.reset()
    env.render()

    for _ in range(1000):
        assert not env._apply_idm.any()
        assert not env._env._graph._neighbor_dict
        _, _, done, _ = env.step(-0.01)

        env.render()
        if done:
            break

    env.close(filestr=str(tmp_path))

def test_idm_last_agent(tmp_path):
    env = IntersimpleLidarFlat(agent=150, use_idm=True, idm_policy=IDM2)
    env.reset()
    env.render()

    for _ in range(1000):
        assert not env._apply_idm.any()
        assert not env._env._graph._neighbor_dict
        _, _, done, _ = env.step(-0.01)

        env.render()
        if done:
            break

    env.close(filestr=str(tmp_path))
