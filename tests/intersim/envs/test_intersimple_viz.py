from intersim.envs import IntersimpleMarker, IntersimpleTargetSpeed
import logging

def test_marker(caplog, tmp_path):
    caplog.set_level(logging.INFO)

    env = IntersimpleMarker()
    env.reset()
    env.render()
    env.step(0)
    env.render()
    env.close(filestr=str(tmp_path/'render'))

    assert f'Highlighting agent {env._agent}' in caplog.text


def test_observations_viz(caplog, tmp_path):
    caplog.set_level(logging.INFO)

    env = IntersimpleMarker()
    env.reset()
    env.render()
    env.step(1)
    env.render()
    env.close(filestr=str(tmp_path/'render'))

    assert 'Drawing observation 0' in caplog.text
    assert 'Drawing observation 1' in caplog.text


def test_actions_viz(caplog, tmp_path):
    caplog.set_level(logging.INFO)

    env = IntersimpleMarker()
    env.reset()
    env.render()
    env.step(1)
    env.render()
    env.close(filestr=str(tmp_path/'render'))

    assert f'Drawing action 0 (a = {1})' in caplog.text
    assert 'Drawing action 1 (no action)' in caplog.text


def test_reward_viz(caplog, tmp_path):
    caplog.set_level(logging.INFO)

    env = IntersimpleMarker()
    env.reset()
    env.render()
    env.step(1)
    env.render()
    env.close(filestr=str(tmp_path/'render'))


def test_visualization_matches_reward(caplog, tmp_path):
    caplog.set_level(logging.INFO)

    env = IntersimpleTargetSpeed()
    env.reset()
    env.render()
    _, r, _, _ = env.step(1)
    env.render()
    env.close(filestr=str(tmp_path/'render'))

    logging.info(f'r = {r}')
    assert f'Drawing reward 0 (r = {r})' in caplog.text
    assert 'Drawing reward 1 (no reward)' in caplog.text
