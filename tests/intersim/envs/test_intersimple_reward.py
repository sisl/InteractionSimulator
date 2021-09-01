from intersim.envs.intersimple import speed_reward, target_speed_reward
from intersim.envs import IntersimpleReward, IntersimpleTargetSpeed
import functools

def test_check_env():
    env = IntersimpleReward()

def test_step_reward_done():
    env = IntersimpleReward()
    env.reset()
    obs, reward, done, info = env.step(0)
    assert obs is not None
    assert reward == 1
    assert done == False
    assert info is not None

def test_reward_until_done():
    env = IntersimpleReward()
    env.reset()
    for _ in range(1000):
        _, reward, done, _ = env.step(1)
        assert reward == 1
        if done: return
    assert False

def test_other_agent():
    env = IntersimpleReward()
    env._agent = 17
    obs = env.reset()
    assert obs is not None

def test_tspeed_check_env():
    env = IntersimpleTargetSpeed()

def test_tspeed_step_reward_done():
    env = IntersimpleTargetSpeed()
    env.reset()
    obs, reward, done, info = env.step(0)
    assert obs is not None
    assert reward < 0
    assert done == False
    assert info is not None
    assert info['collision'] == False

def test_tspeed_reward_until_done():
    env = IntersimpleTargetSpeed()
    env.reset()
    for _ in range(1000):
        _, reward, done, _ = env.step(1)
        assert reward < 0
        if done: return
    assert False

def test_tspeed_other_agent():
    env = IntersimpleTargetSpeed()
    env._agent = 17
    obs = env.reset()
    assert obs is not None

def test_tspeed_collision():
    env = IntersimpleTargetSpeed()
    env.reset()
    for _ in range(1000):
        _, _, done, info = env.step(1)
        # agent 0 should escape
        assert info['collision'] == False
        if done: return
    assert False

def test_tspeed_collision2():
    env = IntersimpleTargetSpeed()
    env._agent = 17
    env.reset()
    for _ in range(1000):
        _, _, done, info = env.step(1)
        # agent 17 should crash into someone
        assert not (done ^ info['collision'])
        if done: return
    assert False

def test_tspeed_reward_value():
    target_speed = 10
    speed_penalty_weight = 0.01
    collision_penalty = 1000

    env = IntersimpleTargetSpeed(
        target_speed=target_speed,
        speed_penalty_weight=speed_penalty_weight,
        collision_penalty=collision_penalty
    )
    obs = env.reset()
    speed = obs[2].item()
    _, reward, _, _ = env.step(0)

    assert reward == -speed_penalty_weight * (speed - target_speed) ** 2
    assert reward == target_speed_reward(
        obs[:5], float('inf'), {'collision': False},
        target_speed=target_speed,
        speed_penalty_weight=speed_penalty_weight,
        collision_penalty=collision_penalty
    )

def test_speed_reward_value():
    env = IntersimpleReward(
        reward=functools.partial(
            speed_reward,
            speed_weight=0.01,
            collision_penalty=1000
        )
    )
    obs = env.reset()
    speed = obs[2].item()
    _, reward, _, _ = env.step(0)

    assert reward == 0.01 * speed
    assert reward == speed_reward(
        obs[:5], float('inf'), {'collision': False},
        speed_weight=0.01,
        collision_penalty=1000
    )
