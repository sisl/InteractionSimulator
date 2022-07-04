import gym
import torch

def test_forward_propagation():
    env = gym.make('intersim:intersim-v0', disable_env_checker=True)
    env.reset()
    for _ in range(20):
        env.step(torch.zeros(env._nv,1))
    action_list = [
        torch.zeros(1, env._nv, 1), # maintain velocity
        torch.zeros(20, env._nv, 1), # maintain velocity for 20
        torch.zeros(100, env._nv, 1), # maintain velocity for 100
        -30*torch.zeros(3, env._nv, 1), # full stop
        10*torch.ones(30, env._nv, 1), # accelerate for 30
    ]

    bools = env.check_future_collisions(action_list)
    assert torch.all(bools == torch.tensor([False, False, True, False, True]))

