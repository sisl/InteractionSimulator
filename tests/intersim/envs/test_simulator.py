import gym
from intersim.utils import df_to_stackedvehicletraj
from intersim.graphs import ConeVisibilityGraph

import torch
cvg = ConeVisibilityGraph(r=20, half_angle=120)
env = gym.make('intersim:intersim-v0', graph=cvg)
env.reset()


def test_relative_state():
    env.reset()
    ob, r, done, info = env.step(env.action_space.sample())


    relstate = ob['relative_state']
    shape = relstate.shape
    assert shape[0] == shape[1]
    assert shape[2] == 6

    assert torch.all(torch.isnan(torch.diagonal(relstate)))

def test_generate_paths():
    # ob, info = env.reset()
    env.reset()
    smax = env.smax[0]
    n = 20

    xpath, ypath = env._generate_paths(delta=2*smax/n, n=n, is_distance=True, override=False)
    # env.state will be nan for all tracks not visible in the current state
    existing_tracks = ~torch.any(torch.isnan(env.state), dim=1)
    xpath, ypath = xpath[existing_tracks], ypath[existing_tracks]
    assert ~torch.any(torch.isnan(xpath))
    assert ~torch.any(torch.isnan(ypath))

    v0 = env.state[0, 1]
    xpath, ypath = env._generate_paths(delta=2*smax/(v0*n), n=n, is_distance=False, override=False)
    # env.state will be nan for all tracks not visible in the current state
    existing_tracks = ~torch.any(torch.isnan(env.state), dim=1)
    xpath, ypath = xpath[existing_tracks], ypath[existing_tracks]
    assert ~torch.any(torch.isnan(xpath))
    assert ~torch.any(torch.isnan(ypath))