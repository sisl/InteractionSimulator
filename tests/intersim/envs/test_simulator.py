import gym
from intersim.utils import df_to_stackedvehicletraj
from intersim.graphs import ConeVisibilityGraph

import torch


def test_relative_state():
    cvg = ConeVisibilityGraph(r=20, half_angle=120)
    env = gym.make('intersim:intersim-v0', graph=cvg)

    env.reset()
    ob, r, done, info = env.step(env.action_space.sample())


    relstate = ob['relative_state']
    shape = relstate.shape
    assert shape[0] == shape[1]
    assert shape[2] == 6

    assert torch.all(torch.isnan(torch.diagonal(relstate)))

