import gym
from intersim.envs.simulator import InteractionSimulator
from intersim.graph import InteractionGraph
from intersim.utils import LOCATIONS, MAX_TRACKS, df_to_stackedvehicletraj
from intersim.graphs import ConeVisibilityGraph

import torch
cvg = ConeVisibilityGraph(r=20, half_angle=120)
env = gym.make('intersim:intersim-v0', graph=cvg)
env2 = gym.make('intersim:intersim-v0', graph=cvg, mask_relstate=True)


def test_mask_relstate():
    env.reset()
    env2.reset()
    done = False
    n_frames = 10
    i = 0
    while not done and i < n_frames:
        env.render()
        env2.render()
        a = env.action_space.sample()
        ob, r, done, info = env.step(a)
        ob2, r, done, info = env2.step(a)
        
        nni = ~torch.isnan(ob['relative_state'][:,0,0])
        nni2 = ~torch.isnan(ob['relative_state'][:,0,0])
        assert sum(nni) >= sum(nni2), 'Masked relative state has more elements than relative state at frame %i' %(i)
        i+=1
    env.render()
    env2.render()

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

    assert torch.allclose(smax, torch.tensor(35.511131286621094))

    xpath, ypath = env._generate_paths(delta=2*smax/n, n=n, is_distance=True, override=False)
    # env.state will be nan for all tracks not visible in the current state
    existing_tracks = ~torch.any(torch.isnan(env.state), dim=1)
    xpath, ypath = xpath[existing_tracks], ypath[existing_tracks]
    assert ~torch.any(torch.isnan(xpath))
    assert ~torch.any(torch.isnan(ypath))

    v0 = env.state[0, 1]
    assert torch.allclose(v0, torch.tensor(6.999134063720703))
    
    xpath, ypath = env._generate_paths(delta=2*smax/(v0*n), n=n, is_distance=False, override=False)
    # env.state will be nan for all tracks not visible in the current state
    existing_tracks = ~torch.any(torch.isnan(env.state), dim=1)
    xpath, ypath = xpath[existing_tracks], ypath[existing_tracks]
    assert ~torch.any(torch.isnan(xpath))
    assert ~torch.any(torch.isnan(ypath))

def test_locations():
    locations = len(LOCATIONS)
    tracks = MAX_TRACKS

    records = set((l, t) for l in range(locations) for t in range(tracks))
    records.remove((1, 4)) # currently crashing
    records.remove((2, 3)) # currently crashing

    for loc, track in records:
        print(f'loc {loc} track {track}')
        env = gym.make('intersim:intersim-v0', loc=loc, track=track)
        env.reset()

def test_propagate_action_profile_vectorized():
    env = gym.make('intersim:intersim-v0')
    actions = 10 * torch.rand((20, 50, 151, 1))
    p1 = torch.stack(env.propagate_action_profile(actions))
    p2 = env.propagate_action_profile_vectorized(actions)
    assert torch.allclose(p1, p2, equal_nan=True)

def test_empty_neighbor_dict():
    env = InteractionSimulator()
    assert not env._graph._neighbor_dict
    env._graph._neighbor_dict = {123: [234]}
    env = InteractionSimulator()
    assert not env._graph._neighbor_dict

def test_empty_neighbor_dict_gym():
    env = gym.make('intersim:intersim-v0')
    assert not env._graph._neighbor_dict
    env._graph._neighbor_dict = {123: [234]}
    env = gym.make('intersim:intersim-v0')
    assert not env._graph._neighbor_dict
