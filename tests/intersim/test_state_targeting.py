# test_simulator.py

import pandas as pd
import numpy as np
import torch
from intersim.utils import get_map_path, get_svt, SVT_to_stateactions
from intersim.viz import animate
from intersim import collisions
import gym

import os
opj = os.path.join
outdir  = opj('tests','output')
if not os.path.isdir(outdir):
    os.mkdir(outdir)
filestr = opj(outdir,'test_state_targeting_states')

def test_state_targeting():

    svt, svt_path = get_svt()
    osm = get_map_path()
    print('SVT path: {}'.format(svt_path))
    print('Map path: {}'.format(osm))
    
    states, actions = SVT_to_stateactions(svt)

    # animate from environment
    env = gym.make('intersim:intersim-v0', svt=svt, map_path=osm, 
        min_acc=-np.inf, max_acc=np.inf)
    env.reset()
    done = False
    i = 0
    obs, infos = [], []
    env_states = []
    nnis = []
    max_devs = []
    while not done and i < len(actions):
        env.render()
        env_state = env.projected_state
        env_states.append(env_state)
        nni = ~torch.isnan(env_state[:,0])
        nnis.append(nni)
        norms = torch.norm(env_state[nni,:2]-states[i,nni,:2], dim=1)
        if len(norms)>0:
            max_devs.append(norms.max())
        ob, r, done, info = env.step(env.target_state(svt.simstate[i+1], mu=0.002))
        obs.append(ob)
        infos.append(info)
        i+=1
    env.render()

    # assert small max norm and no collisions
    
    assert max(max_devs) < 10, "Maximum environment deviation from track: %f m" %(max(max_devs))
    x = torch.stack([ob['state'] for ob in obs])
    cols = collisions.check_collisions_trajectory(x, svt.lengths, svt.widths)
    assert ~torch.any(cols), 'Error: Collisions found at indices {}'.format(cols.nonzero(as_tuple=True))
    
    # env.close(filestr=opj(outdir,'test_state_targeting_env'))
    # animate from states
    # animate(osm, states[:n_frames+1], svt._lengths, svt._widths, filestr=filestr)

