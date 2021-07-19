# test_simulator.py

import pandas as pd
import numpy as np
import torch
from intersim.utils import get_map_path, get_svt, SVT_to_stateactions
from intersim.viz import animate
import gym

import os
opj = os.path.join
def main():

    outdir  = opj('tests','output')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    filestr = opj(outdir,'test_simulator')
    
    svt, svt_path = get_svt()
    osm = get_map_path()
    print('SVT path: {}'.format(svt_path))
    print('Map path: {}'.format(osm))
    
    states, actions = SVT_to_stateactions(svt)
    n_frames = 500

    # animate from environment
    env = gym.make('intersim:intersim-v0', svt=svt, map_path=osm, 
        min_acc=-np.inf, max_acc=np.inf)
    env.reset()
    done = False
    i = 0
    obs, infos = [], []
    env_states = []
    nnis = []
    
    while not done and i < n_frames:
        env.render()
        env_state = env.projected_state
        env_states.append(env_state)
        nni = ~torch.isnan(env_state[:,0])
        nnis.append(nni)
        norm = torch.norm(env_state[nni,:2]-states[i,nni,:2], dim=1)
        print('Step %04i: Max state norm difference = %f'%(i, norm.max()))
        
        ob, r, done, info = env.step(env.target_state(svt.simstate[i+1]))
        obs.append(ob)
        infos.append(info)
        i+=1
    env.render()
    env.close(filestr=opj(outdir,'test_state_targeting_env'))

    # animate from states
    animate(osm, states[:n_frames+1], svt._lengths, svt._widths, filestr=opj(outdir,'test_state_targeting_states'))

if __name__ == '__main__':
    main()