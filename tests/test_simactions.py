# test_simulator.py

import pandas as pd
from intersim.utils import get_map_path, get_svt, SVT_to_sim_stateactions
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
    

    states, actions = SVT_to_sim_stateactions(svt)
    n_frames = 200
    # animate from states
    animate(osm, states[:n_frames], svt._lengths, svt._widths, filestr=opj(outdir,'test_simactions_states'))

    # animate from environment
    env = gym.make('intersim:intersim-v0', svt=svt, map_path=osm)
    env.reset()
    done = False
    i = 0
    while not done and i < n_frames:
        env.render()
        ob, r, done, info = env.step(actions[i])
        i+=1
    env.render()
    env.close(filestr=opj(outdir,'test_simactions_env'))


if __name__ == '__main__':
    main()