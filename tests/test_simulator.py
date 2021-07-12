# test_simulator.py

from intersim.utils import df_to_stackedvehicletraj
from intersim.graphs import ConeVisibilityGraph

import gym

import os
opj = os.path.join
def main():

    outdir  = opj('tests','output')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    filestr = opj(outdir,'test_simulator')

    cvg = ConeVisibilityGraph(r=20, half_angle=120)
    env = gym.make('intersim:intersim-v0', graph=cvg)

    env.reset()
    done = False
    n_frames = 50
    i = 0
    while not done and i < n_frames:
        env.render()
        ob, r, done, info = env.step(env.action_space.sample())
        i+=1
    env.render()
    env.close(filestr=filestr)


if __name__ == '__main__':
    main()