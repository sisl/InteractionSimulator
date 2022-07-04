# test_simulator.py

from intersim.utils import df_to_stackedvehicletraj

import gym

import os
opj = os.path.join
def main():

    outdir  = opj('tests','output')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    filestr = opj(outdir,'test_collision_stop')

    env = gym.make('intersim:intersim-v0', disable_env_checker=True, stop_on_collision=True)
    env.reset()
    done = False
    n_frames = 600
    i = 0
    while not done and i < n_frames:
        env.render()
        ob, r, done, info = env.step(env.action_space.sample())
        i+=1
    env.render()
    env.close(filestr=filestr)
    assert done, "Did not stop for a collision"

if __name__ == '__main__':
    main()