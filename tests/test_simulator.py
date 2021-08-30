# test_simulator.py

import pandas as pd
from intersim.datautils import df_to_stackedvehicletraj, DATASET_DIR
from intersim import RoundaboutSimulator
import matplotlib.animation as animation
from intersim.viz.animatedviz import AnimatedViz
import matplotlib.pyplot as plt

import torch

def main():

    # load a trackfile
    df = pd.read_csv(f'{DATASET_DIR}/trackfiles/DR_USA_Roundabout_FT/vehicle_tracks_000.csv')

    stv = df_to_stackedvehicletraj(df)

    sim = RoundaboutSimulator(stv)

    states = []
    s,_ = sim.reset()
    s = s.reshape(-1,5)
    states.append(s)

    for i in range(100):
        v = s[:,2:3]
        s,_ = sim.step(0.5*(5.-v))

        s = s.reshape(-1,5)

        states.append(s)

    states = torch.stack(states).reshape(101,-1)

    fig = plt.figure()
    ax = plt.axes(
        xlim=(900, 1100), ylim=(900, 1100)
        )
    ax.set_aspect('equal', 'box')

    osm = f'{DATASET_DIR}/maps/DR_USA_Roundabout_FT.osm'

    av = AnimatedViz(ax, osm, states, stv.lengths, stv.widths)

    ani = animation.FuncAnimation(fig, av.animate, frames=len(states),
                   interval=20, blit=True, init_func=av.initfun, 
                   repeat=True)

    plt.show()




if __name__ == '__main__':
    main()
