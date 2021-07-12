# test_animate.py

import pandas as pd
from intersim.datautils import df_to_stackedvehicletraj, SVT_to_simstates
from intersim.default_graphs.conevisibility import ConeVisibilityGraph
from intersim import RoundaboutSimulator
import matplotlib
import matplotlib.animation as animation
from intersim.viz.animatedviz import AnimatedViz
import matplotlib.pyplot as plt

import torch

def main():
    
    savefile = True
    
    if savefile:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    # load a trackfile
    df = pd.read_csv('datasets/trackfiles/DR_USA_Roundabout_FT/vehicle_tracks_000.csv')

    stv = df_to_stackedvehicletraj(df)

    states = SVT_to_simstates(stv)
    cvg = ConeVisibilityGraph(r=20, half_angle=120)
    graphs = []
    for i in range(len(states)):
        s = states[i]
        cvg.update_graph(s.reshape((-1,5)))
        graphs.append(cvg.edges)
        
    fig = plt.figure()
    ax = plt.axes(
        xlim=(900, 1100), ylim=(900, 1100)
        )
    ax.set_aspect('equal', 'box')

    osm = 'datasets/maps/DR_USA_Roundabout_FT.osm'

    av = AnimatedViz(ax, osm, states, stv.lengths, stv.widths, graphs=graphs)

    ani = animation.FuncAnimation(fig, av.animate, frames=len(states),
                   interval=20, blit=True, init_func=av.initfun, 
                   repeat=not savefile)

    ani.save('animation_truegraph.mp4', writer) if savefile else plt.show()


if __name__ == '__main__':
    main()