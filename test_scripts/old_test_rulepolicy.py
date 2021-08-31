# test_idm.py

import pandas as pd
from intersim.datautils import *
from intersim.default_policies.rulepolicy import RulePolicy
from intersim import RoundaboutSimulator
import matplotlib.animation as animation
from intersim.viz.animatedviz import AnimatedViz
import matplotlib.pyplot as plt

def main():
    
    savefile = True
    if savefile:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=1800)
        
    # load a trackfile
    df = pd.read_csv('datasets/trackfiles/DR_USA_Roundabout_FT/vehicle_tracks_000.csv')

    stv = df_to_stackedvehicletraj(df)

    sim = RoundaboutSimulator(stv)

    states = []
    s,_ = sim.reset()
    s = s.reshape(-1,5)
    states.append(s)

    rp = RulePolicy(stv.lengths)

    for i in range(150):
        v = s[:,2:3]
        nni = ~torch.isnan(v)
        a = rp(s.reshape(-1))
        if torch.any(torch.isnan(a[nni])):
            import ipdb
            ipdb.set_trace()
        s,_ = sim.step(a)

        s = s.reshape(-1,5)

        states.append(s)

    states = torch.stack(states).reshape(151,-1)

    fig = plt.figure()
    ax = plt.axes(
        xlim=(900, 1100), ylim=(900, 1100)
        )
    ax.set_aspect('equal', 'box')

    osm = 'datasets/maps/DR_USA_Roundabout_FT.osm'

    av = AnimatedViz(ax, osm, states, stv.lengths, stv.widths)

    ani = animation.FuncAnimation(fig, av.animate, frames=len(states),
                   interval=20, blit=True, init_func=av.initfun, 
                   repeat=not savefile)

    ani.save('rulepolicy.mp4', writer) if savefile else plt.show()


if __name__ == '__main__':
    main()