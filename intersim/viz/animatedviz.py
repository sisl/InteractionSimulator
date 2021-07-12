# animatedviz.py
from numpy import pi
import numpy as np
import torch

import matplotlib
from matplotlib import cm
import matplotlib.patches
import matplotlib.transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from intersim.viz.utils import batched_rotate_around_center, draw_map_without_lanelet, build_map

import os

opj = os.path.join

def animate(osm, states, lengths, widths, graphs=None, filestr='render', **kwargs):
    """
    Wrapper for animating simulation once finished
    Args:
        osm (str): path to .osm map file
        states (torch.tensor): (frames, nv*5) tensor of vehicle states
        lengths (torch.tensor): (nv,) array of vehicle lengths 
        widths (torch.tensor): (nv,) array of vehicle widths
        graphs (list[list[tuple]]): list of list of edges. Outer list indexes frame.
        filestr (str): base file string to save animation to
    """
    fps = kwargs.get('fps', 15)
    bitrate = kwargs.get('bitrate', 1800)
    enc = kwargs.get('encoder', 'ffmpeg')
    iv = kwargs.get('interval', 20)
    blit = kwargs.get('blit', True)

    Writer = animation.writers[enc]
    writer = Writer(fps=fps, bitrate=bitrate)
    fig = plt.figure()
    ax = plt.axes()
    av = AnimatedViz(ax, osm, states, lengths, widths, graphs=graphs)
    ani = animation.FuncAnimation(fig, av.animate, frames=len(states),
                    interval=iv, blit=blit, init_func=av.initfun,repeat=False)
    ani.save(filestr+'_ani.mp4', writer)

class AnimatedViz:
    '''
    Animated visualization of a state sequence
    '''

    def __init__(self, ax, osm, states, lengths, widths, graphs=None):
        '''
        Args:
            ax (plt.Axis): matplotlib Axis
            osm (str): osm file name
            states (torch.tensor): (T, nv*5) states
            lengths (torch.tensor): (nv,) car lengths
            widths (torch.tensor): (nv,) car widths
            graphs (list of list of tuples): graphs[i][j] is the jth edge tuple in the ith frame
        ''' 

        self._T = states.shape[0]
        self._nv = states.shape[1] // 5

        states = states.reshape(self._T,self._nv,5)

        self._ax = ax
        self._osm = osm
        self._map_info, self._point_dict = build_map(osm)
        self._x = states[...,0].detach().numpy()
        self._y = states[...,1].detach().numpy()
        self._psi = states[...,3].detach().numpy()

        self._lengths = lengths.detach().numpy()
        self._widths = widths.detach().numpy()
        
        self._graphs = graphs
        
    @property
    def lines(self):
        return self._carrects + self._edges + [self._text]

    def initfun(self):
        ax = self._ax 

        draw_map_without_lanelet(self._map_info, self._point_dict, ax)

        # init car patches
        carrects = []
        car_colors = ['r']*self._nv
        cmap = cm.get_cmap('jet')
        car_colors = cmap(np.linspace(0,1,num=self._nv))
        np.random.shuffle(car_colors)
        for i in range(self._nv):
            rectpts = np.array([(-1.,-1.), (1.,-1), (1.,1.), (-1.,1.)])
            rect = matplotlib.patches.Polygon(rectpts, closed=True, color=car_colors[i], zorder=2.5, ec='k')
            ax.add_patch(rect)
            carrects.append(rect)
        self._carrects = carrects
        self._edges = []
        self._text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        return self.lines

    def animate(self, i):
        '''
        Args:
            i (int): animation timestep
        '''

        x, y, lengths, widths = self._x, self._y, self._lengths, self._widths
        ax = self._ax
        psi = self._psi

        nni = ~np.isnan(x[i])
        x = x[i, nni]
        y = y[i, nni]
        lengths = lengths[nni]
        widths = widths[nni]
        psi = psi[i, nni]

        self._text.set_text('i=%d' % i)
            
        T = self._T

        i = min(T-1, i)

        lowleft = np.stack([x - lengths / 2., y - widths / 2.], axis=-1)
        lowright = np.stack([x + lengths / 2., y - widths / 2.], axis=-1)
        upright = np.stack([x + lengths / 2., y + widths / 2.], axis=-1)
        upleft = np.stack([x - lengths / 2., y + widths / 2.], axis=-1)

        rotcorners = batched_rotate_around_center(np.stack([lowleft, lowright, upright, upleft],axis=1), 
                np.stack([x, y], axis=-1), 
                yaw=psi).tolist()

        ncars = len(x)

        nnotcars = self._nv - ncars

        rotcorners += [None for j in range(nnotcars)]

        for corners, carrect in zip(rotcorners,self._carrects):
            if corners is None:
                carrect.set_xy(np.array([(-1.,-1.), (1.,-1), (1.,1.), (-1.,1.)]))
                continue
            carrect.set_xy(corners)
        
        edges = []
        if self._graphs is not None:
            for edge in self._edges:
                edge.remove()
            
            graph = self._graphs[i]
            for e in graph:
                stidx, enidx = e
                
                #arrow_func = matplotlib.patches.ConnectionPatch if (enidx, stidx) in graph else matplotlib.patches.Arrow
                ars = '<|-|>' if (enidx, stidx) in graph else '-|>'
                arrow = matplotlib.patches.FancyArrowPatch(posA = (self._x[i,stidx], self._y[i,stidx]),
                    posB = (self._x[i,enidx], self._y[i,enidx]),
                    arrowstyle=ars, mutation_scale=15, color='w', zorder=2.9, ec='k',)

                # arrow = matplotlib.patches.Arrow(self._x[i,stidx], self._y[i,stidx], 
                #                                  self._x[i,enidx] - self._x[i,stidx], 
                #                                  self._y[i,enidx] - self._y[i,stidx],  
                #                                  width=3.0, color='c', zorder=0.1, ec='k')

                ax.add_patch(arrow)
                edges.append(arrow)
        self._edges = edges
        return self.lines