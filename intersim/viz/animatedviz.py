# animatedviz.py

import matplotlib
from matplotlib import cm
import matplotlib.patches
import matplotlib.transforms
from numpy import pi
import numpy as np
import torch

import matplotlib.animation as animation

import matplotlib.pyplot as plt

from intersim.viz.utils import batched_rotate_around_center, draw_map_without_lanelet

import os

opj = os.path.join

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

        draw_map_without_lanelet(self._osm, ax, 0.,0.)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        # init car patches
        carrects = []
        cmap = cm.get_cmap('jet')
        car_colors = cmap(np.linspace(0,1,num=self._nv))
        np.random.seed(0)
        np.random.shuffle(car_colors)
        for i in range(self._nv):
            rectpts = np.array([(-1.,-1.), (1.,-1), (1.,1.), (-1.,1.)])
            rect = matplotlib.patches.Polygon(rectpts, closed=True, color=car_colors[i], zorder=9, ec='k')
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
                np.stack([x, y], axis=-1), yaw=psi)

        all_corners = np.stack([np.array([(-1.,-1.), (1.,-1), (1.,1.), (-1.,1.)])]*self._nv)
        all_corners[nni] = rotcorners

        for corners, carrect in zip(all_corners, self._carrects):
            carrect.set_xy(corners)
        
        edges = []
        if self._graphs is not None:
            for edge in self._edges:
                edge.remove()
            
            graph = self._graphs[i]
            for e in graph:
                stidx, enidx = e
                
                ars = '<|-|>' if (enidx, stidx) in graph else '-|>'
                arrow = matplotlib.patches.FancyArrowPatch(posA = (self._x[i,stidx], self._y[i,stidx]),
                    posB = (self._x[i,enidx], self._y[i,enidx]),
                    arrowstyle=ars, mutation_scale=15, color='w', zorder=9.5, ec='k',)

                ax.add_patch(arrow)
                edges.append(arrow)
        self._edges = edges
        return self.lines