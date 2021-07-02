#closestobstacle.py

import torch
from torch import nn
from intersim.graph import InteractionGraph
import numpy as np

class ClosestObstacleGraph(InteractionGraph):
    """
    Observe the closest vehicle within half-angle degrees.
    """
    
    def __init__(self, half_angle = 20, neighbor_dict = {}):
        """
        Args:
            half_angle (float): half_angle for observation
            neighbor_dict (dict of vertices to edges): dictionary of node neighbors
        """
        InteractionGraph.__init__(self, neighbor_dict)

        self._half_angle = half_angle

    def update_graph(self, state):
        """
        Update the neighbor dict given a new state.
        
        Args:
            state (torch.tensor): (nv*5,) vehicle states
        """
        ndim = state.ndim
        assert ndim==1, "Improper state size to graph updater"
        
        state = state.reshape(-1, 5)
        nv = state.shape[-2]
        
        x = state[...,0:1]
        y = state[...,1:2]
        v = state[...,2:3]
        psi = state[...,3:4]
        psidot = state[...,4:5]

        # compute relative distances, relative velocities
        dx = x.unsqueeze(-2) - x.unsqueeze(-3) #(*, nv, nv, 1) where dx[*,i,j] is x[*,i] - x[*,j]
        dy = y.unsqueeze(-2) - y.unsqueeze(-3)
        dr = (dx ** 2 + dy ** 2).sqrt()
        delpsi = self.to_circle(torch.atan2(dy,dx) - psi.unsqueeze(-3))
        ndist = dr * delpsi.cos()
        ndisth = dr * delpsi.sin()
        
        # now select the vehicles within distance and half-angle requirements
        ndist_ = torch.where((ndist > 0) & (delpsi.abs() < self._half_angle*np.pi/180), ndist, np.inf)

        cndist, inds = ndist_.min(dim=-3)
        
        # generate and store neighbor graph
        neighbor_dict = {}
        for i in range(nv): 
            if cndist[i,0] < np.inf:
                neighbor_dict[i] = [inds[i,0]]
            
        self._neighbor_dict = neighbor_dict
        
    def to_circle(self, x):
        """
        Casts x (in rad) to [-pi, pi)
        
        Args:
            x (torch.tensor): (*) input angles (radians)
            
        Returns:
            y (torch.tensor): (*) x cast to [-pi, pi)
        """
        y = torch.remainder(x + np.pi, 2*np.pi) - np.pi
        return y