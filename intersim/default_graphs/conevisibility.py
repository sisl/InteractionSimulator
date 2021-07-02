#conevisibility.py

import torch
from torch import nn
from intersim.graph import InteractionGraph
import numpy as np

class ConeVisibilityGraph(InteractionGraph):
    """
    Observe another vehicle if they are within r meters and half_angle degrees from your heading.
    """
    
    def __init__(self, r = 20, half_angle = 20, neighbor_dict = {}):
        """
        Args:
            r (float): distance for observation
            half_angle (float): half_angle for observation
            neighbor_dict (dict of vertices to edges): dictionary of node neighbors
        """
        InteractionGraph.__init__(self, neighbor_dict)

        self._r = r
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
        
        # now select the vehicles within distance and half-angle requirements
        valid = (dr < self._r) & (delpsi.abs() < self._half_angle*np.pi/180) & (dr > 0)
        
        # generate and store neighbor graph
        neighbor_dict = {}
        for i in range(nv): 
            if not torch.isnan(x[i,0]):
                neighbor_dict[i] = [j for j in range(nv) if valid[j,i]]
            
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