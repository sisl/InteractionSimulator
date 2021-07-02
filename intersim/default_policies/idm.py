# idm.py

import torch
from torch import nn
from intersim.policy import Policy
import numpy as np
import time

class IDM(Policy, nn.Module):

    def __init__(self, lengths, v0=11.17, s0=3., dth=5., amax=1.5, b=4., half_angle=20):
        """
        Args:
            lengths (torch.tensor): vehicle_lengths
            v0 (float): desired velocity
            s0 (float): minimum spacing
            dth (float): desired time headway
            amax (float): max acceleration
            b (float): comfortable braking deceleration
            half_angle (float): half angle of observation cone
        """
        nn.Module.__init__(self)

        self._v0 = nn.Parameter(torch.tensor([v0]))
        self._s0 = nn.Parameter(torch.tensor([s0]))
        self._dth = nn.Parameter(torch.tensor([dth]))
        self._amax = nn.Parameter(torch.tensor([amax]))
        self._b = nn.Parameter(torch.tensor([b]))
        self._expdel = 4.
        self._lengths = lengths
        self._half_angle = half_angle
        self._times = []
        
    def compute_action(self, state):
        
        t = time.time()
        ndim = state.ndim
        lead_dims = state.shape[:-1]
        state = state.reshape(*(lead_dims + (-1, 5)))
        nv = state.shape[-2]
        lengths = self._lengths.reshape(*([1]*(ndim-1) + [-1, 1]))

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
        dpsi = self.to_circle(psi.unsqueeze(-2) - psi.unsqueeze(-3))
        dcp = dpsi.cos()
        dsp = dpsi.sin()

        ndist = dr * delpsi.cos()
        ndisth = dr * delpsi.sin()

        # now select the distances and velocities of the closest vehicles
        ndist_ = torch.where((ndist > 0) & (delpsi.abs() < self._half_angle*np.pi/180), ndist, np.inf)

        cndist, inds = ndist_.min(dim=-3)

        vx = v * psi.cos()
        vy = v * psi.sin()

        # using the inds of the closest vehicle, compute relative velocities
        dvx = vx.gather(-2, inds) - vx
        dvy = vy.gather(-2, inds) - vy
        dv = (dvx ** 2 + dvy ** 2).sqrt()
        vdelpsi = self.to_circle(torch.atan2(dvy, dvx) - psi)
        
        ndv = dv * vdelpsi.cos()

        sstar = self._s0 + v*self._dth + v*ndv / (2. * (self._amax * self._b).sqrt())
        sal = cndist - lengths
        action_free = self._amax * (1. - (v / self._v0) ** self._expdel)
        action_int =  -self._amax*(sstar/sal)**2

        action = torch.where(torch.isinf(sal) | torch.isnan(sal), action_free, action_free + action_int)
        t = time.time() - t
        self._times.append(t)
        return action

    @property
    def times(self):
        return self._times.copy()
    
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

