# rulepolicy.py

import torch
from torch import nn
from intersim.policy import Policy
import numpy as np

class RulePolicy(Policy, nn.Module):

    def __init__(self, lengths, v0=11.17, s0=3., dth=5., amax=1.5, b=4.):
        """
        Args:
            lengths (torch.tensor): vehicle_lengths
            v0 (float): desired velocity
            s0 (float): minimum spacing
            dth (float): desired time headway
            amax (float): max acceleration
            b (float): comfortable braking deceleration
        """
        nn.Module.__init__(self)

        self._v0 = nn.Parameter(torch.tensor([v0]))
        self._s0 = nn.Parameter(torch.tensor([s0]))
        self._dth = nn.Parameter(torch.tensor([dth]))
        self._amax = nn.Parameter(torch.tensor([amax]))
        self._b = nn.Parameter(torch.tensor([b]))
        self._expdel = 4.
        self._lengths = lengths

    def compute_action(self, state):
        
        ndim = state.ndim
        lead_dims = state.shape[:-1]
        state = state.reshape(*(lead_dims + (-1, 5))) # state (*, nv, 5)
        nv = state.shape[-2]
        lengths = self._lengths.reshape(*([1]*(ndim-1) + [-1, 1]))

        x = state[...,0:1] # (*, nv, 1)
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
        ndist_ = torch.where((ndist > 0) & (delpsi.abs() < 20*np.pi/180), ndist, np.inf)

        cndist, inds = ndist_.min(dim=-3)
        #cndist is closest dist, inds are indices of closest dist , with shape (*, nv, 1)
        
        vx = v * psi.cos() # current x velocity
        vy = v * psi.sin()

        # using the inds of the closest vehicle, compute relative velocities
        dvx = vx.gather(-2,inds) - vx # x velocity to index of closest vehicle in front (*, nv, 1)
        dvy = vy.gather(-2,inds) - vy
        dv = (dvx ** 2 + dvy ** 2).sqrt() # total relative velocity to closest vehicle in front
        vdelpsi = self.to_circle(torch.atan2(dvy, dvx) - psi) # angle between relative velocity vector and heading
        
        ndv = dv * vdelpsi.cos() # component of relative velocity along heading
        
        # select agents to yield right of way and stop immediately
        stop_car = self.stop_car(dr, delpsi, dpsi, v) 
        
        sal = cndist - lengths
        sstar = self._s0 + v*self._dth + v*ndv / (2. * (self._amax * self._b).sqrt())
        action_free = self._amax * (1. - (v / self._v0) ** self._expdel)
        
        # if yielding r-o-w, perform max deceleration
        sstar_sal_ratio = torch.where(stop_car==1., 1., sstar/sal) 
        action_int =  -self._amax*(sstar_sal_ratio)**2

        action = torch.where(torch.isinf(sal) | torch.isnan(sal), action_free, action_free + action_int)
        action = torch.where(stop_car==1., action_free + action_int, action)

        return action # (*, nv, 1)

    def stop_car(self, dr, delpsi, dpsi, v):
        """
        Args:
            dr (torch.tensor): (*, nv, nv, 1) where [..., i, j, 0] indexes the distance from j to i
            delpsi (torch.tensor): (*, nv, nv, 1) where [..., i, j, 0] indexes the angle to i from j's heading            
            dpsi (torch.tensor): (*, nv, nv, 1) where [..., i, j, 0] indexes the difference between j and i's heading angles
            v (torch.tensor): (*, nv, 1): vehicle velocities
            
        Returns:
            stop_car (torch.tensor): (*, nv, 1) tensor which is 1. if a vehicle should halt
        """
        
        stop_car = torch.zeros_like(v) # should default to 0 if vehicle is nan (0 implies follow idm)
        nv = v.size(-2)
        delv = v.unsqueeze(-2) - v.unsqueeze(-3) # (*, nv, nv, 1) where [*,i,j,0] is vj - vi
        
        # stop immediately if there exists a vehicle which satisfies all of the following:
            # - falls within a d meter, r degree cone from you 
            # - is not aligned with your traffic motion
            # - is going faster than you
        
        distance = 20
        space_angle = 60*np.pi/180
        head_angle = 45*np.pi/180
        #stop = torch.any((dr < distance) & (delpsi.abs() < space_angle), -2)
        stop = torch.any((dr < distance) & (delpsi.abs() < space_angle) & (dpsi.abs() > head_angle) & (delv > 0), -3)
         
        stop_car = torch.where(stop, 1., stop_car)
        
        return stop_car
    
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