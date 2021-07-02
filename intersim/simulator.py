# simulator.py

import torch

from intersim.datautils import ssdot_to_simstates
from intersim import Box
from intersim import StackedVehicleTraj
import numpy as np

class RoundaboutSimulator():

    def __init__(self, svt, max_acc=1.0):
        """
        Roundabout simulator.
        Args:
            svt (StackedVehicleTraj): vehicle trajectory to reference
            max_acc (float): maximum acceleration/deceleration
        Notes:
            action_space is [acc] ^ nvehicles
            state_space is [x,y,v,psi,psidot] ^ nvehicles
        """

        self._nv = svt.nv
        self._dt = svt.dt
        self._svt = svt

        super(RoundaboutSimulator, self).__init__()
        self.action_space = Box(low=-max_acc, high=max_acc, shape=(self._nv,1))

        low = torch.tensor([[-np.inf,-np.inf,0.,-np.pi, -np.inf]]).expand(self._nv,5)
        high = torch.tensor([[np.inf,np.inf,np.inf,np.pi, np.inf]]).expand(self._nv,5)
        self.state_space = Box(low=low, high=high, shape=(self._nv,5))

        self._state = torch.zeros(self._nv, 2) * np.nan
        self._t = svt.minT

        self._exceeded = None
    
    @property
    def state(self):
        """
        Return [s, v]
        Returns:
            state (torch.tensor): (nv, 2) state in terms of s, sdot
        """
        return self._state
    
    @property
    def path_coefs(self):
        """
        Return coefficients for computing x(s) and y(s)
        Returns:
            xpoly (torch.tensor): (nv, deg) x polynomial path coefficients
            ypoly (torch.tensor): (nv, deg) y polynomial path coefficients
        """
        svt = self._svt
        return svt.xpoly, svt.ypoly
    
    @property
    def smax(self):
        """
        Return maximum path lengths
        Returns:
            smax (torch.tensor): (nv,) maximum path lengths
        """
        return self._svt.smax
    
    @property
    def projected_state(self):
        """
        Projects [s, v] to [x,y,v,psi,psidot]
        Returns:
            projstate (torch.tensor): (nv, 5) projected state
        """
        svt = self._svt
        projstate = ssdot_to_simstates(self._state[:,0].unsqueeze(0),self._state[:,1].unsqueeze(0),
                svt.xpoly, svt.dxpoly, svt.ddxpoly,
                                svt.ypoly, svt.dypoly, svt.ddypoly)
        return projstate[0].reshape(self._nv, 5)
    

    def step(self, action):
        """
        Step simulation forward 1 time-step.
        Args:
            action (torch.tensor): (nvehicles,1) accelerations
        Returns:
            next_state (torch.tensor): (nvehicles*5,) next state
            info (dict): info
        """

        self._t += self._dt

        # euler step the state
        nextv = self._state[:,1:2] + action * self._dt
        nextv = nextv.clamp(0., np.inf) # not differentiable!
        self._state[:,0:1] = self._state[:,0:1] + 0.5*self._dt*(nextv + self._state[:,1:2])
        self._state[:,1:2] = nextv

        nni = ~torch.isnan(self._state[:,0])      

        # see which states exceeded their maxes
        self._exceeded = (self._state[:,0] > self._svt.smax) | self._exceeded
        self._state[self._exceeded] = np.nan

        # see which new vehicles should spawn
        should_spawn = (self._svt.t0 < self._t) & ~self._exceeded & torch.isnan(self._state[:,0])

        # for now, just spawn them
        spawned = should_spawn
        self._state[spawned,0] = 0.
        self._state[spawned,1] = self._svt.v0[spawned]


        return self.projected_state.reshape(-1), {
                        'exceeded':np.argwhere(self._exceeded.detach().numpy()).flatten(),
                         'should_spawn': np.argwhere(should_spawn.detach().numpy()).flatten(),
                         'spawned': np.argwhere(spawned.detach().numpy()).flatten(),
                         'raw_state': self._state.clone()}

    def reset(self):
        """
        Reset simulation.
        Returns:
            next_state (torch.tensor): (nvehicles*5,) next state
            info (dict): info
        """

        self._state = self._svt.state0
        self._t = self._svt.minT

        self._exceeded = torch.tensor([False] * self._nv)

        return self.projected_state.reshape(-1), {'raw_state': self._state.clone()}

    def render(self):
        raise NotImplementedError