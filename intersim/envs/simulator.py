# simulator.py

import torch
import numpy as np

from intersim.datautils import ssdot_to_simstates
from intersim import Box
from intersim import StackedVehicleTraj
from intersim import InteractionGraph
from intersim.utils import to_circle

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)

class RewardMethod(Enum):
    """
    NONE: 0 reward throughout
    """
    NONE = 'none'   
class ObserveMethod(Enum):
    """
    LIGHT: simple state and graph information
    FULL: simple and relative state information, graph information, and map information
    HIDDEN: FULL state information and hidden environment information

    """
    LIGHT = 'light'
    FULL = 'full'
    HIDDEN = 'hidden' 

class InteractionSimulator(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self, svt: StackedVehicleTraj, 
                 map_object, 
                 graph: InteractionGraph = InteractionGraph(),
                 max_acc=1.0, 
                 reward_method='none', 
                 observe_method='full'):
        """
        Roundabout simulator.
        Args:
            svt (StackedVehicleTraj): vehicle trajectory to reference
            map_object: underlying map object
            graph (InteractionGraph): graph depicting vehicle interactions
            max_acc (float): maximum acceleration/deceleration
        Notes:
            action_space is [acc] ^ nvehicles
            state_space is [x,y,v,psi,psidot] ^ nvehicles
        """

        # simulator fields
        self._nv = svt.nv
        self._dt = svt.dt
        self._svt = svt
        self._state = torch.zeros(self._nv, 2) * np.nan
        self._t = svt.minT
        self._exceeded = torch.tensor([False] * self._nv)
        self._max_acc = max_acc
        self._map_object = map_object
        self._map_info = self.extract_map_info()
        self._graph = graph
        self._reward_method = RewardMethod(reward_method)
        self._observe_method = ObserveMethod(observe_method)

        # gym fields
        # alternatively, spaces.Box
        self.action_space = Box(low=-max_acc, high=max_acc, shape=(self._nv,1))

        low = torch.tensor([[-np.inf,-np.inf,0.,-np.pi, -np.inf]]).expand(self._nv,5)
        high = torch.tensor([[np.inf,np.inf,np.inf,np.pi, np.inf]]).expand(self._nv,5)
        self.state_space = Box(low=low, high=high, shape=(self._nv,5))

        self.observation_space=None
        self.done= False
        super(InteractionSimulator, self).__init__() # is this necessary??

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
        projstate = projstate[0].reshape(self._nv, 5)
        projstate[...,3:5] = to_circle(projstate[...,3:5])
        return projstate

    @property
    def relative_state(self, projstate):
        """
        Compute relative state information from [x,y,v,psi,psidot] projected state
        Args:
            projstate (torch.tensor): (nv,5) projected state
        Returns:
            relstate (torch.tensor): (nv, nv, 5) projected state, where relstate[i,j,k] = projstate[j,k] - projstate[i,k]
        """
        relstate = projstate.unsqueeze(0) - projstate.unsqeeze(1)
        relstate[...,3:5] = to_circle(relstate[...,3:5])
        return relstate
    
    @property
    def map_info(self):
        """
        Return information about environment map
        Returns:
            map_info: information about map
        """
        return self._map_info
    
    def extract_map_info(self):
        """
        Generates information about environment map
        Returns:
            map_info: information about map
        """
        # extract map_info from self._map_object
        return None

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
        
        # TODO: add an isFree checker 
        free = torch.tensor([True] * self._nv)
        should_spawn = (self._svt.t0 < self._t) & ~self._exceeded & ~nni & free

        # for now, just spawn them
        spawned = should_spawn
        self._state[spawned,0] = 0.
        self._state[spawned,1] = self._svt.v0[spawned]
        
        projstate = self.projected_state

        # update interaction graph
        self._graph.update_graph(projstate)

        observation = {}
        observation['state'] = projstate.reshape(-1)
        observation['neighbor_dict'] = self._graph.neighbor_dict

        if self._observe_method in [ObserveMethod.FULL, ObserveMethod.HIDDEN]:
            observation['relative_state'] = self.relative_state(projstate)
            observation['map_info'] = self.map_info
        
        if self._observe_method in [ObserveMethod.HIDDEN]:
            observation['hidden_info'] = {
                'exceeded':np.argwhere(self._exceeded.detach().numpy()).flatten(),
                'should_spawn': np.argwhere(should_spawn.detach().numpy()).flatten(),
                'spawned': np.argwhere(spawned.detach().numpy()).flatten(),
                'raw_state': self._state.clone()
            }
        return observation

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

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError