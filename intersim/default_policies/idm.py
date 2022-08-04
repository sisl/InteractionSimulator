# idm.py

import torch
from torch import nn
from intersim.default_policies.policy import Policy
import numpy as np
import time
from typing import List, Tuple
import jax
import jax.numpy as jnp

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


class IDM2(Policy):
    """
    IDM2 returns action predictions based on an IDM policy.

    The front car is chosen as the closer of:
        - closest car within a 45 degree half angle cone of the ego's heading
        - ''' after propagating the environment forward by `t_future' seconds with
            current headings and velocities

    """

    def __init__(self, env,
        target_speed:float= 8.94,
        t_future:List[float]=[0., 1., 2., 3.],
        half_angle:float=60.):
        """
        Initialize policy with pointer to environment it will run on and target speed

        Args:
            env (Intersimple): intersimple environment which IDM runs on
            target_speed (float): target speed in roundabout (default: 8.94=20 mph)
            t_future (List[float]): list of future time at which to compare closest vehicle
            half_angle (float): half angle to look inside for closest vehicle
        """

        self._env = env
        self.t_future = t_future
        self.half_angle = half_angle

        # Default IDM parameters
        assert target_speed>0, 'negative target speed'
        self.v_max = target_speed
        self.a_max = np.array([3.]) # nominal acceleration
        self.tau = 0.5 # desired time headway
        self.b_pref = 2.5 # preferred deceleration
        self.d_min = 3 #minimum spacing
        self.max_pos_error = 2 # m, for matching vehicles to ego path
        self.max_deg_error = 30 # degree, for matching vehicles to ego path
        self.length = 20 # length of predicted path
        self.step = 0.1 # step size of predicted path

        # for np.remainder nan warnings
        np.seterr(invalid='ignore')

        self._compute_idm_action = jax.jit(lambda x, y, s: compute_idm_action(x, y, s, self.max_pos_error, self.max_deg_error, self.step, self.d_min, self.tau, self.a_max, self.b_pref, self.v_max))

    def compute_action(self, observation:np.ndarray,
        *args, **kwargs) -> Tuple[np.ndarray, None]:
        """
        Predict action from observation

        (But actually generate next action from underlying environment state)

        Args:
            observation (np.ndarray): instantaneous observation from environment

        Returns
            action (np.ndarray): action for controlled agent to take
        """
        x, y = self._env._env._generate_paths(delta=self.step, n=self.length/self.step, is_distance=True)
        full_state = self.ps() #(nv, 5)
        action, leader, valid_leader = self.idmact(x.numpy(), y.numpy(), full_state.numpy())
        return np.array(action), np.array(leader, dtype=int), np.array(valid_leader, dtype=bool)
    
    def ps(self):
        return self._env._env.projected_state
    
    def idmact(self, x, y, state):
        return self._compute_idm_action(x, y, state)

def compute_idm_action(x, y, full_state, max_pos_error, max_deg_error, step, d_min, tau, a_max, b_pref, v_max):
    nv = full_state.shape[0]
    v = full_state[:, 2] # (nv,)
    
    heading = to_circle(jnp.arctan2(jnp.diff(y), jnp.diff(x)))

    paths = jnp.stack([x[:,:-1],y[:,:-1], heading], axis=1) # (nv, 3, (path_length-1))

    # (x,y,phi) of all vehicles
    poses = jnp.expand_dims(full_state[:, [0,1,3]], 2) # (nv, 3, 1)

    diff = paths[:, jnp.newaxis] - poses[jnp.newaxis, :] # (nv, nv, 3, path_length-1)
    diff = diff.at[..., 2, :].set(to_circle(diff[..., 2, :]))

    # Test if position and heading angle are close for some point on the future vehicle track
    pos_close = jnp.sum(diff[..., 0:2, :]**2, -2) <= max_pos_error**2 # (nv, nv, path_length-1)
    heading_close = jnp.abs(diff[..., 2, :]) <= max_deg_error * jnp.pi / 180 # (nv, nv, path_length-1)
    # For all combinations of vehicles get the path points where they are close to each other
    close = jnp.logical_and(pos_close, heading_close) # (nv, nv, path_length-1)
    close = close.at[jnp.arange(nv), jnp.arange(nv), :].set(False) # exclude ego agent

    # Determine vehicle that is closest to each agent in terms of path coordinate
    not_close = 1.0 * (close.cumsum(-1) < 1) # (nv, nv, path_length-1)
    first_idx_close = not_close.sum(-1) # sum will be ==nv if agents are never close

    leader = first_idx_close.argmin(-1) # (nv,)
    min_idx = first_idx_close.min(-1) # (nv,)
    valid_leader = min_idx < nv # (nv,)

    d = jnp.where(valid_leader,
        step * min_idx,
        jnp.inf
    )
    delta_v = v - v[leader]
    d_des = jnp.where(valid_leader,
        jnp.maximum(
            d_min + tau * v + v * delta_v / (2* (a_max*b_pref)**0.5 ),
            d_min
        ),
        d_min
    )
    
    #assert (d_des>= d_min).all()
    action = a_max*(1 - (v/v_max)**4 - (d_des/d)**2)

    assert action.shape==(nv,)
    assert leader.shape==(nv,)
    assert valid_leader.shape==(nv,)
    return action, leader, valid_leader

def to_circle(x: jnp.ndarray) -> jnp.ndarray:
    """
    Casts x (in rad) to [-pi, pi)

    Args:
        x (jnp.ndarray): (*) input angle (radians)

    Returns:
        y (jnp.ndarray): (*) x cast to [-pi, pi)
    """
    y = jnp.remainder(x + jnp.pi, 2*jnp.pi) - jnp.pi
    return y
