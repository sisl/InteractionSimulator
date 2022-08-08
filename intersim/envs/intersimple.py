import gym
import gym.spaces
import numpy as np
import torch
import logging
import random
from intersim.default_policies.policy import Policy
from intersim.default_policies.idm import IDM2
from intersim.viz import make_action_viz, make_marker_viz, make_observation_viz, make_lidar_observation_viz, make_reward_viz, Rasta
import functools
import matplotlib.pyplot as plt
from celluloid import Camera
from intersim.utils import LOCATIONS, MAX_TRACKS
import logging

### Different reward functions
def constant_reward(state, action, info, reward=1., collision_penalty=0.):
    return reward - collision_penalty * info['collision']

def target_speed_reward(state, action, info, target_speed=10., speed_penalty_weight=0.01, collision_penalty=1000.):
    speed = state[2].item()
    collision = info['collision']
    return -speed_penalty_weight * (speed - target_speed)**2 - collision_penalty * collision

def speed_reward(state, action, info, speed_weight=0.01, collision_penalty=1000.):
    speed = state[2].item()
    collision = info['collision']
    return speed_weight * speed - collision_penalty * collision

### Mixin class components
class Intersimple(gym.Env):
    """Single-agent intersim environment with block observation, other vehicles controlled by expert actions."""

    def __init__(self, n_obs:int=5, mu:float=0., random_skip:bool=False, *args, **kwargs):
        """
        Initialize base intersimple environment.

        Args:
            n_obs (int): number of layers in the observation skip
            mu (float): regularizer to smooth the expert acceleration calculation
            random_skip (bool): whether to choose start frames randomly on reset
            use_idm (bool): whether to use IDM to control other agents on impending ego collisions

        """
        super().__init__()
        self._env = gym.make('intersim:intersim-v0', *args, **kwargs) # parent intersim environment
        self._env._mode = None # TODO: move this to intersim
        self.nv = self._env._nv # number of vehicles in parent environment
        self._agent = 0 # agent index which is currently being controlled

        self._mu = mu # regularizing coefficient for expert acceleration smoothing
        self._random_skip = random_skip # whether to choose start frames randomly on environment reset

        self.action_space = gym.spaces.Box(low=self._env._min_acc, high=self._env._max_acc, shape=(1,))
        self.n_relstates = self._env.relative_state.shape[-1]
        self._n_obs = n_obs
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1 + self._n_obs, 1 + self.n_relstates))

        self._reset = False # environment has not been reset

    def _reset_skip(self):
        """
        Reset the environment, while potentially skipping frames if random_skip was true

        Returns:
            obs: intersim observation
            info: intersim info 
        """

        agent_alive = (~self._env._svt.simstate[:, self._agent, :].isnan()).any(1).nonzero().squeeze()
        assert len(agent_alive), f'Start of trajectory of agent {self._agent} not found.'
        start_idx = agent_alive[0].item() if not self._random_skip else random.randint(agent_alive[0].item(), agent_alive[-1].item())
        
        logging.info(f'Skipping to frame {start_idx}.')
        state = self._env._svt.simstate[start_idx].clone()
        assert not state[self._agent].isnan().any(), f'Agent {self._agent} has partially invalid simstates {state[self._agent]}.'
        
        # partially override InteractionSimulator.reset()
        self._env.reset()
        self._env._state = state
        self._env._ind = start_idx
        self._env._exceeded = self._env._svt.simstate[start_idx:, :, :].isnan().all(2).all(0)
        assert not (self._env._exceeded & self._env._svt.simstate[:start_idx, :, :].isnan().all(2).all(0)).any()
        assert not self._env._exceeded.all()
        self._env._graph.update_graph(self._env.projected_state)
        self._env.info['raw_state'] = self._env._state.clone()

        obs = self._env._get_observation(self._env.projected_state)
        info = self._env.info
        
        return obs, info

    def reset(self):
        """
        Reset the intersim environment and return the observation

        Returns:
            obs (np.ndarray): the observation for the controlled agent, as returned by _simple_obs
        """
        self._reset = True
        obs, info = self._reset_skip()
        return self._simple_obs(obs, info)

    def _simple_obs(self, intersim_obs, intersim_info):
        """
        Generate intersimple observation from intersim observation and info.

        Args:
            intersim_obs: the obs returned from the parent intersim env step
            intersim_info: the info returned from the parent intersim env step

        Returns:
            obs (np.ndarray): (n_obs+1, n_relstate+1) the observation for the controlled agent
        """
        ego_state = intersim_obs['state'][self._agent]
        ego_valid = not ego_state.isnan().all()
        if ego_valid and ego_state.isnan().any():
            logging.warning(f'Agent {self._agent} has partially invalid ego state {ego_state} at time step {self._env._ind}.')

        relative_states = intersim_obs['relative_state'][self._agent]
        valid = ~relative_states.isnan().all(-1, keepdim=True)
        if not valid.any():
            logging.info(f'No valid relative states for agent {self._agent} at time step {self._env._ind}.')
        
        _, closest = (relative_states[:, :2]**2).sum(-1).sort()

        n_closest = closest[:self._n_obs]
        obs = np.concatenate((
          np.expand_dims(np.concatenate((ego_state, np.array([0, ego_valid])), -1), axis=0),
          np.concatenate((
              relative_states[n_closest],
              valid[n_closest]
          ), -1)
        ))

        obs = np.where(np.isnan(obs), np.zeros_like(obs), obs)

        return obs
    
    def compute_agent_actions(self):
        if self._env._ind < self._env._svt.simstate.shape[0] - 1:
            next_state = self._env._svt.simstate[self._env._ind + 1]
            gt_action = self._env.target_state(next_state, mu=self._mu) # (nv, 1)
        else:
            gt_action = torch.ones((self.nv, 1))
        
        return gt_action

    def step(self, action):
        """
        Take an intersim step

        Args:
            action (float): the acceleration action for the controlled agent to take
        Returns:
            obs (np.ndarray): the observation for the controlled agent, as returned by _simple_obs
            reward (float): the immediate reward from the step
            done (bool): whether to the episode is finished
            info: environment information
        
        Default observation format is a matrix of shape (n_obs+1, n_rel_state):
        ```
        ┌──────────────────────┬───────┐
        │ ego state,      0    │ valid │
        │ relative_state_1     │ valid │
        │              ...             │
        │ relative_state_n_obs │ valid │
        └──────────────────────┴───────┘
        ```
        """
        assert self._reset, 'Call `reset()` to reset the environment before the first step'

        gt_action = self.compute_agent_actions()
        gt_action[self._agent] = torch.tensor(action)
        observation, reward, done, info = self._env.step(gt_action)

        if observation['state'][self._agent].isnan().all():
            done = True

        info['agent'] = self._agent

        return self._simple_obs(observation, info), reward, bool(done), info
    
    def render(self, mode='post'):
        """
        Overwrite render to call the parent environment render
        """
        return self._env.render(mode)

    def close(self, *args, **kwargs):
        """
        Overwrite close to call the parent environment close
        """
        return self._env.close(*args, **kwargs)

class IDMAgents:

    def __init__(self, *args, use_idm:bool=False, idm_policy:Policy=IDM2, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_idm = use_idm
        self._idm_policy = idm_policy(self)
        self._apply_idm = torch.zeros((self.nv,), dtype=bool)
    
    def compute_agent_actions(self):
        gt_action = super().compute_agent_actions()
            
        if self._use_idm:
            is_ego = torch.zeros((self.nv,), dtype=bool).index_fill_(0, torch.tensor(self._agent), torch.tensor(True))
            exclude_leader = ~(self._apply_idm | is_ego)
            idm_action, idm_leader, idm_leader_valid = self._idm_policy.compute_action(None, exclude_leader=exclude_leader)
            idm_action = torch.from_numpy(idm_action) # (nv,)
            idm_leader = torch.from_numpy(idm_leader) # (nv,)
            idm_leader_valid = torch.from_numpy(idm_leader_valid) # (nv,)

            # if IDM target is ego vehicle or other IDM vehicle, switch to IDM
            enable_idm = idm_leader_valid & ((idm_leader == self._agent) | self._apply_idm[idm_leader])
            # if close enough to GT state (acceleration in bounds) and no collision risk (no IDM target or IDM acceleration larger than GT acceleration), switch back to GT
            disable_idm = (idm_action > self._env._min_acc) & (idm_action < self._env._max_acc) & (~idm_leader_valid | (idm_action > gt_action.squeeze()))
            self._apply_idm = (self._apply_idm | enable_idm) & ~disable_idm # (nv,)

            # Update environment interaction graph with leaders
            # point to self if in IDM mode but without valid IDM target
            idm_leader_or_self = torch.where(idm_leader_valid, idm_leader, torch.arange(len(idm_leader)))
            self._env._graph._neighbor_dict={agent:[leader.item()] for agent, leader in enumerate(idm_leader_or_self) if self._apply_idm[agent]}

            gt_action = torch.where(self._apply_idm.unsqueeze(-1), idm_action.unsqueeze(-1), gt_action) # (nv, 1)
        
        return gt_action

class ImitationCompat:
    """Make environment compatible with `imitation` library, especially `RolloutInfoWrapper`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = None
        self.reward_range = None
        self.spec = None
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info.copy()

class Reward:
    """Extends custom reward function support and makes `stop_on_collision` default to `True`."""

    def __init__(self, reward=constant_reward, stop_on_collision=True, *args, **kwargs):
        super().__init__(stop_on_collision=stop_on_collision, *args, **kwargs)
        self._reward = reward

    def step(self, action):
        state = self._env.projected_state[self._agent]
        obs, _, done, info = super().step(action)

        # collision = done and not self._env._exceeded[self._agent]
        # info.update({ 'collision': collision })
        
        # base reward on current state to avoid passing nan's
        reward = self._reward(state, action, info)
        return obs, reward, done, info

class TargetSpeedReward(Reward):
    """Proved reward through the `target_speed_reward` function."""

    def __init__(self, target_speed=10., speed_penalty_weight=0.01, collision_penalty=1000., *args, **kwargs):
        super().__init__(
            reward=functools.partial(
                target_speed_reward,
                target_speed=target_speed,
                speed_penalty_weight=speed_penalty_weight,
                collision_penalty=collision_penalty
            ),
            *args,
            **kwargs
        )


class FlatObservation:
    """Return a flattened observation vector."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        obs_shape = np.prod(self.observation_space.shape)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,))

    def reset(self):
        return super().reset().reshape((-1,))
    
    def step(self, action):
        observation, reward, done, info = super().step(action)
        return observation.reshape((-1,)), reward, done, info


class LidarRelativeObservation:
    """
    Return observation relative state information based on lidar-like approach:
    Use relative state of closest vehicles in sectors defined by `n_rays`
    """
    def __init__(self, n_rays=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_rays = n_rays
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1 + n_rays, self.n_relstates))

    def _bucket_closest(self, relative_states, buckets):
        distances = (relative_states[:, :2]**2).sum(-1)
        ray_hits = [np.argmin(distances[buckets == b]) for b in buckets]
        return relative_states[ray_hits]

    def _lidar_obs(self, intersim_obs):
        relative_states = intersim_obs['relative_state'][self._agent]

        # dummy observation of unintersected rays
        dummy = np.inf * np.ones((1, self.n_relstates))
        assert (dummy[..., :2]**2).sum() == np.inf

        # filter observations
        valid = ~np.isnan(relative_states.numpy()).any(-1)
        relative_states = relative_states[valid]
        relative_states = np.concatenate((dummy, relative_states))

        # transform relative positions to ego vehicle frame
        psi = intersim_obs['state'][self._agent, 3]
        ego_fwd = np.stack((np.cos(psi), np.sin(psi)), axis=-1)
        ego_left = np.stack((-np.sin(psi), np.cos(psi)), axis=-1)
        ego_tf = np.stack((ego_fwd, ego_left), axis=-2)
        relative_states[..., 1:, :2] = np.matmul(relative_states[..., 1:, :2], ego_tf.T)
        
        # bucket angles
        angles = np.arctan2(relative_states[..., 1], relative_states[..., 0])
        bin_boundaries = np.linspace(-np.pi, np.pi, num=self.n_rays+1)
        bucket_assignments = np.digitize(angles, bins=bin_boundaries[1:-1])
        bucket_matrix = np.expand_dims(bucket_assignments, -2) == np.expand_dims(np.arange(self.n_rays), -1)
        bucket_matrix[..., 0] = True # assign dummy elemement to all buckets

        # take closest vehicle in each bucket
        distances = (relative_states[..., :2]**2).sum(-1)
        bucket_distances = np.where(bucket_matrix, np.expand_dims(distances, -2), np.inf)
        ray_hit_idx = np.argmin(bucket_distances, axis=-1)

        lidar = relative_states[ray_hit_idx]        
        return lidar

    def _simple_obs(self, intersim_obs, intersim_info):
        lidar = self._lidar_obs(intersim_obs)
        ego_state = intersim_obs['state'][self._agent]
        ego_state = np.pad(ego_state, (0, lidar.shape[-1] - ego_state.shape[-1]))
        ego_state = np.expand_dims(ego_state, 0)
        return np.concatenate((ego_state, lidar))


class LidarObservation(LidarRelativeObservation):
    """
    Return observation relative state information based on lidar-like approach, transformed into ego frame:
    Use relative state of closest vehicles in sectors defined by `n_rays`
    """
    def __init__(self, lidar_range=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lidar_range = lidar_range
    
    def _simple_obs(self, intersim_obs, intersim_info):
        obs = super()._simple_obs(intersim_obs, intersim_info)

        # transform relative positions to polar coordinates
        distances = np.linalg.norm(obs[..., 1:, :2], axis=-1)
        angles = np.arctan2(obs[..., 1:, 1], obs[..., 1:, 0])
        obs[..., 1:, 0] = distances
        obs[..., 1:, 1] = angles
        beyond_range = obs[..., 1:, 0] > self.lidar_range

        # transform relative velocities to ego coordinate frame
        psi = obs[..., 0, 3]
        ego_fwd = np.stack((np.cos(psi), np.sin(psi)), axis=-1)
        ego_left = np.stack((-np.sin(psi), np.cos(psi)), axis=-1)
        ego_tf = np.stack((ego_fwd, ego_left), axis=-2)
        obs[..., 1:, 2:4][beyond_range] = 0
        obs[..., 1:, 2:4] = np.matmul(obs[..., 1:, 2:4], ego_tf.T)

        # clip lidar range
        angles = np.linspace(-np.pi, np.pi, obs.shape[-2]-1+1)
        angles = (angles[:-1] + angles[1:]) / 2
        obs[..., 1:, 0][beyond_range] = self.lidar_range
        obs[..., 1:, 1][beyond_range] = angles[beyond_range]
        obs[..., 1:, 2:][beyond_range, :] = 0

        return obs


class RasterizedObservation:
    """Return rasterized observation in ego frame"""

    def __init__(self, height=200, width=200, m_per_px=0.5, raster_fixpoint=(0.5, 0.5), map_color=255, vehicle_color=255, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)
        self._m_per_px = m_per_px
        self._raster_fixpoint = raster_fixpoint
        self._map_color = map_color
        self._vehicle_color = vehicle_color

    def _rasterize(self, intersim_obs, intersim_info):
        canvas = np.zeros(self.observation_space.shape[-2:], dtype=np.uint8)

        if intersim_obs['state'][self._agent].isnan().all():
            return canvas[np.newaxis]
        
        if intersim_obs['state'][self._agent].isnan().any():
            logging.warning(f'Agent {self._agent} has partially invalid ego state {intersim_obs["state"][self._agent]} at time step {self._env._ind}.')
        
        rasta = Rasta(
            m_per_px = self._m_per_px,
            raster_fixpoint = self._raster_fixpoint,
            world_fixpoint = intersim_obs['state'][self._agent, :2],
            camera_rotation = intersim_obs['state'][self._agent, 3]
        )

        # Draw ego agent
        rasta.fill_tilted_rect(
            canvas,
            center=intersim_obs['state'][self._agent, :2],
            length=intersim_info['lengths'][self._agent],
            width=intersim_info['widths'][self._agent],
            rotation=intersim_obs['state'][self._agent, 3],
            color=self._vehicle_color,
        )

        # Draw other agents
        valid = ~intersim_obs['relative_state'][self._agent].isnan().any(1)
        logging.info(f'{valid.sum()} valid observations for agent {self._agent} at timestep {self._env._ind}')
        rasta.fill_tilted_rect(
            canvas,
            center=intersim_obs['state'][self._agent, :2] + intersim_obs['relative_state'][self._agent, valid, :2],
            length=intersim_info['lengths'][valid],
            width=intersim_info['widths'][valid],
            rotation=intersim_obs['state'][self._agent, 3] + intersim_obs['relative_state'][self._agent, valid, 4],
            color=self._vehicle_color,
        )

        # Draw map
        for road_element in intersim_obs['map_info']:
            way_type = road_element['way_type']
            if way_type == 'curbstone':
                vertices = np.stack((
                    road_element['x_list'],
                    road_element['y_list'],
                ), -1)
                rasta.polylines(canvas, vertices, color=self._map_color)

        return canvas[np.newaxis]

    def _simple_obs(self, intersim_obs, intersim_info):
        return self._rasterize(intersim_obs, intersim_info)


class RasterizedRoute:
    """Return rasterized observation in ego frame with route added"""
    def __init__(self, route_thickness=2, height=200, width=200, m_per_px=0.5, raster_fixpoint=(0.5, 0.5), *args, **kwargs):
        super().__init__(
            height=height,
            width=width,
            m_per_px=m_per_px,
            raster_fixpoint=raster_fixpoint,
            *args,
            **kwargs
        )
        channels, _, _ = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(channels+1, height, width),
            dtype=np.uint8
        )
        self._m_per_px = m_per_px
        self._raster_fixpoint = raster_fixpoint
        self._route_thickness = route_thickness

    def _rasterize_route(self, intersim_obs, intersim_info):
        canvas = np.zeros(self.observation_space.shape[-2:], dtype=np.uint8)

        if intersim_obs['state'][self._agent].isnan().all():
            return canvas[np.newaxis]
        
        assert not intersim_obs['state'][self._agent].isnan().any(), f'Agent {self._agent} has partially invalid ego state {intersim_obs["state"][self._agent]} at time step {self._env._ind}.'
        
        rasta = Rasta(
            m_per_px = self._m_per_px,
            raster_fixpoint = self._raster_fixpoint,
            world_fixpoint = intersim_obs['state'][self._agent, :2],
            camera_rotation = intersim_obs['state'][self._agent, 3]
        )

        ego_route = np.stack((
            intersim_obs['paths'][0][self._agent],
            intersim_obs['paths'][1][self._agent],
        ), axis=-1)

        rasta.polylines(canvas, ego_route, thickness=self._route_thickness, color=255)

        return canvas[np.newaxis]

    def _simple_obs(self, intersim_obs, intersim_info):
        img = super()._simple_obs(intersim_obs, intersim_info)
        route = self._rasterize_route(intersim_obs, intersim_info)
        obs = np.concatenate((route, img), axis=0)
        return obs


class NObservations:
    """Return observations which stack multiple past rasterized frames together"""
    def __init__(self, n_frames=5, skip_frames=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_channels, height, width = self.observation_space.shape
        self._framebuffer = np.zeros(((n_frames - 1) * skip_frames + 1, n_channels, height, width), dtype=np.uint8)
        self._skip_frames = skip_frames
        self._n_frames = n_frames
        self._n_channels = n_channels
        self._height = height
        self._width = width
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(n_frames * n_channels, height, width), dtype=np.uint8)
    
    def reset(self):
        self._framebuffer *= 0
        return super().reset()

    def _simple_obs(self, intersim_obs, intersim_info):
        last_frame = super()._simple_obs(intersim_obs, intersim_info)
        self._framebuffer = np.roll(self._framebuffer, shift=1, axis=0)
        self._framebuffer[0] = last_frame
        frames = self._framebuffer[self._skip_frames * np.arange(self._n_frames)]
        frames_flat = np.reshape(frames, (self._n_frames * self._n_channels, self._height, self._width))
        return frames_flat


class ImageObservationAnimation:
    """Overwrite rendering to additionally save a video of the rasterized ego-frame observations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_observation = None
        self._observations = []
    
    def reset(self):
        observation = super().reset()
        self._last_observation = observation
        self._observations = []
        return observation
    
    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._last_observation = observation
        return observation, reward, done, info
    
    def render(self, *args, **kwargs):
        self._observations.append(self._last_observation)
        super().render(*args, **kwargs)
    
    def _save_animation(self, filestr):
        if not self._observations:
            return

        fig = plt.figure()
        camera = Camera(fig)

        for obs in self._observations:
            frame = obs.sum(0)
            plt.imshow(frame)
            camera.snap()
        
        animation = camera.animate(interval=self._env._dt*1000)
        animation.save(filestr + '_observation.mp4')

    def close(self, filestr='render', *args, **kwargs):
        self._save_animation(filestr)
        super().close(filestr=filestr, *args, **kwargs)


class NormalizedActionSpace:
    """Normalize the action space so inputs are in (-1,1) and map to acceleration bounds"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    
    def _unnormalize(self, action):
        if action >= 0:
            return self._env._max_acc * action
        else:
            return self._env._min_acc * -action

    def _normalize(self, action):
        if action >= 0:
            return action / self._env._max_acc
        else:
            return -action / self._env._min_acc

    def step(self, action):
        return super().step(self._unnormalize(action))


class FixedAgent:
    """Fix the agent upon init and reset to control the same agent."""
    def __init__(self, agent=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent = agent


class RandomAgent:
    """Reset to control a random agent."""
    def reset(self):
        self._agent = random.randrange(self.nv)
        return super().reset()


class IncrementingAgent:
    """Increment the controlled agent index on every reset."""
    def __init__(self, start_agent=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent = (start_agent - 1) %  self.nv # assuming going to call reset after initializing environment

    def reset(self):
        self._agent = (self._agent + 1) %  self.nv # increment agent number
        return super().reset()


class FixedLocation:
    """Fix the location/track on initialization."""
    def __init__(self, loc=0, track=0, *args, **kwargs):
        self._location = loc
        self._track = track
        super().__init__(loc=loc, track=track, *args, **kwargs)


class RandomLocation:
    """Reset to a random location/track."""
    records = [(l, t) for l in range(len(LOCATIONS)) for t in range(MAX_TRACKS)]
    # see tests.intersim.envs.test_simulator.test_locations
    records.remove((1, 4))
    records.remove((2, 3))

    def __init__(self, loc=0, track=0, *args, **kwargs):
        self._location = loc
        self._track = track
        self._args = args
        self._kwargs = kwargs
        super().__init__(loc=loc, track=track, *args, **kwargs)

    def reset(self):
        self._location, self._track = random.choice(RandomLocation.records)

        logging.info(f'location {self._location}, track {self._track}')

        self.__init__(
            loc=self._location,
            track=self._track,
            *self._args,
            **self._kwargs,
        )
        
        return super().reset()


class InteractionSimulatorMarkerViz:
    """Mark controlled agent on the animation display."""
    def close(self, *args, **kwargs):
        import intersim.viz.animatedviz
        from intersim.viz.animatedviz import AnimatedViz
        intersim.viz.animatedviz.AnimatedViz = make_marker_viz(
            PredecessorViz=AnimatedViz,
            agent=self._agent
        )
        out = super().close(*args, **kwargs)
        intersim.viz.animatedviz.AnimatedViz = AnimatedViz
        return out


class ObservationVisualization:
    """Update visualization for observations."""
    def reset(self):
        observation = super().reset()
        self._observations = [observation]
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._observations.append(observation)
        return observation, reward, done, info
    
    def close(self, *args, **kwargs):
        import intersim.viz.animatedviz
        from intersim.viz.animatedviz import AnimatedViz
        intersim.viz.animatedviz.AnimatedViz = make_observation_viz(
            PredecessorViz=AnimatedViz,
            observations=self._observations,
        )
        out = super().close(*args, **kwargs)
        intersim.viz.animatedviz.AnimatedViz = AnimatedViz
        return out

class LidarObservationVisualization:
    """Include lidar arrows on the animation display."""
    def reset(self):
        observation = super().reset()
        self._observations = [observation]
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._observations.append(observation)
        return observation, reward, done, info
    
    def close(self, *args, **kwargs):
        import intersim.viz.animatedviz
        from intersim.viz.animatedviz import AnimatedViz
        intersim.viz.animatedviz.AnimatedViz = make_lidar_observation_viz(
            PredecessorViz=AnimatedViz,
            observations=self._observations,
        )
        out = super().close(*args, **kwargs)
        intersim.viz.animatedviz.AnimatedViz = AnimatedViz
        return out

class ActionVisualization:
    """Include the actions on the animation display."""
    def reset(self):
        self._actions = []
        return super().reset()
    
    def step(self, action):
        out = super().step(action) # make sure environment has been reset
        self._actions.append(action)
        return out

    def close(self, *args, **kwargs):
        import intersim.viz.animatedviz
        from intersim.viz.animatedviz import AnimatedViz
        intersim.viz.animatedviz.AnimatedViz = make_action_viz(
            PredecessorViz=AnimatedViz,
            actions=self._actions
        )
        out = super().close(*args, **kwargs)
        intersim.viz.animatedviz.AnimatedViz = AnimatedViz
        return out

class RewardVisualization:
    """Include the rewards on the animation display."""
    def reset(self):
        self._rewards = []
        return super().reset()
    
    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._rewards.append(reward)
        return observation, reward, done, info

    def close(self, *args, **kwargs):
        import intersim.viz.animatedviz
        from intersim.viz.animatedviz import AnimatedViz
        intersim.viz.animatedviz.AnimatedViz = make_reward_viz(
            PredecessorViz=AnimatedViz,
            rewards=self._rewards
        )
        out = super().close(*args, **kwargs)
        intersim.viz.animatedviz.AnimatedViz = AnimatedViz
        return out

class InfoFilter:
    """Filter down info returns by `info_keys` to save time."""
    def __init__(self, info_keys=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if info_keys is None:
            self.info_keys = {
                'projected_state',
                'action_taken',
                'agent',
            }
        else:
            self.info_keys = set(info_keys)

    def step(self, action):
        observation, reward, done, full_info = super().step(action)
        keys = self.info_keys & set(full_info.keys())
        if len(keys) != len(self.info_keys):
            diff = self.info_keys.difference(set(full_info.keys()))
            for k in diff:
                logging.warning("intersimple.InfoFilter: info_key {} does not exist in full_info".format(k))
        info = {k: full_info[k] for k in keys}

        return observation, reward, done, info

### Mixin intersimple environment classes
class IntersimpleMarker(ObservationVisualization, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, IDMAgents, Intersimple):
    pass

class IntersimpleLidar(LidarObservationVisualization, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, LidarObservation, IDMAgents, Intersimple):
    pass

class IntersimpleNormalizedActions(NormalizedActionSpace, IntersimpleMarker):
    pass

class IntersimpleFlat(FlatObservation, IntersimpleNormalizedActions):
    pass

class IntersimpleFlatAgent(FixedAgent, IntersimpleFlat):
    pass

class IntersimpleFlatRandomAgent(RandomAgent, IntersimpleFlat):
    pass

class IntersimpleReward(RewardVisualization, Reward, IntersimpleFlatAgent):
    pass

class IntersimpleLidarFlat(RewardVisualization, Reward, FixedAgent, FlatObservation, NormalizedActionSpace,
                           LidarObservationVisualization, ActionVisualization, InteractionSimulatorMarkerViz,
                           ImitationCompat, LidarObservation, IDMAgents, Intersimple):
    pass

class IntersimpleLidarFlatRandom(RewardVisualization, Reward, RandomAgent, FlatObservation, NormalizedActionSpace,
                           LidarObservationVisualization, ActionVisualization, InteractionSimulatorMarkerViz,
                           ImitationCompat, LidarObservation, IDMAgents, Intersimple):
    pass

class IntersimpleLidarFlatIncrementingAgent(RewardVisualization, Reward, IncrementingAgent, FlatObservation, NormalizedActionSpace,
                           LidarObservationVisualization, ActionVisualization, InteractionSimulatorMarkerViz,
                           ImitationCompat, LidarObservation, IDMAgents, Intersimple):
    pass

class IntersimpleTargetSpeed(RewardVisualization, TargetSpeedReward, IntersimpleFlatAgent):
    pass

class IntersimpleTargetSpeedAgent(RewardVisualization, TargetSpeedReward, IntersimpleFlatAgent):
    pass

class IntersimpleTargetSpeedRandom(RewardVisualization, TargetSpeedReward, IntersimpleFlatRandomAgent):
    pass

class IntersimpleRasterized(RewardVisualization, Reward, ImageObservationAnimation, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, IDMAgents, Intersimple):
    pass

class NRasterized(FixedAgent, RewardVisualization, Reward, ImageObservationAnimation, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, IDMAgents, Intersimple):
    pass

class NRasterizedRoute(FixedAgent, RewardVisualization, Reward, ImageObservationAnimation, RasterizedRoute, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, IDMAgents, Intersimple):
    pass

class NRasterizedRandomAgent(RandomAgent, RewardVisualization, Reward, ImageObservationAnimation, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, IDMAgents, Intersimple):
    pass

class NRasterizedIncrementingAgent(IncrementingAgent, RewardVisualization, Reward, ImageObservationAnimation, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, IDMAgents, Intersimple):
    pass

class NRasterizedRouteIncrementingAgent(IncrementingAgent, RewardVisualization, Reward, ImageObservationAnimation, RasterizedRoute, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, IDMAgents, Intersimple):
    pass

class NRasterizedRouteRandomAgent(RandomAgent, RewardVisualization, Reward, ImageObservationAnimation, RasterizedRoute, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, IDMAgents, Intersimple):
    pass

class NRasterizedRouteRandomAgentLocation(RandomLocation, RandomAgent, RewardVisualization, Reward, ImageObservationAnimation, RasterizedRoute, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, IDMAgents, Intersimple):
    pass

class NRasterizedInfo(InfoFilter, NRasterized):
    pass

class NRasterizedRandomAgentInfo(InfoFilter, NRasterizedRandomAgent):
    pass
