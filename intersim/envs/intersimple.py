import gym
import gym.spaces
import numpy as np
import torch
import logging
import random
from intersim.viz import make_action_viz, make_marker_viz, make_observation_viz, make_reward_viz, Rasta
import functools
import matplotlib.pyplot as plt
from celluloid import Camera

class Intersimple(gym.Env):
    """Single-agent intersim environment with block observation."""

    def __init__(self, n_obs=5, *args, **kwargs):
        super().__init__()
        self._env = gym.make('intersim:intersim-v0', *args, **kwargs)
        self._env._mode = None # TODO: move this to intersim
        self.nv = self._env._nv

        self._agent = 0
        self._n_obs = n_obs
        self.action_space = gym.spaces.Box(low=self._env._min_acc, high=self._env._max_acc, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1 + self._n_obs, 7))

        self._reset = False

    def _reset_skip(self):
        agent_alive = (~self._env._svt.simstate[:, self._agent, :].isnan()).any(1).nonzero().squeeze()
        assert len(agent_alive), f'Start of trajectory of agent {self._agent} not found.'
        start_idx = agent_alive[0].item()
        
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
        self._reset = True
        obs, info = self._reset_skip()
        return self._simple_obs(obs, info)

    def _simple_obs(self, intersim_obs, intersim_info):
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

    def step(self, action):
        """Observation format
        ```
        ┌──────────────────────┬───────┐
        │ ego state            │ valid │
        │ relative state 1     │ valid │
        │              ...             │
        │ relative state n_obs │ valid │
        └──────────────────────┴───────┘
        ```
        """
        assert self._reset, 'Call `reset()` to reset the environment before the first step'

        if self._env._ind < self._env._svt.simstate.shape[0] - 1:
            next_state = self._env._svt.simstate[self._env._ind + 1]
            gt_action = self._env.target_state(next_state)
        else:
            gt_action = torch.ones((self.nv, 1))

        gt_action[self._agent] = torch.tensor(action)
        observation, reward, done, info = self._env.step(gt_action)

        if observation['state'][self._agent].isnan().all():
            done = True

        return self._simple_obs(observation, info), reward, bool(done), info
    
    def render(self, mode='post'):
        return self._env.render(mode)

    def close(self, *args, **kwargs):
        return self._env.close(*args, **kwargs)


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


class Reward:
    """Extends custom reward function support and makes `stop_on_collision` default to `True`."""

    def __init__(self, reward=constant_reward, stop_on_collision=True, *args, **kwargs):
        super().__init__(stop_on_collision=stop_on_collision, *args, **kwargs)
        self._reward = reward

    def step(self, action):
        state = self._env.projected_state[self._agent]
        obs, _, done, info = super().step(action)

        collision = done and not self._env._exceeded[self._agent]
        info.update({ 'collision': collision })
        
        # base reward on current state to avoid passing nan's
        reward = self._reward(state, action, info)
        return obs, reward, done, info


class TargetSpeedReward(Reward):
    
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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=((1 + self._n_obs) * 7,))

    def reset(self):
        return super().reset().reshape((-1,))
    
    def step(self, action):
        observation, reward, done, info = super().step(action)
        return observation.reshape((-1,)), reward, done, info


class RasterizedObservation:

    def __init__(self, height=200, width=200, m_per_px=0.5, raster_fixpoint=(0.5, 0.5), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)
        self._m_per_px = m_per_px
        self._raster_fixpoint = raster_fixpoint

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
            color=255,
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
            color=255,
        )

        # Draw map
        for road_element in intersim_obs['map_info']:
            way_type = road_element['way_type']
            if way_type == 'curbstone':
                vertices = np.stack((
                    road_element['x_list'],
                    road_element['y_list'],
                ), -1)
                rasta.polylines(canvas, vertices, color=255)

        return canvas[np.newaxis]

    def _simple_obs(self, intersim_obs, intersim_info):
        return self._rasterize(intersim_obs, intersim_info)


class RasterizedRoute:

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

    def __init__(self, agent=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent = agent


class RandomAgent:

    def reset(self):
        self._agent = random.randrange(self.nv)
        return super().reset()

class IncrementingAgent:
    
    def __init__(self, start_agent=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent = (start_agent - 1) %  self.nv # assuming going to call reset after initializing environment

    def reset(self):
        self._agent = (self._agent + 1) %  self.nv # increment agent number
        return super().reset()


class InteractionSimulatorMarkerViz:

    def close(self, *args, **kwargs):
        import intersim.viz.animatedviz
        from intersim.viz.animatedviz import AnimatedViz
        intersim.viz.animatedviz.AnimatedViz = make_marker_viz(
            PredecessorViz=AnimatedViz,
            agent=self._agent
        )
        return super().close(*args, **kwargs)


class ObservationVisualization:

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
        return super().close(*args, **kwargs)

class ActionVisualization:

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
        return super().close(*args, **kwargs)


class RewardVisualization:

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
        return super().close(*args, **kwargs)


class IntersimpleMarker(ObservationVisualization, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, Intersimple):
    """Like `Intersimple`, with `imitation` compatibility layer and additional visualizations in animation."""
    pass

class IntersimpleNormalizedActions(NormalizedActionSpace, IntersimpleMarker):
    """Like `IntersimpleMarker`, with symmetric and normalized action space."""
    pass

class IntersimpleFlat(FlatObservation, IntersimpleNormalizedActions):
    """Like `IntersimpleNormalizedActions`, with flattened observation vector."""
    pass

class IntersimpleFlatAgent(FixedAgent, IntersimpleFlat):
    """Like `IntersimpleFlat`, but agent can be selected at instantiation."""
    pass

class IntersimpleFlatRandomAgent(RandomAgent, IntersimpleFlat):
    """Like `IntersimpleFlat`, but controlled agent is chosen randomly at every reset."""
    pass

class IntersimpleReward(RewardVisualization, Reward, IntersimpleFlatAgent):
    """`IntersimpleFlatAgent` with rewards."""
    pass

class IntersimpleTargetSpeed(RewardVisualization, TargetSpeedReward, IntersimpleFlatAgent):
    """Like `IntersimpleFlatAgent`, with speed deviation and collision penalty."""
    pass

class IntersimpleTargetSpeedAgent(RewardVisualization, TargetSpeedReward, IntersimpleFlatAgent):
    """Like `IntersimpleFlatAgent`, with speed deviation and collision penalty."""
    pass

class IntersimpleTargetSpeedRandom(RewardVisualization, TargetSpeedReward, IntersimpleFlatRandomAgent):
    """Like `IntersimpleTargetSpeed`, with random agent."""
    pass

class IntersimpleRasterized(RewardVisualization, Reward, ImageObservationAnimation, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, Intersimple):
    pass

class NRasterized(FixedAgent, RewardVisualization, Reward, ImageObservationAnimation, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, Intersimple):
    pass

class NRasterizedRoute(FixedAgent, RewardVisualization, Reward, ImageObservationAnimation, RasterizedRoute, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, Intersimple):
    pass

class NRasterizedRandomAgent(RandomAgent, RewardVisualization, Reward, ImageObservationAnimation, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, Intersimple):
    pass

class NRasterizedIncrementingAgent(IncrementingAgent, RewardVisualization, Reward, ImageObservationAnimation, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, Intersimple):
    pass

class NRasterizedRouteRandomAgent(RandomAgent, RewardVisualization, Reward, ImageObservationAnimation, RasterizedRoute, NObservations, RasterizedObservation,
                            NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, Intersimple):
    pass
