import logging
from intersim.viz.animatedviz import AnimatedViz
import matplotlib.patches
import numpy as np

def make_marker_viz(PredecessorViz: AnimatedViz, agent):

    class MarkerAnimatedViz(PredecessorViz):

        _agent = agent
        
        def animate(self, i):
            logging.info(f'Highlighting agent {self._agent}')
            self._carrects[self._agent].set_edgecolor('red')
            self._carrects[self._agent].set_fill(False)
            return super().animate(i)
    
    return MarkerAnimatedViz


def make_observation_viz(PredecessorViz: AnimatedViz, observations, width=5):

    class ObservationAnimatedViz(PredecessorViz):

        _observations = observations
        
        @staticmethod
        def _draw(observation):
            x = observation[0, 0]
            y = observation[0, 1]
            arrows = [matplotlib.patches.Arrow(x, y, dx, dy, width) for (dx, dy) in observation[1:, :2]]
            return arrows

        def animate(self, i):
            logging.info(f'Drawing observation {i}')
            super().animate(i)
            for p in self._draw(self._observations[i]):
                self._ax.add_patch(p)
                self._edges.append(p)
            return self.lines
    
    return ObservationAnimatedViz


def make_lidar_observation_viz(PredecessorViz: AnimatedViz, observations, width=5):

    class ObservationAnimatedViz(PredecessorViz):

        _observations = observations
        
        @staticmethod
        def _draw(observation):
            distances = observation[..., 1:, 0]
            angles = observation[..., 1:, 1]
            #angles = np.linspace(-np.pi, np.pi, observation.shape[-2] - 1)
            relx = distances * np.cos(angles)
            rely = distances * np.sin(angles)
            observation[..., 1:, 0] = relx
            observation[..., 1:, 1] = rely

            psi = observation[0, 3]
            fwd = np.stack((np.cos(psi), np.sin(psi)), axis=-1)
            left = np.stack((-np.sin(psi), np.cos(psi)), axis=-1)
            ego_tf = np.stack((fwd, left), axis=-2)
            observation[..., 1:, :2] = np.matmul(observation[..., 1:, :2], ego_tf)

            x = observation[0, 0]
            y = observation[0, 1]
            color = lambda i: (np.sin(i)**2, np.sin(i+1)**2, np.sin(i+2)**2)

            arrows = [matplotlib.patches.Arrow(x, y, dx, dy, width, color=color(i)) for i, (dx, dy) in enumerate(observation[1:, :2])]
            return arrows

        def animate(self, i):
            logging.info(f'Drawing observation {i}')
            super().animate(i)
            for p in self._draw(self._observations[i]):
                self._ax.add_patch(p)
                self._edges.append(p)
            return self.lines
    
    return ObservationAnimatedViz


def make_action_viz(PredecessorViz: AnimatedViz, actions):

    class ActionAnimatedViz(PredecessorViz):

        _actions = actions

        def initfun(self):
            self._action_text = self._ax.text(0.05, 0.85, '', transform=self._ax.transAxes)
            return super().initfun()

        def animate(self, i):
            super().animate(i)
            
            if i < len(self._actions):
                logging.info(f'Drawing action {i} (a = {self._actions[i]})')
                self._action_text.set_text(f'a={self._actions[i]}')
            else:
                logging.info(f'Drawing action {i} (no action)')
                self._action_text.set_text('a=')
            
            return self.lines
            
        @property
        def lines(self):
            return super().lines + [self._action_text]
    
    return ActionAnimatedViz


def make_reward_viz(PredecessorViz: AnimatedViz, rewards):

    class RewardAnimatedViz(PredecessorViz):

        _rewards = rewards

        def initfun(self):
            self._reward_text = self._ax.text(0.05, 0.8, '', transform=self._ax.transAxes)
            return super().initfun()

        def animate(self, i):
            super().animate(i)
            
            if i < len(self._rewards):
                logging.info(f'Drawing reward {i} (r = {self._rewards[i]})')
                self._reward_text.set_text(f'r={self._rewards[i]}')
            else:
                logging.info(f'Drawing reward {i} (no reward)')
                self._reward_text.set_text('r=')
            
            return self.lines
            
        @property
        def lines(self):
            return super().lines + [self._reward_text]
    
    return RewardAnimatedViz