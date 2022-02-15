import gym
from stable_baselines3.common.policies import BasePolicy

class IntersimExpert(BasePolicy):
    
    def __init__(self, intersim_env, mu=0, *args, **kwargs):
        super().__init__(
            observation_space=gym.spaces.Space(),
            action_space=gym.spaces.Space(),
            *args, **kwargs
        )
        self._intersim = intersim_env
        self._mu = mu

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()

    def _action(self):
        target_t = min(self._intersim._ind + 1, len(self._intersim._svt.simstate) - 1)
        target_state = self._intersim._svt.simstate[target_t]
        return self._intersim.target_state(target_state, mu=self._mu)

    def predict(self, *args, **kwargs):
        return self._action(), None

class IntersimpleExpert(BasePolicy):

    def __init__(self, intersimple_env, mu=0, *args, **kwargs):
        super().__init__(
            observation_space=intersimple_env.observation_space,
            action_space=intersimple_env.action_space,
            *args, **kwargs
        )
        self._intersimple = intersimple_env
        self._intersim_expert = IntersimExpert(intersimple_env._env, mu=mu)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()

    def _action(self):
        # RandomLocation mixin re-initializes the intersim sub-env
        self._intersim_expert._intersim = self._intersimple._env
        return self._intersim_expert._action()[self._intersimple._agent]

    def predict(self, *args, **kwargs):
        return self._action(), None

class NormalizedIntersimpleExpert(IntersimpleExpert):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, *args, **kwargs):
        action, _ = super().predict(*args, **kwargs)
        return self._intersimple._normalize(action), None

class DummyVecEnvPolicy(BasePolicy):

    def __init__(self, experts):
        self._experts = [e() for e in experts]

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()
    
    def predict(self, *args, **kwargs):
        predictions = [e.predict() for e in self._experts]
        actions = [p[0] for p in predictions]
        states = [p[1] for p in predictions]
        return actions, states
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()

def save_video(env, expert):
    env.reset()
    env.render()
    done = False
    while not done:
        actions, _ = expert.predict()
        _, _, done, _ = env.step(actions)
        env.render()
    env.close()
