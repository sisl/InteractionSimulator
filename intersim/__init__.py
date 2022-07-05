# __init__.py

from intersim.box import Box
from intersim.vehicletraj import StackedVehicleTraj
from intersim.graph import InteractionGraph
from intersim.utils import LOCATIONS, MAX_TRACKS

from gym.envs.registration import register

register(
    id='intersim-v0',
    entry_point='intersim.envs:InteractionSimulator',
    order_enforce=False,
)
register(
    id='intersimple-v0',
    entry_point='intersim.envs:Intersimple',
    order_enforce=False,
)