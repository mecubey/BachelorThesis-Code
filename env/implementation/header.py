"""
Provides specific types and constants.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any
import numpy as np

# dtypes
DTYPE_INT = np.int64

DTYPE_FLOAT = np.float32

DTYPE_BOOL = np.bool_

# constructed types
IntArr = np.typing.NDArray[DTYPE_INT]

FloatArr = np.typing.NDArray[DTYPE_FLOAT]

BoolArr = np.typing.NDArray[DTYPE_BOOL]

PositionT = np.ndarray[tuple[2], np.dtype[DTYPE_INT]]

PositionTupleT = tuple[int, int]

IntervalT = tuple[int, int]

@dataclass
class EnvParams:
    """
    Holds necessary parameters for the PathTaskEnv.
    """
    num_agents: int
    maze_intensity: float
    spawn_prob: float
    spread_prob: float
    max_num_spread: int
    dir_spread_probs: list[float]
    max_timestep: int
    field_dim: int
    render_mode: str|None
    delay_btw_frames: float

# constants
ORIGIN = -1

ALL_OUTGOING = 5

class GridOffsets(IntEnum):
    """
    Holds offsets used to assign values to grid cells in observations.
    """
    NO_WALL = 0
    ZONE = 1
    AGENT = 2

class Action(IntEnum):
    """
    Describes each possible action.
    """
    MOVE_UP = 0
    MOVE_RIGHT = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    DO_NOTHING = 4

ACT_TO_DIR: dict[Action, PositionT] = {Action.MOVE_UP: np.array([-1, 0]),
                                       Action.MOVE_RIGHT: np.array([0, 1]),
                                       Action.MOVE_DOWN: np.array([1, 0]),
                                       Action.MOVE_LEFT: np.array([0, -1]),
                                       Action.DO_NOTHING: np.array([0, 0])}

DIR_ARR = np.array(list(ACT_TO_DIR.values()), dtype=DTYPE_INT)

ACTS_ARR = np.array(list(ACT_TO_DIR.keys()), dtype=Action)

def dir_to_act(direction: PositionT) -> Action:
    """
    Takes a direction and returns corresponding positions.
    """
    if direction[0] == -1 and direction[1] == 0:
        return Action.MOVE_UP

    if direction[0] == 0 and direction[1] == 1:
        return Action.MOVE_RIGHT

    if direction[0] == 1 and direction[1] == 0:
        return Action.MOVE_DOWN

    if direction[0] == 0 and direction[1] == -1:
        return Action.MOVE_LEFT

    if direction[0] == 0 and direction[1] == 0:
        return Action.MOVE_LEFT

    raise ValueError(f"{direction} doesn't match any actions!")

# methods
def reverse_dir(act: Action) -> PositionT:
    """
    Returns opposite direction coordinates.
    """
    return ACT_TO_DIR[act]*-1

def randomly(l: list[Any], rng: np.random.Generator) -> list[Any]:
    """
    Shuffle int list and return it.
    """
    rng.shuffle(l)
    return l
