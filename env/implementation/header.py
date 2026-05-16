"""
Provides utility types, functions, constants.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any
import numpy as np

# classes
@dataclass(order=True, frozen=True)
class Position:
    """
    Represents a 2D position.
    """
    x: int
    y: int

    def as_ndarray(self):
        """
        Return this position as a NumPy array.
        """
        return np.array([self.x, self.y])

    def length(self) -> int:
        """
        Return Manhattan distance.
        """
        return abs(self.x)+abs(self.y)

    def __add__(self, other: Position|int):
        if isinstance(other, Position):
            return Position(self.x + other.x, self.y + other.y)
        return Position(self.x+other, self.y+other)

    def __sub__(self, other: Position):
        return Position(self.x - other.x, self.y - other.y)

    def __mul__(self, other: int):
        return Position(self.x*other, self.y*other)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __iter__(self):
        yield self.x
        yield self.y

    def __hash__(self):
        return hash((self.x, self.y))

# dtypes
DTYPE_INT = np.int32

DTYPE_FLOAT = np.float32

DTYPE_BOOL = np.bool_

# constructed types
IntArr = np.typing.NDArray[DTYPE_INT]

FloatArr = np.typing.NDArray[DTYPE_FLOAT]

BoolArr = np.typing.NDArray[DTYPE_BOOL]

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

# constants
INVALID_POSITION = Position(-1, -1)

ORIGIN = -1

ALL_OUTGOING = 5

class GridOffsets(IntEnum):
    """
    Holds offsets used to assign values to grid cells in observations.
    """
    NO_WALL = 0
    ZONE = 1
    AGENT = 2

NUM_ACTIONS = 5

class Action(IntEnum):
    """
    Describes each possible action.
    """
    MOVE_UP = 0
    MOVE_RIGHT = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    DO_NOTHING = 4

Actions = list[Action]

ACT_TO_DIR: dict[Action, Position] = {Action.MOVE_UP: Position(-1, 0),
                                      Action.MOVE_RIGHT: Position(0, 1),
                                      Action.MOVE_DOWN: Position(1, 0),
                                      Action.MOVE_LEFT: Position(0, -1),
                                      Action.DO_NOTHING: Position(0, 0)}

ACT_TO_OPPOSITE_ACT: dict[Action, Action] = {Action.MOVE_UP: Action.MOVE_DOWN,
                                             Action.MOVE_RIGHT: Action.MOVE_LEFT,
                                             Action.MOVE_DOWN: Action.MOVE_UP,
                                             Action.MOVE_LEFT: Action.MOVE_RIGHT,
                                             Action.DO_NOTHING: Action.DO_NOTHING}

ACT_TO_DIR_NDARRAY: dict[Action, IntArr] = {Action.MOVE_UP: np.array([-1, 0], dtype=DTYPE_INT),
                                            Action.MOVE_RIGHT: np.array([0, 1], dtype=DTYPE_INT),
                                            Action.MOVE_DOWN: np.array([1, 0], dtype=DTYPE_INT),
                                            Action.MOVE_LEFT: np.array([0, -1], dtype=DTYPE_INT),
                                            Action.DO_NOTHING: np.array([0, 0], dtype=DTYPE_INT)}

DIR_TO_ACT: dict[Position, Action] = {Position(-1, 0): Action.MOVE_UP,
                                      Position(0, 1): Action.MOVE_RIGHT,
                                      Position(1, 0): Action.MOVE_DOWN,
                                      Position(0, -1): Action.MOVE_LEFT,
                                      Position(0, 0): Action.DO_NOTHING}

DIR_ARR = list(ACT_TO_DIR.values())

ACTS_ARR = list(ACT_TO_DIR.keys())

ACTS_ARR_AS_NDARRAY = np.array(ACTS_ARR, dtype=DTYPE_INT)

# methods
def randomly(l: list[Any], rng: np.random.Generator) -> list[Any]:
    """
    Shuffle int list and return it.
    """
    rng.shuffle(l)
    return l

def replace_item_with_multiple(arr: list[Any], elem: Any, new_elems: list[Any]):
    """
    Given a list, an element and multiple new values,
    replaces the element in the list with multiple new values.
    """
    for item in arr:
        if item == elem:
            for new_item in new_elems:
                yield new_item
        else:
            yield item
