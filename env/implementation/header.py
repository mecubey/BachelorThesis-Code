"""
Provides specific types.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, cast
from colorama import Fore, Back, Style
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

@dataclass
class EnvParams:
    """
    Holds necessary parameters for the PathTaskEnv.
    """
    num_agents: int
    num_tasks: int
    obs_radius: int
    agent_capability: float
    maze_intensity: float
    max_num_spread: int
    step_spread_prob: float
    dir_spread_probs: list[float]
    trait_dim: int
    episode_length: int
    field_dim: int
    render_mode: str
    delay_btw_frames: float
    with_debug_infos: bool

@dataclass
class GridOffsets:
    """
    Holds offsets used to assign values to grid cells in observations.
    """
    is_wall: int
    is_zone: int
    agent_occ_b: int
    agent_occ_e: int
    agent_goal_b: int
    agent_goal_e: int

# constants
EPSILON = 1e-16

FIN_TASK_WITH_CONTR = 100

ACTION_LEN = 5

ORIGIN = -1

ALL_OUTGOING = 5

AGENT_DEPOT = -1

class Action(IntEnum):
    """
    Describes each possible action.
    """
    MOVE_UP = 0
    MOVE_RIGHT = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    DO_NOTHING = 4

Act_To_Dir: dict[Action, PositionT] = {Action.MOVE_UP: np.array([-1, 0]),
                                       Action.MOVE_RIGHT: np.array([0, 1]),
                                       Action.MOVE_DOWN: np.array([1, 0]),
                                       Action.MOVE_LEFT: np.array([0, -1]),
                                       Action.DO_NOTHING: np.array([0, 0])}

Act_To_Dir_Arr: IntArr  = np.array([[-1, 0],
                                    [0, 1],
                                    [1, 0],
                                    [0, -1],
                                    [0, 0]], dtype=DTYPE_INT)

Acts_Arr: IntArr = np.array(Action, dtype=DTYPE_INT)

## RENDERING VARS AND METHODS
def rand_color(rng: np.random.Generator):
    """
    Generate a random foreground color.
    """
    r, g, b = rng.integers(0, 256, size=3)
    return f"\033[38;2;{r};{g};{b}m"

AGENT_CHAR: str = "A" + Style.RESET_ALL
COAL_CHAR: str = "L" + Style.RESET_ALL
TASK_CHAR: str = "T" + Style.RESET_ALL
DEPOT_CHAR: str = Fore.YELLOW + "D" + Style.RESET_ALL
WALL_CHAR: str = Fore.WHITE + "\u2588" + Style.RESET_ALL
ZONE_COL: str = Back.LIGHTCYAN_EX
## RENDERING VARS AND METHODS

# methods
def reverse_dir(act: Action) -> PositionT:
    """
    Returns opposite direction coordinates.
    """
    return Act_To_Dir[act]*-1

def randomly(l: list[Any], rng: np.random.Generator) -> list[Any]:
    """
    Shuffle int list and return it.
    """
    rng.shuffle(l)
    return l

def print_infos(d: dict[Any, Any], indent: int = 0):
    """
    Method used to print any dictionary (key:value pairs are on separate lines).\n
    Specifically used to print infos of env.
    """
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            v = cast(dict[Any, Any], v)
            print_infos(v, indent + 2)
            print()
        else:
            print(f"{pad}{k}: {v}")
