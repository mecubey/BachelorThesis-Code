"""
Contains various utility methods, classes, constants, types.
"""

import math
from enum import Enum, StrEnum
from pathlib import Path
from copy import deepcopy
import numpy as np
from .memory import Memory

# classes
class HazardType(StrEnum):
    """
    Specifies hazard type.

    CONSTANT - the probability of getting stuck stays the same
    """
    CONSTANT = "constant"

class Direction(Enum):
    """
    Possible directions (4-connected grid).
    """
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class Position:
    """
    Represents a 2d position.
    """
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y

    def deepcopy(self) -> Position:
        """
        Return a deep copy of this position.

        Note that does everything (besides primitives) by-reference,
        so sometimes you need to copy.
        """
        return deepcopy(self)

    def as_ndarray(self):
        """
        Return the position as a NumPy array.
        """
        return np.array([self.x, self.y])

    def __add__(self, other: Position):
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Position):
        return Position(self.x - other.x, self.y - other.y)

    def __iadd__(self, other: Position):
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other: Position):
        self.x -= other.x
        self.y -= other.y
        return self

    def euclidean(self) -> float:
        """
        Return euclidean distance.

        Returns:
            float: Euclidean distance.
        """
        return math.sqrt(self.x * self.x + self.y * self.y)

    def manhattan(self) -> float:
        """
        Return manhattan distance.

        Returns:
            float: Manhattan distance.
        """
        return abs(self.x) + abs(self.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __eq__(self, other: object):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y

    def __ne__(self, other: object):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"[{self.x}, {self.y}]"

class Map:
    """
    Represents a 2d map.
    """
    def __init__(self, width: int, height: int, cell_type: str):
        self.width: int = width
        self.height: int = height
        self.cell_type = cell_type

        if cell_type not in CELL_TYPES:
            raise ValueError(f"{cell_type} not a valid type-string!")

        if cell_type == "bool":
            self.tiles = np.zeros(width * height, dtype=np.bool_)

        if cell_type == "int":
            self.tiles = np.zeros(width * height, dtype=np.int_)

        self.reset()

    def reset(self):
        """
        Reset every cell.
        """
        if self.cell_type == "bool":
            self.tiles.fill(False)

        if self.cell_type == "int":
            self.tiles.fill(INVALID_AGENT_ID)

    def inside(self, pos: Position):
        """
        Check whether a position is inside the map.

        Args:
            pos (Position): Position to check.

        Returns:
            _type_: `True` if inside, `False` otherwise.
        """
        return (0 <= pos.x < self.height) and (0 <= pos.y < self.width)

    def __getitem__(self, pos: Position):
        return self.tiles[pos.x * self.width + pos.y]

    def __setitem__(self, pos: Position, value: bool|int):
        self.tiles[pos.x * self.width + pos.y] = value

class Agent:
    """
    Represents an agent.
    """
    def __init__(self, *,
                i: int,
                priority: float,
                start_pos: Position,
                goal_pos: Position) -> None:
        self.id = i
        self.memory = Memory()
        self.initial_priority = priority
        self.current_priority = priority
        self.initial_pos = start_pos
        self.current_pos = start_pos.deepcopy()
        self.goal_pos = goal_pos
        self.damage = 0
        self.frozen_for = 0

    def frozen(self) -> bool:
        """
        Check if agent is frozen.

        If an agent is frozen, then it cannot move.
        """
        return self.frozen_for > 0

    def decay_freeze(self) -> None:
        """
        Decrease the freeze time by 1.
        """
        self.frozen_for = max(self.frozen_for-1, 0)

    def freeze(self) -> None:
        """
        Freeze agent. The more damage an agent has taken,
        the longer it is frozen for.
        """
        self.frozen_for = self.damage

    def increase_damage(self) -> None:
        """
        Increase damage of this agent by 1.

        Damage value is clipped to MAX_DAMAGE.
        """
        self.damage = min(self.damage+1, MAX_DAMAGE)

    def move(self, action: Position) -> None:
        """
        Move agent accordin to action.

        Args:
            action (Position): Action agent should take.
        """
        self.current_pos += action

    def on_goal(self) -> bool:
        """
        Check if agent is on its goal.

        Returns:
            bool: True if agent is on its goal, False otherwise.
        """
        return self.current_pos == self.goal_pos

    def reset(self) -> None:
        """
        Reset position and priority of agent.
        """
        self.damage = 0
        self.frozen_for = 0
        self.current_priority = self.initial_priority
        self.current_pos = self.initial_pos.deepcopy()
        self.memory.reset()

Agents = list[Agent]

# constants
BASE_DIR = Path(__file__).resolve().parent.parent
MAPS_DIR = BASE_DIR / "maps"
SCENE_DIR = BASE_DIR / "scenarios"
HAZARD_CONFIGS = BASE_DIR / "configs" / "hazard_configs.yaml"
EXPERIMENT_RESULTS_DIR = BASE_DIR / "experiments" / "results"

DIR_TO_POS: dict[Direction, Position] = {Direction.UP: Position(-1, 0),
                                         Direction.RIGHT: Position(0, 1),
                                         Direction.DOWN: Position(1, 0),
                                         Direction.LEFT: Position(0, -1)}

CELL_TYPES = ["bool", "int"]

INVALID_AGENT_ID = -1

MAX_NUM_INSTANCES = 25

GLOBAL_HAZARD_SEED = 0

GLOBAL_SOLVER_SEED = 0

MAX_DAMAGE = 10

STAY = Position(0, 0)

# types
Positions = list[Position]

# methods
def scene_to_path(scene_name: str) -> str:
    """
    Transform a scenario name into a path relative to the scenario folder.

    Args:
        scene_name (str): Name of scenario.

    Returns:
        str: Path inside scenario folder to that scenario.
    """
    map_name, scenetype, scene_id = scene_name.rsplit("-", 2)

    return f"{map_name}/scen-{scenetype}/{map_name}-{scenetype}-{scene_id}"

def get_scenario_path(scenario_name: str) -> str:
    """
    Get the path of a scenario.

    Args:
        map_name (str): Specified scenario name.

    Raises:
        FileNotFoundError: Raises error if the scenario does not exist.

    Returns:
        str: Path to the scenario.
    """
    path = SCENE_DIR / f"{scene_to_path(scenario_name)}.scen"

    if not path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_name}")

    return str(path)

def get_map_path(map_name: str) -> str:
    """
    Get the path of a map.

    Args:
        map_name (str): Specified map name.

    Raises:
        FileNotFoundError: Raises error if the map does not exist.

    Returns:
        str: Path to the map.
    """
    path = MAPS_DIR / f"{map_name}.map"

    if not path.exists():
        raise FileNotFoundError(f"Map not found: {map_name}")

    return str(path)
