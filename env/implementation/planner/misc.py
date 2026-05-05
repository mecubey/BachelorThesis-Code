"""
Miscelleanous methods and objects for SIPP.
"""

from pprint import pprint
from dataclasses import dataclass
from collections import defaultdict
from typing import Any
from .. import header as h
import numpy as np

@dataclass
class Position:
    """
    General purpose class to hash and calculate with 2D positions.
    """
    x: int
    y: int

    def as_ndarray(self) -> np.ndarray:
        """
        Return position as a NumPy array.
        """
        return np.array([self.x, self.y], dtype=h.DTYPE_INT)

    def __add__(self, other: Any):
        if isinstance(other, Position):
            return Position(self.x + other.x, self.y + other.y)
        raise NotImplementedError(f"Addition not supported for Position and {type(other)}")

    def __sub__(self, other: Any):
        if isinstance(other, Position):
            return Position(self.x - other.x, self.y - other.y)
        raise NotImplementedError(f"Subtraction not supported for Position and {type(other)}")

    def __hash__(self):
        return hash((self.x, self.y))

@dataclass
class Interval:
    """
    Represents an interval.
    """
    start_time: int
    end_time: int

@dataclass
class EntryTimeAndInterval:
    """
    Convenience class to record intervals and their shortest entry times.
    """
    entry_time: int
    interval: Interval

@dataclass
class SIPPNode:
    """
    Represents a node in a SIPP graph.
    """
    position: Position
    t: int
    safe_interval: Interval
    g: float
    h: float
    parent: SIPPNode|None

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, SIPPNode):
            return (self.g + self.h) < (other.g + other.h)
        raise NotImplementedError(f"<= not implemented for SIPPNode and {type(other)}")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SIPPNode):
            return (self.position == other.position) and \
                   (self.safe_interval == other.safe_interval)
        raise NotImplementedError(f"== not implemented for SIPPNode and {type(other)}")

@dataclass
class Trajectory:
    """
    Represents a generic finite trajectory in 2D space.
    Object stays at the last position.
    """
    path_taken: list[Position]

ALL_ACTIONS: list[Position] = [Position(0, 0),
                               Position(1, 0),
                               Position(-1, 0),
                               Position(0, 1),
                               Position(0, -1)]

def walkable(grid: h.BoolArr, pos: Position) -> bool:
    """
    Check if position is walkable.
    """
    # check if inside grid
    if pos.x < 0 or pos.x >= grid.shape[0] or \
       pos.y < 0 or pos.y >= grid.shape[1]:
        return False

    # check if on wall
    if not grid[pos.x, pos.y, h.GridOffsets.NO_WALL]:
        return False

    return True

def get_actions(*,
                grid: h.BoolArr,
                pos: Position) -> list[Position]:
    """
    Given a grid, position and timestep, returns all possible
    actions from that position at that timestep.
    """
    permissible_actions: list[Position] = []

    for action in ALL_ACTIONS:
        new_pos: Position = pos+action

        if not walkable(grid, new_pos):
            continue

        permissible_actions.append(action)

    return permissible_actions

def calculate_heuristic(position: Position, goal: Position) -> int:
    """
    Calculate the heuristic for a given position - Manhattan distance to the goal
    """
    diff = goal - position
    return abs(diff.x) + abs(diff.y)

def get_safe_and_blocked_intervals(trajectories: list[Trajectory],
                                   time_limit: int) -> tuple[dict[Position,
                                                             list[Interval]],
                                                             dict[Position,
                                                                  list[int]]]:
    """
    Returns safe and blocked intervals.
    """
    # TODO: test this method ffs
    blocked_intervals: dict[Position, list[int]] = defaultdict(list)

    for traj in trajectories:
        step: int = 0
        for t, pos in enumerate(traj.path_taken[:-1]):
            # stays on that position for this timestep
            blocked_intervals[pos].append(t)
            step += 1

        # stays on last position until time_limit (fill timeline)
        for j in range(time_limit-step+1):
            blocked_intervals[traj.path_taken[-1]].append(j+step)

    safe_intervals: dict[Position, list[Interval]] = defaultdict(list)

    for pos, blocked_times in blocked_intervals.items():
        blocked_times.sort()
        prev = 0
        res: list[Interval] = []
        for t in blocked_times:
            if t > prev:
                res.append(Interval(prev, t-1))
            prev = t+1

        if prev <= time_limit:
            res.append(Interval(prev, time_limit))

        safe_intervals[pos] = res

    return safe_intervals, blocked_intervals

if __name__ == "__main__":
    trajs = [Trajectory([Position(1, 1),
                         Position(1, 1),
                         Position(1, 1),
                         Position(1, 2),
                         Position(1, 3)])]

    pprint(get_safe_and_blocked_intervals(trajs, 10))
