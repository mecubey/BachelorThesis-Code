"""
Contains node related classes.
"""

from . import header as hd
from typing import Sequence
from .safe_interval_table import SafeIntervalTable
from dataclasses import dataclass
from functools import total_ordering

@dataclass(frozen=True)
@total_ordering
class Node:
    """
    Node class with position, t, g, h, c and parent node.
    """
    position: hd.Position
    t: int
    g: float
    h: float
    c: float
    parent: Node|None

    def __lt__(self, other: Node):
        """
        This is what is used to drive node expansion. 
        The node with the lowest value is expanded next.

        This comparison prioritizes the node with the lowest 
        g_cost + h_cost + c_cost (hazard damage).
        """
        return (self.g+self.h+self.c) < (other.g+other.g+other.c)

    def __eq__(self, other: object):
        if not isinstance(other, Node):
            raise NotImplementedError(f"Cannot compare Node with object of type: {type(other)}")
        return self.position == other.position

class NodePath:
    """
    Represents a path of nodes.
    Contains occupied positions and actions taken at specified timesteps.
    """

    def __init__(self,
                 path: Sequence[Node],
                 positions_at_time: dict[int, hd.Position],
                 actions_taken_at_time: dict[int, hd.Action]):
        self.path: Sequence[Node] = path
        self.positions_at_time: dict[int, hd.Position] = positions_at_time
        self.actions_taken_at_time: dict[int, hd.Action] = actions_taken_at_time

    @classmethod
    def init_and_build_safe_intervals(cls, path: Sequence[Node], safe_intervals: SafeIntervalTable):
        """
        Initialize the NodePath while reserving positions.
        """
        positions_at_time: dict[int, hd.Position] = {}
        actions_taken_at_time: dict[int, hd.Action] = {}

        for j in reversed(range(1, len(path))):
            for t in range(path[j].t, path[j-1].t-1):
                positions_at_time[t] = path[j].position
                actions_taken_at_time[t] = hd.Action.DO_NOTHING
                safe_intervals.reservations.reserve_position(path[j].position, t)
            positions_at_time[path[j-1].t-1] = path[j].position
            safe_intervals.reservations.reserve_position(path[j].position, path[j-1].t-1)
            safe_intervals.build_safe_intervals_at(path[j].position)
            direction: hd.Position = path[j-1].position-path[j].position
            actions_taken_at_time[path[j-1].t-1] = hd.DIR_TO_ACT[direction]
        positions_at_time[path[0].t] = path[0].position
        safe_intervals.reservations.reserve_interval(path[0].position,
                                                     # reserve from t until end of episode
                                                     hd.Interval(path[0].t,
                                                                 safe_intervals.max_timestep))
        safe_intervals.build_safe_intervals_at(path[0].position)
        return cls(path, positions_at_time, actions_taken_at_time)

    def get_position_at(self, t: int) -> hd.Position:
        """
        Get the position of the path at a given time
        """
        return self.positions_at_time.get(t, hd.INVALID_POSITION)

    def get_action_at(self, t: int) -> hd.Action:
        """
        Get the action taken at a specific timestep.
        
        Will be "do nothing" if t is greater than the last occupied timestep.
        """
        return self.actions_taken_at_time.get(t, hd.Action.DO_NOTHING)
