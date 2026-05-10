"""
Safe interval path planner
    This script implements a safe-interval path planner for a 2D 
    grid with dynamic obstacles. It is faster than SpaceTime A* because it 
    reduces the number of redundant node expansions by pre-computing regions of adjacent
    time steps that are safe ("safe intervals") at each position. 
    This allows the algorithm to skip expanding nodes
    that are in intervals that have already been visited earlier.

    References: 
    https://www.cs.cmu.edu/~maxim/files/sipp_icra11.pdf
    https://github.com/AtsushiSakai/PythonRobotics
"""

from collections import defaultdict
from typing import Any
from .. import header as h
from ..node import Node
from ..grid import Grid
from ..reservation_table import ReservationTable
from ..safe_interval_table import SafeIntervalTable
import heapq
from dataclasses import dataclass
from functools import total_ordering

def plan(*,
         grid: Grid,
         t: int,
         time_limit: int,
         reservations: ReservationTable,
         safe_intervals: SafeIntervalTable,
         start: h.Position,
         goal: h.Position) -> list[SIPPNode]:
    """
    Generate a path for a single agent in a dynamic obstacle enviroment.\n
    Stops after a set amount of iterations.
    NOTE: We return the reversed path (nodes[0] is the goal node, ...).
    """

    open_set: list[SIPPNode] = []
    heapq.heappush(open_set, SIPPNode(position=start,
                                      t=t, # we start at time t
                                      safe_interval=safe_intervals.get_safe_intervals_at(start)[0],
                                      g=0,
                                      c=0,
                                      h=calculate_heuristic(start, goal),
                                      parent=None))

    # store shortest visited time per (position, (interval, time))
    visited_entry_interval: dict[h.Position,
                                 dict[h.Interval, int]] = defaultdict(dict)

    while open_set:
        expanded_node: SIPPNode = heapq.heappop(open_set)

        if expanded_node.t + 1 >= time_limit:
            # skip node that is past time limit
            continue

        if expanded_node.position == goal:
            path: list[SIPPNode] = []
            path_walker: SIPPNode = expanded_node
            while True:
                path.append(path_walker)
                if path_walker.parent is None:
                    break
                path_walker = path_walker.parent
            return path

        # if currently expanded node is present, update entry time if better
        if expanded_node.position in visited_entry_interval and \
           expanded_node.safe_interval in visited_entry_interval[expanded_node.position]:
            visited_entry_interval[expanded_node.position][expanded_node.safe_interval] = \
            min(expanded_node.t,
                visited_entry_interval[expanded_node.position][expanded_node.safe_interval])
        else:
            # if currently expanded node of interval not recorded, record it
            visited_entry_interval[expanded_node.position] \
                                  [expanded_node.safe_interval] = expanded_node.t

        successors = generate_successors(grid=grid,
                                         safe_intervals=safe_intervals,
                                         reservations=reservations,
                                         parent_node=expanded_node,
                                         goal=goal,
                                         visited_entry_interval=visited_entry_interval)

        for succ in successors:
            heapq.heappush(open_set, succ)

    raise ValueError("No path found")

def generate_successors(*,
                        grid: Grid,
                        safe_intervals: SafeIntervalTable,
                        reservations: ReservationTable,
                        parent_node: SIPPNode,
                        goal: h.Position,
                        visited_entry_interval: dict[h.Position,
                                                     dict[h.Interval, int]]) -> list[SIPPNode]:
    """
    Generate successors from a given node.
    """
    successors: list[SIPPNode] = []
    for m in h.DIR_ARR:
        cfg: h.Position = parent_node.position+m

        if grid.contains_wall(cfg):
            continue

        parent_interval = parent_node.safe_interval
        start_t = parent_node.t+1
        end_t = parent_interval.end_time+1

        intervals = safe_intervals.get_safe_intervals_at(cfg)

        for i in intervals:
            if i.start_time > end_t or i.end_time < start_t:
                continue

            # if we have already expanded a node in this interval with a <= starting time, skip
            if cfg in visited_entry_interval and \
               i in visited_entry_interval[cfg] and \
               visited_entry_interval[cfg][i] <= parent_node.t+1:
                continue

            # we know there is a node worth expanding
            # generate successor at the earliest possible time the new interval can be entered
            for arrival_t in range(max(parent_node.t+1,  i.start_time),
                                   min(parent_interval.end_time, i.end_time)+1):

                # arrival_t and arrival_t-1 (departure time) need to be in the interval
                # prevents vertex conflict and following conflict
                if reservations.is_reserved(cfg, arrival_t) or \
                   reservations.is_reserved(cfg, arrival_t-1):
                    continue

                successors.append(SIPPNode(position=cfg,
                                           t=arrival_t,
                                           safe_interval=i,
                                           g=parent_node.g+1,
                                           h=calculate_heuristic(cfg, goal),
                                           c=0,
                                           parent=parent_node))

                # break because all t's after this will make nodes with a higher cost,
                # the same heuristic, and are in the same interval
                break
    return successors

@dataclass(frozen=True)
@total_ordering
class SIPPNode(Node):
    """
    Represents a node in a SIPP graph.
    """
    safe_interval: h.Interval
    parent: SIPPNode|None

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SIPPNode):
            return (self.position == other.position) and \
                   (self.safe_interval == other.safe_interval)
        raise NotImplementedError(f"== not implemented for SIPPNode and {type(other)}")

def calculate_heuristic(position: h.Position, goal: h.Position) -> float:
    """
    Calculate the heuristic for a given position - Manhattan distance to the goal
    """
    return (goal-position).length()
