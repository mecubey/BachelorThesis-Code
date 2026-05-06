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
from .. import header as hd
import heapq
from dataclasses import dataclass
from typing import Any
import numpy as np
from typing import Callable


def plan(*,
         start: hd.PositionT,
         goal: hd.PositionT,
         trajs: list[list[hd.PositionT]],
         time_limit: int,
         no_wall: Callable[[hd.PositionT], bool],
         inside_grid: Callable[[hd.PositionT], bool]) -> list[SIPPNode]:
    """
    Generate a path for a single agent in a dynamic obstacle enviroment.\n
    Stops after a set amount of iterations.
    """
    safe_intervals, blocked_intervals = get_safe_and_blocked_intervals(trajs, time_limit)

    start_pos_tuple = (start[0], start[1])
    safe_intervals[start_pos_tuple].append((0, time_limit))

    open_set: list[SIPPNode] = []
    heapq.heappush(open_set, SIPPNode(position=start,
                                      t=0,
                                      safe_interval=safe_intervals[start_pos_tuple][0],
                                      g=hd.DTYPE_FLOAT(0),
                                      h=calculate_heuristic(start, goal),
                                      parent=None))

    # store shortest visited time per (position, interval)
    visited_entry_interval: dict[hd.PositionTupleT,
                                 dict[hd.IntervalT, int]] = defaultdict(dict)

    while open_set:
        expanded_node: SIPPNode = heapq.heappop(open_set)

        if expanded_node.t + 1 >= time_limit:
            # skip node that is past time limit
            continue

        if (expanded_node.position == goal).all():
            path: list[SIPPNode] = []
            path_walker: SIPPNode = expanded_node
            while True:
                path.append(path_walker)
                if path_walker.parent is None:
                    break
                path_walker = path_walker.parent

            # reverse path so it goes start -> goal
            path.reverse()
            return path

        # if currently expanded node is present, update entry time if better
        node_pos_tuple = (expanded_node.position[0],
                          expanded_node.position[1])
        if node_pos_tuple in visited_entry_interval and \
           expanded_node.safe_interval in visited_entry_interval[node_pos_tuple]:
            visited_entry_interval[node_pos_tuple][expanded_node.safe_interval] = \
            min(expanded_node.t,
                visited_entry_interval[node_pos_tuple][expanded_node.safe_interval])
        else:
            # if currently expanded node of interval not recorded, record it
            visited_entry_interval[node_pos_tuple][expanded_node.safe_interval] = expanded_node.t

        successors = generate_successors(parent_node=expanded_node,
                                         safe_intervals=safe_intervals,
                                         blocked_intervals=blocked_intervals,
                                         time_limit=time_limit,
                                         goal=goal,
                                         visited_entry_interval=visited_entry_interval,
                                         no_wall=no_wall,
                                         inside_grid=inside_grid)

        for succ in successors:
            heapq.heappush(open_set, succ)

    raise ValueError("No path found")

def generate_successors(*,
                        parent_node: SIPPNode,
                        safe_intervals: dict[hd.PositionTupleT, list[hd.IntervalT]],
                        blocked_intervals: dict[hd.PositionTupleT, set[int]],
                        time_limit: int,
                        goal: hd.PositionT,
                        visited_entry_interval: dict[hd.PositionTupleT,
                                                     dict[hd.IntervalT, int]],
                        no_wall: Callable[[hd.PositionT], bool],
                        inside_grid: Callable[[hd.PositionT], bool]) -> list[SIPPNode]:
    """
    Generate successors from a given node.
    """
    successors: list[SIPPNode] = []
    for m in hd.DIR_ARR:
        cfg: hd.PositionT = parent_node.position+m

        if not inside_grid(cfg) or not no_wall(cfg):
            continue

        parent_interval = parent_node.safe_interval
        start_t = parent_node.t+1
        end_t = parent_interval[1]+1

        cfg_pos_tuple = (cfg[0], cfg[1])

        intervals = safe_intervals.get(cfg_pos_tuple, [(0, time_limit)])

        for i in intervals:
            if i[0] > end_t or i[1] < start_t:
                continue

            # if we have already expanded a node in this interval with a <= starting time, skip
            if cfg_pos_tuple in visited_entry_interval and \
               i in visited_entry_interval[cfg_pos_tuple] and \
               visited_entry_interval[cfg_pos_tuple][i] <= parent_node.t+1:
                continue

            # we know there is a node worth expanding
            # generate successor at the earliest possible time the new interval can be entered
            for arrival_t in range(max(parent_node.t+1,  i[0]),
                                   min(parent_interval[1], i[1])+1):

                # arrival_t and arrival_t-1 (departure time) need to be in the interval
                # prevents vertex conflict and following conflict
                if arrival_t in blocked_intervals[cfg_pos_tuple] or \
                   arrival_t-1 in blocked_intervals[cfg_pos_tuple]:
                    continue

                successors.append(SIPPNode(position=cfg,
                                           t=arrival_t,
                                           safe_interval=i,
                                           g=parent_node.g+1,
                                           h=calculate_heuristic(cfg, goal),
                                           parent=parent_node))

                # break because all t's after this will make nodes with a higher cost,
                # the same heuristic, and are in the same interval
                break
    return successors

@dataclass
class SIPPNode:
    """
    Represents a node in a SIPP graph.
    """
    position: hd.PositionT
    t: int
    safe_interval: hd.IntervalT
    h: hd.DTYPE_FLOAT
    g: hd.DTYPE_FLOAT
    parent: SIPPNode|None

    def __lt__(self, other: Any) -> hd.DTYPE_BOOL:
        if isinstance(other, SIPPNode):
            return (self.g + self.h) < (other.g + other.h)
        raise NotImplementedError(f"<= not implemented for SIPPNode and {type(other)}")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SIPPNode):
            return (self.position == other.position) and \
                   (self.safe_interval == other.safe_interval)
        raise NotImplementedError(f"== not implemented for SIPPNode and {type(other)}")

def calculate_heuristic(position: hd.PositionT, goal: hd.PositionT) -> hd.DTYPE_FLOAT:
    """
    Calculate the heuristic for a given position - Manhattan distance to the goal
    """
    return np.sum(np.abs(goal-position))

def get_safe_and_blocked_intervals(trajectories: list[list[hd.PositionT]],
                                   time_limit: int) -> tuple[dict[hd.PositionTupleT,
                                                                  list[hd.IntervalT]],
                                                             dict[hd.PositionTupleT,
                                                                  set[int]]]:
    """
    Returns dictionary with position as key and (safe intervals, blocked timesteps) as values.
    """
    blocked_intervals: dict[hd.PositionTupleT, set[int]] = defaultdict(set)

    for traj in trajectories:
        step: int = 0
        for t, pos in enumerate(traj):
            # stays on that position for this timestep
            blocked_intervals[(pos[0], pos[1])].add(t)
            step += 1

        # stays on last position until time_limit (fill timeline)
        blocked_intervals[(traj[-1][0], traj[-1][1])].update([j+step
                                                              for j in range(time_limit-step+1)])

    safe_intervals: dict[hd.PositionTupleT, list[hd.IntervalT]] = defaultdict(list)

    for pos, blocked_times in blocked_intervals.items():
        prev = 0
        for t in sorted(blocked_times):
            if t > prev:
                safe_intervals[(pos[0], pos[1])].append((prev, t-1))
            prev = t+1

        if prev <= time_limit:
            safe_intervals[(pos[0], pos[1])].append((prev, time_limit))

    return safe_intervals, blocked_intervals
