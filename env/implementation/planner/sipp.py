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

from .misc import (Interval,
                    Position,
                    Trajectory,
                    SIPPNode,
                    EntryTimeAndInterval,
                    get_safe_and_blocked_intervals,
                    calculate_heuristic,
                    get_actions)
from collections import defaultdict
from .. import header as h
import heapq

def plan(*,
         grid: h.BoolArr,
         start: Position,
         goal: Position,
         trajs: list[Trajectory],
         time_limit: int) -> list[SIPPNode]:
    """
    Generate a path for a single agent in a dynamic obstacle enviroment.\n
    Stops after a set amount of iterations.
    """
    safe_intervals, blocked_intervals = get_safe_and_blocked_intervals(trajs, time_limit)

    if not start in safe_intervals.keys():
        safe_intervals[start].append(Interval(0, time_limit))

    open_set: list[SIPPNode] = []
    heapq.heappush(open_set, SIPPNode(position=start,
                                      t=0,
                                      safe_interval=safe_intervals[start][0],
                                      g=0,
                                      h=calculate_heuristic(start, goal),
                                      parent=None))

    # store shortest visited time per interval per position
    visited_entry_interval: dict[Position, list[EntryTimeAndInterval]] = defaultdict(list)

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

            # reverse path so it goes start -> goal
            path.reverse()
            return path

        # record currently expanded node
        is_present: bool = False

        # if currently expanded node is present, update entry time if better
        for entry in visited_entry_interval[expanded_node.position]:
            if entry.interval == expanded_node.safe_interval:
                entry.entry_time = min(entry.entry_time, expanded_node.t)
                is_present = True
                break

        # if currently expanded node of interval not recorded, record it
        if not is_present:
            visited_entry_interval[expanded_node.position] \
                                   .append(EntryTimeAndInterval(expanded_node.t,
                                                                expanded_node.safe_interval))

        successors = generate_successors(grid=grid,
                                         parent_node=expanded_node,
                                         safe_intervals=safe_intervals,
                                         blocked_intervals=blocked_intervals,
                                         time_limit=time_limit,
                                         goal=goal,
                                         visited_entry_interval=visited_entry_interval)

        for succ in successors:
            heapq.heappush(open_set, succ)

    raise ValueError("No path found")

def generate_successors(*,
                        grid: h.BoolArr,
                        parent_node: SIPPNode,
                        safe_intervals: dict[Position, list[Interval]],
                        blocked_intervals: dict[Position, list[int]],
                        time_limit: int,
                        goal: Position,
                        visited_entry_interval: dict[Position,
                                                     list[EntryTimeAndInterval]]) -> list[SIPPNode]:
    """
    Generate successors from a given node.
    """
    successors: list[SIPPNode] = []
    for m in get_actions(grid=grid, pos=parent_node.position):
        cfg = parent_node.position+m

        parent_interval = parent_node.safe_interval
        start_t = parent_node.t+1
        end_t = parent_interval.end_time+1

        intervals = safe_intervals.get(cfg, [Interval(0, time_limit)])

        for i in intervals:
            if i.start_time > end_t or i.end_time < start_t:
                continue

            # if we have already expanded a node in this interval with a <= starting time, skip
            better_node_expanded = False
            for visited in visited_entry_interval[cfg]:
                if i == visited.interval and visited.entry_time <= parent_node.t + 1:
                    better_node_expanded = True
                    break
            if better_node_expanded:
                continue

            # we know there is a node worth expanding
            # generate successor at the earliest possible time the new interval can be entered
            for arrival_t in range(max(parent_node.t+1,  i.start_time),
                                   min(parent_interval.end_time, i.end_time)+1):

                # arrival_t and arrival_t-1 (departure time) need to be in the interval
                # prevents vertex conflict and following conflict
                if arrival_t in blocked_intervals[cfg] or \
                   arrival_t-1 in blocked_intervals[cfg]:
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
