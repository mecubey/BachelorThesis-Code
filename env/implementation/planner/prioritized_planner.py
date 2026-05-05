"""
Priority based planner using SIPP.
"""

from .sipp import plan
from .misc import Position, SIPPNode, Trajectory
from .. import header as h
from ..agent import EnvAgent
from pprint import pprint
import numpy as np

class PrioritizedPlanner():
    """
    Contains plan() method of the prioritized planner.
    """
    @staticmethod
    def plan(*,
             grid: h.BoolArr,
             remaining_time_limit: int,
             start_positions: h.IntArr,
             agents: list[EnvAgent],
             rng: np.random.Generator) -> dict[str, list[h.Action]]:
        """
        Get plans for multiple agents.
        """
        agent_idx: list[int] = list(range(len(agents)))

        # random priorities
        rng.shuffle(agent_idx)

        # reserve initial starting positions as trajectories
        reserved_initial_positions: list[Trajectory] = [Trajectory([Position(pos[0], pos[1])])
                                                        for pos in start_positions]

        # reserve goal positions as trajectories
        reserved_goal_positions: list[Trajectory] = [Trajectory([Position(a.goal_pos[0], a.goal_pos[1])])
                                                     for a in agents]

        paths: list[Trajectory] = []

        # format: {"agent_0": path_of_agent_0, "agent_1": path_of_agent_1, ...}
        movements_taken_on_path: dict[str, list[h.Action]] = {}

        for i in agent_idx:
            # TODO: reserve goal positions for all agents

            # remove initial position of this agent
            reserved_initial_positions.remove(Trajectory([Position(start_positions[i][0],
                                                                   start_positions[i][1])]))

            # remove goal position of this agent
            reserved_goal_positions.remove(Trajectory([Position(agents[i].goal_pos[0],
                                                                agents[i].goal_pos[1])]))

            # get path from planner for this agent
            nodes: list[SIPPNode] = plan(grid=grid,
                                         start=Position(start_positions[i][0],
                                                        start_positions[i][1]),
                                         goal=Position(agents[i].goal_pos[0],
                                                       agents[i].goal_pos[1]),
                                         trajs=paths+
                                               reserved_initial_positions+
                                               reserved_goal_positions,
                                         time_limit=remaining_time_limit)

            # get the actual movements taken for each timestep
            movements_taken: list[h.Action] = []

            # get the position the agent is in for each timestep
            node_positions: list[Position] = []

            for j in range(len(nodes)-1):
                for _ in range((nodes[j+1].t-nodes[j].t)-1):
                    node_positions.append(nodes[j].position)
                    movements_taken.append(h.Action.DO_NOTHING)
                node_positions.append(nodes[j].position)
                direction: Position = nodes[j+1].position-nodes[j].position
                movements_taken.append(h.dir_to_act(direction.as_ndarray()))
            node_positions.append(nodes[-1].position)

            movements_taken_on_path[f"agent_{i}"] = movements_taken
            paths.append(Trajectory(node_positions))

        return movements_taken_on_path
