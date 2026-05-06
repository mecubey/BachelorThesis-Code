"""
Priority based planner using SIPP.
"""

from .sipp import plan
from .. import header as h
from .sipp import SIPPNode
from typing import Callable
import numpy as np

class PrioritizedPlanner():
    """
    Prioritized planner utilizing SIPP for multi agent path planning.
    """
    @staticmethod
    def plan(*,
             remaining_time_limit: int,
             start_positions: h.IntArr,
             goal_positions: h.IntArr,
             trajectories: list[list[h.PositionT]],
             rng: np.random.Generator,
             no_wall: Callable[[h.PositionT], bool],
             inside_grid: Callable[[h.PositionT], bool]) -> dict[str, list[h.Action]]:
        """
        Get plans for multiple agents.
        """
        agent_idx: list[int] = list(range(len(start_positions)))

        goal_mask = np.ones(len(goal_positions), h.DTYPE_BOOL)
        start_pos_mask = np.ones(len(goal_positions), h.DTYPE_BOOL)

        paths: list[list[h.PositionT]] = []
        all_movements_taken: dict[str, list[h.Action]] = {}

        # for now, random priorities
        for i in h.randomly(agent_idx, rng):
            # remove initial position of this agent
            start_pos_mask[i] = False

            # remove goal position of this agent
            goal_mask[i] = False

            # get path from planner for this agent
            nodes = plan(start=start_positions[i],
                         goal=goal_positions[i],
                         trajs=paths+
                               np.expand_dims(start_positions[start_pos_mask], axis=1).tolist()+
                               np.expand_dims(goal_positions[goal_mask], axis=1).tolist()+
                               trajectories,
                         time_limit=remaining_time_limit,
                         no_wall=no_wall,
                         inside_grid=inside_grid)

            all_node_positions, movements_taken = PrioritizedPlanner \
                                                  .get_path_and_actions_from_nodes(nodes)

            paths.append(all_node_positions)
            all_movements_taken[f"agent_{i}"] = movements_taken

        return all_movements_taken

    @staticmethod
    def get_path_and_actions_from_nodes(nodes: list[SIPPNode]) -> tuple[list[h.PositionT],
                                                                        list[h.Action]]:
        """
        Given a list of SIPP nodes, computes the positions occupied and
        the actions taken on that path.
        """
        # get the actual movements taken for each timestep
        movements_taken: list[h.Action] = []

        # get the position the agent is in for each timestep
        node_positions: list[h.PositionT] = []

        for j in range(len(nodes)-1):
            for _ in range((nodes[j+1].t-nodes[j].t)-1):
                node_positions.append(nodes[j].position)
                movements_taken.append(h.Action.DO_NOTHING)
            node_positions.append(nodes[j].position)
            direction: h.PositionT = nodes[j+1].position-nodes[j].position
            movements_taken.append(h.dir_to_act(direction))
        node_positions.append(nodes[-1].position)

        return node_positions, movements_taken
