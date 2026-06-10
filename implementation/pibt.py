"""
Contains class definition of PIBT.
"""

from .mapf_instance import MAPFInstance
from .mapf_utils import (Position,
                         Positions,
                         Map,
                         INVALID_AGENT_ID)
import numpy as np

class PIBT:
    """
    Represents a PIBT solver.
    """
    def __init__(self, *,
                 dim: int,
                 seed: int) -> None:
        self.instance: MAPFInstance
        self.seed: int = seed
        self.rng = np.random.default_rng(seed)
        self.occupied_now: Map = Map(dim, dim, "int")
        self.occupied_next: Map = Map(dim, dim, "int")
        self.consider_hazards: bool = True
        self.agent_idxs: list[int]

    def set_instance(self, instance: MAPFInstance):
        """
        Sets instance of solver.

        Should always be called before reset.

        Args:
            instance (MAPFInstance): New instance.
        """
        self.instance = instance

    def reseed(self, seed: int):
        """
        Reseed PIBT solver.

        Args:
            seed (int): New seed for RNG.
        """
        self.rng = np.random.default_rng(seed)

    def set_hazard_awareness(self, val: bool):
        """
        Change hazard awareness of solver.
        """
        self.consider_hazards = val

    def get_vertex_cost(self,
                            i: int,
                            v: Position,
                            estimation: float) -> float:
        """
        Get cost for a specified agent and vertex.

        Args:
            i (int): Agent index.
            v (Position): Specified vertex.
            estimation (float): Estimation of hazard stuck probability.

        Returns:
            float: Cost value for the vertex.
        """
        # FIRST TECHNIQUE: Hazard-Aware Vertex Priorization
        return (self.instance.all_dist_tables[i].get(v) +
                estimation * self.instance.hazard.on_hazard(v) *
                self.consider_hazards)

    def reset(self) -> None:
        """
        Reset solver attributes. Should be used when starting a new episode.
        """
        self.occupied_now.reset()
        self.occupied_next.reset()
        self.reseed(self.seed)
        self.agent_idxs = list(range(self.instance.num_agents))

    def func_pibt(self, q_to: list[Position|None], i: int) -> bool:
        """
        Main function of the PIBT algorithm.

        See: https://kei18.github.io/pibt2/
        """
        a_pos = self.instance.agents[i].current_pos
        candidates: Positions = [a_pos]
        candidates.extend(pos for pos in self.instance.wall_map.neighbour_table[a_pos]
                          if pos is not None)

        # tie breaker
        self.rng.shuffle(candidates)

        candidates.sort(key=lambda v: self.get_vertex_cost(i, v,
                                                               self
                                                               .instance
                                                               .agents[i]
                                                               .memory
                                                               .estimation))

        for candidate in candidates:
            # avoid vertex collision
            if self.occupied_next[candidate] != INVALID_AGENT_ID:
                continue

            j: int = self.occupied_now[candidate]

            # avoid edge collision
            if j != INVALID_AGENT_ID and q_to[j] == a_pos:
                continue

            q_to[i] = candidate
            self.occupied_next[candidate] = i

            if (j != INVALID_AGENT_ID and
                q_to[j] is None and
                not self.func_pibt(q_to, j)):
                continue

            return True

        # failed to secure node
        q_to[i] = a_pos
        self.occupied_next[a_pos] = i
        return False

    def step(self) -> None:
        """
        Executes one step of the PIBT algorithm.
        """
        q_to: list[Position|None] = []
        hazard_prios: list[float] = [0]*self.instance.num_agents

        # freeze agents if they are on a hazard tile
        # NOTE: we loop again... isn't there anything more efficient?
        for i, agent in enumerate(self.instance.agents):
            stuck = self.instance.hazard.is_stuck(agent.current_pos)

            if stuck:
                q_to.append(agent.current_pos)
                self.occupied_next[agent.current_pos] = i
                agent.memory.observed_stucks += 1
                agent.memory.potential_stucks += 1
            else:
                q_to.append(None)

            # SECOND TECHNIQUE: Hazard-Aware Prioritization
            # increase priority of agents on or near hazards
            on_hazard = self.instance.hazard.on_hazard(agent.current_pos)
            if on_hazard and not stuck:
                agent.memory.potential_stucks += 1

            estimation: float = agent.memory.estimation

            if on_hazard:
                hazard_prios[i] += estimation

            for neighbour in self.instance.wall_map.neighbour_table[agent.current_pos]:
                if neighbour is None:
                    continue

                if self.instance.hazard.on_hazard(neighbour):
                    # more dangerous to be on a hazard,
                    # then near a hazard
                    hazard_prios[i] += estimation / 4

                # THIRD TECHNIQUE: Hazard Parameter Estimation
                # agents remember how often they got stuck and
                # share that information with other agents
                neighbour_index: int = self.occupied_now[neighbour]
                if neighbour_index != INVALID_AGENT_ID:
                    self.instance.agents[neighbour_index] \
                    .memory.update_shared_estimation(i, estimation)

        self.agent_idxs.sort(key=lambda i: self.instance.agents[i].current_priority +
                                           hazard_prios[i] * self.consider_hazards,
                             reverse=True)

        for i in self.agent_idxs:
            if q_to[i] is None:
                self.func_pibt(q_to, i)

        # cleanup and goal checking
        for i, a in enumerate(self.instance.agents):
            self.occupied_now[a.current_pos] = INVALID_AGENT_ID

            next_pos = q_to[i]

            # for type checker
            assert next_pos is not None

            self.occupied_now[next_pos] = i
            self.occupied_next[next_pos] = INVALID_AGENT_ID

            if next_pos != a.goal_pos:
                a.current_priority += 1
            else:
                a.current_priority -= np.floor(a.current_priority)
            action = next_pos - a.current_pos
            self.instance.agents[i].move(action)
            self.instance.path_manager.append_action_to_path(i, action)

        self.instance.progress()
