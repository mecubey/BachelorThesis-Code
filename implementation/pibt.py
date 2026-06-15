"""
Contains class definition of PIBT.
"""

import numpy as np
from .mapf_instance import MAPFInstance
from .mapf_utils import (Position,
                         Positions,
                         Map,
                         INVALID_AGENT_ID)

class PIBT:
    """
    Represents a PIBT solver.
    """
    def __init__(self, *,
                 width: int,
                 height: int,
                 seed: int) -> None:
        self.instance: MAPFInstance
        self.seed: int = seed
        self.rng = np.random.default_rng(seed)
        self.occupied_now: Map = Map(width, height, "int")
        self.occupied_next: Map = Map(width, height, "int")
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

    def reset(self) -> None:
        """
        Reset solver attributes. Should be used when starting a new episode.
        """
        self.occupied_now.reset()
        self.occupied_next.reset()
        self.reseed(self.seed)
        self.agent_idxs = list(range(self.instance.num_agents))

    def func_pibt(self,
                  q_to: list[Position | None],
                  i: int,
                  cost_map: Map) -> bool:
        """
        Main function of PIBT.
        """
        current = self.instance.agents[i].current_pos

        candidates = [current]
        candidates.extend(
            p
            for p in self.instance.wall_map.neighbour_table[current]
            if p is not None
        )

        self.rng.shuffle(candidates)

        candidates.sort(key=lambda u: (self.instance.scene.all_dist_tables[i].get(u) +
                                       (
                                           cost_map[u] *
                                           self.instance.hazard.calculate_increased_dmg(
                                               self.instance.agents[i].damage
                                           ) *
                                           self.instance.hazard.config.base_stuck_prob
                                       ) * self.consider_hazards))

        for v in candidates:
            if self.occupied_next[v] != INVALID_AGENT_ID:
                continue

            j = self.occupied_now[v]

            if (
                j != INVALID_AGENT_ID
                and q_to[j] == current
            ):
                continue

            q_to[i] = v
            self.occupied_next[v] = i

            if (
                j != INVALID_AGENT_ID
                and q_to[j] is None
                and not self.func_pibt(q_to, j, cost_map)
            ):
                continue

            return True

        q_to[i] = current
        self.occupied_next[current] = i

        return False

    def step(self) -> Positions:
        """
        Step through PIBT algorithm.
        """
        next_actions: Positions = []

        q_to: list[Position | None] = [None] * self.instance.num_agents

        hazard_prios: list[float] = [0] * self.instance.num_agents

        cost_map = self.instance.calc_cost_map()

        for i, agent in enumerate(self.instance.agents):
            self.occupied_now[agent.current_pos] = i

            if agent.frozen():
                q_to[i] = agent.current_pos
                self.occupied_next[agent.current_pos] = i

            hazard_prios[i] = (cost_map[agent.current_pos] *
                               self.instance.hazard.calculate_increased_dmg(
                                   self.instance.agents[i].damage
                               ) *
                               self.instance.hazard.config.base_stuck_prob)

        self.agent_idxs.sort(
            key=lambda i: (self.instance.agents[i].current_priority +
                           hazard_prios[i] * self.consider_hazards),
            reverse=True
        )

        for i in self.agent_idxs:
            if q_to[i] is None:
                self.func_pibt(q_to, i, cost_map)

        for i, agent in enumerate(self.instance.agents):
            next_pos = q_to[i]
            assert next_pos is not None

            self.occupied_now[agent.current_pos] = INVALID_AGENT_ID
            self.occupied_next[next_pos] = INVALID_AGENT_ID

            action = next_pos - agent.current_pos

            next_actions.append(action)

            self.occupied_now[next_pos] = i

            self.instance.path_manager.append_action_to_path(i, action)

            if next_pos != agent.goal_pos:
                agent.current_priority += 1
            else:
                agent.current_priority -= np.floor(agent.current_priority)

        return next_actions
