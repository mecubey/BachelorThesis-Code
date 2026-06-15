"""
Contains class definition of MAPFInstance and MAPFInstanceManager.
"""

from .hazard import (HazardConfig,
                     Hazard)
from .wall_map import WallMap
from .scene import Scene
from .agent import (Agent,
                    Agents)
from .mapf_utils import (Position,
                         Positions,
                         Map,
                         FRONT_WEIGHT_CUTOFF,
                         FRONT_WEIGHT_DECREASE)
from .dist_table import DistTable
from .path_manager import PathManager

class MAPFInstance:
    """
    Represents a single MAPF instance.
    """
    def __init__(self, *,
                 max_timestep: int,
                 hazard_config: HazardConfig,
                 hazard_seed: int,
                 wall_map: WallMap,
                 scene: Scene) -> None:
        self.wall_map = wall_map
        self.hazard = Hazard(self.wall_map, hazard_config, hazard_seed)
        self.timestep: int = 0
        self.max_timestep: int = max_timestep
        self.agents: Agents = []
        self.num_agents: int = 0
        self.scene = scene
        self.path_manager = PathManager(scene.max_num_agents)

    def change_scene(self, scene: Scene):
        """
        Change scene.

        Args:
            scene (Scene): New scene.
        """
        self.scene = scene

    def move_all_agents(self, actions: Positions):
        """
        Move all agents according to actions.

        Args:
            actions (Positions): List of next actions.
        """
        for i, agent in enumerate(self.agents):
            agent.move(actions[i])

    def change_hazard_config(self, new_config: HazardConfig) -> None:
        """
        Change hazard config of this instance.

        Args:
            new_config (HazardConfig): New config.
        """
        self.hazard.config = new_config

    def finished(self) -> bool:
        """
        Check whether the instance has reached a terminal state.

        An instance is considered finished if all agents reached their goals
        or the maximum allowed timestep has been reached.

        Returns:
            bool: True if the instance is finished, False otherwise.
        """
        return self.succeeded() or self.max_timestep_reached()

    def calc_cost_map(self) -> Map:
        """
        Calculates the hazard cost map.
        """
        cost_map: Map = Map(self.wall_map.width,
                            self.wall_map.height,
                            "float")

        visited: set[Position] = set()
        front: set[Position] = set()

        # we first set the hazard tiles we can see
        for pos in self.hazard.occupied_tiles:
            visited.add(pos)
            for u in self.wall_map.neighbour_table[pos]:
                if u is None or u in self.hazard.occupied_tiles:
                    continue
                front.add(u)
            cost_map[pos] = 1

        front_weight: float = 1
        front_iter: int = 0

        # then, we calculate costs for each cell
        while front_iter < FRONT_WEIGHT_CUTOFF and front:
            next_front: set[Position] = set()
            looked_at_front: set[Position] = set()

            while front:
                pos = front.pop()

                no_spread_prob: float = 1
                for i, u in enumerate(self.wall_map.neighbour_table[pos]):
                    if u is None or u in front or u in looked_at_front:
                        continue

                    if u not in visited:
                        next_front.add(u)
                        continue

                    no_spread_prob *= (1-cost_map[u]*self.hazard.config.dir_spread_probs[i-2])
                cost_map[pos] = front_weight * (1 - no_spread_prob)

                looked_at_front.add(pos)

            visited.update(looked_at_front)
            front_iter += 1
            front_weight = front_weight*(1-FRONT_WEIGHT_DECREASE)
            front = next_front

        return cost_map

    def hazard_step(self) -> None:
        """
        Advance the hazard simulation by one step. 
        """
        self.hazard.step()

        for agent in self.agents:
            agent.decay_freeze()
            if not self.hazard.on_hazard(agent.current_pos):
                agent.decay_dmg()
                continue

            # now we know agent is on hazard tile
            agent.increase_damage()

            if self.hazard.is_stuck(agent.current_pos):
                agent.freeze()

    def get_episode_progress(self) -> float:
        """
        Return episode progress.

        Returns:
            float: Episode progress denoted as a percentage.
        """
        return self.timestep / self.max_timestep

    def progress(self) -> None:
        """
        Progress the episode by one timestep.
        """
        self.timestep += 1

    def full_reset(self) -> None:
        """
        Reset the instance to its initial state and remove all agents.
        """
        self.agents.clear()
        self.num_agents = 0
        self.reset()

    def reset(self) -> None:
        """
        Reset the instance to its initial state.
        """
        for agent in self.agents:
            agent.reset()
        self.hazard.reset()
        self.hazard.reseed(self.hazard.seed)
        self.path_manager.reset()
        self.timestep = 0

    def add_agent(self) -> None:
        """
        Add the next agent to the instance.

        The agent's start position, goal position, distance table, and initial
        priority are taken from the preloaded scenario data.
        """
        new_start_pos: Position = self.scene.all_start_positions[self.num_agents]
        new_goal_pos: Position = self.scene.all_goal_positions[self.num_agents]
        new_dist_table: DistTable = self.scene.all_dist_tables[self.num_agents]
        new_priority: float = new_dist_table.get(new_start_pos) / self.wall_map.width
        self.agents.append(Agent(i=self.num_agents,
                                 priority=new_priority,
                                 start_pos=new_start_pos,
                                 goal_pos=new_goal_pos,
                                 hazard=self.hazard))
        self.num_agents += 1

    def succeeded(self) -> bool:
        """
        Check whether all agents have reached their goals.

        Returns:
            bool: True if every agent is on its goal position, False otherwise.
        """
        num_agents_on_goal: int = 0
        for agent in self.agents:
            num_agents_on_goal += agent.on_goal()
        return num_agents_on_goal == self.num_agents

    def any_collisions(self) -> bool:
        """
        Check for collisions.

        Returns:
            bool: True if any agents are colliding, False otherwise.
        """
        for agent_i in self.agents:
            for agent_j in self.agents:
                if agent_i.id == agent_j.id:
                    continue
                if agent_i.current_pos == agent_j.current_pos:
                    return True
        return False

    def max_timestep_reached(self) -> bool:
        """
        Check whether the maximum allowed timestep has been reached.

        Returns:
            bool: True if the current timestep is greater than or equal to the
            maximum timestep, False otherwise.
        """
        return self.timestep >= self.max_timestep
