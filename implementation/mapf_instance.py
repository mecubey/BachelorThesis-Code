"""
Contains class definition of MAPFInstance and MAPFInstanceManager.
"""

from .memory import Memory
from .hazard import (HazardConfig,
                     Hazard)
from .wall_map import WallMap
from .mapf_utils import (Position,
                         Positions,
                         Agent,
                         Agents,
                         MAX_NUM_INSTANCES,
                         GLOBAL_HAZARD_SEED,
                         get_scenario_path)
from .dist_table import DistTable
from .path_manager import PathManager
from pathlib import Path
from copy import deepcopy

class MAPFInstanceManager:
    """
    Represents a manager class for multiple MAPF instances.
    """
    def __init__(self, *,
                 max_timestep: int,
                 hazard_config: str,
                 wall_map: WallMap,
                 even_or_random: str) -> None:
        self.wall_map: WallMap = wall_map
        self.instances: list[MAPFInstance] = []
        self.scene_names: list[str] = []

        for i in range(1, MAX_NUM_INSTANCES+1): # from 1 ... 25
            self.scene_names.append(f"{wall_map.name}-{even_or_random}-{i}")
            self.instances.append(MAPFInstance(max_timestep=max_timestep,
                                               hazard_config=HazardConfig
                                                             .from_config(hazard_config),
                                               hazard_seed=GLOBAL_HAZARD_SEED,
                                               wall_map=self.wall_map,

                                               # from 0 ... 24
                                               scenario_name=self.scene_names[i-1]))

    @property
    def hazard_name(self):
        """
        Get hazard name.
        """
        return self.instances[0].hazard.config.name

    @property
    def map_name(self):
        """
        Get map name.
        """
        return self.wall_map.name

    def full_reset_all(self) -> None:
        """
        Fully reset all managed instances.
        """
        for inst in self.instances:
            inst.full_reset()

    def change_hazard_configs(self, new_config: str) -> None:
        """
        Change hazard configs of all managed instances.

        Args:
            new_config (HazardConfig): New config.
        """
        for inst in self.instances:
            inst.change_hazard_config(HazardConfig.from_config(new_config))

    def get_scene_name(self, i: int) -> str:
        """
        Get the name of the specified scene.

        Args:
            i (int): Index of the scene.

        Returns:
            str: Name of the scene.
        """
        return self.scene_names[i]

    def get_max_timestep(self) -> int:
        """
        Get the maximum timestep configured for the managed instances.

        Returns:
            int: Maximum timestep.
        """
        return self.instances[0].max_timestep

    def get_max_num_agents(self) -> int:
        """
        Get the maximum number of agents supported by the managed instances.

        Returns:
            int: Maximum number of agents.
        """
        return self.instances[0].max_num_agents

    def deepcopy_instance(self, i: int) -> MAPFInstance:
        """
        Create a deep copy of the specified instance.

        Args:
            i (int): Index of the instance.

        Returns:
            MAPFInstance: Deep copy of the selected instance.
        """
        return deepcopy(self.instances[i])

class MAPFInstance:
    """
    Represents a single MAPF instance.
    """
    def __init__(self, *,
                 max_timestep: int,
                 hazard_config: HazardConfig,
                 hazard_seed: int,
                 wall_map: WallMap,
                 scenario_name: str) -> None:
        self.wall_map = wall_map
        self.hazard = Hazard(self.wall_map, hazard_config, hazard_seed)
        self.timestep: int = 0
        self.max_timestep: int = max_timestep
        self.agents: Agents = []
        self.num_agents: int = 0

        # these should never change for a single MAPF instance
        self.all_start_positions: Positions = []
        self.all_goal_positions: Positions = []
        self.all_dist_tables: list[DistTable] = []
        self.all_optimal_path_lenghts: list[float] = []
        self.max_num_agents: int = 0

        path_to_scene = Path(get_scenario_path(scenario_name))

        # read scenario file
        with path_to_scene.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("version"):
                    continue

                parts = line.split()

                start = Position(int(parts[5]), int(parts[4]))
                goal = Position(int(parts[7]), int(parts[6]))

                self.all_start_positions.append(start)
                self.all_goal_positions.append(goal)
                table = DistTable(self.wall_map, goal)
                self.all_dist_tables.append(table)
                self.all_optimal_path_lenghts.append(table.get(start))
        self.max_num_agents = len(self.all_start_positions)
        self.path_manager = PathManager(self.max_num_agents)

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

    def hazard_step(self) -> None:
        """
        Advance the hazard simulation by one step.

        If no hazard is currently active, an attempt is made to spawn a new
        hazard. Otherwise, the existing hazard is spread. Once the configured
        spread limit is reached, the hazard is reset.
        """
        if self.hazard.empty() and self.hazard.spawn():
            return

        self.hazard.spread()

        if self.hazard.done():
            self.hazard.reset()

    def calc_optimal_makespan(self, for_n_agents: int) -> float:
        """
        Calculate the optimal makespan for the first n agents.

        The makespan is defined as the maximum optimal path length among the
        specified agents.

        Args:
            for_n_agents (int): Number of agents to consider.

        Returns:
            float: Optimal makespan.
        """
        return max(self.all_optimal_path_lenghts[:for_n_agents])

    def calc_optimal_soc(self, for_n_agents: int) -> float:
        """
        Calculate the optimal sum of costs for the first n agents.

        Args:
            for_n_agents (int): Number of agents to consider.

        Returns:
            float: Sum of optimal path lengths.
        """
        soc: float = 0
        for i in range(for_n_agents):
            soc += self.all_optimal_path_lenghts[i]
        return soc

    def episode_progress(self) -> float:
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
        Should be called when changing configs.

        All agents are removed and hazards are reset and the timestep counter is set to zero.
        """
        self.agents.clear()
        self.num_agents = 0
        self.hazard.reset()
        self.path_manager.reset()
        self.timestep = 0

    def reset(self) -> None:
        """
        Reset the instance to its initial state. Should be called
        when starting a new episode.

        All agents and hazards are reset and the timestep counter is set to zero.
        """
        for agent in self.agents:
            agent.reset()
        self.hazard.reset()
        self.path_manager.reset()
        self.timestep = 0

    def add_agent(self) -> None:
        """
        Add the next agent to the instance.

        The agent's start position, goal position, distance table, and initial
        priority are taken from the preloaded scenario data.
        """
        new_start_pos: Position = self.all_start_positions[self.num_agents]
        new_goal_pos: Position = self.all_goal_positions[self.num_agents]
        new_dist_table: DistTable = self.all_dist_tables[self.num_agents]
        new_priority: float = new_dist_table.get(new_start_pos) /self.wall_map.width
        self.agents.append(Agent(id=self.num_agents,
                                 memory=Memory(),
                                 initial_priority=new_priority,
                                 current_priority=new_priority,
                                 initial_pos=new_start_pos,
                                 current_pos=new_start_pos.deepcopy(),
                                 goal_pos=new_goal_pos))
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

    def max_timestep_reached(self) -> bool:
        """
        Check whether the maximum allowed timestep has been reached.

        Returns:
            bool: True if the current timestep is greater than or equal to the
            maximum timestep, False otherwise.
        """
        return self.timestep >= self.max_timestep


if __name__ == "__main__":
    from pibt import PIBT
    from mapf_visualizer import MAPFVisualizer

    manager = MAPFInstanceManager(max_timestep=100,
                                  hazard_config="easy",
                                  wall_map=WallMap("empty-8-8"),
                                  even_or_random="even")

    instance = manager.instances[0]

    for _ in range(10):
        instance.add_agent()

    visualizer = MAPFVisualizer(instance)

    solver = PIBT(dim=instance.wall_map.width,
                  seed=0)
    solver.set_instance(instance)
    solver.set_hazard_awareness(True)
    solver.reset()

    visualizer.render()
    input()

    #print(instance.num_agents)

    solver.consider_hazards = False

    while True:
        instance.hazard_step()
        solver.step()
        visualizer.render()
        print([a.memory.estimation for a in instance.agents])
        input()

        if instance.finished():
            print(f"success: {instance.succeeded()}")
            print(f"soc: {instance.path_manager.calc_soc() / \
                          instance.calc_optimal_soc(instance.num_agents)}")
            print(f"makespan: {instance.path_manager.calc_makespan() / \
                               instance.calc_optimal_makespan(instance.num_agents)}")
            break
