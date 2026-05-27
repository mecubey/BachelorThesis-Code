"""
Contains the implementation of the PathTask enviroment.
"""

from copy import deepcopy
from typing import Any, cast
import numpy as np
from .zone import Zone
from . import header as h
import matplotlib.pyplot as plt
from .grid import Grid
from .logger import Logger
from .pibt.dist_table import DistTable

class PathTaskMultiAgentEnv:
    """
    Contains methods and attributes of the PathTask enviroment.
    """
    def __init__(self, *,
                 args: h.EnvParams):
        self.args: h.EnvParams = args

        self.timestep: int = 0

        self.rng: np.random.Generator

        # AGENTS
        self.agent_positions: list[h.Position]
        self.goal_positions: list[h.Position]

        # RENDERING
        self.figure: Any = None
        self.ax: Any = None
        self.agent_scat: Any = None
        self.goal_scat: Any = None
        self.image: Any = None
        self.colors = plt.colormaps["tab20"](np.linspace(0, 1, args.num_agents))

        # ZONE
        self.zone: Zone

        # GRID
        self.grid: Grid

        # DATA LOGGING
        self.logger: Logger = Logger()
        self.shortest_path_sum: float = 0
        self.longest_shortest_path: float = 0

    def reset(self, *,
              wall_map: h.BoolArr,
              free_tiles: list[h.Position],
              env_seed: int = 0,
              zone_seed: int = 0) -> None:
        """
        Resets the enviroment's attributes. Returns infos.
        """
        self.timestep = 0

        self.rng = np.random.default_rng(env_seed)

        free_tiles_idxs = np.arange(len(free_tiles))

        # AGENTS
        # generate all agent positions (agents cannot overlap with eachother)
        self.agent_positions = []
        for pos_idx in self.rng.choice(free_tiles_idxs,
                                       size=self.args.num_agents,
                                       replace=False):
            pos_idx = cast(int, pos_idx)
            self.agent_positions.append(deepcopy(free_tiles[pos_idx]))

        # GOALS
        # generate all goal positions (goals cannot overlap with eachother)
        self.goal_positions = []
        for pos_idx in self.rng.choice(free_tiles_idxs,
                                       size=self.args.num_agents,
                                       replace=False):
            pos_idx = cast(int, pos_idx)
            self.goal_positions.append(free_tiles[pos_idx])

        # GRID
        self.grid = Grid(wall_map=wall_map,
                         zone_map=np.zeros(wall_map.shape, dtype=h.DTYPE_BOOL),
                         max_timestep=self.args.max_timestep,
                         num_agents=self.args.num_agents,
                         get_episode_progress=lambda: self.timestep / self.args.max_timestep,
                         agent_positions=self.agent_positions,
                         goal_positions=self.goal_positions)

        # ZONE
        self.zone = Zone(grid=self.grid,
                         free_tiles=free_tiles,
                         dir_spread_probs=self.args.dir_spread_probs,
                         max_num_spread=self.args.max_num_spread,
                         dmg_type=self.args.hazard_dmg_type,
                         seed=zone_seed)

        # LOGGING
        self.logger.reset()
        self.shortest_path_sum = 0
        self.longest_shortest_path = -1
        # calculate the sum of all the shortest paths lengths
        # too ugly to connect with PIBT itself, so we will just
        # use the dist tables (inefficient but meh)
        for i in self.grid.agent_idx:
            dist_table = DistTable(self.grid, self.grid.goal_positions[i])
            distance = dist_table.get(self.agent_positions[i])
            self.longest_shortest_path = max(distance, self.longest_shortest_path)
            self.shortest_path_sum += distance

        if self.args.render_mode == "human":
            self.render_setup()

    def step(self, actions: list[h.Action]) -> tuple[bool, bool]:
        """
        Step through the enviroment with a given action dictionary.\n
        The action dictionary has the form
            {"agent_0": action_of_0, "agent_1": action_of_1, ...}
        Returns whether the enviroment has terminated or truncated.
        """
        if self.args.num_agents == 0:
            self.logger.record_episode_end(fin=1)
            self.logger.record_makespan(makespan=0)
            return True, False

        num_agents_on_goal: int = 0

        # we only progress the hazard once we know it has spread to at least one tile
        if not self.zone.empty():
            self.zone.progress()

        # randomly spawn zone if zone is empty
        spawned: bool = False
        if self.zone.empty() and self.rng.random() <= self.args.spawn_prob:
            spawned = True
            self.zone.spawn()

        # randomly spread randomly in cardinal directions if spawned in this timestep
        if self.rng.random() <= self.args.spread_prob and not spawned:
            self.zone.spread()

        # remove all zones after set amount of timesteps
        if self.zone.done():
            self.zone.reset()

        for i in self.grid.agent_idx:
            self.grid.move_agent_in_grid(i, actions[i])

            if self.grid.is_agent_on_goal(i):
                num_agents_on_goal += 1

            self.logger.record_hzd_dmg(self.zone.get_hazard_dmg(self.agent_positions[i])/
                                       self.shortest_path_sum)

            self.logger.record_soc(1/self.shortest_path_sum)

        self.timestep += 1

        if self.args.render_mode == "human":
            self.render()

        truncated = False
        terminated = False

        if num_agents_on_goal == self.args.num_agents:
            self.logger.record_episode_end(fin=1)
            self.logger.record_makespan(makespan=self.timestep/self.longest_shortest_path)
            terminated = True
            return terminated, truncated

        if self.timestep == self.args.max_timestep:
            self.logger.record_episode_end(fin=0)
            self.logger.record_makespan(makespan=np.nan)
            self.logger.soc = np.nan
            self.logger.total_hzd_dmg = np.nan
            truncated = True

        return terminated, truncated

    def gen_img_from_grid(self) -> h.FloatArr:
        """
        Generate an image from the enviroment grid with
        walls and current zone tiles set.
        """

        # base: white free, black wall
        img = np.ones((self.grid.dim, self.grid.dim, 3), dtype=h.DTYPE_FLOAT)
        img[self.grid.wall_map[...] == 1] = [0, 0, 0]

        # zone: light red tint
        img[self.grid.zone_map[...] == 1] = [1.0, 0.7, 0.7]

        return img

    def render_setup(self) -> None:
        """
        Sets up rendering variables.
        """
        plt.ion()  # type: ignore
        self.figure, self.ax = plt.subplots() # type: ignore

        img = self.gen_img_from_grid()

        self.image = self.ax.imshow(img, interpolation='nearest')

        # major ticks at cell centers
        self.ax.set_xticks(np.arange(img.shape[1]))
        self.ax.set_yticks(np.arange(img.shape[0]))

        # minor ticks at cell boundaries
        self.ax.set_xticks(np.arange(-0.5, img.shape[1], 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, img.shape[0], 1), minor=True)

        # draw grid on minor ticks
        self.ax.grid(which="minor", color="gray", linewidth=1.5)

        # show grid lines
        self.ax.set_axisbelow(False)

        # hide labels
        self.ax.tick_params(which='both',
                            bottom=False,
                            left=False,
                            labelbottom=False,
                            labelleft=False)

        # NOTE: plt has (x, y) as in traditional coordinate system
        goal_pos_as_ndarray = np.array([pos.as_ndarray() for pos in self.goal_positions])
        agent_pos_as_ndarray = np.array([pos.as_ndarray() for pos in self.agent_positions])

        if self.args.num_agents > 0:
            self.goal_scat = self.ax.scatter(goal_pos_as_ndarray[:, 1],
                                            goal_pos_as_ndarray[:, 0],
                                            c=self.colors, marker="P", edgecolors="black",
                                            s=3500 / self.grid.dim)
            self.agent_scat = self.ax.scatter(agent_pos_as_ndarray[:, 1],
                                            agent_pos_as_ndarray[:, 0],
                                            c=self.colors, marker="o", edgecolors="black",
                                            s=2000 / self.grid.dim)

    def render(self) -> None:
        """
        Renders the enviroment.
        """
        img = self.gen_img_from_grid()
        self.image.set_data(img)

        goal_pos_as_ndarray = np.array([pos.as_ndarray() for pos in self.goal_positions])
        agent_pos_as_ndarray = np.array([pos.as_ndarray() for pos in self.agent_positions])

        self.goal_scat.set_offsets(np.c_[goal_pos_as_ndarray[:, 1],
                                         goal_pos_as_ndarray[:, 0]])
        self.agent_scat.set_offsets(np.c_[agent_pos_as_ndarray[:, 1],
                                          agent_pos_as_ndarray[:, 0]])
        plt.draw()

    def close(self) -> None:
        """
        Closes generated plt window.
        """
        if self.args.render_mode == "human":
            plt.ioff() # type: ignore
            plt.close(self.figure)
