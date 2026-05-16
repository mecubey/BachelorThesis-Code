"""
Contains the implementation of the PathTask enviroment.
"""

from typing import Any
import numpy as np
from .zone import Zone
from .maze import gen_maze
from . import header as h
import matplotlib.pyplot as plt
from .grid import Grid

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

    def reset(self, *,
              env_seed: int|None = None,
              maze_seed: int|None = None,
              zone_seed: int|None = None) -> None:
        """
        Resets the enviroment's attributes. Returns infos.
        """
        self.rng = np.random.default_rng(env_seed)

        # for a single cell:
        # no_wall (1),
        # zone (1),
        # agent (1)
        grid_dim = 2*self.args.field_dim-1
        field: h.IntArr = np.zeros((grid_dim + 2, # add surrounding wall
                                    grid_dim + 2, # add surrounding wall
                                    3),
                                    dtype=h.DTYPE_INT)

        self.timestep = 0

        # set maze in global_obs and get tiles with no walls
        free_tiles = gen_maze(maze_buffer=field,
                              dim=self.args.field_dim,
                              maze_intensity=self.args.maze_intensity,
                              exp_dim=grid_dim,
                              seed=maze_seed)

        # AGENTS
        # generate all agent positions (agents cannot overlap with eachother)
        self.agent_positions = []
        for pos in self.rng.choice(free_tiles,
                                   size=self.args.num_agents,
                                   replace=False):
            self.agent_positions.append(h.Position(*pos))

        # GOALS
        # generate all goal positions (goals cannot overlap with eachother)
        self.goal_positions = []
        for pos in self.rng.choice(free_tiles,
                                   size=self.args.num_agents,
                                   replace=False):
            self.goal_positions.append(h.Position(*pos))

        # GRID
        self.grid = Grid(field=field,
                         max_timestep=self.args.max_timestep,
                         num_agents=self.args.num_agents,
                         agent_positions=self.agent_positions,
                         goal_positions=self.goal_positions)

        # ZONE
        self.zone = Zone(grid=self.grid,
                         free_tiles=free_tiles,
                         dir_spread_probs=self.args.dir_spread_probs,
                         max_num_spread=self.args.max_num_spread,
                         seed=zone_seed)

        # set agents in grid
        for i in range(self.args.num_agents):
            self.grid.set_agent_in_grid(self.agent_positions[i], True)

        if self.args.render_mode == "human":
            self.render_setup()

    def step(self, actions: list[h.Action]) -> tuple[bool, bool]:
        """
        Step through the enviroment with a given action dictionary.\n
        The action dictionary has the form
            {"agent_0": action_of_0, "agent_1": action_of_1, ...}
        Returns whether the enviroment has terminated or truncated.
        """
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

        for i in self.grid.agent_idx:
            # check for collisions
            if self.grid.contains_multiple_agents(self.agent_positions[i]):
                for j in self.grid.agent_idx:
                    if self.agent_positions[i] == self.agent_positions[j]:
                        # reverse action of agent
                        self.grid.move_agent_in_grid(i,
                                                     h.ACT_TO_OPPOSITE_ACT
                                                     [actions[j]])
            # check for goal completion status
            if self.grid.is_agent_on_goal(i):
                num_agents_on_goal += 1

        truncated = False
        if self.timestep == self.args.max_timestep:
            truncated = True

        terminated = False
        if num_agents_on_goal == self.args.num_agents:
            terminated = True

        if self.args.render_mode == "human":
            self.render()

        self.timestep += 1

        return terminated, truncated

    def gen_img_from_grid(self) -> h.FloatArr:
        """
        Generate an image from the enviroment grid with
        walls and current zone tiles set.
        """

        # base: white free, black wall
        img = np.ones((self.grid.dim, self.grid.dim, 3), dtype=h.DTYPE_FLOAT)
        img[self.grid.field[:, :, h.GridOffsets.NO_WALL] == 0] = [0, 0, 0]

        # zone: light red tint
        zone_mask = self.grid.field[:, :, h.GridOffsets.ZONE] == 1
        img[zone_mask] = [1.0, 0.7, 0.7]

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
