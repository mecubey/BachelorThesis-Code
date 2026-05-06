"""
Contains the implementation of the PathTask enviroment.
"""

import time
from typing import Any, cast
import numpy as np
import matplotlib as plt
from .zone import Zone
from .maze import gen_maze
from . import header as h
import matplotlib.pyplot as plt

class PathTaskMultiAgentEnv:
    """
    Contains methods and attributes of the PathTask enviroment.
    """
    def __init__(self, *, args: h.EnvParams, config: dict[str, Any]|None = None):
        # ENV VARS
        self.args: h.EnvParams = args
        self.timestep: int = 0
        self.rng: np.random.Generator = np.random.default_rng()
        self.config = config
        self.grid_dim: int = 2*args.field_dim-1
        self.free_tiles: h.IntArr

        # for a single cell:
        # no_wall (1),
        # zone (1),
        # agent (1)
        self.global_obs: h.BoolArr = np.zeros((self.grid_dim,
                                               self.grid_dim,
                                               3),
                                               dtype=h.DTYPE_BOOL)

        # REWARD VARS
        self.zone_penalty: h.DTYPE_FLOAT = h.DTYPE_FLOAT(-1)

        # AGENTS
        self.agent_idx = list(range(args.num_agents)) # to randomly iterate through agents
        self.agent_positions: h.IntArr
        self.goal_positions: h.IntArr

        # RENDERING
        self.figure: Any = None
        self.ax: Any = None
        self.agent_scat: Any = None
        self.goal_scat: Any = None
        self.image: Any = None
        self.colors = plt.colormaps["tab20"](np.linspace(0, 1, args.num_agents))

        # ZONE
        self.zone = Zone(dir_spread_probs=args.dir_spread_probs,
                         max_num_spread=args.max_num_spread)

    def on_no_wall(self, pos: h.PositionT) -> bool:
        """
        Returns true if there is no wall on the given position.\n
        False otherwise.
        """
        return self.global_obs[*pos, h.GridOffsets.NO_WALL]

    def on_zone(self, pos: h.PositionT) -> bool:
        """
        Returns true if there is a zone on the given position.\n
        False otherwise.
        """
        return self.global_obs[*pos, h.GridOffsets.ZONE]

    def on_agent(self, pos: h.PositionT) -> bool:
        """
        Returns true if there is an agent on the given position.\n
        False otherwise.
        """
        return self.global_obs[*pos, h.GridOffsets.AGENT]

    def set_zone_grid(self, pos: h.PositionT, val: bool) -> None:
        """
        Sets/Removes zone on the specified position inside the grid.
        """
        self.global_obs[*pos, h.GridOffsets.ZONE] = val

    def set_agent_grid(self, pos: h.PositionT, val: bool) -> None:
        """
        Sets/Removes agent on the specified position inside the grid.
        """
        self.global_obs[*pos, h.GridOffsets.AGENT] = val

    def inside_grid(self, pos: h.PositionT) -> bool:
        """
        Returns true if position is inside the grid.\n
        False otherwise.
        """
        return 0 <= pos[0] < self.grid_dim and 0 <= pos[1] < self.grid_dim

    def walkable(self, pos: h.PositionT) -> bool:
        """
        Returns true if this agent could stand on a given position.\n
        False otherwise.
        """
        # tile cannot contain wall or another agent
        return self.inside_grid(pos) and self.on_no_wall(pos) and not self.on_agent(pos)

    def move_agent(self, *, a_idx: int, direction: h.PositionT) -> None:
        """
        Moves agent according to a specified direction.
        """
        self.set_agent_grid(self.agent_positions[a_idx], False)
        self.agent_positions[a_idx] += direction
        self.set_agent_grid(self.agent_positions[a_idx], True)

    def is_action_valid(self, *, action: h.Action, a_idx: int) -> bool:
        """
        Checks for validity of given action.
        """
        return self.walkable(self.agent_positions[a_idx]+h.ACT_TO_DIR[action])

    def reset(self, *,
              seed: int|None = None) -> None:
        """
        Resets the enviroment's attributes. Returns infos.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.timestep = 0

        self.global_obs.fill(0)

        # set maze in global_obs and get tiles with no walls
        self.free_tiles = gen_maze(maze_buffer=self.global_obs,
                                   dim=self.args.field_dim,
                                   maze_intensity=self.args.maze_intensity,
                                   rng=self.rng,
                                   exp_dim=self.grid_dim)

        num_free_tiles: int = len(self.free_tiles)
        free_tiles_mask: h.BoolArr = np.ones(num_free_tiles, dtype=h.DTYPE_BOOL)

        # AGENTS
        # generate all agent goal positions
        agent_goal_positions_idx = self.rng.choice(num_free_tiles,
                                                   size=self.args.num_agents,
                                                   replace=False)
        self.goal_positions = self.free_tiles[agent_goal_positions_idx]
        free_tiles_mask[agent_goal_positions_idx] = False

        # generate all non-overlapping agent positions
        # make sure agents and goals do not overlap
        self.agent_positions = self.rng.choice(self.free_tiles[free_tiles_mask],
                                               size=self.args.num_agents,
                                               replace=False)

        # set agents in grid and their goal positions
        for i in self.agent_idx:
            self.set_agent_grid(self.agent_positions[i], True)

        if self.args.render_mode == "human":
            self.render_setup()
            time.sleep(self.args.delay_btw_frames)

    def step(self, action_dict: dict[str, h.Action]) -> tuple[bool, bool]:
        """
        Step through the enviroment with a given action dictionary.\n
        The action dictionary has the form
            {"agent_0": action_of_0, "agent_1": action_of_1, ...}
        Returns whether the enviroment has terminated or truncates and infos.
        """
        num_agents_on_goal: int = 0

        # we only progress the hazard once we know it has spread to at least one tile
        if not self.zone.empty():
            self.zone.progress()

        # randomly spawn zone if zone is empty
        spawned: bool = False
        if self.zone.empty() and self.rng.random() <= self.args.spawn_prob:
            spawned = True
            zone_start_pos: h.PositionT = self.rng.choice(self.free_tiles)
            self.zone.spawn(start_pos=zone_start_pos)
            self.set_zone_grid(zone_start_pos, True) # set starting zone inside grid

        # randomly spread randomly in cardinal directions if spawned in this timestep
        if self.rng.random() <= self.args.spread_prob and not spawned:
            new_spread_tiles: list[h.PositionT] = self.zone.spread(rng=self.rng,
                                                                   on_zone=self.on_zone,
                                                                   no_wall=self.on_no_wall,
                                                                   inside_grid=self.inside_grid)
            for tile_pos in new_spread_tiles: # set zone inside grid
                self.set_zone_grid(tile_pos, True)

        if self.zone.done(): # remove all zones after set amount of timesteps
            for tile_pos in self.zone.occupied_tiles:
                self.set_zone_grid(tile_pos, False)
            self.zone.reset()

        for i in self.agent_idx:
            agent_name = f"agent_{i}"
            if self.is_action_valid(action=action_dict[agent_name],
                                    a_idx=i):
                self.move_agent(a_idx=i,
                                direction=h.ACT_TO_DIR[action_dict[agent_name]])

        # check for collisions
        # check for goal completion status
        for i in self.agent_idx:
            collidees: h.IntArr = np.where((self.agent_positions ==
                                            self.agent_positions[i]).all(axis=1))[0]

            # agent collides if more than one agent on agent's cell
            if len(collidees) > 1:
                for c in collidees: # we move all the colliding agents back
                    c = cast(int, c)

                    # reverse last action
                    self.move_agent(a_idx=c,
                                    direction=h.reverse_dir(action_dict[f"agent_{c}"]))

            if (self.agent_positions[i] == self.goal_positions[i]).all():
                num_agents_on_goal += 1

        truncated = False
        if self.timestep == self.args.max_timestep:
            truncated = True

        terminated = False
        if num_agents_on_goal == self.args.num_agents:
            terminated = True

        if self.args.render_mode == "human":
            self.render()
            time.sleep(self.args.delay_btw_frames)

        self.timestep += 1

        return terminated, truncated

    def gen_img_from_grid(self) -> h.FloatArr:
        """
        Generate an image from the enviroment grid with
        walls and current zone tiles set.
        """

        # base: white free, black wall
        img = np.ones((self.global_obs.shape[0], self.global_obs.shape[1], 3), dtype=h.DTYPE_FLOAT)
        img[self.global_obs[:, :, h.GridOffsets.NO_WALL] == 0] = [0, 0, 0]

        # zone: light red tint
        zone_mask = self.global_obs[:, :, h.GridOffsets.ZONE] == 1
        img[zone_mask] = [1.0, 0.7, 0.7]

        return img

    def render_setup(self) -> None:
        """
        Sets up rendering variables.
        """
        plt.ion()  # type: ignore
        self.figure, self.ax = plt.subplots() # type: ignore

        img = self.gen_img_from_grid()

        self.image = self.ax.imshow(img)
        self.ax.axis("off")

        # NOTE: plt has (x, y) as in traditional coordinate system
        self.goal_scat = self.ax.scatter(self.goal_positions[:, 1],
                                         self.goal_positions[:, 0],
                                         c=self.colors, marker="P", edgecolors="black",
                                         s=3500 / self.grid_dim)
        self.agent_scat = self.ax.scatter(self.agent_positions[:, 1],
                                          self.agent_positions[:, 0],
                                          c=self.colors, marker="o", edgecolors="black",
                                          s=2000 / self.grid_dim)

    def render(self) -> None:
        """
        Renders the enviroment.
        """
        img = self.gen_img_from_grid()
        self.image.set_data(img)
        self.agent_scat.set_offsets(np.c_[self.agent_positions[:, 1],
                                          self.agent_positions[:, 0]])
        self.goal_scat.set_offsets(np.c_[self.goal_positions[:, 1],
                                         self.goal_positions[:, 0]])
        plt.draw()

    def close(self) -> None:
        """
        Closes generated plt window.
        """
        if self.args.render_mode == "human":
            plt.ioff() # type: ignore
            plt.close(self.figure)
