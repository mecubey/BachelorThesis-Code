"""
Contains the implementation of the PathTask enviroment.
"""

import time
from typing import Any, cast
import numpy as np
from .agent import EnvAgent
from .zone import Zone
from .maze import gen_maze
from . import header as h

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
        self.agent_positions: h.IntArr # much more efficient to calculate dx, dy this way
        self.agents: list[EnvAgent] = [EnvAgent(i) for i in self.agent_idx]

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

    def set_infos(self, infos: dict[str, Any]):
        """
        Given an empty infos dictionary, sets its infos accordingly to env state.
        """
        assert len(infos) == 0

        for agent in self.agents:
            infos[agent.agent_name] = agent.to_dict(pos=self.agent_positions[agent.agent_id])

    def is_action_valid(self, *, action: h.Action, a_idx: int) -> bool:
        """
        Checks for validity of given action.
        """
        return self.walkable(self.agent_positions[a_idx]+h.ACT_TO_DIR[action])

    def reset(self, *,
              seed: int|None = None) -> dict[str, Any]:
        """
        Resets the enviroment's attributes. Returns infos.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

            # set colors of agents
            for a in self.agents:
                a.agent_color = h.rand_color(self.rng)

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
        free_tiles_mask[agent_goal_positions_idx] = 0

        # generate all non-overlapping agent positions
        # make sure agents and goals do not overlap
        self.agent_positions = self.rng.choice(self.free_tiles[free_tiles_mask],
                                               size=self.args.num_agents,
                                               replace=False)

        # set agents in grid and their goal positions
        for agent in self.agents:
            self.set_agent_grid(self.agent_positions[agent.agent_id], True)
            agent.goal_pos = self.free_tiles[agent_goal_positions_idx[agent.agent_id]]

        infos: dict[str, Any] = {}
        if self.args.with_debug_infos:
            self.set_infos(infos)

        if self.args.render_mode == "human":
            self.render()
            time.sleep(self.args.delay_btw_frames)

        return infos

    def step(self, action_dict: dict[str, h.Action]) -> tuple[bool, bool, dict[str, Any]]:
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

        # randomly spread randomly in cardinal directions if now spawned in this timestep
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
            if self.is_action_valid(action=action_dict[self.agents[i].agent_name],
                                    a_idx=i):
                self.move_agent(a_idx=i,
                                direction=h.ACT_TO_DIR[action_dict[self.agents[i].agent_name]])

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
                                    direction=h.reverse_dir(action_dict[self.agents[c].agent_name]))

            if self.agents[i].on_goal(self.agent_positions[i]):
                num_agents_on_goal += 1

        truncated = False
        if self.timestep == self.args.episode_length:
            truncated = True

        terminated = False
        if num_agents_on_goal == self.args.num_agents:
            terminated = True

        infos: dict[str, Any] = {}
        if self.args.with_debug_infos:
            self.set_infos(infos)

        if self.args.render_mode == "human":
            self.render()
            time.sleep(self.args.delay_btw_frames)

        self.timestep += 1

        return terminated, truncated, infos

    def render(self) -> None:
        """
        Renders the enviroment.
        """
        goal_positions: h.IntArr = np.array([a.goal_pos for a in self.agents],
                                            dtype=h.DTYPE_INT)
        dim: int = self.global_obs.shape[0]

        for row in range(dim):
            for col in range(dim):
                pos: h.PositionT = np.array([row, col])

                # if zone on tile
                tmp_zone_char: str = ""
                if self.on_zone(pos):
                    tmp_zone_char = h.ZONE_COL

                # if agent on tile
                if self.on_agent(pos):
                    a_idx: int = np.where((self.agent_positions == pos).all(axis=1))[0][0]
                    print(tmp_zone_char + self.agents[a_idx].agent_color + h.AGENT_CHAR, end="")
                    continue

                # if goal on tile
                if (goal_positions == pos).all(axis=1).any():
                    g_idx: int = np.where((goal_positions == pos).all(axis=1))[0][0]
                    print(tmp_zone_char + self.agents[g_idx].agent_color + h.GOAL_CHAR, end="")
                    continue

                # if wall on tile
                if not self.on_no_wall(np.array(pos)):
                    print(tmp_zone_char + h.WALL_CHAR, end="")
                    continue

                # if nothing else, print empty space
                print(tmp_zone_char + " " + h.Style.RESET_ALL, end="")

            print()
