"""
Contains the implementation of the PathTask enviroment.
"""

import time
from typing import Any, cast, Optional
from gymnasium.spaces import Dict, Box, Space, Discrete
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
        self.observation_offset: h.PositionT = np.array([args.obs_radius,
                                                         args.obs_radius],
                                                         dtype=h.DTYPE_INT)
        self.grid_offsets = h.GridOffsets(no_wall=0,
                                          zone=1,
                                          agent=2,
                                          neighbour_goal=3,
                                          own_goal=4)
        self.grid_dim: int = 2*args.field_dim-1
        self.max_goal_dist: int = 2*(self.grid_dim-1)
        self.free_tiles: h.IntArr

        # for a single cell:
        # no_wall (1),
        # zone (1),
        # agent (1)
        self.global_obs: h.FloatArr = np.zeros((self.grid_dim + 2*args.obs_radius,
                                                self.grid_dim + 2*args.obs_radius,
                                                3),
                                               dtype=h.DTYPE_FLOAT)
        # in agents' own observation, these are also added:
        # is_neighbour_goals (1)
        # is_own_goal (1)

        # REWARD VARS
        self.all_goals_reached_rwd: h.DTYPE_FLOAT = h.DTYPE_FLOAT(20)
        self.step_penalty: h.DTYPE_FLOAT = h.DTYPE_FLOAT(-0.3)
        self.collision_penalty: h.DTYPE_FLOAT = h.DTYPE_FLOAT(-2)
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
        return self.global_obs[*pos, self.grid_offsets.no_wall]

    def on_zone(self, pos: h.PositionT) -> bool:
        """
        Returns true if there is a zone on the given position.\n
        False otherwise.
        """
        return self.global_obs[*pos, self.grid_offsets.zone]

    def on_agent(self, pos: h.PositionT) -> bool:
        """
        Returns true if there is an agent on the given position.\n
        False otherwise.
        """
        return self.global_obs[*pos, self.grid_offsets.agent]

    def set_zone_grid(self, pos: h.PositionT, val: bool) -> None:
        """
        Sets/Removes zone on the specified position inside the grid.
        """
        self.global_obs[*pos, self.grid_offsets.zone] = val

    def set_agent_grid(self, pos: h.PositionT, val: bool) -> None:
        """
        Sets/Removes agent on the specified position inside the grid.
        """
        self.global_obs[*pos, self.grid_offsets.agent] = val

    def walkable(self, pos: h.PositionT) -> bool:
        """
        Returns true if this agent could stand on a given position.\n
        False otherwise.
        """
        # tile cannot contain wall or another agent
        return self.on_no_wall(pos) and not self.on_agent(pos)

    def move_agent(self, *, a_idx: int, direction: h.PositionT) -> None:
        """
        Moves agent according to a specified direction.
        """
        self.set_agent_grid(self.agent_positions[a_idx], False)
        self.agent_positions[a_idx] += direction
        self.set_agent_grid(self.agent_positions[a_idx], True)

    def get_vec_obs(self, a_idx: int) -> tuple[h.FloatArr, h.DTYPE_FLOAT]:
        """
        Return vector observations (unit vector to goal and euclidean distance).
        """
        goal_vec: h.FloatArr = (self.agents[a_idx].goal_pos -
                                self.agent_positions[a_idx]).astype(h.DTYPE_FLOAT)
        goal_distance: h.DTYPE_FLOAT = np.linalg.norm(goal_vec)+h.EPSILON
        return goal_vec, goal_distance

    def get_agent_obs(self, *,
                      a_idx: int,
                      unit_vec: h.FloatArr,
                      dist: h.DTYPE_FLOAT) -> dict[str, h.FloatArr]:
        """
        Assuming the enviroment is in state s, get observation of agent a_idx of state s.
        """
        begin_obs_radius: h.PositionT = self.agent_positions[a_idx]-self.args.obs_radius
        end_obs_radius: h.PositionT = self.agent_positions[a_idx]+self.args.obs_radius+1

        # get local observation of agent
        grid_obs: h.FloatArr = self.global_obs[begin_obs_radius[0]:end_obs_radius[0],
                                               begin_obs_radius[1]:end_obs_radius[1]]

        # each cell should get two other zeros for neighbour_goals & own_goals
        grid_obs = np.concatenate([grid_obs,
                                   np.zeros((grid_obs.shape[0],
                                             grid_obs.shape[1],
                                             2))],
                                   axis=2).astype(h.DTYPE_FLOAT)

        # get indices of agents that are inside local observation
        # use chebyshev distance
        neighbours: h.IntArr = np.where(np.max(np.abs(self.agent_positions -
                                                      self.agent_positions[a_idx]), axis=1)
                               <= self.args.obs_radius)[0]

        for n in neighbours:
            n = cast(h.DTYPE_INT, n)

            if a_idx == n:
                # put own goal into obs if inside obs
                dist_to_own_goal: h.DTYPE_FLOAT = np.max(np.abs(self.agents[a_idx].goal_pos -
                                                         self.agent_positions[a_idx]))

                if dist_to_own_goal <= self.args.obs_radius:
                    grid_obs[*(self.agents[a_idx].goal_pos-
                               (self.agent_positions[a_idx]-
                                self.observation_offset)), self.grid_offsets.own_goal] = 1
                continue

            # clip neighbour's goals to edge of local observation
            clipped_goal_pos = np.clip(self.agents[n].goal_pos,
                                       self.agent_positions[a_idx]-self.args.obs_radius,
                                       self.agent_positions[a_idx]+self.args.obs_radius)
            clipped_goal_pos -= (self.agent_positions[a_idx]-self.observation_offset)
            grid_obs[*clipped_goal_pos, self.grid_offsets.neighbour_goal] = 1

        return {"grid_obs": grid_obs,
                "vec_obs": np.concatenate([unit_vec / dist, 
                                           np.array([dist], dtype=h.DTYPE_FLOAT)], axis=0)}

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
              seed: int|None = None,
              options: Optional[dict[Any, Any]] = None) -> tuple[h.MultiObsT,
                                                                 dict[str, Any]]:
        """
        Resets the enviroment.
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
                                   obs_radius=self.args.obs_radius,
                                   dim=self.args.field_dim,
                                   maze_intensity=self.args.maze_intensity,
                                   rng=self.rng,
                                   offsets=self.grid_offsets,
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

        observations: h.MultiObsT = {}
        # don't need to randomly iterate because we are assigning obs
        for agent in self.agents:
            unit_vec, dist = self.get_vec_obs(agent.agent_id)
            observations[agent.agent_name] = {"local_obs":
                                              self.get_agent_obs(a_idx=agent.agent_id,
                                                                 unit_vec=unit_vec,
                                                                 dist=dist),
                                              "action_mask": 
                                              agent.get_mask(self.walkable,
                                                             self.agent_positions[agent.agent_id])}

        if self.args.render_mode == "human":
            self.render()
            time.sleep(self.args.delay_btw_frames)

        return observations, infos

    def step(self, action_dict: dict[str, h.Action]) -> tuple[h.MultiObsT,
                                                              dict[str, h.DTYPE_FLOAT],
                                                              bool,
                                                              bool,
                                                              dict[str, Any]]:
        """
        Step through the enviroment with a given action dictionary.\n
        Returns observations, rewards, terminations, trunactions, infos dictionaries.
        """
        num_agents_on_goal: int = 0

        rewards: dict[str, h.DTYPE_FLOAT] = {a.agent_name: self.step_penalty for a in self.agents}

        # randomly spawn zone if zone is zone is empty
        if self.zone.empty() and self.rng.random() <= self.args.spawn_prob:
            zone_start_pos: h.PositionT = self.rng.choice(self.free_tiles)
            self.zone.spawn(start_pos=zone_start_pos)
            self.set_zone_grid(zone_start_pos, True) # set starting zone inside grid

        # randomly spread randomly in cardinal directions
        if self.rng.random() <= self.args.spread_prob:
            new_spread_tiles: list[h.PositionT] = self.zone.spread(rng=self.rng,
                                                                   on_zone=self.on_zone,
                                                                   no_wall=self.on_no_wall)
            for tile_pos in new_spread_tiles: # set zone inside grid
                self.set_zone_grid(tile_pos, True)

        if self.zone.done(): # remove all zones after set amount of timesteps
            for tile_pos in self.zone.occupied_tiles:
                self.set_zone_grid(tile_pos, False)
            self.zone.reset()

        # each agent executes its action if it's valid
        # random sequential action execution
        for i in h.randomly(self.agent_idx, self.rng):
            i = cast(int, i)
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
                    # punish agent for colliding
                    rewards[self.agents[c].agent_name] = self.collision_penalty

                    # reverse last action
                    self.move_agent(a_idx=c,
                                    direction=h.reverse_dir(action_dict[self.agents[c].agent_name]))

            # agents should not step into zones
            if self.on_zone(self.agent_positions[i]):
                rewards[self.agents[i].agent_name] += self.zone_penalty

        observations: h.MultiObsT = {}
        for agent in self.agents:
            num_agents_on_goal += agent.on_goal(self.agent_positions[agent.agent_id])
            unit_vec, dist = self.get_vec_obs(agent.agent_id)
            observations[agent.agent_name] = {"local_obs":
                                              self.get_agent_obs(a_idx=agent.agent_id,
                                                                 unit_vec=unit_vec,
                                                                 dist=dist),
                                              "action_mask": agent.get_mask(self.walkable,
                                                             self.agent_positions[agent.agent_id])}

        truncated = False
        if self.timestep == self.args.episode_length:
            truncated = True

        terminated = False
        if num_agents_on_goal == self.args.num_tasks:
            terminated = True

            # we finished the episode by finishing all tasks, so reward agents
            for i in self.agent_idx:
                rewards[self.agents[i].agent_name] = self.all_goals_reached_rwd

        infos: dict[str, Any] = {}
        if self.args.with_debug_infos:
            self.set_infos(infos)

        if self.args.render_mode == "human":
            self.render()
            time.sleep(self.args.delay_btw_frames)

        return observations, rewards, terminated, truncated, infos

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

    def get_observation_space(self) -> Dict:
        """
        Get the observation space of this enviroment.
        """

        grid_size = 2 * self.args.obs_radius + 1

        return Dict({
            "local_obs": Dict({
                "grid_obs": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(grid_size, grid_size, 4),
                    dtype=h.DTYPE_FLOAT
                ),
                "vec_obs": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=h.DTYPE_FLOAT
                ),
            }),
            "action_mask": Box(
                low=0,
                high=1,
                shape=(5,),
                dtype=h.DTYPE_FLOAT
            ),
        })

    def get_action_space(self) -> Space[np.integer]:
        """
        Get the action space of this enviroment.
        """
        return Discrete(5)
