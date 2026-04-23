"""
Contains the implementation of the PathTask enviroment.
"""

import time
from typing import Any, cast
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
import numpy as np
from .env_agent import EnvAgent
from .task import Task
from .zone import Zone
from .maze import gen_maze
from .task_planner import get_goal
from . import header as h

class PathTaskEnv(MultiAgentEnv):
    """
    Contains methods and attributes of the PathTask enviroment.
    """
    def __init__(self, *, args: h.EnvParams, config: dict[str, Any]|None = None):
        super().__init__()
        # ENV VARS
        # i don't use built-in _np_random, num_agents, agents, possible_agents
        self.args: h.EnvParams = args
        self.timestep: int = 0
        self.rng: np.random.Generator = np.random.default_rng()
        self.config = config
        self.observation_offset: h.PositionT = np.array([args.obs_radius,
                                                         args.obs_radius],
                                                        dtype=h.DTYPE_INT)
        self.grid_offsets = h.GridOffsets(is_wall=0,
                                          is_zone=1,
                                          agent_occ_b=2,
                                          agent_occ_e=2+self.args.num_agents,
                                          agent_goal_b=2+self.args.num_agents,
                                          agent_goal_e=2+2*self.args.num_agents)
        self.grid_dim: int = 2*args.field_dim-1
        self.max_goal_dist: int = 2*(self.grid_dim-1)

        # for a single cell:
        # no_wall (1),
        # is_zone (1),
        # which_agents (num_agents),
        # whose_goal (num_agents)
        self.global_obs: h.FloatArr = np.zeros((self.grid_dim + 2*args.obs_radius,
                                                self.grid_dim + 2*args.obs_radius,
                                                2+2*self.args.num_agents),
                                               dtype=h.DTYPE_FLOAT)
        self.free_tiles: h.IntArr

        # REWARD VARS
        self.all_tasks_finished_rwd: h.DTYPE_FLOAT = h.DTYPE_FLOAT(20)
        self.step_penalty: h.DTYPE_FLOAT = h.DTYPE_FLOAT(-0.3)
        self.collision_penalty: h.DTYPE_FLOAT = h.DTYPE_FLOAT(-2)
        self.zone_penalty: h.DTYPE_FLOAT = h.DTYPE_FLOAT(-1)
        # step() also adds negative distance to goal to reward

        # AGENTS
        self.agent_idx = list(range(args.num_agents))
        self.agent_colors: list[str] = [h.rand_color(self.rng) for _ in range(self.args.num_agents)]
        self.agent_names = [f"agent_{i}" for i in self.agent_idx]
        self.agents = self.possible_agents = [f"agent_{i}" for i in self.agent_idx]
        self.agent_positions: h.IntArr # much more efficient to calculate dx, dy this way
        agent_one_hots: h.FloatArr = np.eye(self.args.num_agents, dtype=h.DTYPE_FLOAT)
        self.env_agents: list[EnvAgent] = [EnvAgent(agent_one_hots[i], i) for i in self.agent_idx]
        self.depot_position: h.PositionT

        # TASKS
        self.tasks: list[Task] = [Task() for _ in range(args.num_tasks)]
        self.num_tasks_finished: int = 0
        self.task_colors: list[str] = [h.rand_color(self.rng) for _ in range(args.num_tasks)]

        # ZONE
        self.zone = Zone(dir_spread_probs=args.dir_spread_probs,
                         max_num_spread=args.max_num_spread)

    def inside_grid(self, pos: h.PositionT) -> bool:
        """
        Returns true if the given position is inside the inner grid.\n
        False otherwise.
        """
        return bool((self.args.obs_radius <= pos).all() and \
                    (pos < self.args.obs_radius+self.grid_dim).all())

    def no_wall(self, pos: h.PositionT) -> bool:
        """
        Returns true if there is no wall on the given position.\n
        False otherwise.
        """
        return self.global_obs[*pos, self.grid_offsets.is_wall]

    def on_zone(self, pos: h.PositionT) -> bool:
        """
        Returns true if there is a zone on the given position.\n
        False otherwise.
        """
        return self.global_obs[*pos, self.grid_offsets.is_zone]

    def set_zone_grid(self, pos: h.PositionT, val: bool) -> None:
        """
        Sets zone on the specified position.
        """
        self.global_obs[*pos, 1] = val

    def agent_count_on(self, pos: h.PositionT) -> h.DTYPE_FLOAT:
        """
        Returns amount of agents on the given position.
        """
        return np.sum(self.global_obs[*pos, self.grid_offsets.agent_occ_b:
                                            self.grid_offsets.agent_occ_e])

    def agents_on(self, pos: h.PositionT) -> h.IntArr:
        """
        Returns indices of agents on the given position.
        """
        return np.where(self.global_obs[*pos, self.grid_offsets.agent_occ_b:
                                              self.grid_offsets.agent_occ_e] == 1)[0]

    def set_agent_occ_grid(self, a_idx: int, pos: h.PositionT, val: bool) -> None:
        """
        Sets agent occupation inside of grid on a given position 
        (occupation changes upon entry/exit of cell).
        """
        self.global_obs[*pos, self.grid_offsets.agent_occ_b+a_idx] = val

    def is_agent_goal_on(self, a_idx: int, pos: h.PositionT) -> bool:
        """
        Returns true if the agent's goal is on the given position.\n
        False otherwise.
        """
        return self.global_obs[*pos, self.grid_offsets.agent_goal_b+a_idx]

    def set_agent_goal_grid(self, a_idx: int, goal: int, want_to_contribute: bool) -> None:
        """
        Sets/Removes agent's goal inside of grid. Also contributes agent's trait to task's
        plan requirements if setting goal.
        """
        if goal == h.AGENT_DEPOT:
            self.global_obs[*self.depot_position,
                            self.grid_offsets.agent_goal_b+a_idx] = want_to_contribute
        else:
            self.global_obs[*self.tasks[goal].position,
                            self.grid_offsets.agent_goal_b+a_idx] = want_to_contribute
            if want_to_contribute:
                self.tasks[goal].contribute_to_plan(self.env_agents[a_idx].traits)

    def walkable(self, pos: h.PositionT) -> bool:
        """
        Returns true if this agent could stand on a given position.\n
        False otherwise.
        """
        return self.no_wall(pos) and self.agent_count_on(pos) == 0

    def move_agent(self, *, a_idx: int, direction: h.PositionT) -> None:
        """
        Moves agent according to a specified direction.
        """
        self.set_agent_occ_grid(a_idx, self.agent_positions[a_idx], False)
        self.agent_positions[a_idx] += direction
        self.set_agent_occ_grid(a_idx, self.agent_positions[a_idx], True)

    def get_vec_obs(self, a_idx: int) -> tuple[h.FloatArr, h.DTYPE_FLOAT]:
        """
        Return vector obsevations (unit vector to goal and euclidean distance).
        """
        goal_vec: h.FloatArr = self.env_agents[a_idx].goal_pos - self.agent_positions[a_idx]
        goal_distance: h.DTYPE_FLOAT = (np.linalg.norm(goal_vec)+h.EPSILON).astype(h.DTYPE_FLOAT)
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
                                               begin_obs_radius[1]:end_obs_radius[1]].copy()

        # zero out goals of other agents (only neighbour goals)
        grid_obs[..., -self.args.num_agents:] = 0

        # get indices of agents that are inside local observation
        # use chebyshev distance
        neighbours: h.IntArr = np.where(np.max(np.abs(self.agent_positions -
                                                      self.agent_positions[a_idx]), axis=1)
                               <= self.args.obs_radius)[0]

        for n in neighbours:
            n = cast(h.DTYPE_INT, n)

            # clip neighbour's goals to edge of local observation
            clipped_goal_pos = np.clip(self.env_agents[n].goal_pos,
                                       self.agent_positions[a_idx]-self.args.obs_radius,
                                       self.agent_positions[a_idx]+self.args.obs_radius)
            clipped_goal_pos -= (self.agent_positions[a_idx]-self.observation_offset)
            grid_obs[*clipped_goal_pos, self.grid_offsets.agent_goal_b+n] = 1

        return {"grid_obs": grid_obs,
                "vec_obs": np.concatenate([unit_vec / dist, 
                                           np.array([dist])], axis=0)}

    def set_infos(self, infos: dict[str, Any]):
        """
        Given an empty infos dictionary, sets its infos accordingly to env state.
        """
        assert len(infos) == 0

        for i in range(self.args.num_tasks):
            infos[f"task_{i}"] = self.tasks[i].to_dict(self.task_colors[i])

        for i in range(self.args.num_agents):
            goal_char = h.DEPOT_CHAR if (self.env_agents[i].goal_pos ==
                                         self.depot_position).all() \
                                     else self.task_colors[self.env_agents[i].goal_idx] + \
                                          h.TASK_CHAR
            infos[self.agent_names[i]] = self.env_agents[i].to_dict(pos=self.agent_positions[i],
                                                                    color=self.agent_colors[i],
                                                                    goal_char=goal_char)
        infos["depot_position"] = self.depot_position.tolist()

    def reset(self, *,
              seed: int|None = None,
              options: dict[Any, Any]|None = None) -> tuple[MultiAgentDict, MultiAgentDict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.agent_colors = [h.rand_color(self.rng) for _ in self.agent_idx]
            self.task_colors = [h.rand_color(self.rng) for _ in range(self.args.num_tasks)]

        self.timestep = 0

        self.num_tasks_finished = 0

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

        # generate agent depot position
        depot_position_idx: int = self.rng.choice(num_free_tiles, size=1)[0]
        self.depot_position = self.free_tiles[depot_position_idx]
        free_tiles_mask[depot_position_idx] = False

        # TASKS
        # generate all non-overlapping task positions
        task_positions_idx: h.IntArr = self.rng.choice(np.arange(num_free_tiles)[free_tiles_mask],
                                                       size=self.args.num_tasks,
                                                       replace=False)
        free_tiles_mask[task_positions_idx] = False

        # generate task requirements
        # make sure no task has requirement [0, ..., 0]
        task_requirements: h.IntArr = np.clip(np.eye(N=self.args.num_tasks,
                                                     M=self.args.trait_dim, dtype=h.DTYPE_INT) +
                                              self.rng.choice([0, 1],
                                                              (self.args.num_tasks,
                                                               self.args.trait_dim)),
                                              0, 1, dtype=h.DTYPE_INT)

        for i in range(self.args.num_tasks):
            self.tasks[i].manual_init(requirement=task_requirements[i],
                                      position=self.free_tiles[task_positions_idx[i]])

        # AGENTS
        # generate all non-overlapping agent positions
        # make sure agents and tasks do not overlap either
        self.agent_positions = self.rng.choice(self.free_tiles[free_tiles_mask],
                                               size=self.args.num_agents,
                                               replace=False)

        # generate agent traits
        # make sure agents have the ability to complete each task AND
        # that every agent should at least have one trait
        agent_traits: h.IntArr = np.clip(np.eye(N=self.args.num_agents,
                                                M=self.args.trait_dim,
                                                dtype=h.DTYPE_INT) +
                                         self.rng.choice([0, 1],
                                                         size=(self.args.num_agents,
                                                               self.args.trait_dim),
                                                         p=[1-self.args.agent_capability,
                                                            self.args.agent_capability]),
                                         0, 1, dtype=h.DTYPE_INT)
        summed_traits = np.sum(agent_traits, axis=0)
        for i in np.where(summed_traits == 0)[0]:
            agent_traits[self.rng.choice(self.args.num_agents)][i] = 1

        for i in h.randomly(self.agent_idx, self.rng):
            i = cast(int, i)

            self.set_agent_occ_grid(i, self.agent_positions[i], True)

            self.env_agents[i].set_traits(agent_traits[i])

            goal: int = get_goal(trait_dim=self.args.trait_dim,
                                 agent_trait=agent_traits[i],
                                 num_tasks=self.args.num_tasks,
                                 tasks=self.tasks)
            self.set_agent_goal_grid(i, goal, True)
            self.env_agents[i].set_goal(new_goal_idx=goal,
                                        tasks=self.tasks,
                                        depot_pos=self.depot_position)

        infos: dict[str, Any] = {}
        if self.args.with_debug_infos:
            self.set_infos(infos)

        observations: dict[str, dict[str, dict[str, h.FloatArr]|h.BoolArr]] = {}
        # don't need to randomly iterate because we are assigning obs
        for i in range(self.args.num_agents):
            self.env_agents[i].edit_mask(self.walkable, self.agent_positions[i])
            unit_vec, dist = self.get_vec_obs(i)
            observations[self.agent_names[i]] = {"observation":
                                                 self.get_agent_obs(a_idx=i,
                                                                    unit_vec=unit_vec,
                                                                    dist=dist),
                                                 "action_mask": self.env_agents[i].mask}

        if self.args.render_mode == "human":
            self.render()
            time.sleep(self.args.delay_btw_frames)

        # return observation dict and infos dict.
        return observations, infos # type: ignore

    def step(self, action_dict: MultiAgentDict) -> tuple[MultiAgentDict,  # obs dict
                                                         MultiAgentDict,  # rwds dict
                                                         MultiAgentDict,  # term dict
                                                         MultiAgentDict,  # trunc dict
                                                         MultiAgentDict]: # infos dict
        rewards: dict[str, h.DTYPE_FLOAT] = {a: self.step_penalty for a in self.agent_names}

        # if the probability of zone spreading hits
        if self.rng.random() < self.args.step_spread_prob:
            if self.zone.empty():
                zone_start_pos: h.PositionT = self.rng.choice(self.free_tiles)
                self.zone.spawn(start_pos=zone_start_pos)
                self.set_zone_grid(zone_start_pos, True) # set starting zone inside grid
            else:
                if self.zone.done(): # remove once zone can't spread anymore
                    for tile_pos in self.zone.occupied_tiles:
                        self.set_zone_grid(tile_pos, False)
                    self.zone.reset()
                else:
                    spread_tiles: list[h.PositionT] = self.zone.spread(rng=self.rng,
                                                                       on_zone=self.on_zone,
                                                                       no_wall=self.no_wall,
                                                                       inside_grid=self.inside_grid)
                    for tile_pos in spread_tiles: # set zone inside grid
                        self.set_zone_grid(tile_pos, True)

        for i in h.randomly(self.agent_idx, self.rng):
            self.move_agent(a_idx=i, direction=h.Act_To_Dir[action_dict[self.agent_names[i]]])

        # check for collisions
        # check for goal completion status
        for i in self.agent_idx:
            collidees: h.IntArr = self.agents_on(self.agent_positions[i])

            # agent collides if more than one agent on agent's cell
            if len(collidees) > 1:
                for c in collidees: # we move all the colliding agents back
                    c = cast(int, c)
                    # punish agent for colliding
                    rewards[self.agent_names[c]] = self.collision_penalty

                    # reverse last action
                    self.move_agent(a_idx=c,
                                    direction=h.reverse_dir(action_dict[self.agent_names[c]]))

            # agents should not step into zones
            if self.on_zone(self.agent_positions[i]):
                rewards[self.agent_names[i]] += self.zone_penalty

            if self.env_agents[i].on_goal(self.agent_positions[i]):
                # we don't contribute anything to depot
                if self.env_agents[i].goal_idx == h.AGENT_DEPOT:
                    continue

                # we know agent is on its goal, so contribute traits
                self.tasks[self.env_agents[i].goal_idx].\
                contribute_to_real(self.env_agents[i].traits)

                if self.tasks[self.env_agents[i].goal_idx].is_finished:
                    self.num_tasks_finished += 1

                # agent has contributes to that task (reached goal),
                # now agent needs a new goal
                new_goal_idx: int = get_goal(trait_dim=self.args.trait_dim,
                                             agent_trait=self.env_agents[i].traits,
                                             num_tasks=self.args.num_tasks,
                                             tasks=self.tasks)

                # set new goal in grid
                self.set_agent_goal_grid(i, self.env_agents[i].goal_idx, False)
                self.env_agents[i].set_goal(new_goal_idx=new_goal_idx,
                                            tasks=self.tasks,
                                            depot_pos=self.depot_position)
                self.set_agent_goal_grid(i, self.env_agents[i].goal_idx, True)

        observations: dict[str, dict[str, dict[str, h.FloatArr]|h.BoolArr]] = {}
        for i in range(self.args.num_agents):
            self.env_agents[i].edit_mask(self.walkable, self.agent_positions[i])
            unit_vec, dist = self.get_vec_obs(i)
            observations[self.agent_names[i]] = {"observation":
                                                 self.get_agent_obs(a_idx=i,
                                                                    unit_vec=unit_vec,
                                                                    dist=dist),
                                                 "action_mask": self.env_agents[i].mask}

            # negative normalized distance to goal is added to reward
            rewards[self.agent_names[i]] += -(dist / self.max_goal_dist).astype(h.DTYPE_FLOAT)

        env_termination = False
        if self.num_tasks_finished == self.args.num_tasks:
            env_termination = True

            # we finished the episode by finishing all tasks, so reward agents
            for i in self.agent_idx:
                rewards[self.agent_names[i]] = self.all_tasks_finished_rwd
        terminations = {"__all__": env_termination}

        env_truncation = False
        if self.timestep == self.args.episode_length:
            env_truncation = True
        truncations = {"__all__": env_truncation}

        infos: dict[str, Any] = {}
        if self.args.with_debug_infos:
            self.set_infos(infos)

        if self.args.render_mode == "human":
            self.render()
            time.sleep(self.args.delay_btw_frames)

        return observations, rewards, terminations, truncations, infos # type: ignore

    def render(self) -> None:
        """
        Renders the enviroment.
        """
        task_positions: h.IntArr = np.array([t.position for t in self.tasks],
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
                if self.agent_count_on(pos) == 1:
                    a_idx: int = np.where((self.agent_positions == pos).all(axis=1))[0][0]
                    print(tmp_zone_char + self.agent_colors[a_idx] + h.AGENT_CHAR, end="")
                    continue

                # print if task on tile
                if (task_positions == pos).all(axis=1).any():
                    t_idx: int = np.where((task_positions == pos).all(axis=1))[0][0]
                    if not self.tasks[t_idx].is_finished:
                        print(tmp_zone_char + self.task_colors[t_idx] + h.TASK_CHAR, end="")
                        continue

                # print if depot on tile
                if (self.depot_position == pos).all():
                    print(tmp_zone_char + h.DEPOT_CHAR, end="")
                    continue

                # print if wall on tile
                if not self.no_wall(np.array(pos)):
                    print(tmp_zone_char + h.WALL_CHAR, end="")
                    continue

                # if nothing else, print empty space
                print(tmp_zone_char + " " + h.Style.RESET_ALL, end="")

            print()
