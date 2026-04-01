import functools
import numpy as np
import time
from itertools import product
from gymnasium import spaces
from pettingzoo import ParallelEnv
from copy import copy
from colorama import Fore, Back
from .task import Task
from .maze import generate_maze, is_dir_avail
from .enums import States
from .zone import Zone

class PathTaskEnv(ParallelEnv):
    metadata = {
        "name": "path_task_env_v0",
    }

    def __init__(self,
                 num_agents, num_tasks, obs_radius, step_rwd,
                 exc_time_limits, rwd_limits, maze_intensity,
                 with_zone, max_num_spread, step_spread_prob, spread_probs, 
                 zone_dmg, trait_dim, episode_length, field_dim, 
                 render_mode, delay_btw_frames, with_task_infos):
        # agents
        self.possible_agents = ["agent_"+str(i) for i in range(num_agents)]
        self.agents = copy(self.possible_agents)
        self.agent_positions = None
        self.agent_traits = None
        self.agent_one_hots =  np.eye(num_agents, dtype=np.float32)[np.arange(num_agents)]
        self.action_masks = np.zeros((self.num_agents, self.action_space().n))

        # tasks
        self.num_tasks = num_tasks
        self.exc_time_limits = exc_time_limits
        self.rwd_limits = rwd_limits
        self.tasks = np.array([Task() for _ in range(num_tasks)])
        self.task_positions = None # easier access to task positions
        self.task_pos_to_idx = None
        self.num_tasks_finished = 0

        # maze
        self.maze_intensity = maze_intensity

        # zone
        self.zone = Zone(max_num_spread, spread_probs, zone_dmg)
        self.with_zone = with_zone
        self.step_spread_prob = step_spread_prob

        # obs attributes
        self.field_dim = field_dim
        self.obs_radius = obs_radius
        self.trait_dim = trait_dim

        # single cell: up, right, down, left, is_task, agent_occ, is_zone
        if with_zone:
            self.padded_grid_obs = np.zeros(shape=(field_dim+obs_radius*2, field_dim+obs_radius*2, 7), dtype=np.float32)
        else:
            self.padded_grid_obs = np.zeros(shape=(field_dim+obs_radius*2, field_dim+obs_radius*2, 6), dtype=np.float32)
        self.grid_offsets = {"dirs": [0, 4],
                             "is_task": 4,
                             "agent_occ": 5,
                             "is_zone": 6}

        # single task: dx, dy, req, rem_req, rwd, exc_time, exc_prg, task_fin
        self.task_enc_len = 6+2*trait_dim
        self.task_obs = np.zeros(shape=(num_agents, num_tasks*self.task_enc_len), dtype=np.float32) 
        req_offsets = 2+2*trait_dim
        self.task_offsets = {"x": 0,
                             "y": 1,
                             "req": [2, req_offsets-trait_dim],
                             "rem_req": [req_offsets-trait_dim, req_offsets],
                             "rwd": req_offsets,
                             "exc_time": req_offsets+1,
                             "exc_prg": req_offsets+2,
                             "fin": req_offsets+3}

        # single agent: dx, dy, traits
        self.agent_enc_len = 2+trait_dim
        self.agent_obs = np.zeros(shape=(num_agents, num_agents*self.agent_enc_len))
        self.agent_offsets = {"x": 0,
                              "y": 1,
                              "traits": [2, 2+trait_dim]}

        # enviroment variables
        self.step_rwd = step_rwd
        self.episode_length = episode_length
        self.timestep = None
        self.with_task_infos = with_task_infos
        self.rng = np.random.default_rng()

        # rendering
        self.render_mode = render_mode
        self.delay_btw_frames = delay_btw_frames
        self.agent_char = Fore.GREEN + "A" + Fore.RESET
        self.coalition_char = Fore.GREEN + "L" + Fore.RESET
        self.task_char = Fore.BLUE + "T" + Fore.RESET
        self.zone_color = Back.RED

    def get_padded_grid_obs_inds(self, i, j):
        return int(i+self.obs_radius), int(j+self.obs_radius)

    def set_grid_agent_occupancy(self, i, val):
        self.padded_grid_obs[*self.get_padded_grid_obs_inds(self.agent_positions[i][0],
                                                            self.agent_positions[i][1]), self.grid_offsets["agent_occ"]] = val

    def edit_grid_agent_occupancy(self, i, val):
        self.padded_grid_obs[*self.get_padded_grid_obs_inds(self.agent_positions[i][0],
                                                            self.agent_positions[i][1]), self.grid_offsets["agent_occ"]] += val

    def set_grid_task(self, i, val):
        self.padded_grid_obs[*self.get_padded_grid_obs_inds(self.task_positions[i][0],
                                                            self.task_positions[i][1]), self.grid_offsets["is_task"]] = val

    def set_agent_rel_positions(self, i):
        agent_rel_positions = self.agent_positions - self.agent_positions[i]
        self.agent_obs[i, self.agent_offsets["x"]::self.agent_enc_len] = agent_rel_positions[:, 0]
        self.agent_obs[i, self.agent_offsets["y"]::self.agent_enc_len] = agent_rel_positions[:, 1]

    def edit_action_mask(self, i):
        grid_pos = self.get_padded_grid_obs_inds(self.agent_positions[i][0], self.agent_positions[i][1])
        self.action_masks[i, 0:4] = self.padded_grid_obs[*grid_pos, self.grid_offsets["dirs"][0]:self.grid_offsets["dirs"][1]]
        self.action_masks[i, 4] = self.on_task(grid_pos)

    def set_task_rel_positions(self, i):
        task_rel_positions = self.task_positions - self.agent_positions[i]
        self.task_obs[i, self.task_offsets["x"]::self.task_enc_len] = task_rel_positions[:, 0]
        self.task_obs[i, self.task_offsets["y"]::self.task_enc_len] = task_rel_positions[:, 1]

    def set_rem_req(self, i, new_req):
        t_start = i*self.task_enc_len
        self.task_obs[:, t_start+self.task_offsets["rem_req"][0]:t_start+self.task_offsets["rem_req"][1]] = new_req
            
    def set_exc_prg(self, i):
        self.task_obs[:, i*self.task_enc_len+self.task_offsets["exc_prg"]] = self.tasks[i].execution_progress
            
    def set_task_fin(self, i):
        self.task_obs[:, i*self.task_enc_len+self.task_offsets["fin"]] = self.tasks[i].finished()

    def on_task(self, pos):
        return self.padded_grid_obs[*pos, self.grid_offsets["is_task"]] == 1

    def on_zone(self, pos):
        return self.padded_grid_obs[*pos, self.grid_offsets["is_zone"]] == 1

    def reset(self, seed=None, options=None):
        if not (seed == None):
            self.rng = np.random.default_rng(seed)
        self.timestep = 0
                  
        self.agents = copy(self.possible_agents)

        # generate maze
        maze = generate_maze(self.field_dim, self.maze_intensity, seed)
        
        # set walls 
        for row in range(self.field_dim):
            for col in range(self.field_dim):
                pos = self.get_padded_grid_obs_inds(row, col)

                # reset grid correctly upon double env reset
                self.padded_grid_obs[*pos, self.grid_offsets["is_task"]] = 0
                self.padded_grid_obs[*pos, self.grid_offsets["agent_occ"]] = 0

                self.padded_grid_obs[*pos, self.grid_offsets["dirs"][0]:self.grid_offsets["dirs"][1]] = \
                [not row == 0 and is_dir_avail(maze, (row, col), States.UP),
                not col == self.field_dim-1 and is_dir_avail(maze, (row, col), States.RIGHT),
                not row == self.field_dim-1 and is_dir_avail(maze, (row, col), States.DOWN),
                not col == 0 and is_dir_avail(maze, (row, col), States.LEFT)]

        # randomize task attributes
        # make sure tasks cannot be on the same tile
        self.num_tasks_finished = 0
        base_indices = np.arange(self.field_dim, dtype=np.float32)
        self.task_positions = self.rng.choice(list(product(base_indices, base_indices)), size=(self.num_tasks))
        self.task_pos_to_idx = {}
        task_infos = {}
        for i in range(self.num_tasks):
            self.task_pos_to_idx[tuple(self.task_positions[i])] = i

            self.tasks[i].randomize(self.rng, self.task_positions[i], 
                                    self.exc_time_limits[0], self.exc_time_limits[1],
                                    self.rwd_limits[0], self.rwd_limits[1],
                                    self.trait_dim)
            self.tasks[i].reset_task_progress()

            # record in grid
            self.set_grid_task(i, 1)

            # edit agent-task observations
            t_start = i*self.task_enc_len
            self.task_obs[:, t_start+self.task_offsets["req"][0]:t_start+self.task_offsets["req"][1]] = self.tasks[i].requirement
            self.set_rem_req(i, self.tasks[i].requirement)
            self.task_obs[:, t_start+self.task_offsets["rwd"]] = self.tasks[i].reward
            self.task_obs[:, t_start+self.task_offsets["exc_time"]] = self.tasks[i].execution_time
            self.set_exc_prg(i)
            self.set_task_fin(i)

            if self.with_task_infos:
                task_infos["task_"+str(i)] = self.tasks[i].attr_dict()

        # generate random agent traits
        self.agent_traits = self.rng.choice(a=[0, 1], size=(self.num_agents, self.trait_dim))
        # we want to make sure that the agents have the ability to complete each task
        summed_traits = np.sum(self.agent_traits, axis=0)
        for i in range(self.trait_dim):
            if summed_traits[i] == 0: # meaning, no agent has ability in this trait
                self.agent_traits[self.rng.choice(self.num_agents)][i] = 1

        # generate random agent positions
        self.agent_positions = self.rng.integers(low=self.field_dim, size=(self.num_agents, 2))

        infos = {}
        for i in range(self.num_agents):
            # encode agent position into grid
            self.edit_grid_agent_occupancy(i, 1)

            # edit agent-agent observation
            self.set_agent_rel_positions(i)

            # set agent traits
            for j in range(self.num_agents):
                self.agent_obs[i, j*self.agent_enc_len+self.agent_offsets["traits"][0]:
                                  j*self.agent_enc_len+self.agent_offsets["traits"][1]] = self.agent_traits[j]

            # edit task-agent observation
            self.set_task_rel_positions(i)

            self.edit_action_mask(i)

            # edit infos
            infos[self.possible_agents[i]] = {"position": self.agent_positions[i], "trait": self.agent_traits[i]}

        observations = {}
        for i in range(self.num_agents):
            pos = tuple(self.agent_positions[i])
            grid_pos = self.get_padded_grid_obs_inds(*pos)
            
            observations[self.possible_agents[i]] = \
            {"action_mask": self.action_masks[i].copy(),
            "observation":
            {"vec_obs": np.concatenate([self.agent_one_hots[i], self.agent_obs[i], self.task_obs[i]], dtype=np.float32),
            "grid_obs": self.padded_grid_obs[pos[0]:grid_pos[0]+self.obs_radius+1,
                                             pos[1]:grid_pos[1]+self.obs_radius+1, :].copy()}}

        if self.with_task_infos:
            infos.update(task_infos)

        if self.render_mode == "human":
            self.render()
            time.sleep(self.delay_btw_frames)

        return observations, infos

    def step(self, actions):
        rewards = {a: self.step_rwd for a in self.agents} # initialize rewards for actions taken in this timestep

        # intialize aggregated trait vectors to check task progress
        aggr_trait_vectors = np.zeros(shape=(self.num_tasks, self.trait_dim))

        infos = {}

        # spread in this timestep according to probability
        if self.with_zone and self.rng.random() <= self.step_spread_prob:
            if self.zone.empty():
                self.zone.spawn(tuple(self.rng.integers(low=self.field_dim, size=2)+self.obs_radius), 
                                self.padded_grid_obs, self.grid_offsets["is_zone"])
            else:
                self.zone.spread(self.padded_grid_obs, self.grid_offsets["dirs"], self.grid_offsets["is_zone"], self.rng)
                if self.zone.finished():
                    self.zone.remove(self.padded_grid_obs, self.grid_offsets["is_zone"])

        for i in range(self.num_agents):
            # edit infos
            infos[self.possible_agents[i]] = {"position": self.agent_positions[i], "trait": self.agent_traits[i]}

            # each action taken is just an int, have to map to an actual action
            decoded_action = self.decode_action(actions[self.possible_agents[i]])

            if decoded_action != States.EXECUTE_TASK:
                # agents wants to move

                # agent does not occupy previous cell anymore
                self.edit_grid_agent_occupancy(i, -1)
                
                # agent moves to new cell            
                self.agent_positions[i][0] += decoded_action[0]
                self.agent_positions[i][1] += decoded_action[1]

                # agent now occupies new cell
                self.edit_grid_agent_occupancy(i, 1)
                continue

            # agent wants to execute task
            
            # due to action mask, we know agent is on a task, so extract task index
            on_task_index = self.task_pos_to_idx[tuple(self.agent_positions[i])]

            # record agent traits
            aggr_trait_vectors[on_task_index] += self.agent_traits[i] 

        task_infos = {}
        # if requirement of task is met, progress task execution only when task isn't finished
        for i in range(self.num_tasks):
            self.set_rem_req(i, self.tasks[i].get_remaining_req(aggr_trait_vectors[i]))
            self.tasks[i].record_aggr_traits(aggr_trait_vectors[i])

            if (aggr_trait_vectors[i] >= self.tasks[i].requirement).all() and not self.tasks[i].finished():
                self.tasks[i].progress()
                self.set_exc_prg(i)

            if self.tasks[i].finished() and not self.tasks[i].reward_given:
                self.set_task_fin(i)
                self.set_grid_task(i, 0) # remove task from grid once finished
                for j in range(self.num_agents):
                    if (self.agent_positions[j] == self.task_positions[i]).all():
                        rewards[self.possible_agents[j]] += self.tasks[i].reward
                self.tasks[i].reward_given = True
                self.num_tasks_finished += 1

            if self.with_task_infos:
                task_infos["task_"+str(i)] = self.tasks[i].attr_dict()

        env_termination = False
        if self.num_tasks_finished == self.num_tasks:
            env_termination = True
        terminations = {}

        env_truncation = False
        if self.timestep == self.episode_length:
            env_truncation = True
        truncations = {}

        done = env_termination or env_truncation

        observations = {}

        for i in range(self.num_agents):
            pos = tuple(self.agent_positions[i])
            grid_pos = self.get_padded_grid_obs_inds(*pos)

            # check if on zone tile
            if self.with_zone and self.on_zone(grid_pos):
                rewards[self.possible_agents[i]] += self.zone.zone_dmg

            self.edit_action_mask(i)

            self.set_task_rel_positions(i)
            self.set_agent_rel_positions(i)

            terminations[self.possible_agents[i]] = env_termination
            
            truncations[self.possible_agents[i]] = env_truncation
            
            observations[self.possible_agents[i]] = \
            {"action_mask": self.action_masks[i].copy(),
            "observation":
            {"vec_obs": np.concatenate([self.agent_one_hots[i], self.agent_obs[i], self.task_obs[i]], dtype=np.float32),
            "grid_obs": self.padded_grid_obs[pos[0]:grid_pos[0]+self.obs_radius+1,
                                             pos[1]:grid_pos[1]+self.obs_radius+1, :].copy()}}

        if self.render_mode == "human":
            self.render()
            time.sleep(self.delay_btw_frames)

        if done:
            eps_rwd_sum = 0
            for i in range(self.num_tasks):
                self.set_grid_task(i, 0) # zero out tasks on grid
                eps_rwd_sum += self.tasks[i].get_episode_reward()

            eps_punisment = -eps_rwd_sum/self.num_tasks * self.timestep/self.episode_length

            for i in range(self.num_agents):
                self.set_grid_agent_occupancy(i, 0) # zero out agents on grid
                rewards[self.possible_agents[i]] += eps_punisment

            self.agents = []

            self.zone.remove(self.padded_grid_obs, self.grid_offsets["is_zone"])

        self.timestep += 1   
            
        if self.with_task_infos:
            infos.update(task_infos)
        
        return observations, rewards, terminations, truncations, infos

    def decode_action(self, action):
        if action == States.UP:
            return [-1, 0]
        elif action == States.RIGHT:
            return [0, 1]
        elif action == States.DOWN:
            return [1, 0]
        elif action == States.LEFT:
            return [0, -1] 
        elif action == States.EXECUTE_TASK:
            return action
        
        raise Exception("Invalid action.")

    def render(self):
        for row in range(self.field_dim):
            print("*", end="")
            for col in range(self.field_dim):
                # check if upper walls exist
                if not self.padded_grid_obs[*self.get_padded_grid_obs_inds(row, col), States.UP]:
                    print("---", end="")
                else:
                    print("   ", end="")
                print("*", end="")

            print()

            for col in range(self.field_dim):
                num_space = 1
                # check if left walls exist
                pos = self.get_padded_grid_obs_inds(row, col)
                if not self.padded_grid_obs[*pos, States.LEFT]:
                    print("|", end="")
                    num_space -= 1

                cell_str = [" ", " ", " "]
                
                # store agent char
                agent_count = np.sum((self.agent_positions == [row, col]).all(axis=1))
                cell_agent_char = self.agent_char
                if agent_count > 1:
                    cell_agent_char = self.coalition_char
                if agent_count >= 1:
                    cell_str[1] = cell_agent_char

                # handle task char, task char + agent char
                if self.on_task(pos):
                    if cell_str[1] == cell_agent_char:
                        cell_str = [self.task_char, " ", cell_agent_char]
                    else:
                        cell_str[1] = self.task_char

                cell_str = "".join(cell_str)
                
                if self.with_zone and self.padded_grid_obs[*pos, self.grid_offsets["is_zone"]]:
                    # add zone as background color
                    cell_str = self.zone_color + cell_str + Back.RESET

                cell_str = " "*num_space + cell_str

                print(cell_str, end="")

            # check last right wall of current row
            print("|")

            # check bottom wall of last row
            if row == self.field_dim-1:
                print("*", end="")
                for col in range(self.field_dim):
                    if not self.padded_grid_obs[*self.get_padded_grid_obs_inds(row, col), States.DOWN]:
                        print("---", end="")
                    else:
                        print("   ", end="")
                    print("*", end="")
                print()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent = None):
        return spaces.Dict({"vec_obs": spaces.Box(
                                low=-np.inf,
                                high=np.inf,
                                shape=(self.num_agents + np.prod(self.agent_obs.shape) + np.prod(self.task_obs.shape),),
                                dtype=np.float32
                            ),
                            "grid_obs": spaces.Box(
                                low=-np.inf,
                                high=np.inf,
                                shape=(self.obs_radius, self.obs_radius, 6 + int(self.with_zone)),
                                dtype=np.float32
                            )})
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent = None):
        return spaces.Discrete(5)