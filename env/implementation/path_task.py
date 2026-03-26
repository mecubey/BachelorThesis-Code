import functools
import numpy as np
import time
from gymnasium import spaces
from pettingzoo import ParallelEnv
from copy import copy
from colorama import Fore, Back, Style
from .task import Task
from .maze import generate_maze, is_dir_avail
from .enums import States
from .zone import Zone

class PathTaskEnv(ParallelEnv):
    metadata = {
        "name": "path_task_env_v0",
    }

    def __init__(self,
                 num_agents,
                 num_tasks, exc_time_limits, rwd_limits, maze_intensity, 
                 with_zone, max_num_spread, step_spread_prob, spread_probs, 
                 zone_dmg, trait_dim, episode_length, field_dim, 
                 render_mode, delay_btw_frames, with_task_infos):
        assert num_agents > 0
        assert num_tasks > 0
        assert field_dim > 0
        assert exc_time_limits[0] > 0
        assert trait_dim > 0
        assert episode_length > 0
        assert delay_btw_frames >= 0
        assert maze_intensity >= 0
        assert render_mode == None or render_mode == "human"
        assert exc_time_limits[0] <= exc_time_limits[1]
        assert rwd_limits[0] <= rwd_limits[1]
        
        # agents
        self.possible_agents = ["agent_"+str(i) for i in range(num_agents)]
        self.agent_positions = None
        self.agent_traits = None
        self.agent_one_hots = [[1 if j == i else 0 for j in range(num_agents)] for i in range(num_agents)]

        # tasks
        self.num_tasks = num_tasks
        self.exc_time_limits = exc_time_limits
        self.rwd_limits = rwd_limits
        self.tasks = [Task() for _ in range(num_tasks)]
        self.task_positions = None # this way, step() doesn't need to loop through all task positions every time
        self.num_tasks_finished = 0

        # maze
        self.maze = None
        self.maze_intensity = maze_intensity

        # zone
        self.zone = Zone(max_num_spread, spread_probs, zone_dmg)
        self.with_zone = with_zone
        self.step_spread_prob = step_spread_prob

        # obs attributes
        # order for one cell: num_agents * (agent_pos_encoding (1) + agent_trait (trait_dim)) + 
        #                     task_requirement (trait_dim) + task_reward (1) + task_execution_time (1) +
        #                     task_execution_progress (1) + task_finished (1) + avail_directions (4) + zone (1)
        self.global_obs = None
        self.trait_dim = trait_dim
        self.field_dim = field_dim
        all_agent_attr = num_agents*(1 + trait_dim)
        self.offsets = {
            "agent_attr": 1 + trait_dim,
            "task_req": [all_agent_attr, all_agent_attr + trait_dim],
            "task_rwd": all_agent_attr + trait_dim,
            "task_exc_time": all_agent_attr + trait_dim+1,
            "task_exc_prg": all_agent_attr + trait_dim+2,
            "task_fin": all_agent_attr + trait_dim+3,
            "dirs": [all_agent_attr + trait_dim+4, all_agent_attr + trait_dim+8],
            "zone": all_agent_attr + trait_dim+8
        }

        # enviroment variables
        self.episode_length = episode_length
        self.render_mode = render_mode
        self.delay_btw_frames = delay_btw_frames
        self.timestep = None
        self.with_task_infos = with_task_infos
        self.rng = np.random.default_rng()
        self.agent_char = Fore.GREEN + "A" + Fore.RESET
        self.coalition_char = Fore.GREEN + "L" + Fore.RESET
        self.task_char = Fore.BLUE + "T" + Fore.RESET
        self.zone_color = Back.RED

    def on_task(self, pos):
        # to check if a tile contains a task, check if tile's task_execution_time > 0
        return self.global_obs[pos[0], pos[1]][self.offsets["task_exc_time"]] > 0

    def on_zone(self, pos):
        return self.global_obs[pos[0], pos[1]][self.offsets["zone"]] == 1

    def edit_action_mask(self, pos, action_mask):
        action_mask[0:self.action_space().n-1] = self.global_obs[pos[0], pos[1]][self.offsets["dirs"][0]:self.offsets["dirs"][1]]
        if not self.on_task(pos):
            action_mask[States.EXECUTE_TASK] = 0
        if self.global_obs[pos[0], pos[1]][self.offsets["task_fin"]] == 1:
            action_mask[States.EXECUTE_TASK] = 0

    def reset(self, seed=None, options=None):
        if not (seed == None):
            self.rng = np.random.default_rng(seed)
        self.timestep = 0
                  
        self.agents = copy(self.possible_agents)

        if not self.with_zone: # if spread_prob == 0, it means we don't want a zone at all
            self.global_obs = np.zeros(shape=(self.field_dim, self.field_dim, self.offsets["zone"]))
        else:
            self.global_obs = np.zeros(shape=(self.field_dim, self.field_dim, self.offsets["zone"]+1))

        # generate maze
        self.maze = generate_maze(self.field_dim, self.maze_intensity, seed)
        for row in range(self.field_dim):
            for col in range(self.field_dim):
                self.global_obs[row, col][self.offsets["dirs"][0]:self.offsets["dirs"][1]] = \
                [val for val in [not row == 0 and is_dir_avail(self.maze, [row, col], States.UP),
                                 not col == self.field_dim-1 and is_dir_avail(self.maze, [row, col], States.RIGHT),
                                 not row == self.field_dim-1 and is_dir_avail(self.maze, [row, col], States.DOWN),
                                 not col == 0 and is_dir_avail(self.maze, [row, col], States.LEFT)]]

        # randomize task attributes
        # make sure tasks cannot be on the same tile
        self.num_tasks_finished = 0
        base_indices = self.rng.choice(self.field_dim*self.field_dim, size=self.num_tasks, replace=False)
        self.task_positions = [[i // self.field_dim, i % self.field_dim] for i in base_indices]
        task_infos = {}
        for i in range(self.num_tasks):
            self.tasks[i].randomize(self.rng, self.task_positions[i], 
                                    self.exc_time_limits[0], self.exc_time_limits[1]+1,
                                    self.rwd_limits[0], self.rwd_limits[1]+1,
                                    self.trait_dim)
            self.tasks[i].reset_task_progress()

            x = self.task_positions[i][0]
            y = self.task_positions[i][1]
            
            self.global_obs[x, y][self.offsets["task_req"][0]:self.offsets["task_req"][1]] = self.tasks[i].requirement
            self.global_obs[x, y][self.offsets["task_rwd"]] = self.tasks[i].reward
            self.global_obs[x, y][self.offsets["task_exc_time"]] = self.tasks[i].execution_time
            self.global_obs[x, y][self.offsets["task_exc_prg"]] = self.tasks[i].execution_progress
            self.global_obs[x, y][self.offsets["task_fin"]] = int(self.tasks[i].finished())

            if self.with_task_infos:
                task_infos["task_"+str(i)] = self.tasks[i].attr_dict()

        # initialize action masks
        action_masks = np.ones((self.max_num_agents, self.action_space().n))

        # generate agent traits
        self.agent_traits = self.rng.choice(a=[0, 1], size=(self.max_num_agents, self.trait_dim))
        # we want to make sure that the agents have the ability to complete each task
        summed_traits = sum(self.agent_traits)
        for i in range(self.trait_dim):
            if summed_traits[i] == 0: # meaning, no agent has ability in this trait
                self.agent_traits[self.rng.choice(self.max_num_agents)][i] = 1

        self.agent_positions = []
        infos = {}
        for i in range(self.max_num_agents):
            # randomize agent position
            a_pos = [self.rng.integers(self.field_dim), self.rng.integers(self.field_dim)]
            self.agent_positions.append(a_pos)

            # encode position encoding and agent traits into obs
            self.global_obs[a_pos[0], a_pos[1]][i*self.offsets["agent_attr"]:(i+1)*self.offsets["agent_attr"]] = \
            [1, *self.agent_traits[i]]

            self.edit_action_mask(a_pos, action_masks[i])

            # edit infos
            infos[self.possible_agents[i]] = {"position": a_pos, "trait": self.agent_traits[i]}

        observations = {}
        for i in range(self.max_num_agents):
            observations[self.possible_agents[i]] = {"observation": 
                                                     {"grid": self.global_obs,
                                                      "one_hot": self.agent_one_hots[i]},
                                                     "action_mask": action_masks[i]}                
        if self.with_task_infos:
            infos.update(task_infos)

        if self.render_mode == "human":
            self.render()
            time.sleep(self.delay_btw_frames)

        return observations, infos

    def step(self, actions):
        rewards = {a: -0.01 for a in self.agents} # initialize rewards for actions taken

        # initialize action mask        
        action_masks = np.ones((self.max_num_agents, self.action_space().n))

        # intialize aggregated trait vectors to check task progress
        aggr_trait_vectors = np.zeros(shape=(self.num_tasks, self.trait_dim))

        infos = {}

        # spread in this timestep according to probability
        if self.with_zone and self.rng.random() <= self.step_spread_prob:
            if self.zone.empty():
                self.zone.spawn(self.rng.integers(low=self.field_dim, size=2), self.global_obs, self.offsets["zone"])
            else:
                self.zone.spread(self.global_obs, self.offsets["dirs"], self.offsets["zone"], self.rng)
                if self.zone.finished():
                    self.zone.remove(self.global_obs, self.offsets["zone"])

        for i in range(self.max_num_agents):
            # edit infos
            infos[self.possible_agents[i]] = {"position": self.agent_positions[i], "trait": self.agent_traits[i]}

            # each action taken is just an int, have to map to an actual action
            decoded_action = self.decode_action(actions[self.possible_agents[i]])

            if decoded_action != States.EXECUTE_TASK:
                # agents wants to move

                # first, remove agent information from currently occupied cell (requirement and pos_encoding)
                self.global_obs[self.agent_positions[i][0], self.agent_positions[i][1]] \
                [i*self.offsets["agent_attr"]:(i+1)*self.offsets["agent_attr"]] = np.zeros(self.trait_dim+1)

                # agent moves to new cell            
                self.agent_positions[i][0] += decoded_action[0]
                self.agent_positions[i][1] += decoded_action[1]

                # add agent information to new cell
                self.global_obs[self.agent_positions[i][0], self.agent_positions[i][1]] \
                [i*self.offsets["agent_attr"]:(i+1)*self.offsets["agent_attr"]] = [1, *self.agent_traits[i]]
                continue

            # agent wants to execute task
            
            # due to action mask, we know agent is on a task, so extract task index
            on_task_index = self.task_positions.index(self.agent_positions[i])

            # record agent traits
            aggr_trait_vectors[on_task_index] += self.agent_traits[i] 

        task_infos = {}
        # if requirement of task is met, progress task execution only when task isn't finished
        for i in range(self.num_tasks):
            row = self.task_positions[i][0]
            col = self.task_positions[i][1]

            if all((aggr_trait_vectors[i][j] >= self.tasks[i].requirement[j]) and 
                   not self.tasks[i].finished() for j in range(self.trait_dim)):
                self.tasks[i].execution_progress += 1
                self.global_obs[row, col][self.offsets["task_exc_prg"]] = self.tasks[i].execution_progress

            ### REWARD CALCULATION
            if self.tasks[i].finished() and not self.tasks[i].reward_given:
                self.global_obs[row, col][self.offsets["task_fin"]] = 1
                for j in range(self.max_num_agents):
                    if self.agent_positions[j] == self.task_positions[i]:
                        rewards[self.possible_agents[j]] += self.tasks[i].reward
                self.tasks[i].reward_given = True
                self.num_tasks_finished += 1
            ### REWARD CALCULATION

            if self.with_task_infos:
                task_infos["task_"+str(i)] = self.tasks[i].attr_dict()

        for i in range(self.max_num_agents):
            # check if on zone tile
            if self.with_zone and self.on_zone(self.agent_positions[i]):
                rewards[self.possible_agents[i]] += self.zone.zone_dmg

            # restrict movement (field edge + walls) and "execute task"
            self.edit_action_mask(self.agent_positions[i], action_masks[i])

        env_termination = False
        if self.num_tasks_finished == self.num_tasks:
            env_termination = True
        terminations = {a: env_termination for a in self.agents}

        env_truncation = False
        if self.timestep == self.episode_length:
            env_truncation = True
        truncations = {a: env_truncation for a in self.agents}

        if env_termination or env_truncation:
            self.agents = []

        self.timestep += 1   

        observations = {}
        for i in range(self.max_num_agents):
            observations[self.possible_agents[i]] = {"observation": 
                                                     {"grid": self.global_obs,
                                                      "one_hot": self.agent_one_hots[i]},
                                                     "action_mask": action_masks[i]}                
        if self.with_task_infos:
            infos.update(task_infos)

        if self.render_mode == "human":
            self.render()
            time.sleep(self.delay_btw_frames)
        
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
                if not self.global_obs[row, col][self.offsets["dirs"][0]:self.offsets["dirs"][1]][States.UP]:
                    print("---", end="")
                else:
                    print("   ", end="")
                print("*", end="")

            print()

            for col in range(self.field_dim):
                num_space = 1
                # check if left walls exist
                if not self.global_obs[row, col][self.offsets["dirs"][0]:self.offsets["dirs"][1]][States.LEFT]:
                    print("|", end="")
                    num_space -= 1

                cell_str = [" ", " ", " "]
                
                # store agent char
                agent_count = self.agent_positions.count([row, col])
                cell_agent_char = self.agent_char
                if agent_count > 1:
                    cell_agent_char = self.coalition_char
                if agent_count >= 1:
                    cell_str[1] = cell_agent_char

                # handle task char, task char + agent char
                if self.on_task([row, col]) and not self.global_obs[row, col][self.offsets["task_fin"]]:
                    if cell_str[1] == cell_agent_char:
                        cell_str = [self.task_char, " ", cell_agent_char]
                    else:
                        cell_str[1] = self.task_char

                cell_str = "".join(cell_str)
                
                if self.with_zone and self.global_obs[row, col][self.offsets["zone"]]:
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
                    if not self.global_obs[row, col][self.offsets["dirs"][0]:self.offsets["dirs"][1]][States.DOWN]:
                        print("---", end="")
                    else:
                        print("   ", end="")
                    print("*", end="")
                print()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent = None):
        return spaces.Dict({"grid": spaces.Box(
                                low=-np.inf,
                                high=np.inf,
                                shape=(self.field_dim, self.field_dim, 
                                       self.max_num_agents * (1+self.trait_dim) + self.trait_dim + 8),
                                dtype=np.double
                            ),
                            "one_hot": spaces.Box(
                                low=0,
                                high=1,
                                shape=(self.max_num_agents,),
                                dtype=np.double
                            )})
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent = None):
        return spaces.Discrete(5)