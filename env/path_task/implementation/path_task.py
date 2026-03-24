import functools
import numpy as np
import time
from gymnasium import spaces
from pettingzoo import ParallelEnv
from copy import copy
from env.task import Task
from env.enums import States

class PathTaskEnv(ParallelEnv):
    metadata = {
        "name": "path_task_env_v0",
    }

    def __init__(self,
                 num_agents,
                 num_tasks, exc_time_limits, rwd_limits,
                 trait_dim, episode_length, field_dim, render_mode, 
                 delay_btw_frames, with_task_infos):
        assert num_agents > 0
        assert num_tasks > 0
        assert field_dim > 0
        assert exc_time_limits[0] > 0
        assert trait_dim > 0
        assert episode_length > 0
        assert delay_btw_frames >= 0
        assert render_mode == None or render_mode == "human"
        assert exc_time_limits[0] <= exc_time_limits[1]
        assert rwd_limits[0] <= rwd_limits[1]
        
        # agents
        self.possible_agents = ["agent_"+str(i) for i in range(num_agents)]
        self.agent_positions = None
        self.agent_traits = None
        self.agent_one_hots = [np.array([1 if j == i else 0 for j in range(num_agents)], np.float32) for i in range(num_agents)]

        # tasks
        self.num_tasks = num_tasks
        self.exc_time_limits = exc_time_limits
        self.rwd_limits = rwd_limits
        self.tasks = [Task() for _ in range(num_tasks)]
        self.task_positions = None # this way, step() doesn't need to loop through all task positions every time

        # obs attributes
        # order for one cell: num_agents * (agent_pos_encoding (1) + agent_trait (trait_dim)) + 
        #                     task_requirement (trait_dim) + task_reward (1) + task_execution_time (1) +
        #                     task_execution_progress (1) + task_finished (1)
        self.global_observation = None
        self.trait_dim = trait_dim
        self.field_dim = field_dim
        self.task_enc_begin = num_agents * (1+trait_dim)
        self.task_enc_len = trait_dim+4

        # enviroment variables
        self.episode_length = episode_length
        self.render_mode = render_mode
        self.delay_btw_frames = delay_btw_frames
        self.timestep = None
        self.with_task_infos = with_task_infos

    def on_task(self, pos):
        # to check if a tile contains a task, check if tile's task_execution_time > 0
        return self.global_observation[pos[0], pos[1]][self.task_enc_begin+self.trait_dim+1] > 0

    def edit_action_mask(self, row, col, agent_pos, action_mask):
        if row == 0:
            action_mask[States.UP] = 0
        if col == self.field_dim-1:
            action_mask[States.RIGHT] = 0
        if row == self.field_dim-1:
            action_mask[States.DOWN] = 0
        if col == 0:
            action_mask[States.LEFT] = 0
        if not self.on_task(agent_pos):
            action_mask[States.EXECUTE_TASK] = 0

    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(seed)
        self.timestep = 0
        self.agents = copy(self.possible_agents)

        self.global_observation = np.zeros(shape=(self.field_dim, self.field_dim, 
                                                  self.max_num_agents * (1+self.trait_dim) + self.trait_dim + 8),
                                                  dtype=np.float32)

        for row in range(self.field_dim):
            for col in range(self.field_dim):
                self.global_observation[row, col][self.task_enc_begin+self.task_enc_len:self.task_enc_begin+self.task_enc_len+4] = \
                [int(val) for val in [row == 0, col == self.field_dim-1, row == self.field_dim-1, col == 0]]

        # randomize task attributes
        # make sure tasks cannot be on the same tile
        base_indices = rng.choice(self.field_dim*self.field_dim, size=self.num_tasks, replace=False)
        self.task_positions = [[i // self.field_dim, i % self.field_dim] for i in base_indices]
        task_infos = {}
        after_trait_enc = self.task_enc_begin+self.trait_dim
        for i in range(self.num_tasks):
            self.tasks[i].randomize(rng, self.task_positions[i], 
                                    self.exc_time_limits[0], self.exc_time_limits[1]+1,
                                    self.rwd_limits[0], self.rwd_limits[1]+1,
                                    self.trait_dim)
            self.tasks[i].reset_task_progress()

            x = self.task_positions[i][0]
            y = self.task_positions[i][1]
            
            self.global_observation[x, y][self.task_enc_begin:after_trait_enc] = self.tasks[i].requirement
            self.global_observation[x, y][after_trait_enc] = self.tasks[i].reward
            self.global_observation[x, y][after_trait_enc+1] = self.tasks[i].execution_time
            self.global_observation[x, y][after_trait_enc+2] = self.tasks[i].execution_progress
            self.global_observation[x, y][after_trait_enc+3] = int(self.tasks[i].finished())

            if self.with_task_infos:
                task_infos["task_"+str(i)] = self.tasks[i].attr_dict()

        # initialize action masks
        action_masks = np.ones((self.max_num_agents, self.action_space().n))

        # generate agent traits
        self.agent_traits = rng.choice(a=[0, 1], size=(self.max_num_agents, self.trait_dim))
        # we want to make sure that the agents have the ability to complete each task
        summed_traits = sum(self.agent_traits)
        for i in range(self.trait_dim):
            if summed_traits[i] == 0: # meaning, no agent has ability in this trait
                self.agent_traits[rng.choice(self.max_num_agents)][i] = 1

        self.agent_positions = []
        infos = {}
        for i in range(self.max_num_agents):
            # randomize agent position
            a_pos = [rng.integers(self.field_dim), rng.integers(self.field_dim)]
            self.agent_positions.append(a_pos)

            # encode position encoding and agent traits into obs
            self.global_observation[a_pos[0]][a_pos[1]][i*(self.trait_dim+1):(i+1)*(self.trait_dim+1)] = [1, *self.agent_traits[i]]

            self.edit_action_mask(a_pos[0], a_pos[1], self.agent_positions[i], action_masks[i])

            # edit infos
            infos[self.possible_agents[i]] = {"position": a_pos, "trait": self.agent_traits[i]}

        observations = {}
        for i in range(self.max_num_agents):
            observations[self.possible_agents[i]] = {"observation": 
                                                     {"grid": self.global_observation,
                                                      "one_hot": self.agent_one_hots[i]},
                                                     "action_mask": action_masks[i]}        
        if self.with_task_infos:
            infos.update(task_infos)

        if self.render_mode == "human":
            self.render()
            time.sleep(self.delay_btw_frames)

        return observations, infos

    def step(self, actions):
        rewards = {a: -0.01 for a in self.agents} # initialize rewards for actions taken (smal step penalty)

        # initialize action mask        
        action_masks = np.ones((self.max_num_agents, self.action_space().n))

        # intialize aggregated trait vectors to check task progress
        aggr_trait_vectors = np.zeros(shape=(self.num_tasks, self.trait_dim))

        infos = {}

        for i in range(self.max_num_agents):
            # edit infos
            infos[self.possible_agents[i]] = {"position": self.agent_positions[i], "trait": self.agent_traits[i]}

            # each action taken is just an int, have to map to an actual action
            decoded_action = self.decode_action(actions[self.possible_agents[i]])

            if decoded_action != States.EXECUTE_TASK:
                # agents wants to move

                # 

                # first, remove agent information from currently occupied cell (requirement and pos_encoding)
                self.global_observation[self.agent_positions[i][0], self.agent_positions[i][1]][i*(self.trait_dim+1):(i+1)*(self.trait_dim+1)] = np.zeros(self.trait_dim+1)

                # agent moves to new cell            
                self.agent_positions[i][0] += decoded_action[0]
                self.agent_positions[i][1] += decoded_action[1]

                # add agent information to new cell
                self.global_observation[self.agent_positions[i][0], self.agent_positions[i][1]][i*(self.trait_dim+1):(i+1)*(self.trait_dim+1)] = [1, *self.agent_traits[i]]
                continue

            # agent wants to execute task
            
            # due to action mask, we know agent is on a task, so extract task index
            on_task_index = self.task_positions.index(self.agent_positions[i])

            # record agent traits
            aggr_trait_vectors[on_task_index] += self.agent_traits[i] 

        task_infos = {}
        # if requirement of task is met, progress task execution only when task isn't finished
        for i in range(self.num_tasks):
            x = self.task_positions[i][0]
            y = self.task_positions[i][1]

            if all((aggr_trait_vectors[i][j] >= self.tasks[i].requirement[j]) and 
                   not self.tasks[i].finished() for j in range(self.trait_dim)):
                self.tasks[i].execution_progress += 1
                self.global_observation[x, y][self.task_enc_begin+self.trait_dim+2] = self.tasks[i].execution_progress

            ### REWARD CALCULATION
            if self.tasks[i].finished() and not self.tasks[i].reward_given:
                self.global_observation[x, y][self.task_enc_begin+self.trait_dim+3] = 1
                for j in range(self.max_num_agents):
                    if self.agent_positions[j] == self.task_positions[i]:
                        rewards[self.possible_agents[j]] += self.tasks[i].reward
                self.tasks[i].reward_given = True
            ### REWARD CALCULATION

            if self.with_task_infos:
                task_infos["task_"+str(i)] = self.tasks[i].attr_dict()

        for i in range(self.max_num_agents):
            # restrict movement (field edge + walls) and "execute task"
            self.edit_action_mask(self.agent_positions[i][0], self.agent_positions[i][1], self.agent_positions[i], action_masks[i])

            if action_masks[i][4] == 0:
                continue

            # now we know agent is on tile of a task, so extract task index
            on_task_index = self.task_positions.index(self.agent_positions[i])

            # check if task is finished; if it is, restrict action "execute task"
            if self.tasks[on_task_index].finished():
                action_masks[i][4] = 0
                continue

        env_termination = False
        if all(t.finished() for t in self.tasks):
            env_termination = True
        terminations = {a: env_termination for a in self.agents}

        env_truncation = False
        if self.timestep >= self.episode_length:
            env_truncation = True
        truncations = {a: env_truncation for a in self.agents}

        if env_termination or env_truncation:
            self.agents = []

        self.timestep += 1   

        observations = {}
        for i in range(self.max_num_agents):
            observations[self.possible_agents[i]] = {"observation": 
                                                     {"grid": self.global_observation,
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
        # print top edge
        print("*", end="")
        for col in range(self.field_dim):
            print("---", end="")
            print("*", end="")
        print()

        # print row-wise
        for row in range(self.field_dim):
            print("|", end="")
            for col in range(self.field_dim):
                if [row, col] in self.agent_positions:
                    print(" A ", end="")
                elif self.on_task([row, col]) and not self.tasks[self.task_positions.index([row, col])].finished():
                    print(" T ", end="")
                else:
                    print("   ", end="")
                if col < self.field_dim-1:
                    print(" ", end="")
                else:
                    print("|", end="")
            print()
            print("*", end="")
            for col in range(self.field_dim):
                if row == self.field_dim-1:
                    print("---", end="") # bottom edge
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
