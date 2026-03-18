import functools
import numpy as np
import time
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv
from copy import copy
from itertools import chain
from gymnasium.spaces import MultiDiscrete, Box
from .task import Task

class PathTaskEnv(ParallelEnv):
    metadata = {
        "name": "path_task_env_v0",
    }

    def __init__(self,
                 num_agents,
                 num_tasks, exc_time_limits, rwd_limits,
                 trait_dim, episode_length, field_dim, render_mode, delay_btw_frames, with_task_infos):
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

        # tasks
        self.num_tasks = num_tasks
        self.exc_time_limits = exc_time_limits
        self.rwd_limits = rwd_limits
        self.tasks = [Task() for _ in range(num_tasks)]
        self.task_positions = None # since it is accessed in reset, step

        # enviroment variables
        self.trait_dim = trait_dim
        self.episode_length = episode_length
        self.field_dim = field_dim
        self.render_mode = render_mode
        self.delay_btw_frames = delay_btw_frames
        self.timestep = None
        self.with_task_infos = with_task_infos

        # part of the observation that never changes
        self.static_obs = None

    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(seed)
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        
        # randomize agent positions
        self.agent_positions = [[rng.integers(self.field_dim), rng.integers(self.field_dim)] for _ in range(self.num_agents)]
        
        # randomize tasks
        # make sure tasks cannot be on the same tile
        base_indices = rng.choice(self.field_dim*self.field_dim, size=self.num_tasks, replace=False)
        self.task_positions = [[i // self.field_dim, i % self.field_dim] for i in base_indices]
        for i in range(self.num_tasks):
            self.tasks[i].randomize(rng, self.task_positions[i], 
                                    self.exc_time_limits[0], self.exc_time_limits[1]+1,
                                    self.rwd_limits[0], self.rwd_limits[1]+1,
                                    self.trait_dim)
            self.tasks[i].reset_task_progress()

        # initialize action masks
        action_masks = np.ones((self.num_agents, self.action_space().n), dtype=np.int8)
        for i in range(len(action_masks)):
            if self.agent_positions[i][0] == 0:
                action_masks[i][0] = 0
            if self.agent_positions[i][1] == self.field_dim-1:
                action_masks[i][1] = 0
            if self.agent_positions[i][0] == self.field_dim-1:
                action_masks[i][2] = 0
            if self.agent_positions[i][1] == 0:
                action_masks[i][3] = 0
            if not (self.agent_positions[i] in self.task_positions):
                action_masks[i][4] = 0

        # randomize agent traits
        self.agent_traits = rng.choice([0, 1], (self.num_agents, self.trait_dim))

        # we want to make sure that the agents have the ability to complete each task
        summed_traits = sum(self.agent_traits)
        for i in range(self.trait_dim):
            if summed_traits[i] == 0: # meaning, no agent has ability in this trait
                self.agent_traits[rng.choice(self.num_agents)][i] = 1

        # same for all agents
        self.static_obs = [*list(chain(*self.agent_traits)), *list(chain(*[[*t.requirement, t.reward, t.execution_time] 
                                                                           for t in self.tasks]))]

        # changes for each agent
        agent_observations = [[*self.static_obs,
                               *list(chain(*[[end_pos[0]-self.agent_positions[i][0], # encode relative coordinates
                                              end_pos[1]-self.agent_positions[i][1]] # to other agents and tasks
                                              for end_pos in self.agent_positions+self.task_positions])),
                               *list(chain(*[[t.execution_progress, int(t.finished())] for t in self.tasks]))]
                               for i in range(self.num_agents)] 

        observations = {a: {"observation": agent_observations[i],
                            "action_mask": mask} for i, mask, a in zip(range(self.num_agents), action_masks, self.agents)}
        
        infos = {self.possible_agents[i]: {"position": self.agent_positions[i],
                                           "trait": self.agent_traits[i]} for i in range(self.num_agents)}
        
        if self.with_task_infos:
            infos.update({"task_"+str(i): t.attr_dict() for i, t in enumerate(self.tasks)})

        if self.render_mode == "human":
            self.render()

        return observations, infos

    def step(self, actions):
        rewards = {a: 0 for a in self.agents} # initialize rewards for actions taken

        # check if task requirements are met by aggregating agent traits
        aggr_task_requirements = [[0]*self.trait_dim for _ in range(self.num_tasks)]
        
        # initialize action mask        
        action_masks = np.ones((self.num_agents, self.action_space().n), dtype=np.int8)

        for i, agent in enumerate(self.agents):
            # each action taken is just an int, have to map to an actual action
            decoded_action = self.decode_action(actions[agent])

            if decoded_action != 4:
                self.agent_positions[i][0] += decoded_action[0]
                self.agent_positions[i][1] += decoded_action[1]
                continue

            # agent wants to execute task
            # due to action mask, we know agent is on a task, so extract task index
            on_task_index = [t.position for t in self.tasks].index(self.agent_positions[i])

            # record that agent worked on this task in this timestep (to calculate reward assignment later on)
            self.tasks[on_task_index].contribution_history.append(agent)

            # add to aggregated agent traits
            aggr_task_requirements[on_task_index] = np.add(aggr_task_requirements[on_task_index], self.agent_traits[i])
        
        # if requirement of task is met, progress task execution only when task isn't finished
        for i in range(self.num_tasks):
            if all((aggr_task_requirements[i][j] >= self.tasks[i].requirement[j]) and 
                   not self.tasks[i].finished() for j in range(self.trait_dim)):
                self.tasks[i].execution_progress += 1

        # edit action masks
        for i in range(len(action_masks)):
            # constrict movement
            if self.agent_positions[i][0] == 0:
                action_masks[i][0] = 0
            if self.agent_positions[i][1] == self.field_dim-1:
                action_masks[i][1] = 0
            if self.agent_positions[i][0] == self.field_dim-1:
                action_masks[i][2] = 0
            if self.agent_positions[i][1] == 0:
                action_masks[i][3] = 0

            # check if agent is on a tile containing a task
            on_task_index = [j for j in range(self.num_tasks) if self.tasks[j].position == self.agent_positions[i]]

            # agent is not on a tile of a task, so constrict action "execute task"
            if len(on_task_index) == 0:
                action_masks[i][4] = 0 
                continue

            # now we know agent is on tile of a task, so extract task index
            on_task_index = on_task_index[0]

            # if agent is on the same tile as a task, check if task is finished
            # if it is, constrict action "execute task"
            if self.tasks[on_task_index].finished():
                action_masks[i][4] = 0
                continue

            # TODO: do not allow agent to choose action "execute task" if
            # requirements of task are already met

        ### REWARD CALCULATION
        # give rewards to agents upon task completion
        for t in self.tasks:
            if t.finished():
                contributed_agents = list(set(t.contribution_history))
                for a in contributed_agents:
                    # reward is scaled by contribution of each agent
                    rewards[a] += t.reward * t.contribution_history.count(a) / t.execution_time
                t.contribution_history = []
        ### REWARD CALCULATION

        if all(t.finished() for t in self.tasks):
            terminations = {a: True for a in self.agents}
        else:   
            terminations = {a: False for a in self.agents}

        truncations = {a: False for a in self.agents}
        if self.timestep >= self.episode_length:
            truncations = {a: True for a in self.agents}

        self.timestep += 1   

        # changes for each agent
        agent_observations = [[*self.static_obs,
                               *list(chain(*[[end_pos[0]-self.agent_positions[i][0], # encode relative coordinates
                                              end_pos[1]-self.agent_positions[i][1]] # to other agents and tasks
                                              for end_pos in self.agent_positions+self.task_positions])),
                               *list(chain(*[[t.execution_progress, int(t.finished())] for t in self.tasks]))]
                               for i in range(self.num_agents)] 

        observations = {a: {"observation": agent_observations[i],
                            "action_mask": mask} for i, mask, a in zip(range(self.num_agents), action_masks, self.agents)}
        
        infos = {self.agents[i]: {"position": self.agent_positions[i],
                                  "trait": self.agent_traits[i]} for i in range(self.num_agents)}

        if self.with_task_infos:
            infos.update({"task_"+str(i): t.attr_dict() for i, t in enumerate(self.tasks)})

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        if self.render_mode == "human":
            self.render()
            time.sleep(self.delay_btw_frames)
        
        return observations, rewards, terminations, truncations, infos

    def decode_action(self, action):
        if action == 0:
            return [-1, 0] # go up
        elif action == 1:
            return [0, 1] # go right
        elif action == 2:
            return [1, 0] # go down
        elif action == 3:
            return [0, -1] # go left
        elif action == 4:
            return action  # execute task (if on tile of a task)
        
        raise Exception("Invalid action.")

    def render(self):
        grid = np.full((self.field_dim, self.field_dim), " ")
        for t in self.tasks:
            if not t.finished():
                grid[t.position[0], t.position[1]] = "T"
        for agent_pos in self.agent_positions:
            grid[agent_pos[0], agent_pos[1]] = "A"
        print(f"{grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent = None):
        return Box(low=-self.field_dim,   # or tighter bound if you know it
                   high=self.field_dim,
                   shape=(self.trait_dim*self.num_agents + 
                         (self.trait_dim+2)*self.num_tasks +
                         2*(self.num_agents+self.num_tasks) +
                         2*self.num_tasks,),
                    dtype=np.int64)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent = None):
        return Discrete(5) # agents can move up, right, down, left and execute a task