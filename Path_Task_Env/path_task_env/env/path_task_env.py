import random
import functools
import numpy as np
import time
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv
from copy import copy

class Task():
    def __init__(self, requirement):
        self.requirement = requirement # TODO: add option to randomize task requirements
        self.position = None 
        self.execution_time = None 
        self.reward = None 
        self.execution_progress = 0

    def randomize(self, rng : np.random.Generator, 
                  position, min_exc_time, max_exc_time, min_rwd, max_rwd, requirement = None):
        # We pass "positions" as a parameter because we have to ensure
        # that no two tasks can have the same position.
        # Optionally, a new requirement can be passed. If requirement = None,
        # then the previous requirement will be used.
        self.position = position
        self.execution_time = rng.integers(low=min_exc_time, high=max_exc_time)
        self.reward = rng.integers(low=min_rwd, high=max_rwd)
        if not (requirement is None):
            self.requirement = requirement

    def reset_task_progress(self):
        self.execution_progress = 0

    def finished(self):
        return self.execution_progress >= self.execution_time

class PathTaskEnv(ParallelEnv):
    metadata = {
        "name": "path_task_env",
    }

    def __init__(self,
                 num_agents, agent_traits,
                 num_tasks, task_requirements, exc_time_limits, rwd_limits,
                 episode_length, field_dim, render_mode, delay_btw_frames):
        assert num_agents == len(agent_traits)
        assert num_tasks == len(task_requirements)

        # agents
        self.possible_agents = ["agent_"+str(i) for i in range(num_agents)]
        self.agent_positions = None
        self.agent_traits = agent_traits # TODO: add option to randomize agent types

        # tasks
        self.num_tasks = num_tasks
        self.exc_time_limits = exc_time_limits
        self.rwd_limits = rwd_limits
        self.tasks = [Task(task_requirements[i]) for i in range(num_tasks)]

        # enviroment variables
        self.trait_dim = len(self.tasks[0].requirement)
        self.episode_length = episode_length
        self.field_dim = field_dim
        self.render_mode = render_mode
        self.delay_btw_frames = delay_btw_frames
        self.timestep = None
        
    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(seed)
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        
        # randomize agent positions
        self.agent_positions = [[rng.integers(self.field_dim), rng.integers(self.field_dim)] for _ in range(self.num_agents)]
        
        # randomize tasks
        # make sure tasks cannot be on the same tile
        base_indices = rng.choice(self.field_dim*self.field_dim, size=self.num_tasks, replace=False)
        task_positions = [[i // self.field_dim, i % self.field_dim] for i in base_indices]
        for i in range(self.num_tasks):
            self.tasks[i].randomize(rng, task_positions[i], 
                                    self.exc_time_limits[0], self.exc_time_limits[1]+1,
                                    self.rwd_limits[0], self.rwd_limits[1]+1)
            self.tasks[i].reset_task_progress()

        observations = {a: self.agent_positions+task_positions for a in self.agents}
        infos = {a: {} for a in self.agents} # dummy infos for proper parallel_to_aec conversion

        return observations, infos

    def step(self, actions):
        rewards = {a: 0 for a in self.agents} # initialize rewards for actions taken

        # check if task requirements are met by aggregating agent traits
        aggr_task_requirements = [[0]*self.trait_dim]*self.num_tasks
        
        # remember which agents worked on which task to give them reward upon task completion
        agents_on_tasks = [[]]*self.num_tasks
        
        for i, agent in enumerate(self.agents):
            # each action taken is just an int, have to map to an actual action
            action_taken = self.decode_action(actions[agent])

            if action_taken != 4:
                self.agent_positions[i][0] = np.clip(self.agent_positions[i][0]+action_taken[0], 0, self.field_dim-1)
                self.agent_positions[i][1] = np.clip(self.agent_positions[i][1]+action_taken[1], 0, self.field_dim-1)
                continue

            print("Agent wants to execute task!")

            # agent wants to execute task
            # first, check if agent is on a tile containing a task
            on_task_index = [j for j in range(self.num_tasks) if self.tasks[j].position == self.agent_positions[i]]
            
            # NOTE: this means agents can choose action "execute task" if they are not on the same tile as a task...
            if len(on_task_index) == 0:
                continue

            # now we know agent is on a task
            # extract task index
            on_task_index = on_task_index[0]
            
            # if the task agent wants to progress has already been finished,
            # then agent wasted an action and nothing happens
            # NOTE: this means we allow agents to execute useless actions. is this a good idea...?
            if self.tasks[on_task_index].finished():
                continue

            # record which agent worked on which task
            agents_on_tasks[on_task_index].append(agent)

            # add to aggregated agent traits
            aggr_task_requirements[on_task_index] = np.add(aggr_task_requirements[on_task_index], self.agent_traits[i])
        
        # if requirement of task is met, progress task execution
        for i in range(self.num_tasks):
            if all(aggr_task_requirements[i][j] >= self.tasks[i].requirement[j] for j in range(self.trait_dim)):
                self.tasks[i].execution_progress += 1
        
        ### REWARD CALCULATION

        # give rewards to agents upon task completion
        for i in range(len(self.tasks)):
            if self.tasks[i].finished():
                for a in agents_on_tasks[i]:
                    rewards[a] += self.tasks[i].reward

        ### REWARD CALCULATION

        if all(t.finished() for t in self.tasks):
            terminations = {a: True for a in self.agents}
        else:   
            terminations = {a: False for a in self.agents}

        truncations = {a: False for a in self.agents}
        if self.timestep >= self.episode_length:
            truncations = {a: True for a in self.agents}

        self.timestep += 1   

        # every task that has been finished will receive position [-1, -1]
        task_positions = [t.position if not t.finished() else [-1, -1] for t in self.tasks]
        
        observations = {a: self.agent_positions+task_positions for a in self.agents}

        # dummy infos
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        if self.render_mode == "human":
            self.render()
            time.sleep(self.delay_btw_frames)
        
        return observations, rewards, terminations, truncations, infos

    def decode_action(self, action):
        if action == 0:
            return [0, 1] # go up
        elif action == 1:
            return [1, 0] # go right
        elif action == 2:
            return [0, -1] # go down
        elif action == 3:
            return [-1, 0] # go left
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
    def observation_space(self, agent):
        return Discrete(self.max_num_agents+self.num_tasks)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5) # agents can move up, right, down, left and execute a task
    
# NOTE: Is it a good idea for agents to be able to execute useless actions? Or should I do action masking? Ask in meeting!