import random
import functools
import numpy as np
import time
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv
from copy import copy

class PathTaskEnv(ParallelEnv):
    metadata = {
        "name": "path_task_env_v0",
    }

    def __init__(self, num_agents, num_tasks, trait_dim, episode_length, field_dims):
        self.num_tasks = num_tasks
        self.trait_dim = trait_dim
        self.episode_length = episode_length
        self.field_dims = field_dims
        self.possible_agents = ["agent_"+str(i) for i in range(num_agents)]
        self.possible_tasks = ["task_"+str(i) for i in range(num_tasks)]
        self.timestep = None
        self.agent_positions = None
        self.task_positions = None
        
    def reset(self, seed=None, options=None):
        random.seed(seed)
        self.agents = copy(self.possible_agents)
        self.possible_agents = ["agent_"+str(i) for i in range(self.num_agents)]
        self.possible_tasks = ["task_"+str(i) for i in range(self.num_tasks)]
        self.timestep = 0
        self.agent_positions = {a: [0, 0] for a in self.possible_agents} # agents all have same starting position
        self.task_positions = {t: [random.randint(0, self.field_dims[0]-1), 
                                   random.randint(0, self.field_dims[0]-1)] for t in self.possible_tasks}
        observations = {a: list(self.agent_positions.values())+list(self.task_positions.values()) for a in self.agents}
        infos = {a: {} for a in self.agents} # dummy infos for proper parallel_to_aec conversion

        return observations, infos

    def step(self, actions):
        rewards = {a: 0 for a in self.agents} # init rewards for actions taken
        for agent in self.agents: # each action taken is just an int, have to map to actual action
            action_taken = self.decode_action(actions[agent])
            self.agent_positions[agent][0] = np.clip(self.agent_positions[agent][0]+action_taken[0], 0, self.field_dims[0]-1)
            self.agent_positions[agent][1] = np.clip(self.agent_positions[agent][1]+action_taken[1], 0, self.field_dims[1]-1)

        terminations = {a: False for a in self.agents}

        # TODO: reward calculation

        truncations = {a: False for a in self.agents}
        if self.timestep >= 100:
            truncations = {a: True for a in self.agents}

        self.timestep += 1            
        observations = {a: list(self.agent_positions.values())+list(self.task_positions.values()) for a in self.agents}

        # dummy infos
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        self.render()
        time.sleep(1)

        return observations, rewards, terminations, truncations, infos

    def decode_action(self, action):
        if action == 0:
            return [0, 1] # go up
        elif action == 1:
            return [1, 0] # go right
        elif action == 2:
            return [0, -1] # go down
        elif action == 3:
            return [-1, 0] # go down
        
        raise Exception("Invalid action.")

    def render(self):
        grid = np.full((self.field_dims[0], self.field_dims[1]), " ")
        for task_pos in self.task_positions.values():
            grid[task_pos[0], task_pos[1]] = "T"
        for agent_pos in self.agent_positions.values():
            grid[agent_pos[0], agent_pos[1]] = "A"
        print(f"{grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(self.num_agents+self.num_tasks)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4) # agents can move up and down
    

# TODO:
# reward calculation (needs trait requirements and tasks should dissappear)
# make test in other folder workable