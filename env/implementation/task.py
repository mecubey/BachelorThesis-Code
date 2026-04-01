import numpy as np

class Task():
    def __init__(self):
        self.requirement = None
        self.position = None 
        self.execution_time = None 
        self.reward = None 
        self.execution_progress = None
        self.reward_given = None
        self.trait_contribution_history = None

    def randomize(self, rng : np.random.Generator, 
                  position, min_exc_time, max_exc_time, min_rwd, max_rwd, trait_dim):
        # We pass "positions" as a parameter because we have to ensure
        # that no two tasks can have the same position.
        # Optionally, a new requirement can be passed. If requirement = None,
        # then the previous requirement will be used.
        self.position = position
        self.execution_time = rng.integers(low=min_exc_time, high=max_exc_time)
        self.reward = rng.uniform(low=min_rwd, high=max_rwd)

        # randomize task requirement
        self.requirement = rng.choice([0, 1], trait_dim)
        
        # we want to ensure that each task requires at least one agent,
        # so no task can have 0 for every requirement attribute
        if sum(self.requirement) == 0:
            self.requirement[rng.choice(range(trait_dim))] = 1

    def reset_task_progress(self):
        self.execution_progress = 0
        self.reward_given = False
        self.trait_contribution_history = np.array([], dtype=np.float32)

    def finished(self):
        return self.execution_progress >= self.execution_time

    def requirements_met(self, traits):
        return all(traits[i] >= self.requirement[i] for i in range(len(self.requirement)))

    def attr_dict(self):
        return {"position": self.position,
                "requirement": self.requirement,
                "reward": self.reward,
                "exec_time": self.execution_time,
                "exec_progress": self.execution_progress,
                "finished": self.finished()}
    
    def record_aggr_traits(self, traits):
        self.trait_contribution_history = np.append(self.trait_contribution_history, traits)

    def get_episode_reward(self):
        # we want to minize the average wasted ability ratio on this task
        # by minimizing it for each individual task, we minimize it for all tasks
        coalition_traits = np.average(self.trait_contribution_history, axis=0)
        summed_diff = np.sum(np.abs(coalition_traits-self.requirement))
        return summed_diff / np.sum(self.requirement) if self.finished() else self.reward

    def get_remaining_req(self, aggr_traits):
        return np.maximum(self.requirement-aggr_traits, 0)

    def progress(self):
        self.execution_progress += 1