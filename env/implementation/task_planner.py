"""
Contains methods that assign given unfinished tasks to agents.
"""

import numpy as np
from .task import Task
from . import header as h

def contribution(*,
                 agent_trait: h.IntArr,
                 trait_dim: int,
                 rem_requirements: h.IntArr) -> h.FloatArr:
    """
    Given an agent trait, task requirement and the task's remaining requirements, 
    calculates a contribution score for the agent.
    """
    score = 0
    for i in range(trait_dim):
        if rem_requirements[i] == 1 and agent_trait[i] == 1:
            # if agent can contribute to something which needs its
            # contribution, we encourage contribution
            score += 1

    return score

def get_goal(*,
             trait_dim: int,
             agent_trait: h.IntArr,
             num_tasks: int,
             tasks: list[Task]) -> int:
    """
    Given an agent trait and task attributes, assigns an unfinished task
    to the agent.
    """
    costs: h.FloatArr = np.zeros(num_tasks, h.DTYPE_FLOAT)

    for i in range(num_tasks):
        if tasks[i].will_get_finished:
            continue

        costs[i] = contribution(agent_trait=agent_trait,
                                trait_dim=trait_dim,
                                rem_requirements=tasks[i].plan_requirements)

    # agent cannot contribute to any task,
    # so its goal will be the agent depot
    if (costs == 0).all():
        return h.AGENT_DEPOT

    return np.argmax(costs).item()
