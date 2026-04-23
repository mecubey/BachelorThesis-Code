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
    fin_contr_score: int = rem_requirements.sum()
    scores: h.FloatArr = np.zeros(2, h.DTYPE_FLOAT)
    for i in range(trait_dim):
        if rem_requirements[i] == agent_trait[i] == 1:
            fin_contr_score -= 1

            # if agent can contribute to something which needs its
            # contribution, we encourage contribution
            scores[0] += 1

        # if agent contributes to something which does not need its
        # contribution, we discourage contribution
        if rem_requirements[i] == 0 and agent_trait[i] == 1:
            scores[1] -= 1

    if fin_contr_score == 0:
        # if agent can finish this task by contributing,
        # then we want the agent to contribute to this task
        scores[0] = scores[1] = np.inf
    return scores

def get_goal(*,
             trait_dim: int,
             agent_trait: h.IntArr,
             num_tasks: int,
             tasks: list[Task]) -> int:
    """
    Given an agent trait and task attributes, assigns an unfinished task
    to the agent.
    """
    costs: h.FloatArr = np.zeros((num_tasks, 2), h.DTYPE_FLOAT)

    for i in range(num_tasks):
        if tasks[i].will_get_finished:
            continue

        costs[i] = contribution(agent_trait=agent_trait,
                                trait_dim=trait_dim,
                                rem_requirements=tasks[i].plan_requirements)

    # agent cannot contribute to any task,
    # so its goal will be the agent depot
    if (costs[:, 0] == 0).all():
        return h.AGENT_DEPOT

    costs = np.sum(costs, axis=1)

    return np.argmax(costs).item()
