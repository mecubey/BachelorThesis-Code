"""
Contains agent related classes, attributes, methods.
"""

from typing import Callable, Any
import numpy as np
from .task import Task
from . import header as h

class EnvAgent():
    """
    Holds agent attributes.\n
    Contains an agent's position, traits, mask, goal, one hot vector.\n
    Note that this class does not contain the actual policy, but is simply
    used as a sort of container class for agent attributes.
    """
    def __init__(self):
        self.traits: h.IntArr
        self.mask: h.FloatArr = np.ones(h.ACTION_LEN, dtype=h.DTYPE_FLOAT)
        self.goal_pos: h.PositionT
        self.goal_idx: int

    def to_dict(self, *, pos: h.PositionT, color: str, goal_char: str) -> dict[str, Any]:
        """
        Returns a dict representation of this class.
        """
        return {"position": pos.tolist(),
                "traits": self.traits.tolist(),
                "color": color + h.AGENT_CHAR,
                "goal": goal_char,
                "goal_pos": self.goal_pos.tolist()}

    def on_goal(self, pos: h.PositionT) -> bool:
        """
        Returns true if agent is on its goal.\n
        False otherwise.
        """
        return (self.goal_pos == pos).all()

    def set_traits(self, new_traits: h.IntArr) -> None:
        """
        Set traits of this agent.\n
        Determines what tasks agent can contribute to.
        """
        self.traits = new_traits

    def edit_mask(self, walkable: Callable[[h.PositionT], bool], pos: h.PositionT) -> None:
        """
        Set valid actions of this agent.
        """
        for i in range(h.ACTION_LEN-1): # agent can always "do nothing"
            self.mask[i] = walkable(pos+h.Act_To_Dir_Arr[i])

    def set_goal(self, *,
                 new_goal_idx: int,
                 tasks: list[Task],
                 depot_pos: h.PositionT) -> None:
        """
        Set new goal (agent depot or task) of agent.
        """
        self.goal_idx = new_goal_idx
        if new_goal_idx == h.AGENT_DEPOT:
            self.goal_pos = depot_pos
            return

        self.goal_pos = tasks[new_goal_idx].position
