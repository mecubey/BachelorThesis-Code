"""
Contains agent related classes, attributes, methods.
"""

from typing import Callable, Any
import numpy as np
from . import header as h

class EnvAgent():
    """
    Holds agent attributes.\n
    Contains an agent's position, traits, mask, goal, one hot vector.\n
    Note that this class does not contain the actual policy, but is simply
    used as a sort of container class for agent attributes.
    """
    def __init__(self, agent_id: int):
        self.agent_id: int = agent_id
        self.agent_name: str = f"agent_{agent_id}"
        self.agent_color: str
        self.goal_pos: h.PositionT

    def to_dict(self, *, pos: h.PositionT) -> dict[str, Any]:
        """
        Returns a dict representation of this class.
        """
        return {"agent": self.agent_color + h.AGENT_CHAR,
                "position": pos.tolist(),
                "goal": self.agent_color + h.GOAL_CHAR,
                "goal_pos": self.goal_pos.tolist()}

    def on_goal(self, pos: h.PositionT) -> bool:
        """
        Returns true if agent is on its goal.\n
        False otherwise.
        """
        return (self.goal_pos == pos).all()

    def get_mask(self, walkable: Callable[[h.PositionT], bool], pos: h.PositionT) -> h.FloatArr:
        """
        Set valid actions of this agent.
        """
        mask: list[bool] = [walkable(pos+h.ACT_TO_DIR_ARR[i]) for i in range(h.ACTION_LEN-1)]
        mask.append(True) # agent can always stay at their current position
        return np.array(mask, h.DTYPE_FLOAT)
