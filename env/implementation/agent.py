"""
Contains agent related classes, attributes, methods.
"""

from typing import Any
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
