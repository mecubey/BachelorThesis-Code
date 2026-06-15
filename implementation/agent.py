"""
Contains class definition of Agent.
"""

from .mapf_utils import (Position,
                         BASE_DAMAGE,)
from .hazard import Hazard

class Agent:
    """
    Represents an agent.
    """
    def __init__(self, *,
                 i: int,
                 priority: float,
                 start_pos: Position,
                 goal_pos: Position,
                 hazard: Hazard,
                 ) -> None:
        self.hazard = hazard
        self.id = i
        self.initial_priority = priority
        self.current_priority = priority
        self.initial_pos = start_pos
        self.current_pos = start_pos.deepcopy()
        self.goal_pos = goal_pos
        self.damage: float = BASE_DAMAGE
        self.frozen_for: int = 0

    def decay_dmg(self) -> None:
        """
        Decay the damage taken so far. Damage is clipped to 0.
        """
        self.damage = self.hazard.calculate_decreased_dmg(self.damage)

    def frozen(self) -> bool:
        """
        Check if agent is frozen.

        If an agent is frozen, then it cannot move.
        """
        return self.frozen_for > 0

    def decay_freeze(self) -> None:
        """
        Decrease the freeze time by 1.
        """
        self.frozen_for = max(self.frozen_for-1, 0)

    def freeze(self) -> None:
        """
        Freeze agent. The more damage an agent has taken,
        the longer it is frozen for.
        """
        self.frozen_for = int(self.damage)

    def increase_damage(self) -> None:
        """
        Increase damage of this agent by 1.

        Damage value is clipped to MAX_DAMAGE.
        """
        self.damage = self.hazard.calculate_increased_dmg(self.damage)

    def move(self, action: Position) -> None:
        """
        Move agent accordin to action.

        Args:
            action (Position): Action agent should take.
        """
        self.current_pos += action

    def on_goal(self) -> bool:
        """
        Check if agent is on its goal.

        Returns:
            bool: True if agent is on its goal, False otherwise.
        """
        return self.current_pos == self.goal_pos

    def reset(self) -> None:
        """
        Reset position and priority of agent.
        """
        self.damage = BASE_DAMAGE
        self.frozen_for = 0
        self.current_priority = self.initial_priority
        self.current_pos = self.initial_pos.deepcopy()

Agents = list[Agent]
