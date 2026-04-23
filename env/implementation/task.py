"""
Contains task related classes, attributes, methods.
"""

from typing import Any
import numpy as np
from . import header as h

class Task():
    """
    Holds task attributes.\n
    Contains remaining requirements, position and finished flag of task.
    """
    def __init__(self):
        self.real_requirements: h.IntArr
        self.plan_requirements: h.IntArr
        self.position: h.PositionT
        self.will_get_finished: bool = False
        self.is_finished: bool = False

    def to_dict(self, color: str) -> dict[str, Any]:
        """
        Returns a dict representation of this class.
        """
        return {"real_requirements": self.real_requirements.tolist(),
                "plan_requirements": self.plan_requirements.tolist(),
                "position": self.position.tolist(),
                "color": color + h.TASK_CHAR,
                "will_get_finished": self.will_get_finished,
                "is_finished": self.is_finished}

    def contribute_to_plan(self, trait: h.IntArr) -> None:
        """
        Contribute to the "plan" requirements. These requirements
        are used to calculate the task planning.
        """
        self.plan_requirements = np.clip(self.plan_requirements-trait, 0, 1)
        if not self.plan_requirements.any():
            self.will_get_finished = True

    def contribute_to_real(self, trait: h.IntArr) -> None:
        """
        Contribute to the "real" requirements. These requirements
        are used to check if the task is actually finished.
        """
        self.real_requirements = np.clip(self.real_requirements-trait, 0, 1)
        if not self.real_requirements.any():
            self.is_finished = True

    def manual_init(self,
                    *,
                    requirement: h.IntArr,
                    position: h.PositionT) -> None:
        """
        Manually initialize task attributes.
        """
        self.is_finished = False
        self.will_get_finished = False
        self.real_requirements = self.plan_requirements = requirement
        self.position = position
