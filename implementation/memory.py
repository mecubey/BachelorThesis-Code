"""
Contains class definition of Memory.
"""

class Memory:
    """
    Represents an agent memory.
    
    Estimates the probability to get stuck on a hazard tile,
    based on previous observations.
    """
    def __init__(self) -> None:
        # we begin with a pessimistic estimate,
        # stuck_prob = 1
        self.observed_stucks: int = 1
        self.potential_stucks: int = 1
        self.shared_estimations: dict[int, float] = {}

    def update_shared_estimation(self, i: int, estimation: float):
        """
        Updates the hazard stuck probability estimation received from another agent.

        Args:
            i (int): Agent index.
            estimation (float): The agent's estimated stuck probability.
        """
        self.shared_estimations[i] = estimation

    @property
    def estimation(self):
        """
        Returns the estimated stuck probability.

        The estimate is computed as the average of this agent's own observation-
        based estimate and all shared estimates received from other agents.

        Returns:
            float: Estimated stuck probability.
        """
        own_estimation = self.observed_stucks / self.potential_stucks
        return ((own_estimation+sum(list(self.shared_estimations.values())))/
                (1+len(self.shared_estimations)))

    def reset(self) -> None:
        """
        Resets all observation counts and shared estimations.
        """
        self.observed_stucks = 1
        self.potential_stucks = 1
        self.shared_estimations.clear()
