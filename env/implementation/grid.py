"""
Grid class to manage reservations and checking cell states.
"""

from . import header as h

class Grid:
    """
    Represents a 2D grid with walls, zones, agents and agents' goals.
    """
    def __init__(self, *,
                 field: h.IntArr,
                 num_agents: int,
                 agent_positions: list[h.Position],
                 goal_positions: list[h.Position]) -> None:
        # each cell contains [is_no_wall, is_zone, is_agent]
        self.field = field
        self.num_agents = num_agents
        self.agent_idx = list(range(num_agents))
        self.agent_names = [f"agent_{i}" for i in self.agent_idx]
        self.agent_positions: list[h.Position] = agent_positions
        self.goal_positions: list[h.Position] = goal_positions

    @property
    def dim(self) -> int:
        """
        Dimension of the grid. 
        Note that the grid is a square and is surrounded by a wall.
        """
        return self.field.shape[0]

    def is_agent_on_goal(self, a_idx: int):
        """
        Returns true if the specified agent is on its' goal,
        false otherwise.
        
        Args:
            a_idx (int): Specified agent.

        Returns:
            _type_: `True` if agent on goal, `False` otherwise.
        """
        return self.agent_positions[a_idx] == self.goal_positions[a_idx]

    def move_agent_in_grid(self, a_idx: int, action: h.Action):
        """Moves the specified agent in the actual grid.

        Args:
            a_idx (int): Index of agent to be moved.
            action (h.Action): Action of agent.

        If action is illegal (moving into wall or another agent), agent stays still.
        """
        if not self.is_action_valid(self.agent_positions[a_idx], action):
            print(f"agent {a_idx} tried to execute illegal action" +
                  f" {action} at position {self.agent_positions[a_idx]}")
            return

        self.set_agent_in_grid(self.agent_positions[a_idx], False)
        self.agent_positions[a_idx] += h.ACT_TO_DIR[action]
        self.set_agent_in_grid(self.agent_positions[a_idx], True)

    def set_zone_in_grid(self, pos: h.Position, val: bool):
        """
        Sets/Removes the zone at a specified position.

        Args:
            pos (h.Position): h.Position where zone is set/removed.
        """
        self.field[*pos, h.GridOffsets.ZONE] = int(val)

    def set_agent_in_grid(self, pos: h.Position, val: bool):
        """
        Sets/Removes an agent at a specified position.

        Args:
            pos (h.Position): h.Position where the agent is set/removed.
        """
        self.field[*pos, h.GridOffsets.AGENT] += 1 if val else -1

    def walkable(self, pos: h.Position) -> bool:
        """
        Returns true if an agent could stand on a given position.\n
        False otherwise.
        """
        return not self.contains_wall(pos) and not self.contains_agent(pos)

    def is_action_valid(self, pos: h.Position, action: h.Action) -> bool:
        """
        Checks for validity of given action.
        """
        if action == h.Action.DO_NOTHING:
            return not self.contains_wall(pos)
        return self.walkable(pos+h.ACT_TO_DIR[action])

    def contains_agent(self, pos: h.Position) -> bool:
        """Returns true if the position contains atleast one agent, false otherwise.

        Args:
            pos (h.Position): h.Position to be checked.

        Returns:
            bool: `True` if agent(s) on `pos`, `False` otherwise.
        """
        return self.field[*pos, h.GridOffsets.AGENT] > 0

    def contains_multiple_agents(self, pos: h.Position) -> bool:
        """Returns true if the position contains more than one agent, false otherwise.

        Args:
            pos (h.Position): h.Position to be checked.

        Returns:
            bool: `True` if agents on `pos`, `False` otherwise.
        """
        return self.field[*pos, h.GridOffsets.AGENT] > 1

    def contains_zone(self, pos: h.Position) -> bool:
        """Returns true if the position contains a zone, false otherwise.

        Args:
            pos (h.Position): h.Position to be checked.

        Returns:
            bool: `True` if zone on `pos`, `False` otherwise.
        """
        return self.field[*pos, h.GridOffsets.ZONE]

    def contains_wall(self, pos: h.Position) -> bool:
        """Returns true if the position contains a wall, false otherwise.

        Args:
            pos (h.Position): h.Position to be checked.

        Returns:
            bool: `True` if wall on `pos`, `False` otherwise.
        """
        return not self.field[*pos, h.GridOffsets.NO_WALL]
