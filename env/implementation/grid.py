"""
Grid class to manage reservations and checking cell states.
"""

from .header import (Position,
                     IntArr,
                     Action,
                     ACT_TO_DIR,
                     GridOffsets,
                     DIR_ARR)

class Grid:
    """
    Represents a 2D grid with walls, zones, agents and agents' goals.
    """
    def __init__(self, *,
                 field: IntArr,
                 max_timestep: int,
                 num_agents: int,
                 agent_positions: list[Position],
                 goal_positions: list[Position]) -> None:
        # each cell contains [is_no_wall, is_zone, is_agent]
        self.field = field
        self.max_timestep = max_timestep
        self.num_agents = num_agents
        self.agent_idx = list(range(num_agents))
        self.agent_positions: list[Position] = agent_positions
        self.goal_positions: list[Position] = goal_positions

    @property
    def dim(self) -> int:
        """
        Dimension of the grid. 
        Note that the grid is a square and is surrounded by a wall.
        """
        return self.field.shape[0]

    def is_agent_on_goal(self, a_idx: int) -> bool:
        """
        Returns true if the specified agent is on its' goal,
        false otherwise.
        
        Args:
            a_idx (int): Specified agent.

        Returns:
            bool: `True` if agent on goal, `False` otherwise.
        """
        return self.agent_positions[a_idx] == self.goal_positions[a_idx]

    def move_agent_in_grid(self, a_idx: int, action: Action):
        """Moves the specified agent in the actual grid.

        Args:
            a_idx (int): Index of agent to be moved.
            action (Action): Action of agent.

        If action is illegal (moving into wall or another agent), agent stays still.
        """
        if not self.is_action_valid(self.agent_positions[a_idx], action):
            print(f"agent {a_idx} tried to execute illegal action" +
                  f" {action} at position {self.agent_positions[a_idx]}")
            return

        self.set_agent_in_grid(self.agent_positions[a_idx], False)
        self.agent_positions[a_idx] += ACT_TO_DIR[action]
        self.set_agent_in_grid(self.agent_positions[a_idx], True)

    def set_zone_in_grid(self, pos: Position, val: bool):
        """
        Sets/Removes the zone at a specified position.

        Args:
            pos (Position): Position where zone is set/removed.
        """
        self.field[*pos, GridOffsets.ZONE] = int(val)

    def set_agent_in_grid(self, pos: Position, val: bool):
        """
        Sets/Removes an agent at a specified position.

        Args:
            pos (Position): Position where the agent is set/removed.
        """
        self.field[*pos, GridOffsets.AGENT] += 1 if val else -1

    def get_neighbours_at(self, pos: Position) -> list[Position]:
        """
        Get valid neighbouring coordinates (4-connected grid).

        Args:
            grid: Grid object representing the map.
            pos: Center position (x, y).

        Returns:
            List of valid neighbouring coordinates in 4 directions (left, right,
            up, down). Empty list of no valid neighbours.
        """
        neighbours: list[Position] = []
        for direction in DIR_ARR:
            new_neighbour: Position = pos+direction
            if not self.contains_wall(new_neighbour):
                neighbours.append(new_neighbour)
        return neighbours

    def is_action_valid(self, pos: Position, action: Action) -> bool:
        """
        Checks for validity of given action.
        """
        return not self.contains_wall(pos+ACT_TO_DIR[action])

    def contains_agent(self, pos: Position) -> bool:
        """Returns true if the position contains atleast one agent, false otherwise.

        Args:
            pos (Position): Position to be checked.

        Returns:
            bool: `True` if agent(s) on `pos`, `False` otherwise.
        """
        return self.field[*pos, GridOffsets.AGENT] > 0

    def contains_multiple_agents(self, pos: Position) -> bool:
        """Returns true if the position contains more than one agent, false otherwise.

        Args:
            pos (Position): Position to be checked.

        Returns:
            bool: `True` if agents on `pos`, `False` otherwise.
        """
        return self.field[*pos, GridOffsets.AGENT] > 1

    def contains_zone(self, pos: Position) -> bool:
        """Returns true if the position contains a zone, false otherwise.

        Args:
            pos (Position): Position to be checked.

        Returns:
            bool: `True` if zone on `pos`, `False` otherwise.
        """
        return self.field[*pos, GridOffsets.ZONE]

    def contains_wall(self, pos: Position) -> bool:
        """Returns true if the position contains a wall, false otherwise.

        Args:
            pos (Position): Position to be checked.

        Returns:
            bool: `True` if wall on `pos`, `False` otherwise.
        """
        return not self.field[*pos, GridOffsets.NO_WALL]
