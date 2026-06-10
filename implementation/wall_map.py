"""
Contains class definition of WallMap.

Maps are taken from mapf.info.
"""

from pathlib import Path
from .mapf_utils import (Position,
                         Positions,
                         Map,
                         Direction,
                         DIR_TO_POS,
                         get_map_path)

class WallMap(Map):
    """
    Represents a map filled with walls.
    """
    def __init__(self, name: str):
        self.name = name
        self.free_tiles: Positions = []
        self.neighbour_table: dict[Position, list[Position|None]] = {}

        path = Path(get_map_path(name))

        lines: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]

        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])

        # find "map" line
        map_start = lines.index("map") + 1
        grid = lines[map_start:map_start + height]

        super().__init__(width, height, "bool")

        for x_, row in enumerate(grid):
            for y_, ch in enumerate(row):
                if ch == ".":
                    self.tiles[x_ * width + y_] = True
                    self.free_tiles.append(Position(x_, y_))

        # fill neighbour table
        for row in range(height):
            for col in range(width):
                current_pos = Position(row, col)

                if self.on_wall(current_pos):
                    self.neighbour_table[current_pos] = [None]*4
                    continue

                neighbours: list[Position|None] = []
                for direction in Direction:
                    neighbour = current_pos+DIR_TO_POS[direction]
                    if self.inside(neighbour) and not self.on_wall(neighbour):
                        neighbours.append(neighbour)
                    else:
                        neighbours.append(None)
                self.neighbour_table[current_pos] = neighbours

    def on_wall(self, pos: Position) -> bool:
        """
        Checks if a given position is on a wall.

        Args:
            pos (Position): Position to be checked.

        Returns:
            bool: True if position is on a wall, False otherwise.
        """
        return not self[pos]

if __name__ == "__main__":
    test = WallMap("random-32-32-20")
    print(test.on_wall(Position(1, 0)))
    print(test.neighbour_table[Position(1, 0)])
