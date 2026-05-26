"""
Pregenerate maps & free tiles.
"""

from .maze import gen_maze
from .header import BoolArr, Position

dims = [16]
maze_seeds: list[int] = [1]
maze_intensities: list[float] = [0.2]
wall_maps: list[BoolArr] = []
all_free_tiles: list[list[Position]] = []

for i, intensity in enumerate(maze_intensities):
    wall_map, free_tiles = gen_maze(dim=dims[i],
                                    maze_intensity=intensity,
                                    seed=maze_seeds[i])
    wall_maps.append(wall_map)
    all_free_tiles.append(free_tiles)
