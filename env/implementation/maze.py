"""
Provides a method to generate a maze and its
corresponding directed graph.
"""

from itertools import product
import numpy as np
from . import header as h
from typing import cast

def gen_maze_graph(*,
                   dim: int,
                   maze_intensity: float,
                   rng: np.random.Generator,
                   iter_c: int = 10) -> h.FloatArr:
    """
    Generates a maze graph (directed graph) according to a specified length.
    """
    assert dim > 0, f"dimension of maze must be greater than 0, got {dim}"
    assert 0 <= maze_intensity <= 1, f"maze_intensity must be between 0 and 1, got {maze_intensity}"
    assert iter_c > 0, f"number of iterations must be non-negative, got {iter_c}"

    if dim == 1:
        return np.array([[h.ALL_OUTGOING]], dtype=h.DTYPE_FLOAT)

    # generate initial perfect maze
    maze: h.FloatArr = np.full(shape=(dim, dim), fill_value=h.Action.MOVE_RIGHT)
    maze[:, dim-1] = h.Action.MOVE_DOWN
    origin_pos: h.Position = h.Position(dim-1, dim-1)
    maze[*origin_pos] = h.ORIGIN

    for i in range(dim*dim*iter_c):
        # step 1: have origin node point to random neigbouring node
        # make sure origin does not point out of maze
        mask: list[bool] = [origin_pos.x != 0, origin_pos.y != dim-1,
                            origin_pos.x != dim-1, origin_pos.y != 0, False]
        new_dir: h.Action = rng.choice(h.ACTS_ARR_AS_NDARRAY[mask])
        maze[*origin_pos] = new_dir

        # don't want origin in output, so break at second to last iteration
        if i == dim*dim*iter_c-1:
            break

        # step 2: that neigbouring node becomes the new origin
        origin_pos += h.ACT_TO_DIR[new_dir]

        # step 3: remove the new origin node's pointer
        maze[*origin_pos] = h.ORIGIN

    # randomly remove ~1-maze_intensity of walls
    n: int = dim * dim
    freed_prctg = int((1 - maze_intensity) * n)
    flat_idx = rng.choice(n, size=freed_prctg, replace=False)
    rows, cols = np.divmod(flat_idx, dim)
    maze[rows, cols] = h.ALL_OUTGOING

    return maze

def gen_maze(*,
             maze_buffer: h.IntArr,
             dim: int,
             maze_intensity: float,
             exp_dim: int,
             seed: int|None = None,
             iter_c: int = 10) -> h.IntArr:
    """
    Fills a maze buffer according to a maze graph (directed graph). \n
    Returns a list of 2D indices which indicate wall-free tiles.
    """
    rng = np.random.default_rng(seed)

    maze_graph: h.FloatArr = gen_maze_graph(dim=dim,
                                            maze_intensity=maze_intensity,
                                            rng=rng,
                                            iter_c=iter_c)

    # randomly remove walls from cells which are not reached by the graph
    unreach_idx_base: h.IntArr = np.arange(2, 1+2*(dim-1), 2, dtype=h.DTYPE_INT)
    unreach_no_walls: h.BoolArr = rng.choice([0, 1],
                                             size=((dim-1)*(dim-1)),
                                             p=[maze_intensity,
                                             1-maze_intensity])
    grid_indices: h.IntArr = np.array(list(product(unreach_idx_base, unreach_idx_base)),
                                      dtype=h.DTYPE_INT)
    maze_buffer[grid_indices[:, 0], grid_indices[:, 1], h.GridOffsets.NO_WALL] = unreach_no_walls
    free_tiles: list[h.IntArr] = grid_indices[unreach_no_walls].tolist()

    for i in range(dim):
        for j in range(dim):
            maze_graph_pos: h.IntArr = np.array([i, j], dtype=h.DTYPE_INT)
            maze_pos: h.IntArr = maze_graph_pos*2

            cell = maze_graph[*maze_graph_pos]

            # open cell centers
            maze_buffer[*(maze_pos+1), h.GridOffsets.NO_WALL] = 1
            free_tiles.append(maze_pos+1)

            if cell == h.ALL_OUTGOING:
                allowed: list[h.Action] = h.ACTS_ARR[:-1]
            else:
                cell = cast(h.Action, cell)
                allowed: list[h.Action] = [cell]

            for d in allowed:
                new_pos: h.IntArr = maze_pos+h.ACT_TO_DIR_NDARRAY[d]
                if 0 <= new_pos[0] < exp_dim and 0 <= new_pos[1] < exp_dim:
                    # remove wall between cells
                    maze_buffer[*(new_pos+1), h.GridOffsets.NO_WALL] = 1
                    free_tiles.append(new_pos+1)

    return np.unique(np.array(free_tiles, dtype=h.DTYPE_INT), axis=0) # remove duplicates
