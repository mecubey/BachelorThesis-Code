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
             dim: int,
             maze_intensity: float,
             seed: int|None = None,
             iter_c: int = 10) -> tuple[h.BoolArr, list[h.Position]]:
    """
    Generate a maze according to a maze graph (directed graph). \n
    Returns maze and a list of 2D indices which indicate wall-free tiles.
    """
    exp_dim: int = 2*dim-1

    maze_buffer: h.BoolArr = np.ones((exp_dim+2, exp_dim+2), dtype=h.DTYPE_BOOL)

    rng = np.random.default_rng(seed)

    maze_graph: h.FloatArr = gen_maze_graph(dim=dim,
                                            maze_intensity=maze_intensity,
                                            rng=rng,
                                            iter_c=iter_c)

    # randomly remove walls from cells which are not reached by the graph
    unreach_idx_base: h.IntArr = np.arange(2, 2*dim-1, 2, dtype=h.DTYPE_INT)
    unreach_no_walls: h.BoolArr = rng.choice([1, 0],
                                             size=((dim-1)*(dim-1)),
                                             p=[maze_intensity,
                                             1-maze_intensity])
    grid_positions: h.IntArr = np.array(list(product(unreach_idx_base, unreach_idx_base)),
                                        dtype=h.DTYPE_INT)
    maze_buffer[grid_positions[:, 0],
                grid_positions[:, 1]] = unreach_no_walls

    free_tiles: set[h.Position] = set()
    for i in range(exp_dim):
        for j in range(exp_dim):
            if maze_buffer[i, j] == 0:
                free_tiles.add(h.Position(i, j))

    for i in range(dim):
        for j in range(dim):
            maze_graph_pos: h.Position = h.Position(i, j)
            maze_pos: h.Position = maze_graph_pos*2

            cell = maze_graph[*maze_graph_pos]

            # open cell centers
            maze_buffer[*(maze_pos+1)] = 0
            free_tiles.add(maze_pos+1)

            if cell == h.ALL_OUTGOING:
                allowed: list[h.Action] = h.ACTS_ARR[:-1]
            else:
                cell = cast(h.Action, cell)
                allowed: list[h.Action] = [cell]

            for d in allowed:
                new_pos: h.Position = maze_pos+h.ACT_TO_DIR[d]
                if 0 <= new_pos.x < exp_dim and 0 <= new_pos.y < exp_dim:
                    # remove wall between cells
                    maze_buffer[*(new_pos+1)] = 0
                    free_tiles.add(new_pos+1)

    return maze_buffer, list(free_tiles)
