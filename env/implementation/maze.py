import numpy as np
from .enums import States

def generate_maze(dim, maze_intensity, seed = None, iter_c = 10):
    rng = np.random.default_rng(seed)

    if dim == 1:
        return np.array([[4]])

    # generate initial perfect maze
    maze = np.array([[1]*(dim-1)+[2] if i != dim-1 else [1]*(dim-1)+[-1] for i in range(dim)])

    for i in range(dim*dim*iter_c):
        # step 1: have origin node point to random neigbouring node
        origin_x, origin_y = np.where(maze == States.ORIGIN)
        origin_x = origin_x[0]
        origin_y = origin_y[0]

        # make sure origin does not point out of maze
        directions = [States.UP, States.RIGHT, States.DOWN, States.LEFT]
        if origin_x == 0:
            directions.remove(0)
        if origin_y == dim-1:
            directions.remove(1)
        if origin_x == dim-1:
            directions.remove(2)
        if origin_y == 0:
            directions.remove(3)
        new_direction = rng.choice(directions)
        maze[origin_x][origin_y] = new_direction

        if i == dim*dim*iter_c-1: # don't want origin in output, so break at second to last iteration
            break
        
        # step 2: that neigbouring node becomes the new origin
        # step 3: remove the new origin node's pointer
        if new_direction == States.UP:
            maze[origin_x-1][origin_y] = States.ORIGIN
        if new_direction == States.RIGHT:
            maze[origin_x][origin_y+1] = States.ORIGIN
        if new_direction == States.DOWN:
            maze[origin_x+1][origin_y] = States.ORIGIN
        if new_direction == States.LEFT:
            maze[origin_x][origin_y-1] = States.ORIGIN

    inds = [[x0, y0] for x0 in list(range(dim)) for y0 in list(range(dim))]
    freed_tiles_inds = rng.choice(inds, replace=False, size=int((1-maze_intensity)*dim*dim))
    for ind in freed_tiles_inds:
        maze[ind[0]][ind[1]] = States.ALL_OUTGOING

    return maze

opp_dirs = {
    States.UP: (States.DOWN, -1, 0),
    States.RIGHT: (States.LEFT, 0, 1),
    States.DOWN: (States.UP, 1, 0),
    States.LEFT: (States.RIGHT, 0, -1),
}

def is_dir_avail(maze, pos, dir):
    opp_dir, opp_row, opp_col = opp_dirs[dir]
    
    if not maze[pos[0]][pos[1]] == dir and not maze[pos[0]+opp_row][pos[1]+opp_col] == opp_dir and \
       not maze[pos[0]][pos[1]] == States.ALL_OUTGOING and not maze[pos[0]+opp_row][pos[1]+opp_col] == States.ALL_OUTGOING:
        return False
    
    return True