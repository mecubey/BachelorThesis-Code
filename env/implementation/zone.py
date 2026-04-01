import numpy as np

class Zone:
    def __init__(self, max_num_spread: int, spread_probs: float, zone_dmg: float):
        self.spread_probs = spread_probs
        self.directions = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        self.occupied_tiles = []
        self.zone_dmg = zone_dmg
        self.spread_progress = 0
        self.max_num_spread = max_num_spread

    def spawn(self, start_pos, grid_obs, zone_offset):
        self.occupied_tiles.append(start_pos)
        grid_obs[*start_pos, zone_offset] = 1

    def spread(self, grid_obs, dirs_offsets, zone_offset, rng: np.random.Generator):
        self.spread_progress += 1
        newly_occupied = []
        for pos in self.occupied_tiles:
            no_walls = grid_obs[*pos, dirs_offsets[0]:dirs_offsets[1]]

            for i in range(len(no_walls)):
                if no_walls[i]: # can only spread if there isn't a wall there
                    new_pos = (pos[0]+self.directions[i][0], pos[1]+self.directions[i][1])
                    # can only spread if next tile hasn't been infected yet, and probability hits
                    if not grid_obs[*new_pos, zone_offset] and rng.random() <= self.spread_probs[i]:
                        grid_obs[*new_pos, zone_offset] = 1
                        newly_occupied.append(new_pos)

        self.occupied_tiles += newly_occupied

    def finished(self):
        return self.spread_progress >= self.max_num_spread
    
    def empty(self):
        return len(self.occupied_tiles) == 0

    def remove(self, grid_obs, zone_offset):
        for pos in self.occupied_tiles:
            grid_obs[*pos, zone_offset] = 0
        self.occupied_tiles = []
        self.spread_progress = 0
