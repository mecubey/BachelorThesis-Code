# PathTaskEnv (RLlib Multi-Agent Environment)

## Overview

`PathTaskMutliAgentEnv` is a multi-agent gridworld enviroment. Multiple agents operate in a shared environment and must coordinate implicitly so that each one of them arrives at their destination.

---

## Environment Structure

### Grid

The environment is based on a square grid:
```
grid_dim = 2 * field_dim - 1
```

Each cell encodes:
```
[is_wall,
 is_zone,
 is_agent]
```

---

## Environment Elements

### Walls (Maze)

- Generated via `gen_maze`.
- Controlled by `maze_intensity`.
- No agent or agent goal is ever fully surrounded by walls.

### Zone

A dynamic hazard with the following behavior:

- With probability `step_spread_prob`, the zone...
  - spawns at a random free tile if inactive.
  - otherwise spreads to neighboring tiles.
- Spread behavior:
  - Controlled by directional probabilities (`dir_spread_probs`).
  - Limited by `max_num_spread`.
- After a set amount of timesteps, the zone disappears.
- Only one zone can be active at a time.

#### Zone Damage Formulations
[INSERT HERE]

---

## Agents

### Properties

Each agent has:
- Position in the grid `(x, y)`
- Goal position `(x, y)`

Agents are named:
```
agent_0, ..., agent_{n-1}
```

### Movement

Agents act each timestep with actions mapped to directions:
```
0: up
1: right
2: down
3: left
4: do nothing
```

- `agent_0` acts first, then `agent_1`, ...
- Movement updates grid occupancy.

### Collisions

- If multiple agents occupy the same tile:
  - Their movement is reversed.

---

## Episode Termination

### Termination

Episode ends once all agents are on their respective goals simultaneously.

### Truncation

Episode is truncated when:
```
timestep == episode_length
```

---

## Rendering
Enviroment grid is rendered through `matplotlib`.

Zones are light red tiles

Walls are black tiles.

Colored circles are agents, colored crosses are goals. Each agent and its' respective goal have the same color.

---