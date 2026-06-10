# PathTaskEnv (RLlib Multi-Agent Environment)

## Overview

`PathTaskMutliAgentEnv` is a multi-agent gridworld enviroment. Multiple agents operate in a shared environment and must coordinate implicitly so that each one of them arrives at their destination at the same time.

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

#### Hazard Damage Formulations
$\lambda$ - hazard damage

Hazard damage itself is expressed as increased cost to travel to a vertex $v$.

Additionally, to encourage agents to finish episodes, an additional factor 

$episode\_progress = \frac{current\_timestep}{max\_timestep}$

can be introduced to decrease hazard damage the farther along agents are in an episode. This makes it so agents are encouraged to keep staying on their goals.
##### Constant
$\lambda$ if agent is on a hazard tile, otherwise `0`.

##### Distance
$max(0, \lambda - \alpha * d(v, c))$

Based on linear distance decay, the damage gradually decreases the farther away an agent is from the zone center $c$.

$\alpha$ controls how much damage dissappears per tile of distance.

$\alpha = \frac{\lambda}{0.1 * map\_width}$

---

## Agents

### Properties

Each agent has:
- Position in the grid `(x, y)`

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



How do you model dynamically spreading hazards (for MAPF)?

How well does HA perform against HUA algorithm?

How do different damage formulations affect the produced plans?