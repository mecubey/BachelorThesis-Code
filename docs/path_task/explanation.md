# PathTaskEnv (RLlib Multi-Agent Environment)

## Overview

`PathTaskEnv` is a cooperative multi-agent gridworld implemented using Ray RLlib’s `MultiAgentEnv` API. Multiple agents operate in a shared environment and must coordinate implicitly to complete spatially distributed tasks based on their trait vectors.

The environment is:
- Discrete-time
- Partially observable
- Multi-agent
- Cooperative

---

## Environment Structure

### Grid

The environment is based on a square grid:
```
grid_dim = 2 * field_dim - 1
```

The internal state (`global_obs`) includes padding for local observations:
```
(grid_dim + 2 * obs_radius,
 grid_dim + 2 * obs_radius,
 2 + 2 * num_agents)
```

Each cell encodes:
```
[is_wall,
 is_zone,
 agent_occupancy (num_agents),
 agent_goals (num_agents)]
```

---

## Environment Elements

### Walls (Maze)

- Generated via `gen_maze`.
- Controlled by `maze_intensity`.
- Ensures full connectivity of the grid.

### Depot

- A single tile sampled from free positions.
- Used as fallback goal when agents cannot contribute to tasks.

### Zone

A dynamic hazard with the following behavior:

- With probability `step_spread_prob`, the zone:
  - Spawns at a random free tile if inactive.
  - Otherwise spreads to neighboring tiles.
- Spread behavior:
  - Controlled by directional probabilities (`dir_spread_probs`).
  - Limited by `max_num_spread`.
- Once spreading is complete, the zone disappears.
- Agents stepping onto zone tiles receive a penalty.

---

## Agents

### Properties

Each agent has:
- Position `(x, y)`
- Trait vector (binary, size `trait_dim`)
- Current goal:
  - Task index or
  - Depot

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
4: don't move
```

- Agents act in random order.
- Movement updates grid occupancy.

### Collisions

- If multiple agents occupy the same tile:
  - All involved agents receive a collision penalty.
  - Their movement is reversed.

---

## Tasks

### Properties

Each task has:
- Position `(x, y)`
- Requirement vector (binary, size `trait_dim`)
- Planned contributions
- Actual contributions
- Completion flag

### Initialization

- Tasks are placed on unique free tiles.
- Requirements are randomly generated but guaranteed solvable.

### Mechanics

- When an agent reaches its assigned task:
  - Its traits are added to the task’s actual contributions.
- A task is completed when its requirements are satisfied.
- Completed tasks are counted globally.

---

## Task Assignment

Goals are assigned via `get_goal`.

### Strategy

The task planner uses the tasks planned contributions.

For each task:
- Evaluate contribution score based on:
  - Matching required traits.
  - Avoiding redundant traits.
- Prefer tasks where the agent meaningfully contributes.
- If an agent's contribution would finish a task:
  - That task is strongly preferred.

Fallback:
- If no task can be contributed to, the agent is assigned the depot.

---

## Observations

Each agent receives:
```
{"observation": 
  {"grid_obs": ...,
   "vec_obs": ...},
 "action_mask": ...}
```

### Grid Observation (`grid_obs`)

Shape:
```
(2 * obs_radius + 1,
 2 * obs_radius + 1,
 2 + 2 * num_agents)
```

Contents:
- Local view centered on the agent.
- Includes:
  - Walls
  - Zones
  - Agent occupancy
  - Goal indicators

Special handling:
- Goals of other agents are removed.
- Nearby agents’ goals are reinserted and clipped to the observation window.

### Vector Observation (`vec_obs`)
```
[dx_normalized, dy_normalized, distance_to_goal]
```

- Direction from agent to goal (normalized).
- Euclidean distance to goal.

### Action Mask

Each agent has an action mask:

- Prevents:
  - Moving into walls.
  - Moving into occupied tiles.
  - Leaving the grid.

---

## Rewards

### Step Reward

Each timestep:
```
step_penalty (default: -0.3)
```

### Distance Shaping

Additional reward:
```
(distance_to_goal / (2 * (grid_dim - 1)))
```

Encourages agents to move toward goals.

### Collision Penalty
```
collision_penalty (default: -2)
```

Applied when agents collide.

### Zone Penalty
```
zone_penalty (default: -1)
```

Applied when agent is on a zone tile.

### Completion Reward

When all tasks are completed:
```
all_tasks_finished_rwd (default: 20)
```

Given to all agents.

---

## Episode Termination

### Termination

Episode ends when:
```
num_tasks_finished == num_tasks
```

### Truncation

Episode is truncated when:
```
timestep == episode_length
```

---

## Execution Flow (Step)

1. Initialize rewards with step penalty.
2. Update or spawn zone.
3. Move agents (random order).
4. Resolve collisions (revert + penalty).
5. Apply zone penalties.
6. Process task contributions.
7. Reassign goals if needed.
8. Compute observations and action masks.
9. Add distance-based reward.
10. Check termination and truncation.

---

## Rendering

ASCII-based rendering:

- Agents: colored `A`
- Tasks: colored `T`
- Depot: `D`
- Walls: `#`
- Zone tiles: colored background

---