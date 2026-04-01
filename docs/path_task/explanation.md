# PathTaskEnv (PettingZoo Parallel Environment)

## Overview

`PathTaskEnv` is a **multi-agent cooperative gridworld environment** implemented using the PettingZoo `ParallelEnv` API. Multiple agents move on a 2D grid and must **collaborate to complete spatially distributed tasks.** Each task requires a combination of agent traits and takes multiple timesteps to complete.

* The environment is a **discrete-time, multi-agent MDP.**
* Due to the nature of the `ParallelEnv` API, all agents act **simultaneously** at each timestep.

---

## Grid World

* Square grid of size:

```
field_dim × field_dim
```

* Agents and tasks occupy discrete grid cells.
* Multiple agents **can occupy the same cell.**
* There are **no two tasks on the same cell.**

### Walls
Additionally, walls can be generated.
* The `maze_intensity` parameter controls the amount of walls generated in the grid. 
  * If `maze_intensity = 0`, there will be no walls at all.
  * If `maze_intensity = 1`, it will be a perfect maze, meaning any space can be accessed from any other space (there are no walled off areas). Note that at any value of `maze_intensity`, there will be no walled off areas.

### Zone
Additionally, zones which affect the agents can be generated.
* A `Zone` will be generated periodically at a random position, spread for a certain amount of timesteps according to a specified probability vector and then dissappear. Only one `Zone` can be active at any time.
* To activate the `Zone`, set `with_zone` to `True`.
* Set `step_spread_prob` to a value between `0` and `1`. This parameter controls the probability in which the `Zone` can spread in a single timestep.
* `max_num_spread` controls how many times a single `Zone` can spread in total.
* `spread_probs` controls the probability for each direction the `Zone` can spread to. It is an array with 4 values.
  * Example: `[0.25, 0.5, 0.8, 0.1]`
  * If a `Zone` is on a tile, it has a 25% chance to spread upwards, a 50% chance to spread to the right, an 80% chance to spread downwards, and a 10% chance to spread to the left.
  * Note that if there is a wall above the tile, the `Zone` cannot spread upwards. The same goes for the other directions.
* `zone_dmg` controls the amount of damage (negative reward) agents receive if they are on the same tile as a `Zone`.

---

## Agents

* Number of agents: `num_agents`
* Agents are named:

```
agent_0, agent_1, ..., agent_{n-1}
```

### Agent Properties

Each agent has:

* A **position** on the grid: `(x, y)`
* A **trait vector**: binary vector of size `trait_dim`

Example:

```
[1, 0, 1]
```

Meaning that this agent possesses the trait at index 0 and 1.

---

## Tasks

Number of tasks: `num_tasks`

Each task has:

* **Position** `(x, y)`
* **Requirement vector** (size `trait_dim`)
* **Execution time**
* **Reward**
* **Execution progress**
* **Finished flag**
* **Trait Contribution History**

### Task Mechanics

* A task progresses only if the **combined traits of agents on its tile meet its requirements** (agents must choose "**execute task**" to contribute their trait vector).
* Completion requires **multiple timesteps** of successful execution
* All tasks are guaranteed to be solvable by the agents present.
* Each time a set of agents (called coalition) contributes their trait vector to a task, their individual traits will be recorded in the task's **trait contribution history.**

Note that the amount of traits of agents and requirements of tasks is determined by `trait_dim` and is the same for all agents and tasks.

---

## Action Space

Each agent has a **discrete action space**:

```
Discrete(5)
```

| Action | Meaning      | Effect       |
| ------ | ------------ | ------------ |
| 0      | Move up      | (-1, 0)      |
| 1      | Move right   | (0, +1)      |
| 2      | Move down    | (+1, 0)      |
| 3      | Move left    | (0, -1)      |
| 4      | Execute task | Work on task |

### Action Mask

Each agent receives an `action_mask` that:

* Prevents moving outside the grid.
* Prevents moving against a wall.
* Disables `execute task` if:

  * Agent is not on a task tile.
  * Task is already completed.

---

## Observation Space

Each agent observes a **dictionary consisting of a vector consisting of a one hot vector and information relative to all other agents and tasks, as well as a local view of the grid**:

```
{"vec_obs": 
 spaces.Box(low=-np.inf,
            high=np.inf,
            shape=(self.num_agents + 
                   np.prod(self.agent_obs.shape) + 
                   np.prod(self.task_obs.shape),),
            dtype=np.float32
            ),
 "grid_obs": 
 spaces.Box(low=-np.inf,
            high=np.inf,
            shape=(self.obs_radius, 
                   self.obs_radius, 
                   6 + int(self.with_zone)),
            dtype=np.float32
            )} 
```

### Structure of Vector Observation

#### 1. One Hot Vector
```
num_agents
```
One hot vector that corresponds to an agent.

#### 2. Relative Agent Information
```
num_agents * (2 + trait_dim)
```
Contains `dx, dy, traits`. 

`dx, dy` are calculated relative to every agent.

#### 3. Relative Task Information
```
num_tasks * (6 + 2 * trait_dim)
```
Contains `dx, dy, requirements, remaining_requirements, reward, execution_time, execution_progress, finished_status`. 

`dx, dy` are calculated relative to every task.

### Structure of Grid Observation

An agent's view of the grid is determined by `obs_radius`.

Example: `obs_radius = 1`
```
*---*---*---*
|           |
*   *   *   *
| 2   3   4 |
*   *   *   *
| 1   A   5 |
*---*---*---*
  6   7   8
```
This agent can observe 1-8 (as well as its' own tile). Note that since 6, 7, 8 are outside of the grid, all their internal cell values would be set to `0`.

#### 1. Directional Encoding
```
1 + 1 + 1 + 1
```
* `1` if the cell has an upper wall, otherwise 0.
* `1` if the has a right wall, otherwise 0.
* `1` if the has a bottom wall, otherwise 0.
* `1` if the cell has a left wall, otherwise 0.

#### 2. Task Information
```
1
```
`1` if a task on that cell, `0` otherwise.

#### 3. Agent Information
```
1
```
`1` if an agent on that cell, `0` otherwise.

#### 4. Zone Information
```
1
```
Set to `1` if the `Zone` has spread to that tile, `0` otherwise.

---

## Rewards

There is a configurable reward of `step_rwd` at each step. To encourage agents to not waste time with useless actions, this should be optimally a small negative value, although this can be set to any value.

Rewards are also given **when a task is completed.** Agents who are on the tile at time of task completion receive the reward.

At the end of the episode, each task will calculates its' episode reward. The reward is negative and its' optimal value is 0. It aims to minimize the average wasted ability ratio on each task, while also encouraging agents to finish all tasks as quickly as possible. For its' exact implementation, please see `env/implementation/task.py` and `env/implementation/path_task.py`.

---

## Episode Termination/Truncation

### Termination

Episode ends when:

* **All tasks are completed.**

### Truncation

Episode is truncated when:

* `timestep >= episode_length`

---

## Rendering

Optional ASCII rendering:

* `A` → agent
* `T` → unfinished task

---

For more information, see 

```
env/path_task/implementation/path_task.py`
```

for the enviroment implementation.

If you wish to configure enviroment parameters and import the enviroment, use

```
env/path_task/path_task_v0.py
```
