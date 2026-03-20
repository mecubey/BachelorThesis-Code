# PathTaskEnv (PettingZoo Parallel Environment)

## Overview

`PathTaskEnv` is a **multi-agent cooperative gridworld environment** implemented using the PettingZoo `ParallelEnv` API. Multiple agents move on a 2D grid and must **collaborate to complete spatially distributed tasks**. Each task requires a combination of agent traits and takes multiple timesteps to complete.

* The environment is a **discrete-time, multi-agent MDP**.
* Due to the nature of the `ParallelEnv` API, agents act **simultaneously** at each timestep.
* Tasks require **joint execution** based on agent capabilities.

---

## Grid World

* Square grid of size:

```
field_dim × field_dim
```

* Agents and tasks occupy discrete grid cells
* Multiple agents **can occupy the same cell**
* There are **no two tasks on the same cell**

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

### Task Mechanics

* A task progresses only if the **combined traits of agents on its tile meet its requirements** (agents must choose "**execute task**" to contribute their trait vector)
* Completion requires **multiple timesteps** of successful execution
* All tasks are guaranteed to be solvable

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

* Prevents moving outside the grid
* Disables `execute task` if:

  * Agent is not on a task tile
  * Task is already completed

---

## Observation Space

Each agent observes a **vector of integers**:

```
Box(low=-field_dim, high=field_dim, shape=(...), dtype=int64)
```

### Observation Structure

#### 1. Static Information

* All agent traits:

```
num_agents × trait_dim
```

* All task descriptions:

```
num_tasks × (trait_dim + 2)
```

Includes:

* Task requirements
* Task reward
* Task execution time

#### 2. Relative Positions

Relative positions to:

* All agents
* All tasks

Encoded as:

```
(dx, dy)
```

Total:

```
2 × (num_agents + num_tasks)
```

#### 3. Task Progress

For each task:

* Execution progress
* Completion flag (0 or 1)

Total:

```
2 × num_tasks
```

### Final Observation Size

```
num_agents * trait_dim
+ num_tasks * (trait_dim + 2)
+ 2 * (num_agents + num_tasks)
+ 2 * num_tasks
```

---

## Rewards

Rewards are given **only when a task is completed**.

For a completed task:

* `R` = task reward
* `T` = execution time
* `c_i` = number of timesteps agent **i** contributed

```
reward_i = R * (c_i / T)
```

---

## Episode Termination/Truncation

### Termination

Episode ends when:

* **All tasks are completed**

### Truncation

Episode is truncated when:

* `timestep >= episode_length`

---

## Transition Dynamics

At each timestep:

1. Agents act simultaneously
2. Movement updates positions
3. Execution actions register contributions
4. Task requirements are checked
5. Tasks progress if requirements are met
6. Rewards are assigned
7. Observations and masks are updated

---

## Rendering

Optional ASCII rendering:

* `A` → agent
* `T` → unfinished task

---

For more information, see 

```
env/path_task_env/implementation/path_task_env.py`
```

for the enviroment implementation.

If you wish to configure enviroment parameters and import the enviroment, use

```
env/path_task_env/path_task_env_v0.py
```
