# Experiment Setup
For each experiment, 30 set seeds were used to generate the corresponding enviroments.

## Independent Variables
- `hazard_consideration`
- `hazard_dmg_type`
- `num_agents`
- `field_dim`
- `maze_intensity`
- `spawn_prob`
- `spread_prob`
- `max_num_spread`
- `dir_spread_probs`
- `max_timestep`

## Baseline Configuration
Every experiment will vary at most one independent variable. The base configuration was chosen based on empirical observations - they do not make the enviroment trivially easy nor impossibly difficult. If otherwise specified, these values will be used:

- `consider_hazards=yes`
- `hazard_dmg_type=CONSTANT`
- `num_agents=300`
- `field_dim=16`
- `maze_intensity=0.2`
- `spawn_prob=0.4`
- `spread_prob=0.6`
- `max_num_spread=20`
- `dir_spread_probs=[0.7,0.7,0.7,0.7]`
- `max_timestep=300`

## Independent Variables to Vary
Each experiment will be done on each damage type for each hazard consideration.

- `spawn_prob`
- `spread_prob`
- `max_num_spread`
- `dir_spread_probs`

## Dependent Variables
### Hazard
- Maximum accumulated hazard damage taken
- Minimum accumulated hazard damage taken
- Average accumulated hazard damage taken

### Makespan
- Minimum amount of timesteps to finish episode
- Maximum amount of timesteps to finish episode
- Average amount of timesteps to finish episode

### Solution Quality
- Sum-Of-Costs divided by the sum over each agent's initial shortest distance to their goal
- Makespan divided by the sum over each agent's initial shortest distance to their goal

### Success Rate
- How many of the 30 seeds were actually sucessfully completed?