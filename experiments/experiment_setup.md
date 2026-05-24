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

- `consider_hazards=True`
- `with_decay=False`
- `hazard_dmg_type=HazardDamageType.CONSTANT`
- `num_agents=200`
- `field_dim=16`
- `maze_intensity=0.2`
- `spawn_prob=0.7`
- `spread_prob=0.4`
- `max_num_spread=10`
- `dir_spread_probs=[0.8, 0.8, 0.8, 0.8]`
- `max_timestep=150`
- `render_mode=None`

## Independent Variables to Vary
Each experiment will be done on each damage type for each hazard consideration.

- `spawn_prob`
- `spread_prob`
- `max_num_spread`
- `dir_spread_probs`

## Dependent Variables
Makespan is calculated only over the successfull episodes.

Every other metric uses all episodes.

### Total Hazard Damage
- Maximum
- Minimum
- Average

### Solution Quality Divided by Sum(dist(s_i, g_i))
- Maximum
- Minimum
- Average

### Makespan Divided by max(dist(s_i, g_i))
- Maximum
- Minimum
- Average

### Success Rate
- How many episodes were successfully completed? (as percentage)