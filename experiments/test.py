import sys
sys.path.insert(0, '')

from implementation.pibt import PIBT
from implementation.mapf_visualizer import MAPFVisualizer
from implementation.mapf_instance import MAPFInstance
from implementation.hazard import HazardConfig, Hazard
from implementation.wall_map import WallMap
from implementation.scene import SceneManager
from implementation.mapf_utils import (GLOBAL_HAZARD_SEED,
                                       GLOBAL_SOLVER_SEED,
                                       Position)
from pprint import pprint
import numpy as np

wall_map = WallMap("random-32-32-10")
scene_manager = SceneManager(wall_map=wall_map,
                             n_scenes=1,
                             even_or_random="random")

instance = MAPFInstance(max_timestep=300,
                        hazard_config=HazardConfig.from_config("additive_difficult"),
                        hazard_seed=GLOBAL_HAZARD_SEED,
                        wall_map=wall_map,
                        scene=scene_manager.scenes[0])

for _ in range(50):
    instance.add_agent()

visualizer = MAPFVisualizer(instance)

solver = PIBT(width=instance.wall_map.width,
              height=instance.wall_map.height,
              seed=GLOBAL_SOLVER_SEED)
solver.set_instance(instance)
solver.reset()
solver.set_hazard_awareness(True)

#visualizer.render()
#input()

while True:
    instance.hazard_step()
    instance.move_all_agents(solver.step())
    instance.progress()
    #visualizer.render()
    #input()

    if instance.finished():
        print(f"success: {instance.succeeded()}")
        print(f"soc: {(instance.path_manager.calc_soc() /
                      instance.scene.calc_optimal_soc(instance.num_agents))}")
        print(f"makespan: {(instance.path_manager.calc_makespan() /
                           instance.scene.calc_optimal_makespan(instance.num_agents))}")
        break
