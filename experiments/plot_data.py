"""
Script to plot data.
"""

import sys
sys.path.insert(0, '')
import pickle
import matplotlib.pyplot as plt
from implementation.mapf_utils import EXPERIMENT_DIR
from pprint import pprint


title = "random-32-32-10_random_(Minefield)_additive"

data = pickle.load(open(EXPERIMENT_DIR / "results" / title, "rb"))

#pprint(data)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

x = data["agents"]

labels = ["HUA", "HA"]

markers = ["o", "s"]

def min_max_norm(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

# ----- Subplot 1 -----
makespan_min = min(data["makespans"][0].min(), data["makespans"][1].min())
makespan_max = max(data["makespans"][0].max(), data["makespans"][1].max())
axes[0].plot(x, min_max_norm(data["makespans"][0], makespan_min, makespan_max),
             label=labels[0], marker=markers[0])
axes[0].plot(x, min_max_norm(data["makespans"][1], makespan_min, makespan_max),
             label=labels[1], marker=markers[1])
axes[0].set_xlabel("agents")
axes[0].set_ylabel("makespan")
axes[0].legend()

# ----- Subplot 2 -----
soc_min = min(data["socs"][0].min(), data["socs"][1].min())
soc_max = max(data["socs"][0].max(), data["socs"][1].max())
axes[1].plot(x, min_max_norm(data["socs"][0], soc_min, soc_max),
             label=labels[0], marker=markers[0])
axes[1].plot(x, min_max_norm(data["socs"][1], soc_min, soc_max),
             label=labels[1], marker=markers[1])
axes[1].set_xlabel("agents")
axes[1].set_ylabel("soc")
axes[1].legend()

# ----- Subplot 3 -----
axes[2].plot(x, data["success_rates"][0], label=labels[0], marker=markers[0])
axes[2].plot(x, data["success_rates"][1], label=labels[1], marker=markers[1])
axes[2].set_xlabel("agents")
axes[2].set_ylabel("success rate")
axes[2].legend()

fig.suptitle(title, fontsize=16)

for ax in axes:
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

plt.tight_layout()
plt.show()
