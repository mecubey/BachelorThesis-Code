"""
Contains method to plot generated data.
"""

from experiment_utils import (LABELS,
                              METRIC_AXES,
                              METRIC_LIMITS,
                              BASE_SPLITS,
                              PARAMETER_LIMITS,
                              MAP_NAMES,
                              Metric,
                              HazardDamageType,
                              PlannerType,
                              Parameter,
                              Data,
                              get_graph_name)
import matplotlib.pyplot as plt
import os
from matplotlib.axes import Axes
import matplotlib.image as mpimg

results_path = os.path.dirname(os.path.abspath(__file__)) + "/results/"
map_path = map_path = os.path.dirname(os.path.abspath(__file__)) + "/map_imgs/"

def subplot_data(*,
                 ax: Axes,
                 xpoints: list[float],
                 ha_ypoints: list[float],
                 max_ha_ypoints: list[float],
                 min_ha_ypoints: list[float],
                 nha_ypoints: list[float],
                 max_nha_ypoints: list[float],
                 min_nha_ypoints: list[float],
                 x_axis_title: str,
                 x_limits: list[int],
                 y_axis_title: str,
                 y_limits: list[int],
                 x_spec_ticks: list[str] = None):
    """
    Plot a subplot of given data.
    Always produces a graph for hazard aware and
    hazard unaware.
    """
    if x_spec_ticks is None:
        x_spec_ticks = ["0", "0.5", "1"]

    ax.plot(xpoints, ha_ypoints, label=LABELS[0])
    ax.fill_between(xpoints,
                    min_ha_ypoints,
                    max_ha_ypoints,
                    color="blue",
                    alpha=0.2)
    ax.plot(xpoints, nha_ypoints, label=LABELS[1])
    ax.fill_between(xpoints,
                    min_nha_ypoints,
                    max_nha_ypoints,
                    color="orange",
                    alpha=0.2)
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_xlim([x_limits[0], x_limits[-1]])
    ax.set_ylim([y_limits[0], y_limits[-1]])
    ax.set_xticks(x_limits, x_spec_ticks)

    if y_axis_title == Metric.SUCCESS_RATE:
        ax.set_yticks([0, 0.5,  1], ["0", "0.5", "1"])
    else:
        ax.set_yticks(y_limits)

def plot_data(*,
              data: Data,
              map_id: int,
              dmg_type: HazardDamageType,
              varied_param: Parameter):
    """
    Plot entire data object.

    Args:
        data (Data): Data object. Loaded from a pickled file.
        map_id (int): ID of the map.
        varied_param (Parameter): Parameter that was varied.
        base_splits (list[float]): Splits of specified parameter.
    """
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.tight_layout(pad=3.0, rect=[0, 0.1, 1, 0.99])

    img = mpimg.imread(f"{map_path}map{map_id}.png")
    ax[0, 0].imshow(img)
    ax[0, 0].axis('off')
    ax[0, 0].set_title(MAP_NAMES[map_id], fontsize=18)

    for m, ax_pos in METRIC_AXES.items():
        subplot_data(ax=ax[ax_pos],
                     xpoints=BASE_SPLITS,
                     ha_ypoints=data[m]["avg"][PlannerType.HAZARD_AWARE],
                     max_ha_ypoints=data[m]["max"][PlannerType.HAZARD_AWARE],
                     min_ha_ypoints=data[m]["min"][PlannerType.HAZARD_AWARE],
                     nha_ypoints=data[m]["avg"][PlannerType.HAZARD_UNAWARE],
                     max_nha_ypoints=data[m]["max"][PlannerType.HAZARD_UNAWARE],
                     min_nha_ypoints=data[m]["min"][PlannerType.HAZARD_UNAWARE],
                     x_axis_title=varied_param,
                     x_limits=PARAMETER_LIMITS[varied_param],
                     y_axis_title=m,
                     y_limits=METRIC_LIMITS[m])

    handles, labels = ax[1, 0].get_legend_handles_labels()

    # figure-level legend
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.01),
               fontsize=16,
               fancybox=True,
               ncol=2)

    plt.savefig(results_path+get_graph_name(varying_param=varied_param,
                                            dmg_type=dmg_type,
                                            map_idx=0), dpi=300)
