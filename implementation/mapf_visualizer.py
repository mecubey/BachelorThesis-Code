"""
Contains MAPFVisualizer class definition.
"""

from mapf_instance import MAPFInstance
from mapf_utils import Position
import numpy as np
import matplotlib.pyplot as plt

class MAPFVisualizer:
    """
    Represents a MAPF visualizer.
    """
    def __init__(self, instance: MAPFInstance):
        self.instance = instance

        self.colors = plt.colormaps["tab20"](np.linspace(0, 1, instance.num_agents))
        self.agent_labels = []
        self.figure = None
        self.ax = None
        self.image = None
        self.goal_scat = None
        self.agent_scat = None

        self.render_setup()

    def gen_img_from_grid(self):
        """
        Generate an image from the enviroment grid with
        walls and current zone tiles set.
        """

        # base: white free, black wall
        img = np.ones((self.instance.wall_map.width,
                       self.instance.wall_map.height, 3))

        for row in range(self.instance.wall_map.width):
            for col in range(self.instance.wall_map.height):
                pos = Position(row, col)
                if self.instance.hazard.on_hazard(pos):
                    img[row, col] = [1.0, 0.7, 0.7]

                if self.instance.wall_map.on_wall(pos):
                    img[row, col] = [0, 0, 0]

        return img

    def render_setup(self) -> None:
        """
        Sets up rendering variables.
        """
        plt.ion()
        self.figure, self.ax = plt.subplots()

        img = self.gen_img_from_grid()

        self.image = self.ax.imshow(img, interpolation="nearest")

        self.ax.set_xticks(np.arange(img.shape[1]))
        self.ax.set_yticks(np.arange(img.shape[0]))

        self.ax.set_xticks(np.arange(-0.5, img.shape[1], 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, img.shape[0], 1), minor=True)

        self.ax.grid(which="minor", color="gray", linewidth=1)
        self.ax.set_axisbelow(False)

        self.ax.tick_params(
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        goal_pos = np.array([a.goal_pos.as_ndarray() for a in self.instance.agents])
        agent_pos = np.array([a.current_pos.as_ndarray() for a in self.instance.agents])

        self.goal_scat = self.ax.scatter(
            goal_pos[:, 1],
            goal_pos[:, 0],
            c=self.colors,
            marker="P",
            edgecolors="black",
            s=3500 / self.instance.wall_map.width,
        )

        self.agent_scat = self.ax.scatter(
            agent_pos[:, 1],
            agent_pos[:, 0],
            c=self.colors,
            marker="o",
            edgecolors="black",
            s=2000 / self.instance.wall_map.width,
        )

        for i, pos in enumerate(agent_pos):
            text = self.ax.text(
                pos[1],           # x
                pos[0],           # y
                str(i),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
            self.agent_labels.append(text)

    def render(self) -> None:
        """
        Renders the environment.
        """
        img = self.gen_img_from_grid()
        self.image.set_data(img)

        goal_pos = np.array([a.goal_pos.as_ndarray() for a in self.instance.agents])
        agent_pos = np.array([a.current_pos.as_ndarray() for a in self.instance.agents])

        self.goal_scat.set_offsets(np.c_[goal_pos[:, 1], goal_pos[:, 0]])
        self.agent_scat.set_offsets(np.c_[agent_pos[:, 1], agent_pos[:, 0]])

        for i, pos in enumerate(agent_pos):
            self.agent_labels[i].set_position((pos[1], pos[0]))

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
