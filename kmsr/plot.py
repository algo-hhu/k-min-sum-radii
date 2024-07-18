from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def plot_cluster(
    points: Sequence[Sequence[float]],
    clusters: Optional[Sequence[int]] = None,
    balls: Optional[Sequence[Sequence[float]]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    if len(points[0]) > 2:
        raise ValueError("Only 2D data is supported for plotting.")

    fig, ax = plt.subplots(figsize=(10, 10))

    _points = np.array(points)

    ax.scatter(_points[:, 0], _points[:, 1], c=clusters, s=50, cmap="Set2")

    if balls is not None:
        for x, y, radius in balls:
            circle = Circle(x, y, radius, fill=False, edgecolor="black")
            ax.add_patch(circle)
            ax.plot(x, y, "+", color="black")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    ax.set_aspect("equal")

    if output_path is not None:
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    if show:
        plt.show()

    plt.close()
