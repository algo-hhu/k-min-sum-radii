from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def plot_result(
    points: Sequence[Sequence[float]],
    clusters: Optional[Sequence[int]] = None,
    centers: Optional[Sequence[Sequence[float]]] = None,
    radii: Optional[Sequence[float]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    if len(points[0]) > 2:
        raise ValueError("Only 2D data is supported for plotting.")

    fig, ax = plt.subplots(figsize=(10, 10))

    _points = np.array(points)

    ax.scatter(_points[:, 0], _points[:, 1], c=clusters, s=50, cmap="Set2")

    if centers is not None and radii is not None:
        for i, ((x, y), radius) in enumerate(zip(centers, radii)):
            circle = Circle(
                (x, y), radius, fill=False, edgecolor="black", linestyle="--"
            )
            ax.add_patch(circle)
            ax.plot(x, y, "+", color="black")

            if clusters is not None:
                points_in_cluster = _points[clusters == i]
                furthest_point = points_in_cluster[
                    np.argmax(np.linalg.norm(points_in_cluster - centers[i], axis=1))
                ]
                ax.add_line(
                    plt.Line2D(
                        [centers[i][0], furthest_point[0]],
                        [centers[i][1], furthest_point[1]],
                        color="black",
                        linestyle="--",
                    )
                )

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
