from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def plot_2d_ax(
    points: np.ndarray,
    clusters: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    ax: plt.Axes,
    title: Optional[str] = None,
) -> None:
    ax.scatter(points[:, 0], points[:, 1], c=clusters, s=50, cmap="Set2")

    if centers is not None and radii is not None:
        for i, ((x, y), radius) in enumerate(zip(centers, radii)):
            circle = Circle(
                (x, y), radius, fill=False, edgecolor="black", linestyle="--"
            )
            ax.add_patch(circle)
            ax.plot(x, y, "+", color="black")

            if clusters is not None:
                points_in_cluster = points[clusters == i]
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
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    ax.set_aspect("equal")


def plot_result(
    points: Sequence[Sequence[float]],
    clusters: Optional[Sequence[int]] = None,
    centers: Optional[Sequence[Sequence[float]]] = None,
    radii: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
    transparent: bool = True,
) -> None:
    if len(points[0]) > 2:
        raise ValueError("Only 2D data is supported for plotting.")

    fig, ax = plt.subplots(figsize=(10, 10))

    plot_2d_ax(np.array(points), clusters, centers, radii, ax, title)

    if output_path is not None:
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            transparent=transparent,
        )
    if show:
        plt.show()

    plt.close()


def plot_multiple_results(
    points: Sequence[Sequence[float]],
    clusterings: Optional[Sequence[Optional[Sequence[int]]]] = None,
    centers: Optional[Sequence[Optional[Sequence[Sequence[float]]]]] = None,
    radii: Optional[Sequence[Optional[Sequence[float]]]] = None,
    titles: Optional[Optional[Sequence[str]]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
    transparent: bool = True,
) -> None:
    if len(points[0]) > 2:
        raise ValueError("Only 2D data is supported for plotting.")

    if clusterings is not None:
        ll = len(clusterings)
    elif centers is not None:
        ll = len(centers)
    elif radii is not None:
        ll = len(radii)
    else:
        ll = 1

    fig, axs = plt.subplots(1, ll, figsize=(10 * ll, 10))

    if clusterings is None:
        clusterings = [None] * ll

    if centers is None:
        centers = [None] * ll

    if radii is None:
        radii = [None] * ll

    if titles is None:
        titles = [f"Plot {i}" for i in range(ll)]

    for params in zip(clusterings, centers, radii, axs, titles):
        plot_2d_ax(np.array(points), *params)

    if output_path is not None:
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            transparent=transparent,
        )
    if show:
        plt.show()
