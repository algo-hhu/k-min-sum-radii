from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle


def plot_2d_ax(
    points: np.ndarray,
    ax: plt.Axes,
    clusters: Optional[Sequence[int]] = None,
    centers: Optional[Sequence[Sequence[float]]] = None,
    radii: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    edge_color: str = "black",
    cmap: str = "plasma",
) -> None:
    ax.scatter(points[:, 0], points[:, 1], c=clusters, s=20, cmap=cmap)
    _centers = np.array(centers)
    ax.scatter(_centers[:, 0], _centers[:, 1], c="red", s=50, alpha=0.5, marker="X")

    if centers is not None and radii is not None:
        for i, ((x, y), radius) in enumerate(zip(centers, radii)):
            circle = Circle(
                (x, y),
                radius,
                fill=False,
                edgecolor=edge_color,
                linestyle="--",
                alpha=0.75,
            )
            ax.add_patch(circle)

            if clusters is not None:
                _clusters = np.array(clusters)
                points_in_cluster = points[_clusters == i]
                furthest_point = points_in_cluster[
                    np.argmax(np.linalg.norm(points_in_cluster - centers[i], axis=1))
                ]
                ax.add_line(
                    plt.Line2D(
                        [centers[i][0], furthest_point[0]],
                        [centers[i][1], furthest_point[1]],
                        color=edge_color,
                        linestyle="--",
                    )
                )
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    ax.set_aspect("equal")


def plot_3d_ax(
    points: np.ndarray,
    ax: plt.Axes,
    clusters: Optional[Sequence[int]] = None,
    centers: Optional[Sequence[Sequence[float]]] = None,
    radii: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    edge_color: str = "black",
    cmap: str = "plasma",
) -> None:
    ax.scatter(  # type: ignore
        points[:, 0], points[:, 1], points[:, 2], c=clusters, s=20, cmap=cmap
    )
    _centers = np.array(centers)
    ax.scatter(  # type: ignore
        _centers[:, 0],
        _centers[:, 1],
        _centers[:, 2],
        c="red",
        s=50,
        alpha=0.5,
        marker="X",
    )

    if centers is not None and radii is not None:
        for i, ((c_x, c_y, c_z), radius) in enumerate(zip(centers, radii)):
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]  # type: ignore
            x = c_x + radius * np.cos(u) * np.sin(v)
            y = c_y + radius * np.sin(u) * np.sin(v)
            z = c_z + radius * np.cos(v)
            ax.plot_wireframe(  # type: ignore
                x, y, z, linestyle="--", color=edge_color, alpha=0.3
            )

            if clusters is not None:
                _clusters = np.array(clusters)
                points_in_cluster = points[_clusters == i]
                furthest_point = points_in_cluster[
                    np.argmax(np.linalg.norm(points_in_cluster - centers[i], axis=1))
                ]

                ax.plot(
                    [_centers[i, 0], furthest_point[0]],
                    [_centers[i, 1], furthest_point[1]],
                    [_centers[i, 2], furthest_point[2]],
                    color=edge_color,
                    linestyle="--",
                )
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")  # type: ignore

    ax.set_aspect("equal")


def plot_ax(points: np.ndarray, *args: Any, **kwargs: Any) -> None:
    if len(points[0]) == 2:
        plot_2d_ax(points, *args, **kwargs)
    elif len(points[0]) == 3:
        plot_3d_ax(points, *args, **kwargs)
    else:
        raise ValueError("Only 2D and 3D data is supported for plotting.")


def get_projection(dim: int) -> Optional[str]:
    if dim == 2:
        return None
    elif dim == 3:
        return "3d"
    else:
        raise ValueError("Only 2D and 3D data is supported for plotting.")


def plot_result(
    points: Sequence[Sequence[float]],
    clusters: Optional[Sequence[int]] = None,
    centers: Optional[Sequence[Sequence[float]]] = None,
    radii: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    *,
    cmap: str = "plasma",
    output_path: Optional[Path] = None,
    transparent: bool = True,
    show: bool = True,
    close: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw=dict(projection=get_projection(len(points[0])))
    )

    plot_ax(np.array(points), ax, clusters, centers, radii, title, cmap=cmap)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            transparent=transparent,
        )
    if show:
        plt.show()

    if close:
        plt.close()

    return fig, ax


def plot_multiple_results(
    points: Sequence[Sequence[float]],
    clusters: Optional[Sequence[Optional[Sequence[int]]]] = None,
    centers: Optional[Sequence[Optional[Sequence[Sequence[float]]]]] = None,
    radii: Optional[Sequence[Optional[Sequence[float]]]] = None,
    title: Optional[Optional[Sequence[str]]] = None,
    *,
    cmap: str = "plasma",
    output_path: Optional[Path] = None,
    transparent: bool = True,
    show: bool = True,
    close: bool = True,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    proj = get_projection(len(points[0]))

    if clusters is not None:
        ll = len(clusters)
    elif centers is not None:
        ll = len(centers)
    elif radii is not None:
        ll = len(radii)
    else:
        ll = 1

    if ll == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=proj))
        axs: List[Axes] = [ax]
    else:
        fig, axs = plt.subplots(  # type: ignore
            1,
            ll,
            figsize=(10 * ll, 10),
            subplot_kw=dict(projection=proj),
        )

    if clusters is None:
        clusters = [None] * ll

    if centers is None:
        centers = [None] * ll

    if radii is None:
        radii = [None] * ll

    if title is None:
        title = [f"Plot {i}" for i in range(ll)]

    for params in zip(axs, clusters, centers, radii, title):
        plot_ax(np.array(points), *params, cmap=cmap)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            transparent=transparent,
        )

    if show:
        plt.show()

    if close:
        plt.close()

    return fig, axs
