import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def plot_cluster(cluster_file, ball_file, plot_file, radii, algorithm):
    plt.close("all")
    plt.clf()

    points = pd.read_csv(cluster_file, header=None, names=["x", "y", "Center"])
    balls = pd.read_csv(ball_file, header=None)

    plt.scatter(points["x"], points["y"], c=points["Center"])
    ax = plt.gca()

    for _, ball in balls.iterrows():
        center = ball[:2].values
        furthest_point = ball[2:2*2].values
        radius = ball[2*2]

        # Plot circle
        kreis = Circle(
            center, radius, fill=False, edgecolor="black", linestyle='--', alpha=0.75
        )
        ax.add_patch(kreis)
        ax.plot(center[0], center[1], "+", color="black")

        # Linie vom Zentrum zum am weitesten entfernten Punkt zeichnen
        ax.plot([center[0], furthest_point[0]], [center[1], furthest_point[1]], color="black", linestyle='--', alpha=0.75)

    plt.axis("equal")
    plt.title(f'Algorithmus={algorithm}  Summe der Radien={radii.round(7)}')
    plt.savefig(plot_file)


def plot_3d_cluster(cluster_file, ball_file, plot_file, radii, algorithm):

    plt.close("all")
    plt.clf()

    points = pd.read_csv(cluster_file, header=None, names=["x", "y", "z", "Center"])
    balls = pd.read_csv(ball_file, header=None, names=["x", "y", "z", "p1", "p2", "p3", "Radius"])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 3D-Punkte plotten
    scatter = ax.scatter(points["x"], points["y"], points["z"], c=points["Center"])

    # Kugeln plotten
    for _, ball in balls.iterrows():
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x = ball["x"] + ball["Radius"] * np.cos(u) * np.sin(v)
        y = ball["y"] + ball["Radius"] * np.sin(u) * np.sin(v)
        z = ball["z"] + ball["Radius"] * np.cos(v)
        ax.plot_wireframe(x, y, z, color="r", alpha=0.3)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(f'Algorithmus={algorithm}  Summe der Radien={radii.round(7)}')
    plt.savefig(plot_file)
