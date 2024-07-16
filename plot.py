import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def plot_cluster(cluster_file, ball_file, plot_file):

    plt.close("all")

    plt.clf()

    points = pd.read_csv(cluster_file, header=None, names=["x", "y", "Center"])
    balls = pd.read_csv(ball_file, header=None, names=["x", "y", "Radius"])

    plt.scatter(points["x"], points["y"], c=points["Center"])
    ax = plt.gca()

    for _, ball in balls.iterrows():
        kreis = Circle(
            (ball["x"], ball["y"]), ball["Radius"], fill=False, edgecolor="black"
        )
        ax.add_patch(kreis)
        ax.plot(ball["x"], ball["y"], "+", color="black")

    plt.axis("equal")
    plt.savefig(plot_file)


def plot_3d_cluster(cluster_file, ball_file, plot_file):

    plt.close("all")
    plt.clf()

    points = pd.read_csv(cluster_file, header=None, names=["x", "y", "z", "Center"])
    balls = pd.read_csv(ball_file, header=None, names=["x", "y", "z", "Radius"])

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

    plt.savefig(plot_file)
