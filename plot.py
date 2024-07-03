import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_cluster(cluster_file, ball_file, plot_file):

    plt.close('all')

    plt.clf()

    points = pd.read_csv(cluster_file, header=0, names=['x', 'y', 'Center'])
    balls = pd.read_csv(ball_file, header=0, names=['x', 'y', 'Radius'])

    plt.scatter(points['x'], points['y'], c=points['Center'])
    ax = plt.gca()

    for _, ball in balls.iterrows():
        kreis = Circle((ball['x'], ball['y']), ball['Radius'], fill=False, edgecolor='black')
        ax.add_patch(kreis)
        ax.plot(ball['x'], ball['y'], '+', color='black')


    plt.axis('equal')
    plt.savefig(plot_file)