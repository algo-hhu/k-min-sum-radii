import argparse
import math
import random
import os


# Funktion zur Generierung von Zufallszentren basierend auf der Konfiguration
def generate_centers(config):
    centers = []
    for _ in range(config['number_centers']):
        # Erzeugt ein Zentrum mit zufälligen Koordinaten in jeder Dimension
        center = [random.uniform(0, config['max_value']) for _ in range(config['dimensions'])]
        centers.append(center)

    return centers


# Funktion zur Generierung von Punkten um ein gegebenes Zentrum basierend auf der Konfiguration
def generate_points_around_center(center, config):
    # Bestimmen der Anzahl der Punkte im Cluster und des Cluster-Radius
    number_points = random.randint(
        config['min_points_per_cluster'], config['max_points_per_cluster']
    )
    cluster_radius = random.uniform(
        config['min_cluster_radius'], config['max_cluster_radius']
    )
    points = []
    count = 0
    while count < number_points:
        # Generieren eines zufälligen Radius und zufälliger Winkel für jede Dimension
        radius = random.uniform(0, cluster_radius)
        angles = [random.uniform(0, 2 * math.pi) for _ in range(config['dimensions'])]
        point = []
        valid_point = True
        for i in range(config['dimensions']):
            # Berechnen der Koordinate unter Verwendung des Radius und des Winkels
            coord = radius * math.cos(angles[i]) + center[i]
            # Überprüfen, ob die Koordinate innerhalb der gültigen Grenzen liegt
            if coord < 0 or coord > config['max_value']:
                valid_point = False
                break
            point.append(coord)
        if valid_point:
            # Hinzufügen des Punktes zur Liste, wenn er gültig ist
            points.append(point)
            count += 1

    return points


# Funktion zur Generierung von Clustern basierend auf der Konfiguration
def generate_clusters(config):
    # Generieren der Zentren
    centers = generate_centers(config)
    points = []
    for center in centers:
        # Generieren der Punkte um jedes Zentrum
        points = points + generate_points_around_center(center, config)

    return centers, points


# Funktion zum Schreiben der generierten Punkte in eine CSV-Datei
def write_clusters_to_file(points, config, file_index):

    # Sicherstellen, dass das Verzeichnis existiert
    if not os.path.exists(f'Data/{config['dimensions']}/{config['k']}/Points'):
        os.makedirs(f'Data/{config['dimensions']}/{config['k']}/Points')

    # Erzeugen des Dateinamens
    file_name = f'Data/{config['dimensions']}/{config['k']}/Points/points_{file_index}.csv'

    if not os.path.exists(file_name):
        file = open(file_name, 'w')
        for point in points:
            # Schreiben der Dimension und der Koordinaten des Punktes in die Datei
            file.write(f'{len(point)}')
            for coord in point:
                file.write(f',{coord}')
            file.write('\n')
        file.close()


# Funktion zur Verarbeitung der Befehlszeilenargumente
def handle_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-nf',
        '--number-files',
        type=int,
        help='number of files that will be generated with the given config',
        default=1
    )
    parser.add_argument(
        '-mv',
        '--max-value', 
        type=float, 
        help='maximum value for any dimension', 
        default=10
    )
    parser.add_argument(
        '-d',
        '--dimensions',
        type=int,
        help='number of dimensions for the points and centers', 
        default=2
    )
    parser.add_argument(
        '-n',
        '--number-centers',
        type=int,
        help='number of centers/clusters that will be generated',
        default=3
    )
    parser.add_argument(
        '-minp',
        '--min-points-per-cluster',
        type=int,
        help='minimum number of points per cluster (center excluded)',
        default=25
    )
    parser.add_argument(
        '-maxp',
        '--max-points-per-cluster',
        type=int,
        help='maximum number of points per cluster (center excluded)',
        default=25
    )
    parser.add_argument(
        '-minr',
        '--min-cluster-radius',
        type=float,
        help='minimum radius of generated clusters',
        default=0.5
    )
    parser.add_argument(
        '-maxr',
        '--max-cluster-radius',
        type=float,
        help='minimum radius of generated clusters',
        default=2
    )
    parser.add_argument(
        '-k',
        '--k',
        type=int,
        help='number of clusters',
        default=3
    )
    args = parser.parse_args()

    # Gültigkeitsprüfungen für die Argumente
    assert args.dimensions > 0
    assert args.max_value > 0
    assert args.number_centers > 0
    assert args.min_points_per_cluster > 0
    assert args.min_points_per_cluster <= args.max_points_per_cluster
    assert args.min_cluster_radius > 0
    assert args.min_cluster_radius <= args.max_cluster_radius

    return vars(args)


# Hauptfunktion zum Generieren der Daten
def generate_data(config):
    random.seed(1234)
    for i in range(config['number_files']):
        # Generieren der Cluster
        centers, points = generate_clusters(config)
        # Schreiben der generierten Daten in eine Datei
        write_clusters_to_file(centers + points, config, i)


def main():
    config = handle_arguments()
    generate_data(config)


if __name__ == '__main__':
    main()
