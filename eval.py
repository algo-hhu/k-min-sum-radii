import generator
import plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import csv
import ctypes
import time
import miniball

# Laden der C-Bibliothek
lib = ctypes.CDLL('./clustering.so')


# Definition von PointData-Struktur für die Verwendung in der C-Bibliothek
class PointData(ctypes.Structure):
    _fields_ = [('coordinates', ctypes.POINTER(ctypes.c_double)),
                ('dimension', ctypes.c_int)]


# Definition von ClusterData-Struktur für die Verwendung in der C-Bibliothek
class ClusterData(ctypes.Structure):
    _fields_ = [('points', ctypes.POINTER(PointData)),
                ('num_points', ctypes.c_int)]


# Definition der Argument- und Rückgabetypen für die C-Funktionen
lib.schmidt_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)
]
lib.schmidt_wrapper.restype = ctypes.POINTER(ClusterData)

lib.heuristic_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)
]
lib.heuristic_wrapper.restype = ctypes.POINTER(ClusterData)

lib.gonzales_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)
]
lib.gonzales_wrapper.restype = ctypes.POINTER(ClusterData)

lib.kmeans_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)
]
lib.kmeans_wrapper.restype = ctypes.POINTER(ClusterData)

lib.free_cluster_data.argtypes = [ctypes.POINTER(ClusterData), ctypes.c_int]
lib.free_cluster_data.restype = None


# Funktion zur Berechnung des minimalen umgebenden Balls
def calculate_miniball(points):
    mb = miniball.Miniball(points)
    radius = np.sqrt(mb.squared_radius())
    return mb.center(), radius


# Funktion zum Speichern der Cluster in einer CSV-Datei
def save_clusters_to_csv(clusters, output_file):
    data_to_save = []
    for cluster_idx, cluster in enumerate(clusters):
        for point in cluster:
            data_to_save.append(list(point) + [cluster_idx])

    with open(output_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'cluster'])
        writer.writerows(data_to_save)


# Funktion zum Extrahieren der Cluster aus der C-Struktur
def extract_clusters(cluster_ptr, num_clusters_value):
    clusters = cluster_ptr[:num_clusters_value]
    cluster_points_list = []

    for cluster_idx in range(num_clusters_value):
        cluster = clusters[cluster_idx]
        cluster_points = []
        for point_idx in range(cluster.num_points):
            point = cluster.points[point_idx]
            coords = [point.coordinates[dim] for dim in range(point.dimension)]
            cluster_points.append(coords)
        cluster_points_list.append(np.array(cluster_points))

    return np.array(cluster_points_list, dtype=object)


# Funktion zum Berechnen und Speichern der MEBs in einer CSV-Datei
def save_mebs_to_csv(clusters, output_file):
    meb_data_to_save = []
    sum_of_radii = 0
    for _, cluster in enumerate(clusters):
        if len(cluster) > 0:
            center, radius = calculate_miniball(np.array(cluster))
            sum_of_radii += radius
            meb_data_to_save.append([center[0], center[1], radius])

    with open(output_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['center_x', 'center_y', 'radius'])
        writer.writerows(meb_data_to_save)

    return sum_of_radii


# Funktion zum Lesen von Punkten aus einer CSV-Datei und Umwandlung in ein C-Array
def read_points_from_csv(input_file):
    points_array = np.loadtxt(input_file, delimiter=',', usecols=[1, 2])
    points_array = np.ascontiguousarray(points_array)

    c_array = points_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    numPoints = len(points_array)

    return c_array, numPoints


# Funktion zum Aufrufen der C-Funktionen
def call_clustering_function(clustering_func, c_array, numPoints, dimensions, k, num_clusters):
    cluster_ptr = clustering_func(
        c_array,
        numPoints,
        dimensions,
        k,
        num_clusters
    )
    return cluster_ptr


# Funktion zur Durchführung des Clusterings mit verschiedenen Algorithmen
def cluster(point_files, config, ball_directory, cluster_directory, plot_directory, point_directory, algorithm, c_function):
    # Anzahl der Cluster
    k = config['k']
    dimension = config['dimensions']

    # Verzeichnisse erstellen, falls sie nicht existieren
    os.makedirs(os.path.join(ball_directory, algorithm), exist_ok=True)
    os.makedirs(os.path.join(cluster_directory, algorithm), exist_ok=True)
    os.makedirs(os.path.join(plot_directory, algorithm), exist_ok=True)
    os.makedirs(os.path.join(
        f'Data/{dimension}/Results', algorithm), exist_ok=True)

    # Regex-Muster zum Extrahieren der Nummer aus dem Dateinamen
    pattern = r"points_(\d+)\.csv"

    results = []

    for point_file in point_files:
        # Nummer aus dem Dateinamen extrahieren
        match = re.search(pattern, point_file)
        if match:
            number = match.group(1)

        # Definieren der Pfade für die Point-, Ball-, Cluster- und Plot-Dateien
        point_path = os.path.join(point_directory, point_file)
        ball_path = os.path.join(
            ball_directory, algorithm, f'balls_{number}.csv')
        cluster_path = os.path.join(
            cluster_directory, algorithm, f'cluster_{number}.csv')
        plot_path = os.path.join(
            plot_directory, algorithm, f'plot_{number}.png')

        # Lesen der Punkte aus der CSV-Datei und Umwandeln in ein C-Array
        c_array, numPoints = read_points_from_csv(point_path)

        num_clusters = ctypes.c_int()

        start_time = time.time()

        # Aufrufen der Clustering-Funktion aus der C-Bibliothek
        cluster_ptr = call_clustering_function(
            c_function,
            c_array,
            numPoints,
            dimension,
            k,
            ctypes.byref(num_clusters)
        )

        end_time = time.time()

        # Berechne Dauer des Durchlaufes
        duration = end_time - start_time

        # Extrahieren der Cluster aus der C-Struktur
        cluster = extract_clusters(cluster_ptr, num_clusters.value)

        # Freigeben des Speichers, der von der C-Bibliothek belegt wird
        lib.free_cluster_data(cluster_ptr, num_clusters)

        # Speichern der Cluster in einer CSV-Datei
        save_clusters_to_csv(cluster, cluster_path)

        # Berechnen und Speichern der MEBs
        radii = save_mebs_to_csv(cluster, ball_path)

        # Ergebnisse speichern
        results.append((point_file, duration,  radii))

        # Cluster plotten, wenn Dimension = 2
        if dimension == 2:
            plot.plot_cluster(cluster_path, ball_path, plot_path)

    # Ergebnisse sortieren nach Dateiname
    results.sort(key=lambda x: (x[0]))

    # Ergebnisse in eine CSV-Datei schreiben
    with open(f'Data/{dimension}/Results/{algorithm}/results.csv', 'w') as f:
        f.write('Datei,Dauer (Sekunden),Summe_der_Radien\n')
        for point_file, duration, radii in results:
            f.write(f"{point_file},{duration},{radii}\n")


def schmidt(point_files, config, ball_directory, cluster_directory, plot_directory, point_directory):
    # Werte für epsilon und u definieren
    epsilon_values = [0.5, 0.4]
    u_values = [1, 10, 100, 1000, 3000]
    num_radii_values = [5]
    # Anzahl der Cluster und Dimension der Punkte
    k = config['k']
    dimension = config['dimensions']

    # Verzeichnisse erstellen, falls sie nicht existieren
    os.makedirs(os.path.join(ball_directory, "Schmidt"), exist_ok=True)
    os.makedirs(os.path.join(cluster_directory, "Schmidt"), exist_ok=True)
    os.makedirs(os.path.join(plot_directory, "Schmidt"), exist_ok=True)
    os.makedirs(os.path.join(
        f'Data/{dimension}/Results', "Schmidt"), exist_ok=True)

    # Regex-Muster zum Extrahieren der Nummer aus dem Dateinamen
    pattern = r"points_(\d+)\.csv"

    results = []
    count = 1
    for point_file in point_files:
        # Nummer aus dem Dateinamen extrahieren
        match = re.search(pattern, point_file)
        if match:
            number = match.group(1)

        point_path = os.path.join(point_directory, point_file)

        for epsilon in epsilon_values:
            for num_radii in num_radii_values:
                for u in u_values:
                    # Definieren der Pfade für die Ball-, Cluster- und Plot-Dateien
                    ball_path = os.path.join(ball_directory, 'Schmidt', f'balls_{number}_u{
                                             u}_epsilon{epsilon}_num_radii{num_radii}.csv')
                    cluster_path = os.path.join(cluster_directory, 'Schmidt', f'cluster_{number}_u{
                                                u}_epsilon{epsilon}_num_radii{num_radii}.csv')
                    plot_path = os.path.join(plot_directory, 'Schmidt', f'plot_{number}_u{
                                             u}_epsilon{epsilon}_num_radii{num_radii}.png')

                    # Lesen der Punkte aus der CSV-Datei und Umwandeln in ein C-Array
                    c_array, numPoints = read_points_from_csv(point_path)

                    num_clusters = ctypes.c_int()

                    start_time = time.time()

                    # Rufe die C-Funktion auf
                    cluster_ptr = lib.schmidt_wrapper(
                        c_array,
                        numPoints,
                        dimension,
                        k,
                        epsilon,
                        u,
                        num_radii,
                        ctypes.byref(num_clusters)
                    )

                    end_time = time.time()

                    # Berechne Dauer des Durchlaufes
                    duration = end_time - start_time

                    # Extrahieren der Cluster aus der C-Struktur
                    cluster = extract_clusters(cluster_ptr, num_clusters.value)

                    # Freigeben des Speichers, der von der C-Bibliothek belegt wird
                    lib.free_cluster_data(cluster_ptr, num_clusters)

                    # Speichern der Cluster in einer CSV-Datei
                    save_clusters_to_csv(cluster, cluster_path)

                    # Berechnen und Speichern der MEBs
                    radii = save_mebs_to_csv(cluster, ball_path)

                    # Ergebnisse speichern
                    results.append(
                        (point_file, u, epsilon, num_radii, duration, radii))

                    # Cluster plotten und speichern
                    plot.plot_cluster(cluster_path, ball_path, plot_path)

        print(count)
        count += 1

    # Ergebnisse nach Dateiname, 'u' und 'epsilon' sortieren
    results.sort(key=lambda x: (x[0], x[1], -x[2]))

    # Ergebnisse in eine CSV-Datei schreiben
    with open(f'Data/{dimension}/Results/Schmidt/results.csv', 'w') as f:
        f.write('Datei,u,epsilon,num_radii,Dauer (Sekunden),Summe_der_Radien\n')
        for point_file, u, epsilon, num_radii, duration, radii in results:
            f.write(f"{point_file},{u},{epsilon},{
                    num_radii},{duration},{radii}\n")


def analyze_results_schmidt(config):
    # Ergebnisse in einen DataFrame umwandeln
    df = pd.read_csv(
        f'Data/{config['dimensions']}/Results/Schmidt/results.csv')

    # Boxplot der Radien nach 'u' und 'epsilon'
    plt.figure(figsize=(10, 6))
    df.boxplot(column='Summe_der_Radien', by=['u', 'epsilon'])
    plt.title('Verteilung der Radien nach u und epsilon')
    plt.suptitle('')
    plt.xlabel('u, epsilon')
    plt.ylabel('Summe_der_Radien')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'Data/{config['dimensions']
                        }/Results/Schmidt/radii_boxplot_all.png')
    plt.close()

    # Boxplots für konstantes u und variierendes epsilon
    for u_val in df['u'].unique():
        df_u = df[df['u'] == u_val]
        plt.figure(figsize=(10, 6))
        df_u.boxplot(column='Summe_der_Radien', by='epsilon')
        plt.title(f'Verteilung der Radien für u={
                  u_val} und variierendes epsilon')
        plt.suptitle('')
        plt.xlabel('epsilon')
        plt.ylabel('Summe_der_Radien')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(
            f'Data/{config['dimensions']}/Results/Schmidt/radii_boxplot_u{u_val}.png')
        plt.close()

    # Boxplots für konstantes epsilon und variierendes u
    for epsilon_val in df['epsilon'].unique():
        df_epsilon = df[df['epsilon'] == epsilon_val]
        plt.figure(figsize=(10, 6))
        df_epsilon.boxplot(column='Summe_der_Radien', by='u')
        plt.title(f'Verteilung der Radien für epsilon={
                  epsilon_val} und variierendes u')
        plt.suptitle('')
        plt.xlabel('u')
        plt.ylabel('Summe_der_Radien')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(
            f'Data/{config['dimensions']}/Results/Schmidt/radii_boxplot_epsilon{epsilon_val}.png')
        plt.close()

    # Boxplot der Dauer der Durchläufe nach 'u'
    plt.figure(figsize=(10, 6))
    df.boxplot(column='Dauer (Sekunden)', by='u')
    plt.title('Verteilung der Dauer nach u')
    plt.suptitle('')
    plt.xlabel('u')
    plt.ylabel('Dauer (Sekunden)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'Data/{config["dimensions"]
                        }/Results/Schmidt/duration_boxplot.png')
    plt.close()

    # Berechnung der Verbesserung des Radius
    min_u = min(df['u'])
    max_epsilon = max(df['epsilon'])
    min_num_radii = min(df['num_radii'])
    df_min = df[(df['u'] == min_u) & (df['epsilon'] == max_epsilon)
                & (df['num_radii'] == min_num_radii)]

    improvements = []
    for u in df['u'].unique():
        for r in df['num_radii'].unique():
            for epsilon in df['epsilon'].unique():
                if u != min_u or epsilon != max_epsilon or r != min_num_radii:
                    df_u_num_radii_epsilon = df[(df['u'] == u) & (
                        df['num_radii'] == r) & (df['epsilon'] == epsilon)]
                    merged = pd.merge(df_min, df_u_num_radii_epsilon, on='Datei', suffixes=(
                        '_min', f'_u{u}_num_radii{r}_epsilon{epsilon}'))
                    merged['improvement'] = (merged['Summe_der_Radien_min'] - merged[f'Summe_der_Radien_u{
                                             u}_num_radii{r}_epsilon{epsilon}']) / merged['Summe_der_Radien_min'] * 100
                    improvements.append(
                        (u, r, epsilon, merged['improvement'].mean()))

    improvement_df = pd.DataFrame(
        improvements, columns=['u', 'num_radii', 'epsilon', 'Improvement (%)'])

    # Verbesserungen anzeigen
    print(f"Verbesserung des Radius im Vergleich zu u={
          min_u}, num_radii={min_num_radii} und epsilon={max_epsilon}:")
    print(improvement_df)

    # Verbesserungen in eine CSV-Datei schreiben
    improvement_df.to_csv(
        f'Data/{config['dimensions']}/Results/Schmidt/radius_improvement.csv', index=False)

    # Beste Kombination von u und epsilon basierend auf dem kleinsten Durchschnitt der Radien
    best_u, best_num_radii, best_epsilon = df.groupby(['u', 'num_radii', 'epsilon'])[
        'Summe_der_Radien'].mean().idxmin()
    best_mean_radius = df[(df['u'] == best_u) & (df['num_radii'] == best_num_radii) & (
        df['epsilon'] == best_epsilon)]['Summe_der_Radien'].mean()
    print(f"Die beste Kombination ist u={best_u}, num_radii={best_num_radii} und epsilon={
          best_epsilon} mit dem kleinsten durchschnittlichen Radius={best_mean_radius:.6f}.")

    # Boxplot der Radien nach num_radiis für best_u und best_epsilon
    plt.figure(figsize=(10, 6))
    df_best = df[(df['u'] == best_u) & (df['epsilon'] == best_epsilon)]
    df_best.boxplot(column='Summe_der_Radien', by='num_radii')
    plt.title(f'Verteilung der Radien nach Anzahl der Radien (bestes u={
              best_u}, bestes epsilon={best_epsilon})')
    plt.suptitle('')
    plt.xlabel('num_radii')
    plt.ylabel('Summe_der_Radien')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'Data/{config["dimensions"]
                        }/Results/Schmidt/num_radii_boxplot.png')
    plt.close()

    # Markdown-Datei erstellen und Ergebnisse formatieren
    with open(f"Data/{config['dimensions']}/Results/Schmidt/radius_improvement.md", 'w') as md_file:
        md_file.write("# Verbesserung des Radius\n")
        md_file.write(f"Vergleich der Verbesserungen des Radius im Vergleich zu u={
                      min_u}, num_radii={min_num_radii} und epsilon={max_epsilon}:\n\n")
        md_file.write("| u | num_radii | epsilon | Improvement (%) |\n")
        md_file.write("|---|---------------|---------|-----------------|\n")
        for row in improvements:
            md_file.write(f"| {row[0]} | {row[1]} | {
                          row[2]} | {row[3]:.2f} |\n")
        md_file.write("\n")
        md_file.write(f"Die beste Kombination ist u={best_u}, num_radii={best_num_radii} und epsilon={
                      best_epsilon} mit dem kleinsten durchschnittlichen Radius={best_mean_radius:.6f}.\n")


def compare_algorithms(config):
    # Lade die Ergebnisse
    schmidt_results = pd.read_csv(
        f'Data/{config["dimensions"]}/Results/Schmidt/results.csv')
    gonzales_results = pd.read_csv(
        f'Data/{config["dimensions"]}/Results/Gonzales/results.csv')
    kmeans_results = pd.read_csv(
        f'Data/{config["dimensions"]}/Results/KMeansPlusPlus/results.csv')
    heuristik_results = pd.read_csv(
        f'Data/{config["dimensions"]}/Results/Heuristik/results.csv')

    # Beste Kombination von u, num_radii und epsilon basierend auf dem kleinsten Durchschnitt der Radien
    best_u, best_num_radii, best_epsilon = schmidt_results.groupby(
        ['u', 'num_radii', 'epsilon'])['Summe_der_Radien'].mean().idxmin()

    # Berechnen der Durchschnittswerte der Radien für jeden Algorithmus
    best_schmidt_avg_radius = schmidt_results[(schmidt_results['u'] == best_u) & (
        schmidt_results['num_radii'] == best_num_radii) & (schmidt_results['epsilon'] == best_epsilon)]['Summe_der_Radien'].mean()
    gonzales_avg_radius = gonzales_results['Summe_der_Radien'].mean()
    kmeans_avg_radius = kmeans_results['Summe_der_Radien'].mean()
    heuristik_avg_radius = heuristik_results['Summe_der_Radien'].mean()

    # Anzeigen der Durchschnittswerte
    print(f"Durchschnittlicher Radius für Schmidt: {
          best_schmidt_avg_radius:.6f}")
    print(f"Durchschnittlicher Radius für Gonzales: {gonzales_avg_radius:.6f}")
    print(f"Durchschnittlicher Radius für KMeans++: {kmeans_avg_radius:.6f}")
    print(f"Durchschnittlicher Radius für Heuristik: {
          heuristik_avg_radius:.6f}")

    all_comparison_results = []
    worse_schmidt_points = []
    # Schleife über alle Kombinationen von u und epsilon
    for u_val in schmidt_results['u'].unique():
        for num_radii_val in schmidt_results['num_radii'].unique():
            for epsilon_val in schmidt_results['epsilon'].unique():

                # Zusammensetzen der Dataframes
                paired_results = pd.merge(
                    schmidt_results[(schmidt_results['u'] == u_val) & (schmidt_results['num_radii'] == num_radii_val) & (
                        schmidt_results['epsilon'] == epsilon_val)][['Datei', 'Summe_der_Radien']],
                    gonzales_results[['Datei', 'Summe_der_Radien']], on='Datei', suffixes=('_Schmidt', '_Gonzales'))

                paired_results = pd.merge(paired_results, kmeans_results[[
                                          'Datei', 'Summe_der_Radien']], on='Datei')
                paired_results.rename(
                    columns={'Summe_der_Radien': 'Summe_der_Radien_KMeans'}, inplace=True)
                paired_results = pd.merge(paired_results, heuristik_results[[
                                          'Datei', 'Summe_der_Radien']], on='Datei')
                paired_results.rename(
                    columns={'Summe_der_Radien': 'Summe_der_Radien_Heuristik'}, inplace=True)

                # Runden der Radien
                paired_results['Summe_der_Radien_Schmidt'] = paired_results['Summe_der_Radien_Schmidt'].round(
                    7)
                paired_results['Summe_der_Radien_Gonzales'] = paired_results['Summe_der_Radien_Gonzales'].round(
                    7)
                paired_results['Summe_der_Radien_KMeans'] = paired_results['Summe_der_Radien_KMeans'].round(
                    7)
                paired_results['Summe_der_Radien_Heuristik'] = paired_results['Summe_der_Radien_Heuristik'].round(
                    7)

                # Zählen, wie oft Schmidt besser als alle anderen ist
                paired_results['Schmidt_better_than_all'] = paired_results.apply(
                    lambda row: row['Summe_der_Radien_Schmidt'] < row[[
                        'Summe_der_Radien_Gonzales', 'Summe_der_Radien_KMeans', 'Summe_der_Radien_Heuristik']].min(),
                    axis=1
                )

                # Zählen, wie oft Schmidt schlechter als alle anderen ist
                paired_results['Schmidt_worse_than_all'] = paired_results.apply(
                    lambda row: row['Summe_der_Radien_Schmidt'] > row[[
                        'Summe_der_Radien_Gonzales', 'Summe_der_Radien_KMeans', 'Summe_der_Radien_Heuristik']].max(),
                    axis=1
                )
                schmidt_better_count = paired_results['Schmidt_better_than_all'].sum(
                )
                schmidt_worse_count = paired_results['Schmidt_worse_than_all'].sum(
                )

                # Zählt wie oft Schmidt besser ist als Gonzales
                schmidt_vs_gonzales_better = (
                    paired_results['Summe_der_Radien_Schmidt'] < paired_results['Summe_der_Radien_Gonzales']).sum()

                # Zählt wie oft Schmidt schlechter ist als Gonzales
                schmidt_vs_gonzales_worse = (
                    paired_results['Summe_der_Radien_Schmidt'] > paired_results['Summe_der_Radien_Gonzales']).sum()

                # Zählt wie oft Schmidt besser ist als KMeans++
                schmidt_vs_kmeans_better = (
                    paired_results['Summe_der_Radien_Schmidt'] < paired_results['Summe_der_Radien_KMeans']).sum()

                # Zählt wie oft Schmidt schlechter ist als KMeans++
                schmidt_vs_kmeans_worse = (
                    paired_results['Summe_der_Radien_Schmidt'] > paired_results['Summe_der_Radien_KMeans']).sum()

                # Zählt wie oft Schmidt besser ist als die Heuristik
                schmidt_vs_heuristik_better = (
                    paired_results['Summe_der_Radien_Schmidt'] < paired_results['Summe_der_Radien_Heuristik']).sum()

                # Zählt wie oft Schmidt schlechter ist als die Heuristik
                schmidt_vs_heuristik_worse = (
                    paired_results['Summe_der_Radien_Schmidt'] > paired_results['Summe_der_Radien_Heuristik']).sum()

                total_count = paired_results.shape[0]

                schmidt_better_percentage = (
                    schmidt_better_count / total_count) * 100
                schmidt_worse_percentage = (
                    schmidt_worse_count / total_count) * 100

                schmidt_vs_gonzales_better_percentage = (
                    schmidt_vs_gonzales_better / total_count) * 100
                schmidt_vs_gonzales_worse_percentage = (
                    schmidt_vs_gonzales_worse / total_count) * 100

                schmidt_vs_kmeans_better_percentage = (
                    schmidt_vs_kmeans_better / total_count) * 100
                schmidt_vs_kmeans_worse_percentage = (
                    schmidt_vs_kmeans_worse / total_count) * 100

                schmidt_vs_heuristik_better_percentage = (
                    schmidt_vs_heuristik_better / total_count) * 100
                schmidt_vs_heuristik_worse_percentage = (
                    schmidt_vs_heuristik_worse / total_count) * 100

                # Ergebnisse sammeln
                all_comparison_results.append({
                    'u': u_val,
                    'epsilon': epsilon_val,
                    'num_radii': num_radii_val,
                    'Schmidt vs Alle Besser (%)': schmidt_better_percentage,
                    'Schmidt vs Alle Schlechter (%)': schmidt_worse_percentage,
                    'Schmidt vs Gonzales Besser (%)': schmidt_vs_gonzales_better_percentage,
                    'Schmidt vs Gonzales Schlechter (%)': schmidt_vs_gonzales_worse_percentage,
                    'Schmidt vs KMeans++ Besser (%)': schmidt_vs_kmeans_better_percentage,
                    'Schmidt vs KMeans++ Schlechter (%)': schmidt_vs_kmeans_worse_percentage,
                    'Schmidt vs Heuristik Besser (%)': schmidt_vs_heuristik_better_percentage,
                    'Schmidt vs Heuristik Schlechter (%)': schmidt_vs_heuristik_worse_percentage
                })

                # Speichern der Dateien, bei denen Schmidt schlechter als alle anderen ist
                if u_val == best_u and epsilon_val == best_epsilon and num_radii_val == best_num_radii:
                    worse_files = paired_results[paired_results['Schmidt_worse_than_all'] == True]['Datei'].tolist(
                    )
                    worse_schmidt_points.extend(worse_files)

    # Ergebnisse in ein DataFrame packen
    comparison_df = pd.DataFrame(all_comparison_results)

    # Ergebnisse als Tabelle speichern
    comparison_df.to_csv(
        f'Data/{config["dimensions"]}/Results/comparison_all_us_epsilons.csv', index=False)

    # Markdown-Datei erstellen
    with open(f'Data/{config["dimensions"]}/Results/comparison_all_us_epsilons.md', 'w') as md_file:
        md_file.write(
            "# Paarweiser Vergleich der Clustering-Algorithmen für alle u- und epsilon-Werte\n\n")
        md_file.write("| u  | epsilon | num_radii | Schmidt vs Alle Besser (%) | Schmidt vs Alle Schlechter (%) | Schmidt vs Gonzales Besser (%) | Schmidt vs Gonzales Schlechter (%) | Schmidt vs KMeans++ Besser (%) | Schmidt vs KMeans++ Schlechter (%) | Schmidt vs Heuristik Besser (%) | Schmidt vs Heuristik Schlechter (%) |\n")
        md_file.write("|----|---------|---------------|----------------------------|--------------------------------|--------------------------------|------------------------------------|--------------------------------|------------------------------------|---------------------------------|-------------------------------------|\n")
        for _, row in comparison_df.iterrows():
            md_file.write(f"| {int(row['u'])} | {row['epsilon']} | {int(row['num_radii'])} |{row['Schmidt vs Alle Besser (%)']:.2f} | {row['Schmidt vs Alle Schlechter (%)']:.2f} | {row['Schmidt vs Gonzales Besser (%)']:.2f} | {row['Schmidt vs Gonzales Schlechter (%)']:.2f} | {
                          row['Schmidt vs KMeans++ Besser (%)']:.2f} | {row['Schmidt vs KMeans++ Schlechter (%)']:.2f} | {row['Schmidt vs Heuristik Besser (%)']:.2f} | {row['Schmidt vs Heuristik Schlechter (%)']:.2f} |\n")

    # Speichern der Dateien, bei denen Schmidt schlechter als alle anderen ist
    with open(f'Data/{config["dimensions"]}/Results/Schmidt/worse_schmidt_points.md', 'w') as md_file:
        md_file.write(f"u: {best_u}, num_radii: {
                      best_num_radii}, epsilon: {best_epsilon}\n")
        md_file.write("Dateien:\n")
        for datei in worse_schmidt_points:
            md_file.write(f"{datei}\n")
        md_file.write("\n")


def main(config):
    # Verzeichnisse definieren
    directory = f'Data/{config["dimensions"]}'
    point_directory = os.path.join(directory, 'Points')
    ball_directory = os.path.join(directory, 'Balls')
    cluster_directory = os.path.join(directory, 'Cluster')
    plot_directory = os.path.join(directory, 'Plots')

    # Überprüfen, ob die Punkte-Dateien existieren, andernfalls generieren
    if not os.path.exists(os.path.join(point_directory, f'points_{config["number_files"] - 1}.csv')):
        generator.generate_data(config)

    # Liste der Punkte-Dateien im Verzeichnis
    point_files = [f for f in os.listdir(
        point_directory) if f.endswith('.csv')]

    # Ausführung der Algorithmen
    schmidt(point_files, config, ball_directory,
            cluster_directory, plot_directory, point_directory)
    cluster(point_files, config, ball_directory, cluster_directory,
            plot_directory, point_directory, 'Heuristik', lib.heuristic_wrapper)
    cluster(point_files, config, ball_directory, cluster_directory,
            plot_directory, point_directory, 'Gonzales', lib.gonzales_wrapper)
    cluster(point_files, config, ball_directory, cluster_directory,
            plot_directory, point_directory, 'KMeansPlusPlus', lib.kmeans_wrapper)


if __name__ == "__main__":
    # Argumente aus der Konfiguration holen
    config = generator.handle_arguments()

    # Clustering-Algorithmen ausführen
    main(config)

    # Analyse und Vergleich der Ergebnisse von Schmidt
    analyze_results_schmidt(config)

    # Vergleich der Ergebnisse der verschiedenen Algorithmen
    compare_algorithms(config)
