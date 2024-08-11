from random import seed
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
from typing import List


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
lib.FPT_Heuristic_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int
]
lib.FPT_Heuristic_wrapper.restype = ctypes.POINTER(ClusterData)

lib.heuristic_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)
]
lib.heuristic_wrapper.restype = ctypes.POINTER(ClusterData)

lib.gonzalez_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int
]
lib.gonzalez_wrapper.restype = ctypes.POINTER(ClusterData)

lib.kmeans_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int
]
lib.kmeans_wrapper.restype = ctypes.POINTER(ClusterData)

lib.free_cluster_data.argtypes = [ctypes.POINTER(ClusterData), ctypes.c_int]
lib.free_cluster_data.restype = None


def find_furthest_point(points, center):
    largest_distance = 0
    furthest_point = points[0]
    for point in points:
        distance = np.linalg.norm(point - center)
        if distance > largest_distance:
            largest_distance = distance
            furthest_point = point

    return furthest_point


# Funktion zur Berechnung des minimalen umgebenden Balls
def calculate_miniball(points):
    mb = miniball.Miniball(points)
    radius = np.sqrt(mb.squared_radius())
    center = mb.center()
    furthest_point = find_furthest_point(points, center)
    return center, furthest_point, radius


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


# Funktion zum Speichern der Cluster in einer CSV-Datei
def save_clusters_to_csv(clusters, output_file):
    data_to_save = []
    for cluster_idx, cluster in enumerate(clusters):
        for point in cluster:
            data_to_save.append(list(point) + [cluster_idx])

    with open(output_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerows(data_to_save)

# Funktion zum Berechnen und Speichern der MEBs in einer CSV-Datei
def save_mebs_to_csv(clusters, output_file):
    meb_data_to_save = []
    sum_of_radii = 0
    for _, cluster in enumerate(clusters):
        if len(cluster) > 0:
            center, furthest_point, radius = calculate_miniball(np.array(cluster))
            meb_data_to_save.append(list(center) + list(furthest_point) + [radius])
            sum_of_radii += radius

    with open(output_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerows(meb_data_to_save)

    return sum_of_radii


# Funktion zum Lesen von Punkten aus einer CSV-Datei und Umwandlung in ein C-Array
def read_points_from_csv(input_file):
    points_array = np.loadtxt(input_file, delimiter=',')
    points_array = np.ascontiguousarray(points_array)

    c_array = points_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    numPoints = len(points_array)

    return c_array, numPoints


# Funktion zum Aufrufen der C-Funktionen
def call_clustering_function(clustering_func, c_array, numPoints, dimensions, k, num_clusters, seed=None):
    if seed is None:
        cluster_ptr = clustering_func(
            c_array,
            numPoints,
            dimensions,
            k,
            num_clusters
        )
    else:
        cluster_ptr = clustering_func(
            c_array,
            numPoints,
            dimensions,
            k,
            num_clusters,
            seed
        )
    return cluster_ptr


# Funktion zur Durchführung des Clusterings mit verschiedenen Algorithmen
def cluster(point_files, dimension, k, directory, point_directory, algorithm, c_function, seed=None):
    if seed is None:
        ball_directory = os.path.join(directory, f'{algorithm}/Balls')
        cluster_directory = os.path.join(directory, f'{algorithm}/Cluster')
        plot_directory = os.path.join(directory, f'{algorithm}/Plots')
        result_directory = os.path.join(directory, f'{algorithm}/Results')
    else:
        ball_directory = os.path.join(directory, f'{algorithm}/Seed={seed}/Balls')
        cluster_directory = os.path.join(directory, f'{algorithm}/Seed={seed}/Cluster')
        plot_directory = os.path.join(directory, f'{algorithm}/Seed={seed}/Plots')
        result_directory = os.path.join(directory, f'{algorithm}/Seed={seed}/Results')

    # Verzeichnisse erstellen, falls sie nicht existieren
    os.makedirs(ball_directory, exist_ok=True)
    os.makedirs(cluster_directory, exist_ok=True)
    os.makedirs(plot_directory, exist_ok=True)
    os.makedirs(result_directory, exist_ok=True)

    # Regex-Muster zum Extrahieren der Nummer aus dem Dateinamen
    pattern = r'points_(\d+)\.csv'

    results = []

    for point_file in point_files:
        # Nummer aus dem Dateinamen extrahieren
        match = re.search(pattern, point_file)
        if match:
            number = match.group(1)

        # Definieren der Pfade für die Point-, Ball-, Cluster- und Plot-Dateien
        point_path = os.path.join(point_directory, point_file)
        ball_path = os.path.join(ball_directory, f'balls_{number}.csv')
        cluster_path = os.path.join(cluster_directory, f'cluster_{number}.csv')
        plot_path = os.path.join(plot_directory, f'plot_{number}.pdf')

        # Lesen der Punkte aus der CSV-Datei und Umwandeln in ein C-Array
        c_array, numPoints = read_points_from_csv(point_path)

        num_clusters = ctypes.c_int()

        start_time = time.time()

        # Aufrufen der Clustering-Funktion aus der C-Bibliothek
        if algorithm == 'Heuristik':
            cluster_ptr = call_clustering_function(
                c_function,
                c_array,
                numPoints,
                dimension,
                k,
                ctypes.byref(num_clusters),
                None
            )
        else:
            cluster_ptr = call_clustering_function(
                c_function,
                c_array,
                numPoints,
                dimension,
                k,
                ctypes.byref(num_clusters),
                seed
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

        # Cluster plotten, wenn Dimension = 2 oder 3
        if dimension == 2:
            plot.plot_cluster(cluster_path, ball_path, plot_path, radii, algorithm)
        if dimension == 3:
            plot.plot_3d_cluster(cluster_path, ball_path, plot_path, radii, algorithm)

    # Ergebnisse sortieren nach Dateiname
    results.sort(key=lambda x: (x[0]))

    # Ergebnisse in eine CSV-Datei schreiben
    with open(f'{result_directory}/results.csv', 'w') as f:
        f.write('Datei,Dauer (Sekunden),Summe_der_Radien\n')
        for point_file, duration, radii in results:
            f.write(f'{point_file},{duration},{radii}\n')


def FPT_Heuristic(point_files, dimension, k, epsilon_values, u_values, num_radii_values, directory, point_directory, seed):

    ball_directory = os.path.join(directory, f'FPT_Heuristic/Seed={seed}/Balls')
    cluster_directory = os.path.join(directory, f'FPT_Heuristic/Seed={seed}/Cluster')
    plot_directory = os.path.join(directory, f'FPT_Heuristic/Seed={seed}/Plots')
    result_directory = os.path.join(directory, f'FPT_Heuristic/Seed={seed}/Results')

    # Verzeichnisse erstellen, falls sie nicht existieren
    os.makedirs(ball_directory, exist_ok=True)
    os.makedirs(cluster_directory, exist_ok=True)
    os.makedirs(plot_directory, exist_ok=True)
    os.makedirs(result_directory, exist_ok=True)

    # Regex-Muster zum Extrahieren der Nummer aus dem Dateinamen
    pattern = r'points_(\d+)\.csv'

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
                    ball_path = os.path.join(ball_directory, f'balls_{number}_u{u}_epsilon{epsilon}_num_radii{num_radii}.csv')
                    cluster_path = os.path.join(cluster_directory, f'cluster_{number}_u{u}_epsilon{epsilon}_num_radii{num_radii}.csv')
                    plot_path = os.path.join(plot_directory, f'plot_{number}_u{u}_epsilon{epsilon}_num_radii{num_radii}.pdf')

                    # Lesen der Punkte aus der CSV-Datei und Umwandeln in ein C-Array
                    c_array, numPoints = read_points_from_csv(point_path)

                    num_clusters = ctypes.c_int()

                    start_time = time.time()

                    # Rufe die C-Funktion auf
                    cluster_ptr = lib.FPT_Heuristic_wrapper(
                        c_array,
                        numPoints,
                        dimension,
                        k,
                        epsilon,
                        u,
                        num_radii,
                        ctypes.byref(num_clusters),
                        seed
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
                    if dimension == 2:
                        plot.plot_cluster(cluster_path, ball_path, plot_path, radii, 'FPT-Heuristik')
                    if dimension == 3:
                        plot.plot_3d_cluster(cluster_path, ball_path, plot_path, radii, 'FPT-Heuristik')

        print(count)
        count += 1

    # Ergebnisse nach Dateiname, 'u' und 'epsilon' sortieren
    results.sort(key=lambda x: (x[0], x[1], -x[2]))

    # Ergebnisse in eine CSV-Datei schreiben
    with open(f'{result_directory}/results.csv', 'w') as f:
        f.write('Datei,u,epsilon,num_radii,Dauer (Sekunden),Summe_der_Radien\n')
        for point_file, u, epsilon, num_radii, duration, radii in results:
            f.write(f'{point_file},{u},{epsilon},{
                    num_radii},{duration},{radii}\n')


def analyze_results_FPT_Heuristic(dimensions, ks, seeds, data_directory):
    all_results = []
    
    # Alle Ergebnisse in einem DataFrame zusammenführen
    for dimension in dimensions:
        for k in ks:
            for seed in seeds:
                result_directory = f'{data_directory}/Dimension={dimension}/k={k}/FPT_Heuristic/Seed={seed}/Results'
                df = pd.read_csv(f'{result_directory}/results.csv')
                df['Dimension'] = dimension
                df['k'] = k 
                df['seed'] = seed
                all_results.append(df)
    
    combined_df = pd.concat(all_results)

    # Berechne die besten Kombinationen für jede Dimension und jedes k
    for dimension in dimensions:
        for k in ks:
            df_subset = combined_df[(combined_df['Dimension'] == dimension) & (combined_df['k'] == k)]
            plot_directory = f'{data_directory}/Dimension={dimension}/k={k}/FPT_Heuristic'

            mean_results = df_subset.groupby(['Datei', 'u', 'epsilon', 'num_radii']).agg({'Dauer (Sekunden)': 'mean', 'Summe_der_Radien': 'mean'}).reset_index()
            
            num_radii_results = mean_results.groupby('num_radii').agg({'Summe_der_Radien': 'mean'}).reset_index()
            u_results = mean_results.groupby('u').agg({'Summe_der_Radien': 'mean'}).reset_index()
            epsilon_results = mean_results.groupby('epsilon').agg({'Summe_der_Radien': 'mean'}).reset_index()


            best_num_radii = num_radii_results.loc[num_radii_results['Summe_der_Radien'].idxmin(), 'num_radii']
            best_u = u_results.loc[u_results['Summe_der_Radien'].idxmin(), 'u']
            best_epsilon = epsilon_results.loc[epsilon_results['Summe_der_Radien'].idxmin(), 'epsilon']


            analyse_u(mean_results, best_num_radii, best_epsilon, plot_directory, dimension, k)   
            analyze_num_radii(mean_results, best_u, best_epsilon, plot_directory, dimension, k)   
            analyze_epsilon(mean_results, best_u, best_num_radii, plot_directory, dimension, k)     


def analyse_u(df, best_num_radii, best_epsilon, output_directory, dimension, k):
    best_df = df[(df['num_radii'] == best_num_radii) & (df['epsilon'] == best_epsilon)]
    
    duration_stats = best_df.groupby('u').agg(
        mean_duration=('Dauer (Sekunden)', 'mean'),
        median_duration=('Dauer (Sekunden)', 'median'),
        std_duration=('Dauer (Sekunden)', 'std'),
        min_duration=('Dauer (Sekunden)', 'min'),
        max_duration=('Dauer (Sekunden)', 'max')
    ).reset_index()

    sum_of_radii_stats = best_df.groupby('u').agg(
        mean_sum=('Summe_der_Radien', 'mean'),
        median_sum=('Summe_der_Radien', 'median'),
        std_sum=('Summe_der_Radien', 'std'),
        min_sum=('Summe_der_Radien', 'min'),
        max_sum=('Summe_der_Radien', 'max')
    ).reset_index()

    plt.figure(figsize=(12, 8))
    best_df.boxplot(column='Dauer (Sekunden)', by='u')
    plt.title(f'Laufzeit nach num_u\n(num_radii={best_num_radii}, epsilon={best_epsilon}, Dimension={dimension}, k={k})', fontsize=14, pad=20)
    plt.suptitle('')
    plt.xlabel('num_u', fontsize=12)
    plt.ylabel('Dauer (Sekunden)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{output_directory}/u_runtime_boxplot.pdf')
    plt.close('all')

    # Erstellen des Liniendiagramms
    plt.figure(figsize=(12, 8))
    line_data = best_df.groupby('u').agg(
        avg_dauer=('Dauer (Sekunden)', 'mean')
    ).reset_index()
    plt.plot(line_data['u'], line_data['avg_dauer'], marker='o', linestyle='-', color='b')
    plt.title(f'Laufzeit nach num_u\n(num_radii={best_num_radii}, epsilon={best_epsilon}, Dimension={dimension}, k={k})', fontsize=25, pad=20)
    plt.xlabel('num_u', fontsize=20)
    plt.ylabel('Dauer (Sekunden)', fontsize=20)
    plt.xticks(line_data['u'], fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{output_directory}/u_runtime_lineplot.pdf')
    plt.close('all')

    plt.figure(figsize=(12, 8))
    best_df.boxplot(column='Summe_der_Radien', by='u')
    plt.title(f'Verteilung der Summe der Radien nach Anzahl von num_u\n (num_radii={best_num_radii}, epsilon={best_epsilon}, Dimension={dimension}, k={k})', fontsize=14, pad=20)
    plt.suptitle('')
    plt.xlabel('num_u', fontsize=12)
    plt.ylabel('Summe der Radien', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{output_directory}/u_sum_of_radii_boxplot.pdf')
    plt.close('all') 

    # Bestimmen des minimalen u-Werts
    min_u = best_df['u'].min()
    min_df = best_df[best_df['u'] == min_u]
    max_radius = min_df['Summe_der_Radien'].mean()

    # Liste zur Speicherung der Verbesserungen
    improvements = []

    # Iteration über alle einzigartigen u-Werte in best_df
    for u in best_df['u'].unique():
        df_u = best_df[best_df['u'] == u]
        u_radius = df_u['Summe_der_Radien'].mean()
        improvement = (max_radius - u_radius) / max_radius * 100

        improvements.append((u, improvement, u_radius))

    improvement_df = pd.DataFrame(improvements, columns=['u', 'Improvement (%)', 'Durchschnitt der Summe der Radien'])


    tex_file = f'{output_directory}/u.tex'

    with open(tex_file, 'w') as f:
        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write(
            '\\textit{num\\_u} & Mittelwert (Sek.) & Median (Sek.) & Standard-abweichung & Minimum (Sek.) & Maximum (Sek.) \\\\ \\hline\n')

        for _, row in duration_stats.iterrows():
            f.write(f'{int(row["u"])} & {row["mean_duration"]:.4f} & {row["median_duration"]:.4f} & {row["std_duration"]:.4f} & {row["min_duration"]:.4f} & {row["max_duration"]:.4f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(
            f'\\caption{{Statistische Zusammenfassung der Laufzeit für verschiedene Werte von \\textit{{num\\_u}} (num\\_radii={best_num_radii}, epsilon={best_epsilon}, Dimension={dimension}, k={k}).}}\n')
        f.write(f'\\label{{tab:stats_u_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')

        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write(
            '\\textit{num\\_u} & Mittelwert & Median & Standard-abweichung & Minimum & Maximum \\\\ \\hline\n')

        for _, row in sum_of_radii_stats.iterrows():
            f.write(f'{int(row["u"])} & {row["mean_sum"]:.4f} & {row["median_sum"]:.4f} & {row["std_sum"]:.4f} & {row["min_sum"]:.4f} & {row["max_sum"]:.4f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(
            f'\\caption{{Statistische Zusammenfassung der Summe der Radien für verschiedene Werte von \\textit{{num\\_u}} (num\\_radii={best_num_radii}, epsilon={best_epsilon}, Dimension={dimension}, k={k}).}}\n')
        f.write(f'\\label{{tab:summe_der_radien_u_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')


        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|}\n')
        f.write('\\hline\n')
        f.write(
            '\\textit{num\\_u} & Mittelwert Summe der Radien & Verbesserung \\%  \\\\ \\hline\n')

        for _, row in improvement_df.iterrows():
            f.write(f'{int(row['u'])} & {row['Durchschnitt der Summe der Radien']:.6f} & {row['Improvement (%)']:.2f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(
            f'\\caption{{Verbesserung des Mittelwerts der Summe der Radien für verschiedene Werte von \\textit{{num\\_u}} (num\\_radii={best_num_radii}, epsilon={best_epsilon}, Dimension={dimension}, k={k}).}}\n')
        f.write(f'\\label{{tab:summe_der_radien_u_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')


def analyze_num_radii(df, best_u, best_epsilon, output_directory, dimension, k):
    best_df = df[(df['u'] == best_u) & (df['epsilon'] == best_epsilon)]

    duration_stats = best_df.groupby('num_radii').agg(
        mean_duration=('Dauer (Sekunden)', 'mean'),
        median_duration=('Dauer (Sekunden)', 'median'),
        std_duration=('Dauer (Sekunden)', 'std'),
        min_duration=('Dauer (Sekunden)', 'min'),
        max_duration=('Dauer (Sekunden)', 'max')
    ).reset_index()

    sum_of_radii_stats = best_df.groupby('num_radii').agg(
        mean_sum=('Summe_der_Radien', 'mean'),
        median_sum=('Summe_der_Radien', 'median'),
        std_sum=('Summe_der_Radien', 'std'),
        min_sum=('Summe_der_Radien', 'min'),
        max_sum=('Summe_der_Radien', 'max')
    ).reset_index()

    plt.figure(figsize=(12, 8))
    best_df.boxplot(column='Dauer (Sekunden)', by='num_radii')
    plt.title(f'Laufzeit nach num_radii\n(num_u={best_u}, epsilon={best_epsilon}, Dimension={dimension}, k={k})', fontsize=14, pad=20)
    plt.suptitle('')
    plt.xlabel('num_radii', fontsize=12)
    plt.ylabel('Dauer (Sekunden)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{output_directory}/num_radii_runtime_boxplot.pdf')
    plt.close('all')

    plt.figure(figsize=(12, 8))
    best_df.boxplot(column='Summe_der_Radien', by='num_radii')
    plt.title(f'Verteilung der Summe der Radien nach Anzahl von num_radii\n (num_u={best_u}, epsilon={best_epsilon}, Dimension={dimension}, k={k})', fontsize=13, pad=20)
    plt.suptitle('')
    plt.xlabel('num_radii', fontsize=12)
    plt.ylabel('Summe der Radien', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{output_directory}/num_radii_sum_of_radii_boxplot.pdf')
    plt.close('all') 

    # Erstellen des Liniendiagramms
    plt.figure(figsize=(12, 8))
    line_data = best_df.groupby('num_radii').agg(
        avg_dauer=('Dauer (Sekunden)', 'mean')
    ).reset_index()
    plt.plot(line_data['num_radii'], line_data['avg_dauer'], marker='o', linestyle='-', color='b')
    plt.title(f'Laufzeit nach num_radii\n(num_u={best_u}, epsilon={best_epsilon}, Dimension={dimension}, k={k})', fontsize=25, pad=20)
    plt.xlabel('num_radii', fontsize=20)
    plt.ylabel('Dauer (Sekunden)', fontsize=20)
    plt.xticks(line_data['num_radii'], fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{output_directory}/num_radii_runtime_lineplot.pdf')
    plt.close('all')

    min_num_radii = min(df['num_radii'])
    min_df = best_df[best_df['num_radii'] == min_num_radii]
    max_radius = min_df['Summe_der_Radien'].mean()


    improvements = []

    for num_radii in best_df['num_radii'].unique():
        df_num_radii = best_df[(best_df['num_radii'] == num_radii)]
        num_radii_radius = df_num_radii[f'Summe_der_Radien'].mean()
        improvement = (max_radius - num_radii_radius) / max_radius * 100

        improvements.append((num_radii, improvement, num_radii_radius))

    improvement_df = pd.DataFrame(improvements, columns=['num_radii', 'Improvement (%)', 'Durchschnitt der Summe der Radien'])


    tex_file = f'{output_directory}/num_radii.tex'

    with open(tex_file, 'w') as f:
        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write(
            '\\textit{num\\_radii} & Mittelwert (Sek.) & Median (Sek.) & Standard-abweichung & Minimum (Sek.) & Maximum (Sek.) \\\\ \\hline\n')

        for _, row in duration_stats.iterrows():
            f.write(f'{int(row['num_radii'])} & {row['mean_duration']:.4f} & {row['median_duration']:.4f} & {
                    row['std_duration']:.4f} & {row['min_duration']:.4f} & {row['max_duration']:.4f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(
            f'\\caption{{Statistische Zusammenfassung der Laufzeit für verschiedene Werte von \\textit{{num\\_radii}} (num\\_u={best_u}, epsilon={best_epsilon}, Dimension={dimension}, k={k}).}}\n')
        f.write(f'\\label{{tab:stats_num_radii_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')

        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write(
            '\\textit{num\\_radii} & Mittelwert & Median & Standard-abweichung & Minimum & Maximum \\\\ \\hline\n')

        for _, row in sum_of_radii_stats.iterrows():
            f.write(f'{int(row['num_radii'])} & {row['mean_sum']:.4f} & {row['median_sum']:.4f} & {
                    row['std_sum']:.4f} & {row['min_sum']:.4f} & {row['max_sum']:.4f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(
            f'\\caption{{Statistische Zusammenfassung der Summe der Radien für verschiedene Werte von \\textit{{num\\_radii}} (num\\_u={best_u}, epsilon={best_epsilon}, Dimension={dimension}, k={k}).}}\n')
        f.write(f'\\label{{tab:summe_der_radien_num_radii_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')


        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|}\n')
        f.write('\\hline\n')
        f.write(
            '\\textit{num\\_radii} & Mittelwert Summe der Radien & Verbesserung \\% \\\\ \\hline\n')

        for _, row in improvement_df.iterrows():
            f.write(f'{int(row['num_radii'])} & {row['Durchschnitt der Summe der Radien']:.6f} & {row['Improvement (%)']:.2f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(
            f'\\caption{{Verbesserung des Mittelwerts der Summe der Radien für verschiedene Werte von \\textit{{num\\_radii}} (num\\_u={best_u}, epsilon={best_epsilon}, Dimension={dimension}, k={k}).}}\n')
        f.write(f'\\label{{tab:summe_der_radien_num_radii_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')


def analyze_epsilon(df, best_u, best_num_radii, output_directory, dimension, k):
    best_df = df[(df['u'] == best_u) & (df['num_radii'] == best_num_radii)]

    duration_stats = best_df.groupby('epsilon').agg(
        mean_duration=('Dauer (Sekunden)', 'mean'),
        median_duration=('Dauer (Sekunden)', 'median'),
        std_duration=('Dauer (Sekunden)', 'std'),
        min_duration=('Dauer (Sekunden)', 'min'),
        max_duration=('Dauer (Sekunden)', 'max')
    ).reset_index()

    sum_of_radii_stats = best_df.groupby('epsilon').agg(
        mean_sum=('Summe_der_Radien', 'mean'),
        median_sum=('Summe_der_Radien', 'median'),
        std_sum=('Summe_der_Radien', 'std'),
        min_sum=('Summe_der_Radien', 'min'),
        max_sum=('Summe_der_Radien', 'max')
    ).reset_index()

    plt.figure(figsize=(12, 8))
    best_df.boxplot(column='Dauer (Sekunden)', by='epsilon')
    plt.title(f'Laufzeit nach epsilon\n(num_u={best_u}, num_radii={best_num_radii}, Dimension={dimension}, k={k})', fontsize=14, pad=20)
    plt.suptitle('')
    plt.xlabel('epilon', fontsize=12)
    plt.ylabel('Dauer (Sekunden)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{output_directory}/epsilon_runtime_boxplot.pdf')
    plt.close('all')

    # Erstellen des Liniendiagramms
    plt.figure(figsize=(12, 8))
    line_data = best_df.groupby('epsilon').agg(
        avg_dauer=('Dauer (Sekunden)', 'mean')
    ).reset_index()
    plt.plot(line_data['epsilon'], line_data['avg_dauer'], marker='o', linestyle='-', color='b')
    plt.title(f'Laufzeit nach epsilon\n(num_u={best_u}, num_radii={best_num_radii}, Dimension={dimension}, k={k})', fontsize=25, pad=20)
    plt.xlabel('epsilon', fontsize=20)
    plt.ylabel('Dauer (Sekunden)', fontsize=20)
    plt.xticks(line_data['epsilon'], fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{output_directory}/epsilon_runtime_lineplot.pdf')
    plt.close('all')

    plt.figure(figsize=(12, 8))
    best_df.boxplot(column='Summe_der_Radien', by='epsilon')
    plt.title(f'Verteilung der Summe der Radien nach epsilon\n (num_u={best_u}, num_radii={best_num_radii}, Dimension={dimension}, k={k})', fontsize=14, pad=20)
    plt.suptitle('')
    plt.xlabel('epsilon', fontsize=12)
    plt.ylabel('Summe der Radien', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{output_directory}/epsilon_sum_of_radii_boxplot.pdf')
    plt.close('all')

    min_epsilon = min(df['epsilon'])
    min_df = best_df[best_df['epsilon'] == min_epsilon]
    max_radius = min_df['Summe_der_Radien'].mean()


    improvements = []

    for epsilon in best_df['epsilon'].unique():
        df_epsilon = best_df[(best_df['epsilon'] == epsilon)]
        epsilon_radius = df_epsilon[f'Summe_der_Radien'].mean()
        improvement = (max_radius - epsilon_radius) / max_radius * 100

        improvements.append((epsilon, improvement, epsilon_radius))

    improvement_df = pd.DataFrame(improvements, columns=['epsilon', 'Improvement (%)', 'Durchschnitt der Summe der Radien'])

    tex_file = f'{output_directory}/epsilon.tex'

    with open(tex_file, 'w') as f:
        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write(
            '\\textit{epsilon} & Mittelwert (Sek.) & Median (Sek.) & Standard-abweichung & Minimum (Sek.) & Maximum (Sek.) \\\\ \\hline\n')

        for _, row in duration_stats.iterrows():
            f.write(f'{row['epsilon']} & {row['mean_duration']:.4f} & {row['median_duration']:.4f} & {
                    row['std_duration']:.4f} & {row['min_duration']:.4f} & {row['max_duration']:.4f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(
            f'\\caption{{Statistische Zusammenfassung der Laufzeit für verschiedene Werte von \\textit{{epsilon}} (num\\_u={best_u}, num\\_radii={best_num_radii}, Dimension={dimension}, k={k}).}}\n')
        f.write(f'\\label{{tab:stats_epsilon_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')

        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write(
            '\\textit{epsilon} & Mittelwert & Median & Standard-abweichung & Minimum & Maximum \\\\ \\hline\n')

        for _, row in sum_of_radii_stats.iterrows():
            f.write(f'{row['epsilon']} & {row['mean_sum']:.4f} & {row['median_sum']:.4f} & {
                    row['std_sum']:.4f} & {row['min_sum']:.4f} & {row['max_sum']:.4f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(
            f'\\caption{{Statistische Zusammenfassung der Summe der Radien für verschiedene Werte von \\textit{{epsilon}} (num\\_u={best_u}, num\\_radii={best_num_radii}, Dimension={dimension}, k={k}).}}\n')
        f.write(f'\\label{{tab:summe_der_radien_epsilon_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')


        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('\\textit{epsilon} & Mittelwert Summe der Radien & Verbesserung \\% \\\\ \\hline\n')

        for _, row in improvement_df.iterrows():
            f.write(f'{row['epsilon']} & {row['Durchschnitt der Summe der Radien']:.6f} & {row['Improvement (%)']:.2f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(
            f'\\caption{{Verbesserung des Mittelwerts der Summe der Radien für verschiedene Werte von \\textit{{epsilon}} (num\\_u={best_u}, num\\_radii={best_num_radii}, Dimension={dimension}, k={k}).}}\n')
        f.write(f'\\label{{tab:summe_der_radien_epsilon_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')

def analyze_runtime_for_dimensions_and_k(df, dimensions, ks, best_u, best_num_radii, best_epsilon, output_directory):
    # Filter für die besten Parameter
    filtered_df = df[(df['u'] == best_u) & 
                     (df['num_radii'] == best_num_radii) & 
                     (df['epsilon'] == best_epsilon)]

    runtime_stats = filtered_df.groupby(['Dimension', 'k']).agg(
        mean_duration=('Dauer (Sekunden)', 'mean'),
        median_duration=('Dauer (Sekunden)', 'median'),
        std_duration=('Dauer (Sekunden)', 'std'),
        min_duration=('Dauer (Sekunden)', 'min'),
        max_duration=('Dauer (Sekunden)', 'max')
    ).reset_index()

    plt.figure(figsize=(14, 10))
    for dimension in dimensions:
        subset = runtime_stats[runtime_stats['Dimension'] == dimension]
        plt.plot(subset['k'], subset['mean_duration'], label=f'Dimension {dimension}', marker='o')

    plt.title('Durchschnittliche Laufzeit für verschiedene Dimensionen und k', fontsize=16)
    plt.xlabel('k', fontsize=14)
    plt.xticks(ks)
    plt.ylabel('Durchschnittliche Laufzeit (Sekunden)', fontsize=14)
    plt.legend(title='Dimension')
    plt.tight_layout()
    plt.savefig(f'{output_directory}/runtime_dimension_k_plot.pdf')
    plt.close('all')

    # Detailierte Statistik für jede Kombination von Dimension und k
    detailed_stats = []
    for dimension in dimensions:
        for k in ks:
            subset = df[(df['Dimension'] == dimension) & (df['k'] == k)]
            stats = subset['Dauer (Sekunden)'].describe().to_dict()
            stats.update({'Dimension': dimension, 'k': k})
            detailed_stats.append(stats)

    detailed_stats_df = pd.DataFrame(detailed_stats)
    detailed_stats_df.to_csv(f'{output_directory}/detailed_runtime_stats.csv', index=False)

    print('Laufzeitanalyse abgeschlossen und Ergebnisse gespeichert.')


def compare_algorithms(dimension, k, seeds, data_directory):

    fpt_heuristic_list = []
    gonzalez_list = []
    kmeans_list = []

    for seed in seeds:
        fpt_heuristic_results = pd.read_csv(f'{data_directory}/Dimension={dimension}/k={k}/FPT_Heuristic/Seed={seed}/Results/results.csv')
        fpt_heuristic_results['Seed'] = seed
        fpt_heuristic_list.append(fpt_heuristic_results)
        
        gonzalez_results = pd.read_csv(f'{data_directory}/Dimension={dimension}/k={k}/Gonzalez/Seed={seed}/Results/results.csv')
        gonzalez_results['Seed'] = seed
        gonzalez_list.append(gonzalez_results)
        
        kmeans_results = pd.read_csv(f'{data_directory}/Dimension={dimension}/k={k}/KMeansPlusPlus/Seed={seed}/Results/results.csv')
        kmeans_results['Seed'] = seed
        kmeans_list.append(kmeans_results)

    fpt_heuristic_df = pd.concat(fpt_heuristic_list, ignore_index=True)
    gonzalez_df = pd.concat(gonzalez_list, ignore_index=True)
    kmeans_df = pd.concat(kmeans_list, ignore_index=True)
    string = 'mean'

    fpt_heuristic_df = fpt_heuristic_df.groupby(['Datei', 'u', 'epsilon', 'num_radii']).agg({'Dauer (Sekunden)': string, 'Summe_der_Radien': string}).reset_index()
    gonzalez_df = gonzalez_df.groupby('Datei').agg({'Dauer (Sekunden)': string, 'Summe_der_Radien': string}).reset_index()
    kmeans_df = kmeans_df.groupby('Datei').agg({'Dauer (Sekunden)': string, 'Summe_der_Radien': string}).reset_index()
    heuristik_df = pd.read_csv(f'{data_directory}/Dimension={dimension}/k={k}/Heuristik/Results/results.csv')

    best_parameter_df = fpt_heuristic_df.groupby(['u', 'num_radii', 'epsilon']).agg({'Dauer (Sekunden)': string, 'Summe_der_Radien': string}).reset_index()
    best_row = best_parameter_df.loc[best_parameter_df['Summe_der_Radien'].idxmin()]

    best_u = best_row['u']
    best_num_radii = best_row['num_radii']
    best_epsilon = best_row['epsilon']


    # Berechnen der Durchschnittswerte der Radien für jeden Algorithmus
    best_FPT_Heuristic_avg_radius = best_row['Summe_der_Radien']
    gonzalez_avg_radius = gonzalez_df['Summe_der_Radien'].mean()
    kmeans_avg_radius = kmeans_df['Summe_der_Radien'].mean()
    heuristik_avg_radius = heuristik_df['Summe_der_Radien'].mean()

    # Anzeigen der Durchschnittswerte
    print(f'Durchschnittlicher Radius für FPT_Heuristic: {best_FPT_Heuristic_avg_radius:.6f}')
    print(f'Durchschnittlicher Radius für Gonzalez: {gonzalez_avg_radius:.6f}')
    print(f'Durchschnittlicher Radius für KMeans++: {kmeans_avg_radius:.6f}')
    print(f'Durchschnittlicher Radius für Heuristik: {heuristik_avg_radius:.6f}')

    all_comparison_results = []
    worse_FPT_Heuristic_points = []
    # Schleife über alle Kombinationen von u und epsilon
    for u_val in fpt_heuristic_df['u'].unique():
        for num_radii_val in fpt_heuristic_df['num_radii'].unique():
            for epsilon_val in fpt_heuristic_df['epsilon'].unique():

                # Zusammensetzen der Dataframes
                merged_df = pd.merge(
                    fpt_heuristic_df[(fpt_heuristic_df['u'] == u_val) & (fpt_heuristic_df['num_radii'] == num_radii_val) & (fpt_heuristic_df['epsilon'] == epsilon_val)][['Datei','Dauer (Sekunden)', 'Summe_der_Radien']],
                    gonzalez_df[['Datei', 'Dauer (Sekunden)', 'Summe_der_Radien']], on='Datei', suffixes=('_FPT_Heuristic', '_Gonzalez'))

                merged_df = pd.merge(merged_df, kmeans_df[['Datei', 'Dauer (Sekunden)', 'Summe_der_Radien']], on='Datei')
                merged_df.rename(columns={'Summe_der_Radien': 'Summe_der_Radien_KMeans', 'Dauer (Sekunden)': 'Dauer (Sekunden)_KMeans'}, inplace=True)
                merged_df = pd.merge(merged_df, heuristik_df[['Datei', 'Dauer (Sekunden)', 'Summe_der_Radien']], on='Datei')
                merged_df.rename(columns={'Summe_der_Radien': 'Summe_der_Radien_Heuristik', 'Dauer (Sekunden)': 'Dauer (Sekunden)_Heuristik'}, inplace=True)

                # Runden der Radien
                merged_df['Summe_der_Radien_FPT_Heuristic'] = merged_df['Summe_der_Radien_FPT_Heuristic'].round(7)
                merged_df['Summe_der_Radien_Gonzalez'] = merged_df['Summe_der_Radien_Gonzalez'].round(7)
                merged_df['Summe_der_Radien_KMeans'] = merged_df['Summe_der_Radien_KMeans'].round(7)
                merged_df['Summe_der_Radien_Heuristik'] = merged_df['Summe_der_Radien_Heuristik'].round(7)

                duration_fpt_heuristik = merged_df['Dauer (Sekunden)_FPT_Heuristic'].mean().round(7)
                duration_gonzalez = merged_df['Dauer (Sekunden)_Gonzalez'].mean().round(7)
                duration_kmeans = merged_df['Dauer (Sekunden)_KMeans'].mean().round(7)
                duration_heuristik = merged_df['Dauer (Sekunden)_Heuristik'].mean().round(7)

                # Zählen, wie oft FPT_Heuristic besser als alle anderen ist
                merged_df['FPT_Heuristic_better_than_all'] = merged_df.apply(lambda row: row['Summe_der_Radien_FPT_Heuristic'] < row[['Summe_der_Radien_Gonzalez',
                                                                                                                                                'Summe_der_Radien_KMeans', 'Summe_der_Radien_Heuristik']].min(),axis=1)

                # Zählen, wie oft FPT_Heuristic schlechter als alle anderen ist
                merged_df['FPT_Heuristic_worse_than_all'] = merged_df.apply(lambda row: row['Summe_der_Radien_FPT_Heuristic'] > row[['Summe_der_Radien_Gonzalez', 
                                                                                                                                               'Summe_der_Radien_KMeans', 'Summe_der_Radien_Heuristik']].max(),axis=1)
                FPT_Heuristic_better_count = merged_df['FPT_Heuristic_better_than_all'].sum()
                FPT_Heuristic_worse_count = merged_df['FPT_Heuristic_worse_than_all'].sum()

                # Zählt wie oft FPT_Heuristic besser ist als Gonzalez
                FPT_Heuristic_vs_gonzalez_better = (merged_df['Summe_der_Radien_FPT_Heuristic'] < merged_df['Summe_der_Radien_Gonzalez']).sum()

                FPT_Heuristic_vs_gonzalez_equal = (merged_df['Summe_der_Radien_FPT_Heuristic'] == merged_df['Summe_der_Radien_Gonzalez']).sum()

                # Zählt wie oft FPT_Heuristic schlechter ist als Gonzalez
                FPT_Heuristic_vs_gonzalez_worse = (merged_df['Summe_der_Radien_FPT_Heuristic'] > merged_df['Summe_der_Radien_Gonzalez']).sum()

                # Zählt wie oft FPT_Heuristic besser ist als KMeans++
                FPT_Heuristic_vs_kmeans_better = (merged_df['Summe_der_Radien_FPT_Heuristic'] < merged_df['Summe_der_Radien_KMeans']).sum()

                FPT_Heuristic_vs_kmeans_equal = (merged_df['Summe_der_Radien_FPT_Heuristic'] == merged_df['Summe_der_Radien_KMeans']).sum()

                # Zählt wie oft FPT_Heuristic schlechter ist als KMeans++
                FPT_Heuristic_vs_kmeans_worse = (merged_df['Summe_der_Radien_FPT_Heuristic'] > merged_df['Summe_der_Radien_KMeans']).sum()

                # Zählt wie oft FPT_Heuristic besser ist als die Heuristik
                FPT_Heuristic_vs_heuristik_better = (merged_df['Summe_der_Radien_FPT_Heuristic'] < merged_df['Summe_der_Radien_Heuristik']).sum()

                FPT_Heuristic_vs_heuristik_equal = (merged_df['Summe_der_Radien_FPT_Heuristic'] == merged_df['Summe_der_Radien_Heuristik']).sum()

                # Zählt wie oft FPT_Heuristic schlechter ist als die Heuristik
                FPT_Heuristic_vs_heuristik_worse = (merged_df['Summe_der_Radien_FPT_Heuristic'] > merged_df['Summe_der_Radien_Heuristik']).sum()


                total_count = merged_df.shape[0]

                FPT_Heuristic_better_percentage = (FPT_Heuristic_better_count / total_count) * 100
                FPT_Heuristic_worse_percentage = (FPT_Heuristic_worse_count / total_count) * 100

                FPT_Heuristic_vs_gonzalez_better_percentage = (FPT_Heuristic_vs_gonzalez_better / total_count) * 100
                FPT_Heuristic_vs_gonzalez_equal_percentage = (FPT_Heuristic_vs_gonzalez_equal / total_count) * 100
                FPT_Heuristic_vs_gonzalez_worse_percentage = (FPT_Heuristic_vs_gonzalez_worse / total_count) * 100

                FPT_Heuristic_vs_kmeans_better_percentage = (FPT_Heuristic_vs_kmeans_better / total_count) * 100
                FPT_Heuristic_vs_kmeans_equal_percentage = (FPT_Heuristic_vs_kmeans_equal / total_count) * 100
                FPT_Heuristic_vs_kmeans_worse_percentage = (FPT_Heuristic_vs_kmeans_worse / total_count) * 100

                FPT_Heuristic_vs_heuristik_better_percentage = (FPT_Heuristic_vs_heuristik_better / total_count) * 100
                FPT_Heuristic_vs_heuristik_equal_percentage = (FPT_Heuristic_vs_heuristik_equal / total_count) * 100
                FPT_Heuristic_vs_heuristik_worse_percentage = (FPT_Heuristic_vs_heuristik_worse / total_count) * 100

                # Ergebnisse sammeln
                all_comparison_results.append({
                    'u': u_val,
                    'epsilon': epsilon_val,
                    'num_radii': num_radii_val,
                    'Laufzeit FPT-Heuristik': duration_fpt_heuristik, 
                    'Laufzeit Gonzalez': duration_gonzalez, 
                    'Laufzeit KMeans': duration_kmeans,
                    'Laufzeit Heuristik': duration_heuristik, 
                    'FPT_Heuristic vs Alle Besser (%)': FPT_Heuristic_better_percentage,
                    'FPT_Heuristic vs Alle Schlechter (%)': FPT_Heuristic_worse_percentage,
                    'FPT_Heuristic vs Gonzalez Besser (%)': FPT_Heuristic_vs_gonzalez_better_percentage,
                    'FPT_Heuristic vs Gonzalez Schlechter (%)': FPT_Heuristic_vs_gonzalez_worse_percentage,
                    'FPT_Heuristic vs Gonzalez Gleich (%)': FPT_Heuristic_vs_gonzalez_equal_percentage,
                    'FPT_Heuristic vs KMeans++ Besser (%)': FPT_Heuristic_vs_kmeans_better_percentage,
                    'FPT_Heuristic vs KMeans++ Schlechter (%)': FPT_Heuristic_vs_kmeans_worse_percentage,
                    'FPT_Heuristic vs KMeans++ Gleich (%)': FPT_Heuristic_vs_kmeans_equal_percentage,
                    'FPT_Heuristic vs Heuristik Besser (%)': FPT_Heuristic_vs_heuristik_better_percentage,
                    'FPT_Heuristic vs Heuristik Schlechter (%)': FPT_Heuristic_vs_heuristik_worse_percentage,
                    'FPT_Heuristic vs Heuristik Gleich (%)': FPT_Heuristic_vs_heuristik_equal_percentage
                })

                # Speichern der Dateien, bei denen FPT_Heuristic schlechter als alle anderen ist
                if u_val == best_u and epsilon_val == best_epsilon and num_radii_val == best_num_radii:
                    worse_files = merged_df[merged_df['FPT_Heuristic_worse_than_all'] == True]['Datei'].tolist()
                    worse_FPT_Heuristic_points.extend(worse_files)

    # Ergebnisse in ein DataFrame packen
    comparison_df = pd.DataFrame(all_comparison_results)

    # Ergebnisse als Tabelle speichern
    os.makedirs(f'{data_directory}/Dimension={dimension}/k={k}/ComparisonResults', exist_ok=True)
    comparison_df.to_csv(f'{data_directory}/Dimension={dimension}/k={k}/ComparisonResults/comparison_all_us_epsilons.csv', index=False)

    # Markdown-Datei erstellen
    with open(f'{data_directory}/Dimension={dimension}/k={k}/ComparisonResults/comparison_all_us_epsilons.md', 'w') as md_file:
        md_file.write('# Paarweiser Vergleich der Clustering-Algorithmen für alle u- und epsilon-Werte\n\n')
        md_file.write('| u  | epsilon | num_radii | FPT_Heuristic vs Alle Besser (%) | FPT_Heuristic vs Alle Schlechter (%) | FPT_Heuristic vs Gonzalez Besser (%) | FPT_Heuristic vs Gonzalez Schlechter (%) | FPT_Heuristic vs KMeans++ Besser (%) | FPT_Heuristic vs KMeans++ Schlechter (%) | FPT_Heuristic vs Heuristik Besser (%) | FPT_Heuristic vs Heuristik Schlechter (%) |\n')
        md_file.write('|----|---------|---------------|----------------------------|--------------------------------|--------------------------------|------------------------------------|--------------------------------|------------------------------------|---------------------------------|-------------------------------------|\n')
        for _, row in comparison_df.iterrows():
            md_file.write(f'| {int(row['u'])} | {row['epsilon']} | {int(row['num_radii'])} |{row['FPT_Heuristic vs Alle Besser (%)']:.2f} | {row['FPT_Heuristic vs Alle Schlechter (%)']:.2f} | {row['FPT_Heuristic vs Gonzalez Besser (%)']:.2f} | {row['FPT_Heuristic vs Gonzalez Schlechter (%)']:.2f} | {
                          row['FPT_Heuristic vs KMeans++ Besser (%)']:.2f} | {row['FPT_Heuristic vs KMeans++ Schlechter (%)']:.2f} | {row['FPT_Heuristic vs Heuristik Besser (%)']:.2f} | {row['FPT_Heuristic vs Heuristik Schlechter (%)']:.2f} |\n')

    # Speichern der Dateien, bei denen FPT_Heuristic schlechter als alle anderen ist
    with open(f'{data_directory}/Dimension={dimension}/k={k}/ComparisonResults/worse_FPT_Heuristic_points.md', 'w') as md_file:
        md_file.write(f'u: {best_u}, num_radii: {best_num_radii}, epsilon: {best_epsilon}\n')
        md_file.write('Dateien:\n')
        for datei in worse_FPT_Heuristic_points:
            md_file.write(f'{datei}\n')
        md_file.write('\n')


    with open(f'{data_directory}/Dimension={dimension}/k={k}/ComparisonResults/FPT_Heuristik_vs_Gonzalez.tex', 'w') as f:
        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('\\textit{num\\_u} & Besser (\\%) & Gleich (\\%) & Schlechter (\\%)\\\\ \\hline\n')

        for _, row in comparison_df.iterrows():
            f.write(f'{int(row['u'])} & {row['FPT_Heuristic vs Gonzalez Besser (%)']:.2f} & {row['FPT_Heuristic vs Gonzalez Gleich (%)']:.2f} & {row['FPT_Heuristic vs Gonzalez Schlechter (%)']:.2f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Ergebnisse der FPT-Heuristik und Gonzalez für Daten der Dimension={dimension} und \\textit{{k}}={k}}}\n')
        f.write(f'\\label{{tab:fpt_heuristic_vs_gonzalez_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')


        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('\\textit{num\\_u} & Laufzeit FPT-Heuristik & Laufzeit Gonzalez\\\\ \\hline\n')

        for _, row in comparison_df.iterrows():
            f.write(f'{int(row['u'])} & {row['Laufzeit FPT-Heuristik']} & {row['Laufzeit Gonzalez']} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Laufzeit der FPT-Heuristik und Gonzalez für Daten der Dimension={dimension} und \\textit{{k}}={k} (Sekunden)}}\n')
        f.write(f'\\label{{tab:fpt_heuristic_vs_gonzalez_laufzeit_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')
    
    with open(f'{data_directory}/Dimension={dimension}/k={k}/ComparisonResults/FPT_Heuristik_vs_KMeans.tex', 'w') as f:
        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('\\textit{num\\_u} & Besser (\\%) & Gleich (\\%) & Schlechter (\\%)\\\\ \\hline\n')

        for _, row in comparison_df.iterrows():
            f.write(f'{int(row['u'])} & {row['FPT_Heuristic vs KMeans++ Besser (%)']:.2f}& {row['FPT_Heuristic vs KMeans++ Gleich (%)']:.2f}  & {row['FPT_Heuristic vs KMeans++ Schlechter (%)']:.2f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Ergebnisse der FPT-Heuristik und \\textit{{k}}-Means++ für Daten der Dimension={dimension} und \\textit{{k}}={k}}}\n')
        f.write(f'\\label{{tab:fpt_heuristic_vs_kmeans_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')

        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('\\textit{num\\_u} & Laufzeit FPT-Heuristik & Laufzeit \\textit{k}-Means++\\\\ \\hline\n')

        for _, row in comparison_df.iterrows():
            f.write(f'{int(row['u'])} & {row['Laufzeit FPT-Heuristik']} & {row['Laufzeit KMeans']} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Laufzeit der FPT-Heuristik und \\textit{{k}}-Means++ für Daten der Dimension={dimension} und \\textit{{k}}={k} (Sekunden)}}\n')
        f.write(f'\\label{{tab:fpt_heuristic_vs_kmeans_laufzeit_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')
    
    with open(f'{data_directory}/Dimension={dimension}/k={k}/ComparisonResults/FPT_Heuristik_vs_Heuristik.tex', 'w') as f:
        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('\\textit{num\\_u} & Besser (\\%) & Gleich (\\%) & Schlechter (\\%)\\\\ \\hline\n')

        for _, row in comparison_df.iterrows():
            f.write(f'{int(row['u'])} & {row['FPT_Heuristic vs Heuristik Besser (%)']:.2f} & {row['FPT_Heuristic vs Heuristik Gleich (%)']:.2f} & {row['FPT_Heuristic vs Heuristik Schlechter (%)']:.2f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Ergebnisse der FPT-Heuristik und Heuristik für Daten der Dimension={dimension} und \\textit{{k}}={k}}}\n')
        f.write(f'\\label{{tab:fpt_heuristic_vs_heuristik_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')

        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('\\textit{num\\_u} & Laufzeit FPT-Heuristik & Laufzeit Heuristik\\\\ \\hline\n')

        for _, row in comparison_df.iterrows():
            f.write(f'{int(row['u'])} & {row['Laufzeit FPT-Heuristik']} & {row['Laufzeit Heuristik']} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Laufzeit der FPT-Heuristik und Heuristik für Daten der Dimension={dimension} und \\textit{{k}}={k} (Sekunden)}}\n')
        f.write(f'\\label{{tab:fpt_heuristic_vs_heuristik_laufzeit_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')

def compare_heuristic_with_other_algorithms(dimensions, ks, seeds, data_directory):

    comparison_results= []
    mean_results = []
    for dimension in dimensions:
        for k in ks:
            gonzalez_list = []
            kmeans_list = []
            heuristik_list = []
            for seed in seeds:        
                gonzalez_results = pd.read_csv(f'{data_directory}/Dimension={dimension}/k={k}/Gonzalez/Seed={seed}/Results/results.csv')
                gonzalez_results['Seed'] = seed
                gonzalez_list.append(gonzalez_results)
                kmeans_results = pd.read_csv(f'{data_directory}/Dimension={dimension}/k={k}/KMeansPlusPlus/Seed={seed}/Results/results.csv')
                kmeans_results['Seed'] = seed
                kmeans_list.append(kmeans_results)
            
            heuristik_results = pd.read_csv(f'{data_directory}/Dimension={dimension}/k={k}/Heuristik/Results/results.csv')
            heuristik_results['Dimension'] = dimension
            heuristik_results['k'] = k
            heuristik_list.append(heuristik_results)

            gonzalez_df = pd.concat(gonzalez_list, ignore_index=True)
            gonzalez_df['Dimension'] = dimension
            gonzalez_df['k'] = k
            kmeans_df = pd.concat(kmeans_list, ignore_index=True)
            kmeans_df['Dimension'] = dimension
            kmeans_df['k'] = k
            heuristik_df = pd.concat(heuristik_list, ignore_index=True)


            gonzalez_df = gonzalez_df.groupby(['Datei', 'k', 'Dimension']).agg({'Dauer (Sekunden)': 'mean', 'Summe_der_Radien': 'mean'}).reset_index()
            kmeans_df = kmeans_df.groupby(['Datei', 'k', 'Dimension']).agg({'Dauer (Sekunden)': 'mean', 'Summe_der_Radien': 'mean'}).reset_index()
            heuristik_df = heuristik_df.groupby(['Datei', 'k', 'Dimension']).agg({'Dauer (Sekunden)': 'mean', 'Summe_der_Radien': 'mean'}).reset_index()

            merged_df = pd.merge(gonzalez_df, heuristik_df, on='Datei', suffixes=('_Gonzalez', '_Heuristik'))
            merged_df = pd.merge(merged_df, kmeans_df, on='Datei')
            merged_df.rename(columns={'Summe_der_Radien': 'Summe_der_Radien_KMeans', 'Dauer (Sekunden)': 'Dauer (Sekunden)_KMeans', 'k': 'k_KMeans', 'Dimension': 'Dimension_KMeans'}, inplace=True)
            
            merged_df['Summe_der_Radien_Gonzalez'] = merged_df['Summe_der_Radien_Gonzalez'].round(7)
            merged_df['Summe_der_Radien_KMeans'] = merged_df['Summe_der_Radien_KMeans'].round(7)
            merged_df['Summe_der_Radien_Heuristik'] = merged_df['Summe_der_Radien_Heuristik'].round(7)

            mean_duration_heuristik = merged_df['Dauer (Sekunden)_Heuristik'].mean()
            mean_sum_radii_heuristik = merged_df['Summe_der_Radien_Heuristik'].mean()

            mean_duration_gonzalez = merged_df['Dauer (Sekunden)_Gonzalez'].mean()
            mean_sum_radii_gonzalez = merged_df['Summe_der_Radien_Gonzalez'].mean()

            mean_duration_kmeans = merged_df['Dauer (Sekunden)_KMeans'].mean()
            mean_sum_radii_kmeans = merged_df['Summe_der_Radien_KMeans'].mean()


            Heuristik_vs_gonzalez_better = (merged_df['Summe_der_Radien_Heuristik'] < merged_df['Summe_der_Radien_Gonzalez']).sum()
            Heuristik_vs_gonzalez_equal = (merged_df['Summe_der_Radien_Heuristik'] == merged_df['Summe_der_Radien_Gonzalez']).sum()
            Heuristik_vs_gonzalez_worse = (merged_df['Summe_der_Radien_Heuristik'] > merged_df['Summe_der_Radien_Gonzalez']).sum()

            Heuristik_vs_kmeans_better = (merged_df['Summe_der_Radien_Heuristik'] < merged_df['Summe_der_Radien_KMeans']).sum()
            Heuristik_vs_kmeans_equal = (merged_df['Summe_der_Radien_Heuristik'] == merged_df['Summe_der_Radien_KMeans']).sum()
            Heuristik_vs_kmeans_worse = (merged_df['Summe_der_Radien_Heuristik'] > merged_df['Summe_der_Radien_KMeans']).sum()

            # Filtern der Dateien, bei denen die Summe der Radien der Heuristik größer ist als die von Gonzalez
            filtered_df = merged_df[merged_df['Summe_der_Radien_Heuristik'] > merged_df['Summe_der_Radien_KMeans']]

            # Ausgabe der Dateinamen
            file_names = filtered_df['Datei'].tolist()

            total_count = merged_df.shape[0]

            Heuristik_vs_gonzalez_better_percentage = (Heuristik_vs_gonzalez_better / total_count) * 100
            Heuristik_vs_gonzalez_equal_percentage = (Heuristik_vs_gonzalez_equal / total_count) * 100
            Heuristik_vs_gonzalez_worse_percentage = (Heuristik_vs_gonzalez_worse / total_count) * 100
            Heuristik_vs_kmeans_better_percentage = (Heuristik_vs_kmeans_better / total_count) * 100
            Heuristik_vs_kmeans_equal_percentage = (Heuristik_vs_kmeans_equal / total_count) * 100
            Heuristik_vs_kmeans_worse_percentage = (Heuristik_vs_kmeans_worse / total_count) * 100

            mean_results.append({
                'Dimension': dimension,
                'k': k,
                'Dauer (Sekunden)_Heuristik': mean_duration_heuristik,
                'Summe_der_Radien_Heuristik': mean_sum_radii_heuristik,
                'Dauer (Sekunden)_Gonzalez': mean_duration_gonzalez,
                'Summe_der_Radien_Gonzalez': mean_sum_radii_gonzalez,
                'Dauer (Sekunden)_KMeans': mean_duration_kmeans,
                'Summe_der_Radien_KMeans': mean_sum_radii_kmeans
            })

            comparison_results.append({
                'Dimension': dimension,
                'k': k,
                'Heuristik vs Gonzalez Besser (%)': Heuristik_vs_gonzalez_better_percentage,
                'Heuristik vs Gonzalez Schlechter (%)': Heuristik_vs_gonzalez_worse_percentage,
                'Heuristik vs Gonzalez Gleich (%)': Heuristik_vs_gonzalez_equal_percentage,
                'Heuristik vs KMeans++ Besser (%)': Heuristik_vs_kmeans_better_percentage,
                'Heuristik vs KMeans++ Schlechter (%)': Heuristik_vs_kmeans_worse_percentage,
                'Heuristik vs KMeans++ Gleich (%)': Heuristik_vs_kmeans_equal_percentage,
            })

    mean_df = pd.DataFrame(mean_results)
    comparison_df = pd.DataFrame(comparison_results)
    os.makedirs(f'{data_directory}/ComparisonResultsHeuristik', exist_ok=True)


    with open(f'{data_directory}/ComparisonResultsHeuristik/Heuristik_vs_Gonzalez.tex', 'w') as f:
        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('Dimension & k & Besser (\\%) & Gleich (\\%) & Schlechter (\\%)\\\\ \\hline\n')

        for _, row in comparison_df.iterrows():
            f.write(f'{int(row['Dimension'])} & {int(row['k'])} & {row['Heuristik vs Gonzalez Besser (%)']:.2f} & {row['Heuristik vs Gonzalez Gleich (%)']:.2f} & {row['Heuristik vs Gonzalez Schlechter (%)']:.2f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Ergebnisse von der Heuristik und Gonzalez}}\n')
        f.write(f'\\label{{tab:heuristik_vs_gonzalez_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')




        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('Dimension & k & Summe der Radien Heuristik & Summe der Radien Gonzalez \\\\ \\hline\n')

        for _, row in mean_df.iterrows():
            f.write(f'{int(row['Dimension'])} & {int(row['k'])} & {row['Summe_der_Radien_Heuristik']:.7f} & {row['Summe_der_Radien_Gonzalez']:.7f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Summe der Radien von der Heuristik und Gonzalez}}\n')
        f.write(f'\\label{{tab:heuristik_vs_gonzalez_summe_der_radien_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')




        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('Dimension & k & Laufzeit Heuristik & Laufzeit Gonzalez \\\\ \\hline\n')

        for _, row in mean_df.iterrows():
            f.write(f'{int(row['Dimension'])} & {int(row['k'])} & {row['Dauer (Sekunden)_Heuristik']:.7f} & {row['Dauer (Sekunden)_Gonzalez']:.7f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Laufzeit von der Heuristik und Gonzalez (Sekunden)}}\n')
        f.write(f'\\label{{tab:heuristik_vs_gonzalez_laufzeit_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')

    with open(f'{data_directory}/ComparisonResultsHeuristik/Heuristik_vs_KMeans.tex', 'w') as f:
        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('Dimension & k & Besser (\\%) & Gleich (\\%) & Schlechter (\\%)\\\\ \\hline\n')

        for _, row in comparison_df.iterrows():
            f.write(f'{int(row['Dimension'])} & {int(row['k'])} & {row['Heuristik vs KMeans++ Besser (%)']:.2f} & {row['Heuristik vs KMeans++ Gleich (%)']:.2f} & {row['Heuristik vs KMeans++ Schlechter (%)']:.2f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Ergebnisse von der Heuristik und \\textit{{k}}-Means++}}\n')
        f.write(f'\\label{{tab:heuristik_vs_kmeans_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')  


        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('Dimension & k & Summe der Radien Heuristik & Summe der Radien \\textit{k}-Means++ \\\\ \\hline\n')

        for _, row in mean_df.iterrows():
            f.write(f'{int(row['Dimension'])} & {int(row['k'])} & {row['Summe_der_Radien_Heuristik']:.7f} & {row['Summe_der_Radien_KMeans']:.7f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Summe der Radien von der Heuristik und \\textit{{k}}-Means++}}\n')
        f.write(f'\\label{{tab:heuristik_vs_kemans_summe_der_radien_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')




        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabularx}{\\textwidth}{|X|X|X|X|}\n')
        f.write('\\hline\n')
        f.write('Dimension & k & Laufzeit Heuristik & Laufzeit \\textit{k}-Means++ \\\\ \\hline\n')

        for _, row in mean_df.iterrows():
            f.write(f'{int(row['Dimension'])} & {int(row['k'])} & {row['Dauer (Sekunden)_Heuristik']:.7f} & {row['Dauer (Sekunden)_KMeans']:.7f} \\\\ \\hline\n')

        f.write('\\end{tabularx}\n')
        f.write(f'\\caption{{Vergleich der Laufzeit von der Heuristik und \\textit{{k}}-Means++ (Sekunden)}}\n')
        f.write(f'\\label{{tab:heuristik_vs_kmeans_laufzeit_d{dimension}_k{k}}}\n')
        f.write('\\end{table}\n')
        f.write('\n')


def clustering(config, dimensions, ks, seeds, epsilon_values, u_values, num_radii_values, number_files, data_directory):
    for dimension in dimensions:
        for k in ks:
            # Verzeichnisse definieren
            directory = f'{data_directory}/Dimension={dimension}/k={k}'
            point_directory = f'{directory}/Points'

            os.makedirs(point_directory, exist_ok=True)

            # Überprüfen, ob die Punkte-Dateien existieren, andernfalls generieren
            if not os.path.exists(os.path.join(point_directory, f'points_{number_files - 1}.csv')):
                generator.generate_data(config, dimension, k, number_files, data_directory)

            # Liste der Punkte-Dateien im Verzeichnis
            point_files = [f for f in os.listdir(point_directory) if f.endswith('.csv')]

            for seed in seeds:
        
                # Ausführung der Algorithmen
                FPT_Heuristic(point_files, dimension, k, epsilon_values, u_values, num_radii_values, directory, point_directory, seed)
                cluster(point_files, dimension, k, directory, point_directory, 'Gonzalez', lib.gonzalez_wrapper, seed)
                cluster(point_files, dimension, k, directory, point_directory, 'KMeansPlusPlus', lib.kmeans_wrapper, seed)


            cluster(point_files, dimension, k, directory, point_directory, 'Heuristik', lib.heuristic_wrapper)



if __name__ == '__main__':
    epsilon_values = [0.5]
    u_values = [100, 1000, 3000, 5000, 10000, 30000, 50000]
    num_radii_values = [5]
    number_files = 100
    dimensions = [2]
    ks = [3, 4, 6]
    seeds = range(5)

    data_directory = 'Data_d2_k3_k4_k6'
    os.makedirs(data_directory, exist_ok=True)

    # Argumente aus der Konfiguration holen
    config = generator.handle_arguments()

    # Clustering-Algorithmen ausführen
    clustering(config, dimensions, ks, seeds, epsilon_values, u_values, num_radii_values, number_files, data_directory)

    # Analyse und Vergleich der Ergebnisse von FPT_Heuristic
    analyze_results_FPT_Heuristic(dimensions, ks, seeds, data_directory)

    compare_heuristic_with_other_algorithms(dimensions, ks, seeds, data_directory)

    for dimension in dimensions:
        for k in ks:
            # Vergleich der Ergebnisse der verschiedenen Algorithmen
            compare_algorithms(dimension, k, seeds, data_directory)
