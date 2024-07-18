import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Function for generating random centers based on the configuration
def generate_centers(
    number_centers: int, max_value: float, dimension: int
) -> List[List[float]]:
    centers = []
    for _ in range(number_centers):
        # Create a center with random coordinates in each dimension
        center = [random.uniform(0, max_value) for _ in range(dimension)]
        centers.append(center)

    return centers


# Function for generating points around a given center based on the configuration
def generate_points_around_center(
    center: List[float],
    min_points_per_cluster: int,
    max_points_per_cluster: int,
    min_cluster_radius: float,
    max_cluster_radius: float,
    max_value: float,
    dimension: int,
) -> List[List[float]]:
    # Determine the number of points in the cluster and the cluster radius
    number_points = random.randint(min_points_per_cluster, max_points_per_cluster)
    cluster_radius = random.uniform(min_cluster_radius, max_cluster_radius)
    points = []
    count = 0
    while count < number_points:
        # Generate a random radius and random angle for each dimension
        radius = random.uniform(0, cluster_radius)
        angles = [random.uniform(0, 2 * math.pi) for _ in range(dimension)]
        point = []
        valid_point = True
        for i in range(dimension):
            # Calculate the coordinate using the radius and angle
            coord = radius * math.cos(angles[i]) + center[i]
            # Check if the coordinate is within the valid limits
            if coord < 0 or coord > max_value:
                valid_point = False
                break
            point.append(coord)
        if valid_point:
            # Add the point to the list if it is valid
            points.append(point)
            count += 1

    return points


# Function for generating clusters based on the configuration
def generate_clusters(
    number_centers: int,
    max_value: float,
    min_points_per_cluster: int,
    max_points_per_cluster: int,
    min_cluster_radius: float,
    max_cluster_radius: float,
    dimension: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    # Generate the centers
    centers = generate_centers(number_centers, max_value, dimension)
    points: List[List[float]] = []
    for center in centers:
        # Generate points around each center
        points = points + generate_points_around_center(
            center=center,
            min_points_per_cluster=min_points_per_cluster,
            max_points_per_cluster=max_points_per_cluster,
            min_cluster_radius=min_cluster_radius,
            max_cluster_radius=max_cluster_radius,
            max_value=max_value,
            dimension=dimension,
        )

    return centers, points


# Function for writing the generated points to a CSV file
def write_clusters_to_file(output_path: Path, points: List[List[float]]) -> None:
    with output_path.open("r") as file:
        for point in points:
            # Use join to concatenate the coordinates with commas
            line = ",".join(map(str, point))
            file.write(line + "\n")


# Function for handling command line arguments
def handle_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-mv",
        "--max-value",
        type=float,
        help="maximum value for any dimension",
        default=10,
    )
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        help="number of dimensions for the points and centers",
        default=2,
    )
    parser.add_argument(
        "-n",
        "--number-centers",
        type=int,
        help="number of centers/clusters that will be generated",
        default=3,
    )
    parser.add_argument(
        "-minp",
        "--min-points-per-cluster",
        type=int,
        help="minimum number of points per cluster (center excluded)",
        default=25,
    )
    parser.add_argument(
        "-maxp",
        "--max-points-per-cluster",
        type=int,
        help="maximum number of points per cluster (center excluded)",
        default=25,
    )
    parser.add_argument(
        "-minr",
        "--min-cluster-radius",
        type=float,
        help="minimum radius of generated clusters",
        default=0.5,
    )
    parser.add_argument(
        "-maxr",
        "--max-cluster-radius",
        type=float,
        help="minimum radius of generated clusters",
        default=2,
    )
    parser.add_argument("-k", "--k", type=int, help="number of clusters", default=3)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file for the generated data",
    )
    args = parser.parse_args()

    # Validations for the arguments
    assert args.dimensions > 0
    assert args.max_value > 0
    assert args.number_centers > 0
    assert args.min_points_per_cluster > 0
    assert args.min_points_per_cluster <= args.max_points_per_cluster
    assert args.min_cluster_radius > 0
    assert args.min_cluster_radius <= args.max_cluster_radius

    return vars(args)


# Main function for generating the data
def generate_data(config: Dict[str, Any]) -> None:
    random.seed(1234)
    # Generate the clusters
    centers, points = generate_clusters(
        number_centers=config["number_centers"],
        max_value=config["max_value"],
        min_points_per_cluster=config["min_points_per_cluster"],
        max_points_per_cluster=config["max_points_per_cluster"],
        min_cluster_radius=config["min_cluster_radius"],
        max_cluster_radius=config["max_cluster_radius"],
        dimension=config["dimensions"],
    )
    # Write the generated data to a file
    write_clusters_to_file(
        config["output_path"].parent / (config["output_path"].name + ".data"), points
    )
    write_clusters_to_file(
        config["output_path"].parent / (config["output_path"].name + ".centers"),
        centers,
    )


def main() -> None:
    config = handle_arguments()
    generate_data(config)


if __name__ == "__main__":
    main()
