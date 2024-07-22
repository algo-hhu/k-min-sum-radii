import random
import unittest
from typing import Any

import numpy as np

from kmsr import KMSR
from kmsr.generate_points import generate_clusters


def check_cost(inertia: float, radii: np.ndarray) -> Any:
    return np.isclose(inertia, sum(radii))


EXPECTED_VALUES = {
    2: {  # Dim
        2: {  # K
            "fpt-heuristic": (45.91, 45.93),
            "heuristic": (48.25, 48.27),
            "gonzalez": (48.25, 48.27),
            "kmeans": (48.25, 48.27),
        },
        3: {
            "fpt-heuristic": (55.86, 56.17),
            "heuristic": (57.73, 57.75),
            "gonzalez": (57.73, 57.75),
            "kmeans": (57.73, 57.75),
        },
        4: {
            "fpt-heuristic": (50.53, 50.95),
            "heuristic": (52.34, 52.36),
            "gonzalez": (52.34, 52.36),
            "kmeans": (52.34, 52.36),
        },
    },
    3: {
        2: {
            "fpt-heuristic": (41.24, 42.84),
            "heuristic": (42.96, 42.98),
            "gonzalez": (42.96, 44.81),
            "kmeans": (42.96, 44.81),
        },
        3: {
            "fpt-heuristic": (51.79, 51.81),
            "heuristic": (51.79, 51.81),
            "gonzalez": (51.79, 60.65),
            "kmeans": (51.79, 73.38),
        },
        4: {
            "fpt-heuristic": (66.37, 66.39),
            "heuristic": (68.64, 68.66),
            "gonzalez": (68.64, 68.66),
            "kmeans": (68.64, 68.66),
        },
    },
}


class TestKMSR(unittest.TestCase):
    def test_fit(self) -> None:
        random.seed(42)
        s = "{"
        for k in range(2, 5):
            for dim in range(2, 4):
                _, points = generate_clusters(
                    number_centers=k,
                    max_value=100,
                    min_points_per_cluster=20,
                    max_points_per_cluster=100,
                    min_cluster_radius=0.1,
                    max_cluster_radius=50,
                    dimensions=dim,
                )
                for algo in ["fpt-heuristic", "heuristic", "gonzalez", "kmeans"]:
                    with self.subTest(msg=f"{algo}_k={k}_dim={dim}"):
                        # costs = []
                        for i in range(20):
                            kmsr = KMSR(n_clusters=3, algorithm=algo, random_state=i)
                            kmsr.fit_predict(points)
                            # costs.append(kmsr.inertia_)

                            self.assertTrue(
                                check_cost(kmsr.inertia_, kmsr.cluster_radii_)
                            )
                            assert (
                                EXPECTED_VALUES[dim][k][algo][0]
                                <= kmsr.inertia_
                                <= EXPECTED_VALUES[dim][k][algo][1]
                            ), (
                                f"{EXPECTED_VALUES[dim][k][algo][0]} "
                                f"<= {kmsr.inertia_} "
                                f"<= {EXPECTED_VALUES[dim][k][algo][1]}"
                            )

        print(s)


if __name__ == "__main__":
    unittest.main()
