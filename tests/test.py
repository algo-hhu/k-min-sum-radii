import random
import unittest
from typing import Any

import numpy as np

from kmsr import KMSR
from kmsr.generate_points import generate_clusters


def check_cost(inertia: float, radii: np.ndarray) -> Any:
    return np.isclose(inertia, sum(radii))


class TestKMSR(unittest.TestCase):
    def test_fit(self) -> None:
        random.seed(42)
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
                for algo in ["schmidt", "heuristic", "gonzales", "kmeans"]:
                    with self.subTest(msg=f"{algo}_k={k}_dim={dim}"):
                        # costs = []
                        for i in range(20):
                            kmsr = KMSR(n_clusters=3, algorithm=algo, random_state=i)
                            kmsr.fit_predict(points)
                            # costs.append(kmsr.inertia_)

                            self.assertTrue(
                                check_cost(kmsr.inertia_, kmsr.cluster_radii_)
                            )
                        # print(algo, sum(costs) / len(costs))


if __name__ == "__main__":
    unittest.main()
