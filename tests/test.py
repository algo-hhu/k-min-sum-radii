import random
import unittest

from kmsr import KMSR, generate_clusters


class TestFLSPP(unittest.TestCase):
    def test_fit(self) -> None:
        random.seed(42)
        _, points = generate_clusters(
            number_centers=3,
            max_value=100,
            min_points_per_cluster=25,
            max_points_per_cluster=100,
            min_cluster_radius=0.5,
            max_cluster_radius=2,
            dimensions=2,
        )
        for algo in ["schmidt", "heuristic", "gonzales", "kmeans"]:
            with self.subTest(msg=f"{algo}"):
                costs = []
                for i in range(50):
                    kmsr = KMSR(n_clusters=3, algorithm=algo, random_state=i)
                    kmsr.fit(points)
                    costs.append(kmsr.inertia_)
                # print(algo, sum(costs) / len(costs))


if __name__ == "__main__":
    unittest.main()
