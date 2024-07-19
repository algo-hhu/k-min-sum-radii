[![Build Status](https://github.com/algo-hhu/k-min-sum-radii/actions/workflows/mypy-flake-test.yml/badge.svg)](https://github.com/algo-hhu/k-min-sum-radii/actions)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Supported Python version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Stable Version](https://img.shields.io/pypi/v/kmsr?label=stable)](https://pypi.org/project/kmsr/)

# K-Min-Sum-Radii


TODO

## Installation

```bash
pip install kmsr
```

## Example

```python
from kmsr import KMSR
from kmsr.plot import plot_multiple_results
from sklearn.datasets import make_blobs
from time import time

points, ground_truth = make_blobs(
    n_samples=100,
    n_features=2,
    centers=2,
    cluster_std=0.6,
    shuffle=True,
    random_state=42,
)

labels = []
centers = []
radii = []
titles = []
for algo in ["Schmidt", "Heuristic", "Gonzales", "KMeans"]:
    kmsr = KMSR(
        n_clusters=5,
        algorithm=algo,
        epsilon=0.5,
        n_u=10000,
        n_test_radii=10,
        random_state=42,
    )
    start = time()
    kmsr.fit(points)
    end = time() - start
    labels.append(kmsr.labels_)
    centers.append(kmsr.cluster_centers_)
    radii.append(kmsr.cluster_radii_)
    titles.append(f"{algo}: {sum(kmsr.cluster_radii_):.3f}, Time: {end:.3f}s")

plot_multiple_results(
    points,
    clusters=labels,
    centers=centers,
    radii=radii,
    title=titles,
)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/algo-hhu/k-min-sum-radii/main/images/example.png" alt="Comparison of the Different Methods for kMSR"/>
</p>

## Development

Install [poetry](https://python-poetry.org/docs/#installation)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install clang
```bash
sudo apt-get install clang
```

Set clang variables
```bash
export CXX=/usr/bin/clang++
export CC=/usr/bin/clang
```

Install the package
```bash
poetry install
```

If the installation does not work and you do not see the C++ output, you can build the package to see the stack trace
```bash
poetry build
```

Run the tests
```bash
poetry run python -m unittest discover tests -v
```

## Citation

If you use this cose, please cite the following bachelor thesis:

```
N. Lenßen, "Experimentelle Analyse von Min-Sum-Radii Approximationsalgorithmen". Bachelorarbeit, Heinrich-Heine-Universität Düsseldorf, 2024.
```

Moreover, depending on the selection of the `algorithm` parameter, you should also cite the [following paper](https://doi.org/10.1007/978-3-031-49815-2_9) for `algorithm='schmidt'`:

```
L. Drexler, A. Hennes, A. Lahiri, M. Schmidt, and J. Wargalla, "Approximating Fair K-Min-Sum-Radii in Euclidean Space," in Lecture notes in computer science, 2023, pp. 119–133. doi: 10.1007/978-3-031-49815-2_9.
```

the [following paper](https://doi.org/10.1016/0304-3975(85)90224-5) for `algorithm='gonzales'`:

```
T. F. Gonzalez, "Clustering to minimize the maximum intercluster distance," Theoretical Computer Science, vol. 38, pp. 293–306, Jan. 1985, doi: 10.1016/0304-3975(85)90224-5.
```

and the [following paper](https://doi.org/10.5555/1283383.1283494) for `algorithm='kmeans'`:

```
D. Arthur and S. Vassilvitskii, "k-means++: the advantages of careful seeding," Symposium on Discrete Algorithms, pp. 1027–1035, Jan. 2007, doi: 10.5555/1283383.1283494.
```
