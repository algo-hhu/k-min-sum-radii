import ctypes
from time import time
from typing import Any, Optional, Sequence, Type

import numpy as np
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import validate_data

import kmsr._core  # type: ignore

_DLL = ctypes.cdll.LoadLibrary(kmsr._core.__file__)


class KMSR(BaseEstimator, ClusterMixin, ClassNamePrefixFeaturesOutMixin):
    NOT_FITTED_ERROR = (
        "This KMSR instance is not fitted yet. "
        "Call 'fit' with appropriate arguments before using this estimator."
    )

    def __init__(
        self,
        n_clusters: int = 3,
        algorithm: str = "auto",
        epsilon: float = 0.5,
        n_u: int = 1000,
        n_test_radii: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.algorithm = algorithm
        self.n_u = n_u
        self.n_test_radii = n_test_radii

    @property
    def inertia_(self) -> float:
        if not hasattr(self, "_inertia"):
            raise NotFittedError(self.NOT_FITTED_ERROR)
        return self._inertia

    @property
    def labels_(self) -> np.ndarray:
        if not hasattr(self, "_labels"):
            raise NotFittedError(self.NOT_FITTED_ERROR)
        return self._labels

    @property
    def cluster_centers_(self) -> np.ndarray:
        if not hasattr(self, "_cluster_centers"):
            raise NotFittedError(self.NOT_FITTED_ERROR)
        return self._cluster_centers

    @property
    def cluster_radii_(self) -> np.ndarray:
        if not hasattr(self, "_cluster_radii"):
            raise NotFittedError(self.NOT_FITTED_ERROR)
        return self._cluster_radii

    @property
    def real_n_clusters_(self) -> int:
        if not hasattr(self, "_real_n_clusters"):
            raise NotFittedError(self.NOT_FITTED_ERROR)
        return self._real_n_clusters

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Any = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "KMSR":
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not supported")

        return self._fit(X)

    def _fit(
        self,
        X: Sequence[Sequence[float]],
    ) -> "KMSR":
        self._validate_params()

        _algorithm = (
            "fpt-heuristic" if self.algorithm == "auto" else self.algorithm.lower()
        )
        _seed = int(time()) if self.random_state is None else self.random_state
        _X = validate_data(
            self,
            X,
            accept_sparse=False,
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
            copy=False,
            reset=True,
        )

        n_samples = _X.shape[0]

        _X = np.ascontiguousarray(_X, dtype=np.float64)
        c_array = _X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        c_n = ctypes.c_int(n_samples)
        c_features = ctypes.c_int(self.n_features_in_)
        c_clusters = ctypes.c_int(self.n_clusters)
        found_clusters = ctypes.c_int()
        c_seed = ctypes.c_int(_seed)

        labels = np.empty(n_samples, dtype=np.int32, order="C")
        centers = np.empty(
            (self.n_clusters, self.n_features_in_), dtype=np.float64, order="C"
        )
        radii = np.empty(self.n_clusters, dtype=np.float64, order="C")

        c_labels = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        c_centers = centers.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_radii = radii.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Define common argtypes for the wrappers
        common_argtypes: list[Type[ctypes._CData]] = [
            ctypes.POINTER(ctypes.c_double),  # X
            ctypes.c_int,  # n_samples
            ctypes.c_int,  # n_features
            ctypes.c_int,  # n_clusters
            ctypes.POINTER(ctypes.c_int),  # found_clusters
            ctypes.POINTER(ctypes.c_int),  # labels
            ctypes.POINTER(ctypes.c_double),  # centers
            ctypes.POINTER(ctypes.c_double),  # radii
            ctypes.c_int,  # seed
        ]

        if _algorithm == "fpt-heuristic":
            c_epsilon = ctypes.c_double(self.epsilon)
            c_u = ctypes.c_int(self.n_u)
            c_num_radii = ctypes.c_int(self.n_test_radii)

            schmidt_argtypes = (
                common_argtypes[:4]
                + [ctypes.c_double, ctypes.c_int, ctypes.c_int]
                + common_argtypes[4:]
            )

            _DLL.schmidt_wrapper.argtypes = schmidt_argtypes  # type: ignore[assignment]

            _DLL.schmidt_wrapper.restype = ctypes.c_double

            self._inertia: float = _DLL.schmidt_wrapper(
                c_array,
                c_n,
                c_features,
                c_clusters,
                c_epsilon,
                c_u,
                c_num_radii,
                ctypes.byref(found_clusters),
                c_labels,
                c_centers,
                c_radii,
                c_seed,
            )
        else:
            if _algorithm == "heuristic":
                wrapper_function = _DLL.heuristic_wrapper
            elif _algorithm == "gonzalez":
                wrapper_function = _DLL.gonzalez_wrapper
            elif _algorithm == "kmeans":
                wrapper_function = _DLL.kmeans_wrapper
            else:
                raise ValueError(f"Invalid algorithm: {self.algorithm}")

            wrapper_function.argtypes = common_argtypes  # type: ignore[assignment]
            wrapper_function.restype = ctypes.c_double

            self._inertia = wrapper_function(
                c_array,
                c_n,
                c_features,
                c_clusters,
                ctypes.byref(found_clusters),
                c_labels,
                c_centers,
                c_radii,
                c_seed,
            )

        self._real_n_clusters = found_clusters.value

        self._cluster_centers = centers
        self._cluster_radii = radii

        # Crop the centers and the radii in case the algorithm found less clusters
        self._cluster_centers = self.cluster_centers_[: self._real_n_clusters]
        self._cluster_radii = self.cluster_radii_[: self._real_n_clusters]

        self._labels = labels

        return self

    def _validate_params(self) -> None:
        if not isinstance(self.n_clusters, (int, np.integer)):
            raise TypeError(
                f"n_clusters must be an integer, got {type(self.n_clusters).__name__}"
            )
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}")

        valid_algos = {"auto", "fpt-heuristic", "heuristic", "gonzalez", "kmeans"}
        if self.algorithm.lower() not in valid_algos:
            raise ValueError(
                f"algorithm must be one of {valid_algos}; got {self.algorithm}"
            )

        if not isinstance(self.epsilon, (float, np.floating)):
            raise TypeError(
                f"epsilon must be a float, got {type(self.epsilon).__name__}"
            )
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be >= 0; got {self.epsilon}")

        if not isinstance(self.n_u, (int, np.integer)):
            raise TypeError(f"n_u must be an integer," f"got {type(self.n_u).__name__}")
        if self.n_u < 1:
            raise ValueError(f"n_u must be >= 1, got {self.n_u}")
        if not isinstance(self.n_test_radii, (int, np.integer)):
            raise TypeError(
                f"n_test_radii must be an integer,"
                f"got {type(self.n_test_radii).__name__}"
            )
        if self.n_test_radii < 1:
            raise ValueError(f"n_test_radii must be >= 1, got {self.n_test_radii}")

        if self.random_state is not None:
            if not isinstance(self.random_state, (int, np.integer)):
                raise TypeError(
                    f"random_state must be None or an integer,"
                    f"got {type(self.random_state).__name__}"
                )
            if self.random_state < 0:
                raise ValueError(f"random_state must be >= 0, got {self.random_state}")
