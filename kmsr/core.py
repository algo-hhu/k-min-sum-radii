import ctypes
from numbers import Integral, Real
from time import time
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    _fit_context,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_random_state

import kmsr._core  # type: ignore

_DLL = ctypes.cdll.LoadLibrary(kmsr._core.__file__)


class KMSR(BaseEstimator, ClusterMixin, ClassNamePrefixFeaturesOutMixin):
    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "algorithm": [
            StrOptions({"auto", "fpt-heuristic", "heuristic", "gonzalez", "kmeans"})
        ],
        "epsilon": [Interval(Real, 0, None, closed="left")],
        "n_u": [Interval(Integral, 1, None, closed="left")],
        "n_test_radii": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

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
        self._seed = int(time()) if random_state is None else random_state
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.algorithm = "fpt-heuristic" if algorithm == "auto" else algorithm.lower()
        self.n_u = n_u
        self.n_test_radii = n_test_radii
        self.random_state = check_random_state(self._seed)

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

    @_fit_context(prefer_skip_nested_validation=True)
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
        self._check_feature_names(X, reset=True)

        _X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64],
            order="C",
            accept_large_sparse=False,
            copy=False,
        )

        n_samples, self.n_features_in_ = _X.shape

        c_array = _X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        c_n = ctypes.c_int(n_samples)
        c_features = ctypes.c_int(self.n_features_in_)
        c_clusters = ctypes.c_int(self.n_clusters)
        found_clusters = ctypes.c_int()
        c_seed = ctypes.c_int(self._seed)

        c_labels = (ctypes.c_int * n_samples)()
        c_centers = (ctypes.c_double * self.n_features_in_ * self.n_clusters)()
        c_radii = (ctypes.c_double * self.n_clusters)()

        if self.algorithm == "fpt-heuristic":
            c_epsilon = ctypes.c_double(self.epsilon)
            c_u = ctypes.c_int(self.n_u)
            c_num_radii = ctypes.c_int(self.n_test_radii)
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
            if self.algorithm == "heuristic":
                wrapper_function = _DLL.heuristic_wrapper
            elif self.algorithm == "gonzalez":
                wrapper_function = _DLL.gonzalez_wrapper
            elif self.algorithm == "kmeans":
                wrapper_function = _DLL.kmeans_wrapper
            else:
                raise ValueError(f"Invalid algorithm: {self.algorithm}")
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

        self._cluster_centers = np.ctypeslib.as_array(
            c_centers, shape=(self.n_clusters, self.n_features_in_)
        )
        self._cluster_radii = np.ctypeslib.as_array(c_radii)

        # Crop the centers and the radii in case the algorithm found less clusters
        self._cluster_centers = self.cluster_centers_[: self._real_n_clusters]
        self._cluster_radii = self.cluster_radii_[: self._real_n_clusters]

        self._labels = np.ctypeslib.as_array(c_labels)

        return self
