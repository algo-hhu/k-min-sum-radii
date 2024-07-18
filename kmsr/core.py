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
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_random_state

import kmsr._core  # type: ignore

_DLL = ctypes.cdll.LoadLibrary(kmsr._core.__file__)


class KMSR(BaseEstimator, ClusterMixin, ClassNamePrefixFeaturesOutMixin):
    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "algorithm": [
            StrOptions({"auto", "schmidt", "heuristic", "gonzales", "kmeans"})
        ],
        "epsilon": [Interval(Real, 0, None, closed="left")],
        "u": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_clusters: int = 3,
        algorithm: str = "auto",
        epsilon: float = 0.5,
        u: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        self._seed = int(time()) if random_state is None else random_state
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.algorithm = "schmidt" if algorithm == "auto" else algorithm
        self.u = u
        self.random_state = check_random_state(self._seed)
        self.num_radii = 5  # TODO: what does this do?

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: Sequence[Sequence[float]],
        sample_weight: Optional[Sequence[float]] = None,
        y: Any = None,
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
            dtype=[np.float64, np.float32],
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

        c_labels = (ctypes.c_int * n_samples)()
        c_centers = (ctypes.c_double * self.n_features_in_ * self.n_clusters)()

        if self.algorithm == "schmidt":
            c_epsilon = ctypes.c_double(self.epsilon)
            c_u = ctypes.c_int(self.u)
            c_num_radii = ctypes.c_int(self.num_radii)
            _DLL.schmidt_wrapper.restype = ctypes.c_double

            self.inertia_ = _DLL.schmidt_wrapper(
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
            )
        else:
            if self.algorithm == "heuristic":
                wrapper_function = _DLL.heuristic_wrapper
            elif self.algorithm == "gonzales":
                wrapper_function = _DLL.gonzales_wrapper
            elif self.algorithm == "kmeans":
                wrapper_function = _DLL.kmeans_wrapper
            else:
                raise ValueError(f"Invalid algorithm: {self.algorithm}")
            wrapper_function.restype = ctypes.c_double

            self.inertia_ = wrapper_function(
                c_array,
                c_n,
                c_features,
                c_clusters,
                ctypes.byref(found_clusters),
                c_labels,
                c_centers,
            )

        self.real_clusters_ = found_clusters.value

        self.cluster_centers_ = np.ctypeslib.as_array(
            c_centers, shape=(self.n_clusters, self.n_features_in_)
        )
        self.labels_ = np.ctypeslib.as_array(c_labels)

        return self
