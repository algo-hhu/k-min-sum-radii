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
        epsilon: float = 0.1,
        u: int = 1,
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

    def _fit_schmidt(
        self,
        X: Any,
        n: int,
        centers: Any,
        labels: Any,
    ) -> int:

        _DLL.schmidt_wrapper.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        # _DLL.schmidt_wrapper.restype = ctypes.POINTER(ClusterData)

        found_clusters = ctypes.c_int()

        _DLL.schmidt_wrapper(
            X,
            n,
            self.n_features_in_,
            self.n_clusters,
            self.epsilon,
            self.u,
            self.num_radii,
            ctypes.byref(found_clusters),
            labels,
            centers,
        )

        return found_clusters.value

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
        c_labels = (ctypes.c_int * n_samples)()
        c_centers = (ctypes.c_double * self.n_features_in_ * self.n_clusters)()

        real_clusters = self._fit_schmidt(c_array, n_samples, c_centers, c_labels)

        self.cluster_centers_ = np.ctypeslib.as_array(
            c_centers, shape=(self.n_clusters, self.n_features_in_)
        )
        self.real_clusters_ = real_clusters
        self.labels_ = np.ctypeslib.as_array(c_labels)

        return self
