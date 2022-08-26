from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import randomized_svd


class MCA(TransformerMixin, BaseEstimator):
    def __init__(self,
                 n_components: int = None,
                 n_oversamples: int = 10,
                 iterated_power: Union[int, str] = "auto",
                 power_iteration_normalizer: str = "auto",
                 n_iter: [int, str] = "auto",
                 copy: bool = True,
                 random_state=None
                 ):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.iterated_power = iterated_power
        self.power_iteration_normalizer = power_iteration_normalizer
        self.n_iter = n_iter
        self.copy = copy
        self.random_state = random_state

    def fit(self, X, y=None):

        self._fit(X)
        return self

    def _fit(self, X):
        self._validate_data(
            X,
            dtype=[np.int32, np.int64],
            ensure_2d=True,
            copy=self.copy
        )

        self.N_ = X.sum()
        self.Z_ = X/self.N_
        r = X.sum(axis=0)
        c = X.sum(axis=1)
        self.Dr_ = np.diag(r)
        self.Dc_ = np.diag(c)
        self.rcT_ = r @ c.T
        M = np.inv(np.sqrt(self.Dr_)) @ (self.Z_ - self.rcT_) @ np.inv(np.sqrt(self.Dc_))
        P, D, QT = randomized_svd(
            M,
            n_components=self.n_components,
            n_oversamples=self.n_oversamples,
            n_iter=self.iterated_power,
            power_iteration_normalizer=self.power_iteration_normalizer,
            flip_sign=True,
            random_state=self.random_state
        )

        self.n_components_ = self.n_components

        self.P_ = P[:, : self.n_components_]
        self.Delta_ = D[: self.n_components_]
        self.QT_ = QT[self.n_components_, :]

        self.Z_eig_ = self.Delta_**2
        self.factor_coordinates_ = np.inv(np.sqrt(self.Dr_)) @ self.P_ @ self.Delta_
        self.variable_coordinates_ = np.inv(np.sqrt(self.Dc_)) @ self.QT_.T @ self.Delta_

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, dtype=[np.int64, np.int32], reset=False)
        X_transformed = np.dot(X, self.QT_.T)
        return X_transformed

    def fit_transform(self, X, y=None):
        self._fit(X)
        return self.P_ * self.Delta_
