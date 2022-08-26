import numbers
import warnings


import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import _init_arpack_v0
from sklearn.utils import check_scalar, check_random_state
from sklearn.utils.extmath import svd_flip, randomized_svd
from tqdm import tqdm


class RPCA(PCA):
    """Robust principal component analysis (RPCA)

    Linear dimensionality reduction that decomposes a data matrix X into a low-rank matrix L
    and a sparse matrix S containing outliers. In short, X = L + S.

    The optimization objective is to find L and S
    that minimizes the rank of L plus the 1-norm of aS subject to X = L + S,
    where a is one divided-by the square-root of max(n,m), where n and m are the dimensions
    of X. """

    def __init__(self,
                 n_components: int = None,
                 *,
                 copy: bool = True,
                 svd_solver: str = "auto",
                 svt_svd_solver: str = "auto",
                 tol: float = 0.0,
                 svt_tol: float = 10**(-7),
                 iterated_power: str = "auto",
                 n_oversamples: int = 10,
                 power_iteration_normalizer: str = "auto",
                 random_state=None,
                 max_iters: int = 1000,
                 stop_on_failure_to_converge: bool = False,
                 add_sparse: bool = False
                 ):
        self.n_components = n_components
        self.copy = copy
        self.svd_solver = svd_solver
        self.svt_svd_solver = svt_svd_solver
        self.tol = tol
        self.svt_tol = svt_tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        self.max_iters = max_iters
        self.stop_on_failure_to_converge = stop_on_failure_to_converge
        self.add_sparse = add_sparse

    def fit(self, X, y=None):

        check_scalar(
            self.n_oversamples,
            "n_oversamples",
            min_val=1,
            target_type=numbers.Integral
        )

        self._fit(X)
        return self

    def _fit(self, X):
        MU, LAMBDA, THRESHOLD = self._compute_rpca_parameters(X, self.svt_tol)

        X = self._validate_data(
            X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy
        )

        # Handle n_components==None
        if self.n_components is None:
            if self.svt_svd_solver != "arpack":
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        self._fit_svt_svd_solver = self.svt_svd_solver
        if self._fit_svt_svd_solver == "auto":
            # Small problem or n_components == 'mle', just call full PCA
            if max(X.shape) <= 500 or n_components == "mle":
                self._fit_svt_svd_solver = "full"
            elif 1 <= n_components < 0.8 * min(X.shape):
                self._fit_svt_svd_solver = "randomized"
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svt_svd_solver = "full"

        L, Sp = self._alternating_directions_solver(X, MU, LAMBDA, THRESHOLD, n_components)
        self.L_ = L[:, : self.n_components_]
        self.Sp_ = Sp[: self.n_components_]

        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == "auto":
            # Small problem or n_components == 'mle', just call full PCA
            if max(X.shape) <= 500 or n_components == "mle":
                self._fit_svd_solver = "full"
            elif 1 <= n_components < 0.8 * min(X.shape):
                self._fit_svd_solver = "randomized"
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svd_solver = "full"

        if self._fit_svd_solver == "full":
            return super()._fit_full(self.L_, n_components)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            return super()._fit_truncated(self.L_, n_components, self._fit_svd_solver)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'".format(self._fit_svd_solver)
            )

    def _alternating_directions_solver(self, X, mu, lambd, threshold, n_components):
        S = np.zeros_like(X)
        Y = np.zeros_like(X)
        L = np.zeros_like(X)

        iteration = 0
        if self._fit_svt_svd_solver == "full":
            for i in tqdm(range(self.max_iters)):
                if np.linalg.norm(X-L-S) <= threshold:
                    break
                L = self._svt_fit_full(X-S+(1/mu)*Y, 1/mu)
                S = self._shrink(X-L+(1/mu)*Y, lambd/mu)
                Y = Y + mu*(X-L-S)
                iteration += 1
        elif self._fit_svt_svd_solver in ["arpack", "randomized"]:
            for i in tqdm(range(self.max_iters)):
                if np.linalg.norm(X-L-S) <= threshold:
                    break
                L = self._svt_fit_truncated(X-S+(1/mu)*Y, 1/mu, n_components, self._fit_svt_svd_solver)
                S = self._shrink(X-L+(1/mu)*Y, lambd/mu)
                Y = Y + mu*(X-L-S)
                iteration += 1
        else:
            raise ValueError(
                "Unrecognized svt_svd_solver='{0}'".format(self._fit_svt_svd_solver)
            )

        if np.linalg.norm(X-L-S) > threshold:
            warnings.warn("Alternating Direction Solver failed to converge. X-L-S='{0}'".format(np.linalg.norm(X-L-S)))
            if self.stop_on_failure_to_converge:
                raise InterruptedError  # TODO: Replace with exception for failure to converge
        else:
            print("SUCCESS: Convergence at Iteration '{0}'!".format(iteration))

        return L, S

    def _svt_fit_full(self, X, tau):
        x_mean = np.mean(X, axis=0)

        U, S, VT = linalg.svd(X - x_mean, full_matrices=False)
        U, VT = svd_flip(U, VT)

        return U @ np.diag(self._shrink(S, tau)) @ VT

    def _svt_fit_truncated(self, X, tau, n_components, svd_solver):

        n_samples, n_features = X.shape

        if isinstance(n_components, str):
            raise ValueError(
                "n_components=%r cannot be a string with svt_svd_solver='%s'"
                % (n_components, svd_solver)
            )
        elif not 1 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                "n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svt_svd_solver='%s'"
                % (n_components, min(n_samples, n_features), svd_solver)
            )
        elif not isinstance(n_components, numbers.Integral):
            raise ValueError(
                "n_components=%r must be of type int "
                "when greater than or equal to 1, was of type=%r"
                % (n_components, type(n_components))
            )
        elif svd_solver == "arpack" and n_components == min(n_samples, n_features):
            raise ValueError(
                "n_components=%r must be strictly less than "
                "min(n_samples, n_features)=%r with "
                "svt_svd_solver='%s'"
                % (n_components, min(n_samples, n_features), svd_solver)
            )

        random_state = check_random_state(self.random_state)
        x_mean = np.mean(X, axis=0)

        if svd_solver == "arpack":
            v0 = _init_arpack_v0(min(X.shape), random_state)
            U, S, VT = svds(X - x_mean, k=n_components, tol=self.svt_tol, v0=v0)
            S = S[::-1]
            U, VT = svd_flip(U[:, ::-1], VT[::-1])

        elif svd_solver == "randomized":
            U, S, VT = randomized_svd(
                X - x_mean,
                n_components=n_components,
                n_oversamples=self.n_oversamples,
                n_iter=self.iterated_power,
                power_iteration_normalizer=self.power_iteration_normalizer,
                flip_sign=True,
                random_state=random_state
            )

        return U @ np.diag(self._shrink(S, tau)) @ VT

    @staticmethod
    def _compute_rpca_parameters(X, svt_tol):
        mu = X.shape[0]*X.shape[1]/(4*np.sum(np.abs(X.reshape(-1))))
        lambd = 1/np.sqrt(max(X.shape))
        thresh = svt_tol * np.linalg.norm(X)
        return mu, lambd, thresh

    @staticmethod
    def _shrink(X, tau):
        Y = np.abs(X) - tau
        return np.sign(X) * np.maximum(Y, np.zeros_like(Y))

    def transform(self, X):
        X_transformed = np.dot(X, self.components_.T)
        if self.add_sparse:
            X_transformed += self.Sp_
        return X_transformed

    def fit_transform(self, X, y=None):
        LU, LS, LVT = self._fit(X)
        LU = LU[:, : self.n_components_]
        LU *= LS[: self.n_components_]

        return LU
