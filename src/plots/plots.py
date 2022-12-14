from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from prince import MCA

from src.optimal_SVHT.optimal_svht import compute_gavish_donoho_threshold
from src.decomposition.RPCA import RPCA

datamatrix = Union[np.ndarray, pd.DataFrame]

plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams.update({'font.size': 18})


def plot_singular_values_vs_rank(x: datamatrix, method: str = "standard", max_iters: int = 1000, svt_tol: float = 10**(-7))\
        -> int:
    scaler = StandardScaler()
    if method == "standard":
        pca = PCA()
    elif method == "rpca":
        pca = RPCA(max_iters=max_iters, stop_on_failure_to_converge=True, svt_tol=svt_tol)
    elif method == "mca":
        pca = MCA(n_components=x.shape[1])
    else:
        raise ValueError("'{}' is not an implemented method.".format(method))

    if x.shape[0] == x.shape[1]:
        aspect_ratio = 1.0
    elif x.shape[0] > x.shape[1]:
        aspect_ratio = float(x.shape[1]/x.shape[0])
    else:
        aspect_ratio = float(x.shape[0]/x.shape[1])

    thresh = compute_gavish_donoho_threshold(beta=aspect_ratio)
    x_scaled = scaler.fit_transform(x) if method != "mca" else x
    pca.fit(x_scaled)

    if method == "mca":
        s = pca.s_
    else:
        s = pca.singular_values_
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].semilogy(s, '-o', color='k')
    axs[0].set_xlabel("r")
    axs[0].set_ylabel("Singular Values")
    axs[0].axhline(y=thresh)
    axs[1].plot(np.cumsum(s)/np.sum(s), '-o', color='k')
    axs[1].set_xlabel("r")
    axs[1].set_ylabel("Cumulative Explained Variance")
    r = np.max(np.where(s > thresh))
    axs[1].axvline(x=r)

    fig.suptitle("Singular Value Summaries by Rank of Matrix Approximations")
    fig.subplots_adjust(wspace=0.4)
    plt.show()

    return r
