from typing import Union, get_args

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

pca_type = Union[PCA, IncrementalPCA]

plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams.update({'font.size': 18})


def check_is_pca(pca_model: pca_type) -> None:
    if not isinstance(pca_model, get_args(pca_type)):
        raise TypeError("model must be an instance of sklearn PCA or sklearn IncrementalPCA")
    if not check_is_fitted(pca_model, ["singular_values_", "mean_"]):
        raise NotFittedError("model has not been fitted")


def plot_singular_values_vs_rank(pca_model: pca_type) -> None:
    check_is_pca(pca_model)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0, 0].semilogy(pca_model.singular_values_, '-o', color='k')
    axs[0, 0].xlabel("r")
    axs[0, 0].ylabel("Singular Values")
    axs[0, 1].plot(pca_model.explained_variance_ratio_, '-o', color='k')
    axs[0, 1].xlabel("r")
    axs[0, 1].ylabel("Explained Variances Ratios")

    plt.show()

