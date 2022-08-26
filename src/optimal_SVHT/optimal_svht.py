import os

from oct2py import octave


def compute_gavish_donoho_threshold(beta: float, sigma_known: bool = False) -> float:
    known = 1 if sigma_known else 0
    octave.addpath(os.path.join("src", "optimal_SVHT"))
    return octave.optimal_SVHT_coef(beta, known)
