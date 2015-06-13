#!/usr/bin/python
import numpy as np


def cmatrix(disc_traj, tau=1):
    """ Simple count matrix
    Input: discrete trajectory
    """

    n_centers = int(disc_traj.max()) + 1
    C = np.zeros((n_centers, n_centers))
    for i in range(0, len(disc_traj) - 1, tau):
        C[disc_traj[i], disc_traj[i+1]] += 1
    return C


def tmatrix(cmatrix):
    """ simple non-reversible transition matrix estimator """

    # check if cmatrix is integer valued (will otherwise return zeros)
    if issubclass(cmatrix.dtype.type, int):
        _cmatrix = np.array(cmatrix, dtype=float)
    else:
        _cmatrix = np.copy(cmatrix)

    T = np.zeros_like(_cmatrix)
    for row in range(len(_cmatrix[0, :])):
        T[row, :] = _cmatrix[row, :] / sum(_cmatrix[row, :])
    return T
