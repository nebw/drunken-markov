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
        cmatrix = cmatrix.astype(dtype=float, copy=False)

    T = np.zeros_like(cmatrix, dtype=float)
    for row in range(cmatrix.shape[0]):
        T[row, :] = cmatrix[row, :] / float(sum(cmatrix[row, :]))
    return T
