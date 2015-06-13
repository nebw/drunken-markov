#!/usr/bin/python
import numpy as np


def cmatrix(clusters):
    """ Simple count matrix
    Input: clusters object from clustering algorithm
    """

    C = np.zeros((clusters.ncenters, clusters.ncenters))
    membership = clusters.get_centers()
    for i in range(len(membership) - 1):
        C[membership[i], membership[i+1]] += 1
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
