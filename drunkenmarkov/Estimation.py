#!/usr/bin/python

import numpy as np
import sys


def cmatrix(disc_traj, tau=1, sliding_window=False):
    """ Simple count matrix
    Input: discrete trajectory
    """

    n_centers = int(disc_traj.max()) + 1
    C = np.zeros((n_centers, n_centers))

    # Without sliding window: evaluate every tau steps and count transition
    # from state i to state i + tau.
    # With sliding window: evaluate every step and count transition from state
    # i to state i + tau.
    for i in range(0, len(disc_traj) - tau, sliding_window * (1 - tau) + tau):
        C[disc_traj[i], disc_traj[i + tau]] += 1
    return C


def estimate_nonreversible(C):
    """ simple non-reversible transition matrix estimator """

    # check if cmatrix is integer valued (will otherwise return zeros)
    if issubclass(C.dtype.type, int):
        C = C.astype(dtype=float, copy=False)

    T = np.zeros_like(C, dtype=float)
    for row in range(C.shape[0]):
        T[row, :] = C[row, :] / sum(C[row, :])
    return T


def estimate_reversible(C):
    """Maximum likelihood estimator of reversible transition matrix"""

    assert(C.shape[0] == C.shape[1])

    # use non-reversible transition matrix as initial guess
    T = estimate_nonreversible(C)

    # iterate until convergence
    max_delta = sys.float_info.max
    while max_delta > 1e-10:
        max_delta = 0.
        T_old = np.copy(T)
        for i in range(C.shape[0]):
            # c_i / x_i(k)
            ci_over_ti = sum(C[i, :]) / sum(T_old[i, :])
            for j in range(C.shape[1]):
                # c_j / x_j(k)
                cj_over_tj = sum(C[j, :]) / sum(T_old[j, :])
                T[i, j] = (C[i, j] + C[j, i]) / (ci_over_ti + cj_over_tj)
                delta = T[i, j] - T_old[i, j]
                # use only the maximum delta in the current iteration as
                # convergence critera
                if (abs(delta) > max_delta):
                    max_delta = abs(delta)
    # normalize
    for i in range(C.shape[0]):
        T[i, :] /= sum(T[i, :])

    return T
