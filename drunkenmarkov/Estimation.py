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
	""" simple transition matrix estimator """

    T = np.zeros_like(cmatrix)
    for row in range(len(cmatrix[0, :])):
        T[row, :] = cmatrix[row, :] / sum(cmatrix[row, :])
    return T

