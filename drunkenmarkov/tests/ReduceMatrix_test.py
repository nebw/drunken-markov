#!/usr/bin/python

import numpy as np

from unittest import TestCase

from drunkenmarkov.Analysis import MarkovStateModel


class ReduceMatrixTests(TestCase):
    def test_ReduceMatrixRowsum(self):
        """
        Tests if the rowsum of a reduced matrix is one. Uses 3x3 test matrix.
        """

        T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])

        MSM = MarkovStateModel(T)
        T_reduced = MSM.reduce_matrix(2)[0]

        self.assertTrue(np.allclose(T_reduced.sum(axis=1), 1., rtol=1.e-5))
