#!/usr/bin/python

import numpy as np
from unittest import TestCase
import drunkenmarkov
from drunkenmarkov.Estimation import tmatrix

class TMatrixTest(TestCase):
    def test_TMatrix_rowsums(self):
        """
        Tests if the row sums of a random 10x10 transition matrix yield one.
        """

        C = np.random.randint(1000, size=(10,10))
        T = tmatrix(C)
        T_rowsums = T.sum(axis=1)

        self.assertTrue(np.allclose(T_rowsums, np.ones_like(T_rowsums), rtol=1.e-5))

    def test_TMatrix_simple(self):
        """
        Compare 2x2 transition matrix by its analytical result.
        """
        C = np.array([[9, 1], [7, 3]])
        T_computed = tmatrix(C)
        T_analytical = np.array([[.9, .1], [.7, .3]])

        self.assertTrue(np.allclose(T_computed, T_analytical))