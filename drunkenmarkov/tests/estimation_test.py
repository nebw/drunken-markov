#!/usr/bin/python

import numpy as np
from unittest import TestCase
from drunkenmarkov.Estimation import estimate_nonreversible, \
    estimate_reversible, cmatrix


class EstimatorTest(TestCase):
    def test_nonreversible_estimator_rowsums(self):
        """
        Tests if the row sums of a random 10x10 transition matrix yield one.
        """

        C = np.random.randint(1000, size=(10, 10))
        T = estimate_nonreversible(C)
        T_rowsums = T.sum(axis=1)

        self.assertTrue(np.allclose(T_rowsums, np.ones_like(T_rowsums), rtol=1.e-5))

    def test_nonreversible_estimator_simple(self):
        """
        Compare 2x2 transition matrix by its analytical result.
        """
        C = np.array([[9, 1], [7, 3]])
        T_computed = estimate_nonreversible(C)
        T_analytical = np.array([[.9, .1], [.7, .3]])

        self.assertTrue(np.allclose(T_computed, T_analytical))

    def test_reversible_estimator_rowsums(self):
        """
        Tests if the row sums of a random 10x10 transition matrix yield one.
        """

        C = np.random.randint(1000, size=(10, 10))
        T = estimate_reversible(C)
        T_rowsums = T.sum(axis=1)

        self.assertTrue(np.allclose(T_rowsums, np.ones_like(T_rowsums), rtol=1.e-5))


class CMatrixTest(TestCase):
    def test_cmatrix_simple(self):
        """
        Simple count matrix test.
        """

        test_dtraj = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        cmatrix_compare = np.array([[2., 2.], [1., 4.]])
        cmatrix_computed = cmatrix(test_dtraj)
        self.assertTrue(np.allclose(cmatrix_compare, cmatrix_computed))
