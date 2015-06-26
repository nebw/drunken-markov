#!/usr/bin/python

import numpy as np
from unittest import TestCase
from drunkenmarkov.Analysis import MarkovStateModel
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

    def test_reversible_estimator_detailed_balance(self):
        """
        Test whether the estimated transition matrix fulfills the detailed
        balance criteritum.
        """
        C = np.random.randint(1000, size=(10, 10))
        T = estimate_reversible(C)
        MSM = MarkovStateModel(T)
        P = MSM.stationary_distribution

        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                self.assertTrue(np.allclose(P[i] * T[i, j], P[j] * T[j, i]))


class CMatrixTest(TestCase):
    def test_cmatrix_simple(self):
        """
        Simple count matrix test.
        """

        test_dtraj = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        cmatrix_compare = np.array([[2., 2.], [1., 4.]])
        cmatrix_computed = cmatrix(test_dtraj)
        self.assertTrue(np.allclose(cmatrix_compare, cmatrix_computed))

    def test_cmatrix_lagtime(self):
        """
        Tests if using a lagtime reproduces the analytical result
        of a simple example trajectory.
        """
        test_dtraj = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        cmatrix_compare = np.array([[0., 2.], [1., 1.]])
        cmatrix_computed = cmatrix(test_dtraj, tau=2, sliding_window=False)

        self.assertTrue(np.allclose(cmatrix_compare, cmatrix_computed))

    def test_cmatrix_slidingwindow(self):
        """
        Tests if using the sliding window keyword reproduces the
        analytical result of a simple example trajectory.
        """
        test_dtraj = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        cmatrix_compare = np.array([[1., 3.], [2., 2.]])
        cmatrix_computed = cmatrix(test_dtraj, tau=2, sliding_window=True)

        self.assertTrue(np.allclose(cmatrix_compare, cmatrix_computed))

    def test_cmatrix_list(self):
        """
        Test count matrix list support.
        """

        test_dtraj = [np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1]), \
                    np.array([0,1,1,1,0])]

        cmatrix_compare = np.array([[2., 3.], [2., 6.]])
        cmatrix_computed = cmatrix(test_dtraj)
        self.assertTrue(np.allclose(cmatrix_compare, cmatrix_computed))
