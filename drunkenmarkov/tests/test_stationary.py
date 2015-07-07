#!/usr/bin/python

import numpy as np

from unittest import TestCase

from drunkenmarkov.Analysis import MarkovStateModel


class StationaryTests(TestCase):
    def test_stationary(self):
        """
        Tests if the implemented stationary distribution method
        (analysis.stationary_distribution) works with
        -a simple transition matrix (regular, reversible)
        -a singular transition matrix (reversible)
        -a non-reversible transition matrix
        """
        T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
        U = np.array([[1.0/3, 2.0/3], [1.0/3, 2.0/3]])
        V = np.array([[0.4, 0.2, 0.4], [0.6, 0.2, 0.2], [0.0, 0.6, 0.4]])

        markovmodel_1 = MarkovStateModel(T)
        markovmodel_2 = MarkovStateModel(U)
        markovmodel_3 = MarkovStateModel(V)
        pi_1 = markovmodel_1.stationary_distribution
        pi_2 = markovmodel_2.stationary_distribution
        pi_3 = markovmodel_3.stationary_distribution

        # test if stationary distribution is a stochastic vector (has norm 1 and no negative entries)
        self.assertTrue(np.sum(pi_1),1.0)
        for i in range(0, len(T[0, :])):
            self.assertGreaterEqual(pi_1[i],0)

        self.assertTrue(np.sum(pi_2),1.0)
        for i in range(0, len(U[0, :])):
            self.assertGreaterEqual(pi_2[i],0)

        # test if stationary distribution * transition matrix = stationary distribution
        self.assertTrue(np.allclose(np.dot(pi_1, T), pi_1, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(np.dot(pi_2, U), pi_2, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(np.dot(pi_3, V), pi_3, rtol=1e-05, atol=1e-08))


        #while we're at it, we might as well just check whether is_reversible works
        self.assertTrue(markovmodel_1.is_reversible)
        self.assertTrue(markovmodel_2.is_reversible)
        self.assertFalse(markovmodel_3.is_reversible)