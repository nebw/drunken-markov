#!/usr/bin/python

import numpy as np

from unittest import TestCase

from drunkenmarkov.Analysis import MarkovStateModel


class StationaryTests(TestCase):
    def test_stationary(self):
        """
        Tests if the implemented stationary distribution method
        (analysis.stationary_distribution) works with a simple
        transition matrix.
        """
        T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])

        markovmodel = MarkovStateModel(T)
        pi = markovmodel.stationary_distribution

        # test if stationary distribution is a stochastic vector (has norm 1 and no negative entries)
        self.assertTrue(np.sum(pi),1.0)
        for i in range(0, len(T[0, :])):
            self.assertGreaterEqual(pi[i],0)

        # test if stationary distribution * transition matrix = stationary distribution
        self.assertTrue(np.allclose(np.dot(pi, T), pi, rtol=1e-05, atol=1e-08))
        #while we're at it, we might as well just check whether is_reversible works
        self.assertTrue(markovmodel.is_reversible)