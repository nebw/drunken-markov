#!/usr/bin/python

import numpy as np
from unittest import TestCase

from drunkenmarkov.Analysis import MarkovStateModel


class StochasiticityTest(TestCase):
    def test_is_stochastic(self):
        """
        Test whether test returns expected result when transition matrix is
        stochastic.
        """
        T = np.array([
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.49, 0.01, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.01, 0.49, 0.5],
            [0.0, 0.0, 0.0, 0.5, 0.5]], dtype=np.float64)

        msm = MarkovStateModel(T)
        self.assertTrue(msm.is_transition_matrix)

    def test_is_not_stochastic(self):
        """
        Test that the MarkovStateModel init function throws an exeption when
        the transition matrix is not stochastic.
        """
        T = np.array([
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.49, 0.05, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.01, 0.49, 0.5],
            [0.0, 0.0, 0.0, 0.5, 0.5]], dtype=np.float64)

        self.assertRaises(ValueError, MarkovStateModel, T)

    def test_is_not_quadratic(self):
        """
        Test that the MarkovStateModel init function throws an exeption when
        the transition matrix is not quadratic.
        """
        T = np.array([
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.49, 0.05, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5]], dtype=np.float64)

        self.assertRaises(ValueError, MarkovStateModel, T)

    def test_is_not_bidimensional(self):
        """
        Test that the MarkovStateModel init function throws an exeption when
        the transition matrix is not bidimensional.
        """
        T = np.ones((2, 2, 2), dtype=np.float64)

        self.assertRaises(ValueError, MarkovStateModel, T)
