#!/usr/bin/python

import numpy as np
from unittest import TestCase

from drunkenmarkov.Analysis import MarkovStateModel


class StochasiticityTest(TestCase):
    def test_is_stochastic(self):
        """
        Test whether test for stochasiticity returns expected result when
        transition matrix is stochastic
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
        Test whether test for stochasiticity returns expected result when
        transition matrix isn't stochastic
        """
        T = np.array([
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.49, 0.05, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.01, 0.49, 0.5],
            [0.0, 0.0, 0.0, 0.5, 0.5]], dtype=np.float64)

        msm = MarkovStateModel(T)
        self.assertFalse(msm.is_transition_matrix)
