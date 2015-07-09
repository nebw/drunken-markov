#!/usr/bin/python

import numpy as np

from unittest import TestCase

from drunkenmarkov.Analysis import MarkovStateModel
from drunkenmarkov.Util import gcd


class PeriodTests(TestCase):
    def test_period(self):
        """
        Tests if the implemented period computation method
        (analysis.period) works with different simple
        transition matrices.
        """
        # simple aperiodic Markov chain
        S = np.array([[0.2, 0.8],[0.5, 0.5]])

        # simple periodic Markov chains
        T = np.array([[0.0, 1.0],[1.0, 0.0]])
        U = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 0.5],[1.0, 0.0, 0.0, 0.0]])

        markovmodel_1 = MarkovStateModel(S)
        markovmodel_2 = MarkovStateModel(T)
        markovmodel_3 = MarkovStateModel(U)
        p_1 = markovmodel_1.period
        p_2 = markovmodel_2.period
        p_3 = markovmodel_3.period

        # test if the computed periods fulfill the requirements
        self.assertEqual(1.0,p_1)
        self.assertEqual(2.0,p_2)
        self.assertEqual(2.0,p_3)

        # test if the greatest common divisor function works:
        self.assertEqual(gcd(2,-1),1)
        self.assertEqual(gcd(1,0),1)
        self.assertEqual(gcd(10,15),5)