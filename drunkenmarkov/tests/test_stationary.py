#!/usr/bin/python

import numpy as np

from unittest import TestCase

from drunkenmarkov.Analysis import msm

class StationaryTests(TestCase):
    def test_stationary(self):
        """
        Tests if the implemented timescales method (analysis.timescales) works
        with a simple transition matrix (taken from http://www.pythonhosted.org/
        pyEMMA/api/generated/pyemma.msm.analysis.timescales.html)
        """
        T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])

        markovmodel = msm(T)
        print markovmodel.stationary_distribution