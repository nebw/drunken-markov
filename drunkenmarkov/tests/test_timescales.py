#!/usr/bin/python

import numpy as np

from unittest import TestCase

from drunkenmarkov.Analysis import msm

class TimescaleTests(TestCase):
    def test_timescales_simple(self):
        """
        Tests if the implemented timescales method (analysis.timescales) works
        with a simple transition matrix (taken from http://www.pythonhosted.org/
        pyEMMA/api/generated/pyemma.msm.analysis.timescales.html)
        """
        T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])

        markovmodel = msm(T)
        comp_timescales = markovmodel.timescales
        ref_timescales = np.array([np.inf, 9.49122158, 0.43429448])
        
        self.assertTrue(np.allclose(comp_timescales, ref_timescales, rtol=1.e-5))