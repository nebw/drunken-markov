#!/usr/bin/python

import numpy as np
from unittest import TestCase
from drunkenmarkov.Analysis import MarkovStateModel
import sys

def onlypython2(fun):
    """
    This is a decorator that effectively disables a unit test when using
    python 2 by replacing the test function with a lambda function that always
    returns True.

    pyEMMA is not available for python 3. Therefore we have to disable tests
    based on pyEMMA when running python 3.
    """
    if sys.version_info >= (3, 0):
        return lambda: True
    else:
        return fun

@onlypython2
class ReduceMatrixTests(TestCase):
    def test_ReduceMatrixRowsum(self):
        """
        Tests if the rowsum of a reduced matrix is one. Uses 3x3 test matrix.
        """

        T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])

        MSM = MarkovStateModel(T)
        T_reduced = MSM.reduce_matrix(2)[0]

        self.assertTrue(np.allclose(T_reduced.sum(axis=1), 1., rtol=1.e-5))
