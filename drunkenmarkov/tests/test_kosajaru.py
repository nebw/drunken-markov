#!/usr/bin/python

import numpy as np
from unittest import TestCase

from drunkenmarkov.Analysis import MarkovStateModel


class KosajaruTest(TestCase):
    T_single = np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.49, 0.01, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.01, 0.49, 0.5],
        [0.0, 0.0, 0.0, 0.5, 0.5]], dtype=np.float64)

    T_multiple = np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.51, 0.49, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.49, 0.51],
        [0.0, 0.0, 0.0, 0.5, 0.5]], dtype=np.float64)

    def test_single_communication_class(self):
        """
        Test kosajaru algorithm for the case that there exists only one
        communication class.
        """
        msm = MarkovStateModel(self.T_single)

        # order of elements in communication classes is not relevant
        classes = [set(x) for x in msm.communication_classes]

        self.assertTrue(set([0, 1, 2, 3, 4]) in classes)

    def test_multiple_communication_class(self):
        """
        Test kosajaru algorithm for the case that there exists three
        communication classes.
        """
        msm = MarkovStateModel(self.T_multiple)

        # order of elements in communication classes is not relevant
        classes = [set(x) for x in msm.communication_classes]

        self.assertTrue(set([1, 0]) in classes)
        self.assertTrue(set([4, 3]) in classes)
        self.assertTrue(set([2]) in classes)

    def test_connected(self):
        """
        Test whether the check for connectedness is correct.
        """
        msm = MarkovStateModel(self.T_single)
        self.assertTrue(msm.is_connected)

        msm = MarkovStateModel(self.T_multiple)
        self.assertFalse(msm.is_connected)