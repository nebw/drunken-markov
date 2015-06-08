#!/usr/bin/python

import numpy as np
import scipy

from unittest import TestCase

from drunkenmarkov.Analysis import TransitionPathTheory


class TPTTests(TestCase):
    def test_fcom_simple(self):
        """
        Tests the implemented forward committor function fcom.
        Example taken from the pyEMMA workshop by the Noe group
        in early 2015 at FU Berlin.
        """

        rate_matrix = np.array([[-1, 1, 0], [100, -300, 200], [0, 1, -1]])
        K = scipy.linalg.expm(rate_matrix)
        TPT = TransitionPathTheory(K, [0], [1])

        fcom = TPT.fcom
        ref_fcom = np.array([0, 1., 0.0156032])

        self.assertTrue(np.allclose(fcom, ref_fcom, rtol=1.e-5))

    def test_bcom_simple(self):
        """
        Tests the implemented backward committor function bcom.
        Example taken from the pyEMMA workshop by the Noe group
        in early 2015 at FU Berlin.
        """

        rate_matrix = np.array([[-1, 1, 0], [100, -300, 200], [0, 1, -1]])
        K = scipy.linalg.expm(rate_matrix)
        TPT = TransitionPathTheory(K, [0], [1])

        bcom = TPT.bcom
        ref_bcom = np.array([1.,  0., 0.9843968])

        self.assertTrue(np.allclose(bcom, ref_bcom, rtol=1.e-5))
