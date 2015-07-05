#!/usr/bin/python

import numpy as np
import scipy.linalg
import sys

from unittest import TestCase

from drunkenmarkov.Analysis import TransitionPathTheory

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

if sys.version_info < (3, 0):
    import pyemma.msm.flux as msmflux

class TPTTests(TestCase):
    def test_fcom_simple(self):
        """
        Tests the implemented forward committor function fcom.
        Example taken from the pyEMMA workshop by the Noe group
        in early 2015 at FU Berlin.
        """

        rate_matrix = np.array([[-1, 1, 0], [100, -300, 200], [0, 1, -1]]).astype(float)
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

        rate_matrix = np.array([[-1, 1, 0], [100, -300, 200], [0, 1, -1]]).astype(float)
        K = scipy.linalg.expm(rate_matrix)
        TPT = TransitionPathTheory(K, [0], [1])

        bcom = TPT.bcom
        ref_bcom = np.array([1.,  0., 0.9843968])
        self.assertTrue(np.allclose(bcom, ref_bcom, rtol=1.e-5))

    @onlypython2
    def test_prob_current_simple(self):
        """
        Tests the implemented probability current function probability_current.
        Example taken from the pyEMMA workshop by the Noe group
        in early 2015 at FU Berlin.
        """

        rate_matrix = np.array([[-1, 1, 0], [100, -300, 200], [0, 1, -1]]).astype(float)
        K = scipy.linalg.expm(rate_matrix)
        TPT = TransitionPathTheory(K, [0], [1])
        TPT_emma = msmflux.tpt(K, [0], [1])

        probcurrent = TPT.probability_current
        ref_prob = TPT_emma.flux
        self.assertTrue(np.allclose(probcurrent, ref_prob, rtol=1.e-5))


    @onlypython2
    def test_effective_prob_current_simple(self):
        """
        Tests the implemented effective probability current function effective_probability_current.
        Example taken from the pyEMMA workshop by the Noe group
        in early 2015 at FU Berlin.
        """

        rate_matrix = np.array([[-1, 1, 0], [100, -300, 200], [0, 1, -1]]).astype(float)
        K = scipy.linalg.expm(rate_matrix)
        TPT = TransitionPathTheory(K, [0], [1])
        TPT_emma = msmflux.tpt(K, [0], [1])

        #probcurrent = TPT.probability_current
        effec_probcurrent = TPT.effective_probability_current
        ref_net_prob = TPT_emma.net_flux
        self.assertTrue(np.allclose(effec_probcurrent, ref_net_prob, rtol=1.e-5))


    @onlypython2
    def test_flux_simple(self):
        """
        Tests the implemented FLUX function flux.
        Example taken from the pyEMMA workshop by the Noe group
        in early 2015 at FU Berlin.
        """

        rate_matrix = np.array([[-1, 1, 0], [100, -300, 200], [0, 1, -1]]).astype(float)
        K = scipy.linalg.expm(rate_matrix)
        TPT = TransitionPathTheory(K, [0], [1])
        TPT_emma = msmflux.tpt(K, [0], [1])

        flux = TPT.flux
        ref_total_flux = TPT_emma.total_flux
        self.assertTrue(np.allclose(flux, ref_total_flux, rtol=1.e-5))


    @onlypython2
    def test_rate_simple(self):
        """
        Tests the implemented rate function transition_rate.
        Example taken from the pyEMMA workshop by the Noe group
        in early 2015 at FU Berlin.
        """

        rate_matrix = np.array([[-1, 1, 0], [100, -300, 200], [0, 1, -1]]).astype(float)
        K = scipy.linalg.expm(rate_matrix)
        TPT = TransitionPathTheory(K, [0], [1])
        TPT_emma = msmflux.tpt(K, [0], [1])

        transition_rate = TPT.transition_rate
        ref_rate = TPT_emma.rate
        self.assertTrue(np.allclose(transition_rate, ref_rate, rtol=1.e-5))


    @onlypython2
    def test_mean_first_passage_time_simple(self):
        """
        Tests the implemented test_mean first passage time function test_mean_first_passage_time.
        Example taken from the pyEMMA workshop by the Noe group
        in early 2015 at FU Berlin.
        """

        rate_matrix = np.array([[-1, 1, 0], [100, -300, 200], [0, 1, -1]]).astype(float)
        K = scipy.linalg.expm(rate_matrix)
        TPT = TransitionPathTheory(K, [0], [1])
        TPT_emma = msmflux.tpt(K, [0], [1])

        mfpt = TPT.mean_first_passage_time
        ref_mfpt = TPT_emma.mfpt

        self.assertTrue(np.allclose(mfpt, ref_mfpt, rtol=1.e-5))

    @onlypython2
    def test_dominant_pathway(self):
        rate_matrix = np.array([[0,2./8.,1./8.,5./8.,0,0],
                [0.3,0.1,0.2,0.1,0.2,0.1],
               [0.3,0.1,0.1,0.3,0.2,0],
               [0,0,3./4.,0,1./4.,0],
               [0.1,0.3,0.5,0,0,0.1],
               [0,2./8.,0,3./8.,0,3./8.]],
              dtype = float)
        TPT = TransitionPathTheory[rate_matrix,[0],[5]]
        self.assertTrue(np.allclose(TPT.dominant_pathways, [0,3,4,5]))