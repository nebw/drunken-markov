#!/usr/bin/python
import numpy as np
from pyemma.msm.analysis import eigenvalues


class msm:
    def __init__(self, T, lagtime=1.):
        self.T = T
        # compute eigenvalues (function to be exchanged by our own...)
        self.eigenv = eigenvalues(T)
        self.lagtime = lagtime
        # only compute the timescales if they are called explicitly.
        # This might not be necessary here, but can be useful at some
        # other point of the project.
        self._timescales = None

    @property
    def timescales(self):
        """
        Compute the time scales of a given transition matrix T.

        Keyword arguments:
        lagtime tau (default 1.0)
        """
        if self._timescales is None:
            # test for complex eigenvalues
            ev_is_cmplx = np.where(np.imag(self.eigenv) > 0.)
            if sum(ev_is_cmplx) > 0:
                print('Complex eigenvalues found!')

            re_eigenv = np.real(self.eigenv)
            # continue with real part only
            self._timescales = np.zeros_like(re_eigenv)

            # take care of devision by zero (EV = 1) and compute time scales
            # for loop to be replaced by something faster
            for ii in range(len(re_eigenv)):
                if (re_eigenv[ii] - 1.)**2 < 1e-5:
                    self._timescales[ii] = np.inf
                else:
                    self._timescales[ii] = -self.lagtime / np.log(abs(re_eigenv[ii]))

        return self._timescales

# Sorry, das ist am falschen Ort, aber bei dummy_test.py gibt's bei
# mir nen Fehler (import drunkenmarkov fehlgeschlagen). (Tim)


def test_timescales_simple():
    """
    Tests if the implemented timescales method (analysis.timescales) works
    with a simple transition matrix (taken from http://www.pythonhosted.org/
    pyEMMA/api/generated/pyemma.msm.analysis.timescales.html)
    """
    T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])

    markovmodel = msm(T)
    comp_timescales = markovmodel.timescales
    ref_timescales = np.array([np.inf, 9.49122158, 0.43429448])

    return np.allclose(comp_timescales, ref_timescales, rtol=1.e-5)
