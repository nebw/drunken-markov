#!/usr/bin/python
import numpy as np

class msm:
    def __init__(self, T, lagtime=1.):
        self.T = T
        # compute eigenvalues
        W, _ = np.linalg.eig(T)
        self.eigenv = sorted(W, reverse=True)
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