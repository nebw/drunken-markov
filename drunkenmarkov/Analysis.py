#!/usr/bin/python
import numpy as np
from pyemma.msm.analysis import eigenvalues


def timescales(T, lagtime=1.):
	"""
	Compute the time scales of a given transition matrix T.

	Keyword arguments:
	lagtime tau (default 1.0)
	"""

	# compute eigenvalues (function to be exchanged by our own...)
	eigenv = eigenvalues(T)

	# test for complex eigenvalues
	ev_is_cmplx = np.where(np.imag(eigenv) > 0.)
	if sum(ev_is_cmplx) > 0: print 'Complex eigenvalues found!'
	eigenv = np.real(eigenv)

	# continue with real part only
	timescales = np.zeros_like(eigenv)

	# take care of devision by zero (EV = 1) and compute time scales
	# for loop to be replaced by something faster
	for ii in range(len(eigenv)):
		if (eigenv[ii] - 1.)**2 < 1e-5:
			timescales[ii] = np.inf
		else:
			timescales[ii] = -lagtime / np.log(abs(eigenv[ii]))

	return timescales

# Sorry, das ist am falschen Ort, aber bei dummy_test.py gibt's bei
# mir nen Fehler (import drunkenmarkov fehlgeschlagen). (Tim)

def test_timescales_simple():
	"""
	Tests if the implemented timescales method (analysis.timescales) works
	with a simple transition matrix (taken from http://www.pythonhosted.org/
	pyEMMA/api/generated/pyemma.msm.analysis.timescales.html)
	"""
	T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])

	comp_timescales = timescales(T)
	ref_timescales = np.array([np.inf, 9.49122158, 0.43429448])

	return np.allclose(comp_timescales, ref_timescales, rtol=1.e-5)
