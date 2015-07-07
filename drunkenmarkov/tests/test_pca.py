import numpy as np
from unittest import TestCase
import drunkenmarkov.Clustering
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

if sys.version_info < (3, 0):
    from pyemma.coordinates import pca as emma_pca

@onlypython2
class pcaTests(TestCase):
	def test_pca(self):
		"""
		Testing the implemented PCA class against the pyemma result
		Create deterministic data in [1,1] direction with noise in perpendiculate [1,-1] direction and apply pca
		"""
		data = np.zeros((1000,2))
		rand = (np.random.rand(1000)-1.)*5
		for i in range(0,1000):
			data[i,:] = [i,i]
			data[i,:] += [rand[i],-rand[i]]
		pca = drunkenmarkov.pca(data)
		emmapca = emma_pca(data)
		self.assertTrue(np.allclose(pca.reduced_data(L = 2),emmapca.get_output()))
		self.assertTrue(np.allclose(pca.covariance, emmapca.covariance_matrix))
