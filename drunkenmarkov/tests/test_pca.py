import numpy as np
from unittest import TestCase
import drunkenmarkov.Clustering
from pyemma.coordinates import pca as emma_pca

class PCATests(TestCase):
	def test_PCA(self):
		"""
		Testing the implemented PCA class against the pyemma result
		Create deterministic data in [1,1] direction with noise in perpendiculate [1,-1] direction and apply pca
		"""
		data = np.zeros((1000,2))
		rand = (np.random.rand(1000)-1.)*5
		for i in range(0,1000):
			data[i,:] = [i,i]
			data[i,:] += [rand[i],-rand[i]]
		pca = drunkenmarkov.PCA(data)
		emmapca = emma_pca(data)
		self.assertTrue(np.allclose(pca.get_reduced_data(L = 2),emmapca.get_output()))
		self.assertTrue(np.allclose(pca.get_covariance, emmapca.covariance_matrix))