#!/usr/bin/python

import numpy as np
from unittest import TestCase
import drunkenmarkov
from drunkenmarkov.clustering import kmeans

class KmeansTests(TestCase):
	def test_kmeans_coverage(self):
		"""
		This is a simple code-coverage test for the kmeans algorithm.
		"""
		try:
			data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 2, 3]])
			clusters = drunkenmarkov.cluster(data, algorithm = kmeans.KMeans(k = 2))
			clusters.getClusterIDs()
			clusters.getClusterCenters()
			clusters[0]
			
			# clustering defaults to kmeans
			new_clusters = drunkenmarkov.cluster(data)
		except Exception as e:
			self.fail("Kmeans clustering raised exception: {}".format(e))
