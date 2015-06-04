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
			self.assertTrue(clusters.getClusterCenters() == clusters.centers)
			self.assertTrue(len(clusters.getClusterCenters()) == clusters.ncenters)
			clusters[0]
			
			# clustering defaults to kmeans
			new_clusters = drunkenmarkov.cluster(data)
			new_clusters = drunkenmarkov.cluster(data, ncenters=2)
			
			# cluster some additional (actually the old) points
			cluster_membership = new_clusters.discretise(data)
			for index, datapoint in enumerate(data):
				self.assertTrue(cluster_membership[index] == new_clusters[index])
			
		except Exception as e:
			self.fail("Kmeans clustering raised exception: {}".format(e))
