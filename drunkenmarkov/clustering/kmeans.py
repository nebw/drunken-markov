#!/usr/bin/python
from ..Clustering import Clusters

import numpy as np
# for speeding up lookups for assigning a center to a data point
from scipy.spatial import KDTree

# Clusters a dataset using the k-means algorithm.
# The clusters will have IDs enumerated from 0 to k-1.
class KMeans(object):
	# the number of clusters for k-means
	k = None
	# the final tree of the clustering algorithm
	_kdtree = None
	
	# creates a new k-means clustering object with a specified k
	def __init__(self, k = None):
		# sanity checks
		if k is None or k <= 0:
			raise ValueError("k needs to be a positive integer.")
			
		self.k = k
	
	# discretizes data points with the previously calculated cluster centers
	def discretise(self, data):
		_, clusters = self._kdtree.query(data)
		return clusters
		
	# clusters a matrix of observations and returns an Clusters object
	def cluster(self, data):
		# maps data point to cluster index; optimizes stopping criteria
		data_length = len(data)
		cluster_map = [-1] * data_length
		# start off with a few random cluster centers;
		# this is the Forgy method and usually OK for standard k-means.
		current_cluster_centers = data[np.random.choice(data_length, size=self.k, replace=False), :]

		# repeat the loop until the cluster assignments do not change anymore
		at_least_one_assignment_changed = True
		
		while at_least_one_assignment_changed:
			# consult a KD tree for performance improvement
			cluster_center_tree = KDTree(current_cluster_centers, leafsize=5)
			_, data_membership  = cluster_center_tree.query(data)

			# update stopping condition
			# manual loop instead of equality operator as the latter always checks all elements, ...
			at_least_one_assignment_changed = False
			for i in range(data_length):
				if cluster_map[i] == data_membership[i]: continue
				at_least_one_assignment_changed = True
				break
			
			# and finally update the cluster centers to be the centroids of the voronoi subsets
			cluster_map = data_membership
			current_cluster_centers = [np.median(data[cluster_map == i, :], axis=0) for i in range(self.k)]
			
			# remember the kd tree to be able to cluster additional data later on
			self._kdtree = cluster_center_tree
			
		return Clusters(data = data, 
						membership = cluster_map, 
						cluster_centers = current_cluster_centers,
						algorithm = self)
