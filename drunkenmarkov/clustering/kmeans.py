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
	
	# stopping criterion
	# how many iterations are allowed with a threshold of change below _patience_threshold?
	_patience = 10
	# the ratio of changes in the assignments necessary to continue the fitting;
	# set to 0 to disable early stopping
	_patience_threshold = 0.1
	
	# creates a new k-means clustering object with a specified k
	def __init__(self, k = None, _patience = None, _patience_threshold = None):
		# sanity checks
		if k is None or k <= 0:
			raise ValueError("k needs to be a positive integer.")
			
		self.k = k
		self._patience = _patience or self._patience
		self._patience_threshold = _patience_threshold or self._patience_threshold 
	
	# discretizes data points with the previously calculated cluster centers
	def discretise(self, data):
		_, clusters = self._kdtree.query(data)
		return clusters
		
	# clusters a matrix of observations and returns an Clusters object
	def cluster(self, data):
		# special case: the data might be a simple list - convert it to a standard feature matrix first!
		if (not isinstance(data, np.ndarray)) or len(data.shape) < 2:
			data = np.transpose(np.array([data]))
		# maps data point to cluster index; optimizes stopping criteria
		data_length = len(data)
		cluster_map = [-1] * data_length
		# start off with a few random cluster centers;
		# this is the Forgy method and usually OK for standard k-means.
		current_cluster_centers = data[np.random.choice(data_length, size=self.k, replace=False), :]

		# repeat the loop until the cluster assignments do not change anymore
		stopping_criterion_met = False
		current_patience = self._patience
		
		while not stopping_criterion_met:
			# consult a KD tree for performance improvement
			cluster_center_tree = KDTree(current_cluster_centers, leafsize=5)
			_, data_membership  = cluster_center_tree.query(data)

			# update stopping condition
			if self._patience_threshold > 0.0:
				# calculate ratio of changes and apply threshold & patience
				same_assignment_count = np.sum(cluster_map == data_membership)
				change_ratio  = 1.0 - same_assignment_count / float(data_length)
				if change_ratio < self._patience_threshold:
					current_patience -= 1
					if current_patience <= 0:
						stopping_criterion_met = True
			else:
				# check whether there is at least ONE assignment that did not change
				# manual loop instead of equality operator as the latter always checks all elements, ...
				stopping_criterion_met = True
				for i in range(data_length):
					if cluster_map[i] == data_membership[i]: continue
					stopping_criterion_met = False
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
