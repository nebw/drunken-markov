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
	
	# how many runs of k-means will be executed to select the best one
	_number_of_runs = None
	# how many parallel processes will be spawned for the different runs of k-means
	_number_of_processes = None
	# this will temporarily hold the original data as a means of passing it to other processes
	_data = None
	
	# creates a new k-means clustering object with a specified k
	def __init__(self, k = None, n_runs = 3, n_processes = 1, _patience = None, _patience_threshold = None):
		# sanity checks
		if k is None or k <= 0:
			raise ValueError("k needs to be a positive integer.")
		
		self.k = k
		self._patience = _patience or self._patience
		self._patience_threshold = _patience_threshold or self._patience_threshold
		
		self._number_of_runs = n_runs if n_runs > 0 else 1
		self._number_of_processes = n_processes
	
	# discretizes data points with the previously calculated cluster centers
	def discretise(self, data):
		_, clusters = self._kdtree.query(data)
		return clusters
	
	# clusters a matrix of observations and returns an Clusters object
	def cluster(self, data):
		# special case: the data might be a simple list - convert it to a standard feature matrix first!
		if (not isinstance(data, np.ndarray)) or len(data.shape) < 2:
			data = np.transpose(np.array([data]))
		
		# do not only one run but allow parallel runs to select the best one afterwards
		candidates = []
		# pass the data as a temporary member of this "configuration" instance
		self._data = data
		
		if self._number_of_processes <= 1: # just do that locally
			for i in range(self._number_of_runs):
				candidates.append(_one_clustering_run(self))
		else:
			# for allowing true parallel k-means runs
			import multiprocessing
			
			# provide a custom context manager to also support python 2.*
			from contextlib import contextmanager
			@contextmanager
			def auto_close_pool(pool):
				try:
					yield pool
				finally:
					pool.terminate()
			# and finally map to processes
			with auto_close_pool(multiprocessing.Pool(processes=self._number_of_processes)) as pool:
				candidates = pool.map(_one_clustering_run, [self] * self._number_of_runs)
		self._data = None
		# figure out best candidate from all possibilities
		assert (len(candidates) >= 1)
		assert (len(candidates) == self._number_of_runs)
		
		best_candidate = 0
		if self._number_of_runs > 1:
			min_distance = None
			for index in range(self._number_of_runs):
				distance = candidates[index][0]
				
				if (min_distance is None) or (distance < min_distance):
					best_candidate = index
					min_distance = distance
					
		(_, cluster_centers) = candidates[best_candidate]
		# remember the kd tree to be able to cluster additional data later on
		self._kdtree = KDTree(cluster_centers, leafsize=5)
		_, data_membership = self._kdtree.query(data)
		
		return Clusters(data = data, 
						membership = data_membership,
						cluster_centers = cluster_centers,
						algorithm = self)
						
# This might be run as a separate process and returns (avg. distance, centers) of one cluster run.
# Note that this will NOT modify the passed kmeans object and only use it to access the configuration.
# This was supposed to be a class-method, which sadly doesn't work with python multiprocessing.
def _one_clustering_run(configuration):
	data = configuration._data
	# maps data point to cluster index; optimizes stopping criteria
	data_length = len(data)
	cluster_map = [-1] * data_length
	# start off with a few random cluster centers;
	# this is the Forgy method and usually OK for standard k-means.
	current_cluster_centers = data[np.random.choice(data_length, size=configuration.k, replace=False), :]

	# repeat the loop until the cluster assignments do not change anymore
	stopping_criterion_met = False
	current_patience = configuration._patience
	
	while not stopping_criterion_met:
		# consult a KD tree for performance improvement
		cluster_center_tree = KDTree(current_cluster_centers, leafsize=5)
		_, data_membership  = cluster_center_tree.query(data)

		# update stopping condition
		if configuration._patience_threshold > 0.0:
			# calculate ratio of changes and apply threshold & patience
			same_assignment_count = np.sum(cluster_map == data_membership)
			change_ratio  = 1.0 - same_assignment_count / float(data_length)
			if change_ratio < configuration._patience_threshold:
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
		current_cluster_centers = [np.median(data[cluster_map == i, :], axis=0) for i in range(configuration.k)]
	
	# now finally calculate the score as the avg. euclidian distance to the centers
	avg_distance = 0.0
	for index, point in enumerate(data):
		avg_distance += np.linalg.norm(data[index,:] - data[cluster_map[index], :])
	avg_distance /= data.shape[0]
	
	return avg_distance, current_cluster_centers
