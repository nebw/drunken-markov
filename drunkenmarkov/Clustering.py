#!/usr/bin/python
import numpy as np

# Object that is returned by the clustering algorithms.
# It can be used as a mapping from sample index to cluster or to retrieve general informations about the clusters.
class Clusters(object):
	# maps sample index to cluster ID
	_map = None
	# contains the IDs of the clusters; usually just integers
	_cluster_ids = None
	# a reference to the original data; not copied!
	_data = None
	# optionally saves the cluster centers
	_cluster_centers = None
	# the clustering algorithm, used to discretize additional data points.
	_algorithm = None
	
	# constructs a new Clusters object from data and a membership map
	def __init__(self, data, membership, cluster_centers = None, algorithm = None):
		self._data = data
		self._map = membership
		self._cluster_centers = cluster_centers
		self._algorithm = algorithm
		
	# returns an unordered list of the cluster IDs
	def getClusterIDs(self):
		if self._cluster_ids is None:
			self._cluster_ids = np.unique(self._map)
		return self._cluster_ids
		
	# returns the center of a cluster or None;
	# important: not every clustering algorithm uses centers, so this does not have to return a value.
	def getClusterCenter(self, cluster_id):
		if self._cluster_centers is None:
			return None
		return self._cluster_centers[cluster_id]
	
	# returns all cluster centers if available or None otherwise
	def getClusterCenters(self):
		return self._cluster_centers
	
	# returns a list with the assigned cluster IDs for the original data
	def getDataMapping(self):
		return self._map
	
	# discretizes additional data points with the previously used algorithm and clustering parameters
	def discretise(self, data):
		return self._algorithm.discretise(data)
		
	# maps a sample index to a cluster ID
	def __getitem__(self, index):
		# sanity
		if index < 0 or index >= len(self._map):
			raise ValueError("index {} out of range 0-{}".format(index, len(self._map)))
		
		return self._map[index]
	
	# returns a textual summary about the cluster object
	def __repr__(self):
		return "Cluster(datasize = {}, clustersize = {}, data_dimensions = {})".format(
				len(self._data), 
				len(self.getClusterIDs()), 
				self._data[0].shape if (not (self._data is None) and (len(self._data) > 0)) else "??")
	
	
	# to adhere to interface specifications
	
	# Returns the number of cluster centers (if available).
	@property
	def ncenters(self):
		return len(self.getClusterCenters())
		
	# Returns the cluster centers (if available).
	@property
	def centers(self):
		return self.getClusterCenters()

	def get_centers(self):
		import warnings
		warnings.warn("Function deprecated.. use getDataMapping()")
		return self.getDataMapping()

# Clusters a matrix of observations with a specified clustering algorithm.
# returns a Clusters object.
def cluster(data, algorithm=None, **kwargs):
	
	# by default, use the k-means algorithm for clustering
	if algorithm is None:
		from .clustering import kmeans
		ncenters = kwargs["ncenters"] if "ncenters" in kwargs else 3
		algorithm = kmeans.KMeans(k = ncenters)
		
	clusters = algorithm.cluster(data)
	return clusters