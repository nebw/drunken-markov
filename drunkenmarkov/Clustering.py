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

	# returns a reference to the original data used to instantiate this cluster.
	def getOriginalData(self):
		return self._data

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


#Performs PCA on an array of data
#note that this does not support a list of trajectories(yet)
class PCA:
	def __init__(self, data):
		if not isinstance(data, np.ndarray):
			raise TypeError("Data must be numpy array")
		#reject one dimensional data
		if len(data.shape) ==  1:
			raise TypeError("Data is already one dimensional")
		#subtract mean from data
		self.X = data - data.mean(axis = 0)
		#calculate the covariance matrix. Factor of N-1 due to Bessels correction.
		self.C = np.dot(np.transpose(self.X),self.X)/(self.X.shape[0] - 1)
		self.sigma = np.zeros(self.C.shape[0])
		self.W = np.zeros_like(self.C)
		self.sigma[::-1],self.W[:,::-1] = np.linalg.eigh(self.C)
		#TODO: sort eigenvalues by size and eigenvectors respectively if they are not sorted yet
		self.Sigma = np.diag(self.sigma)
		self.Y = np.dot(self.X, self.W)
		self.s = np.cumsum(self.sigma) / np.sum(self.sigma)

	#return projection of the data on the first L pricipal axis or specify L by a cutoff of the cummulative variance
	#if neither is given return full projection on eigenspace of the covariance matrix
	def get_reduced_data(self, cutoff = 1, L = None):
		if L is not None:
			return self.Y[:,0:L]
		return self.Y[:,np.where(self.s <= cutoff)[0]]

	#returns the cummulative variance
	@property
	def get_cummulative_variance(self):
		return self.s

	#returns transformation matrix
	@property
	def get_transformation_matrix(self):
		return self.W

	#returns variance vector
	@property
	def get_variance(self):
		return self.sigma

	#return covariance matrix of trajectories in non-transformed space
	@property
	def get_covariance(self):
			return self.C

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