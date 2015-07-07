#!/usr/bin/python
import numpy as np
from scipy.linalg import eigh

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


#Performs principal component analysis on an array of data
#note that this does not support a list of trajectories(yet)
class pca:
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
		self.sigma,self.W = np.linalg.eigh(self.C)
		#sort eigenvalues and - vectors descending
		idx = np.argsort(np.absolute(self.sigma))
		self.sigma = self.sigma[idx[::-1]]
		self.W = self.W[:,idx[::-1]]
		#calculate transformed data
		self.Y = np.dot(self.X, self.W)
		#calculate cummulative variance as cutoff criterion
		self.s = np.cumsum(self.sigma) / np.sum(self.sigma)

	#return projection of the data on the first L pricipal axis or specify L by a cutoff of the cummulative variance
	#if neither is given return full projection on eigenspace of the covariance matrix
	def reduced_data(self, cutoff = 1., L = None):
		if L is not None:
			return self.Y[:,0:L]
		return self.Y[:,np.where(self.s <= cutoff)[0]]

	#returns the cummulative variance
	@property
	def cummulative_variance(self):
		return self.s

	#returns transformation matrix
	@property
	def transformation_matrix(self):
		return self.W

	#returns variance vector
	@property
	def variance(self):
		return self.sigma

	#return covariance matrix of trajectories in non-transformed space
	@property
	def covariance(self):
			return self.C

#Performs tica on an array of data
#note that this does not support a list of trajectories(yet)
class tica:
	def __init__(self, data, lag = 10):
		if not isinstance(data, np.ndarray):
			raise TypeError("Data must be numpy array")
		#reject one dimensional data
		self.dim = data.shape[1]
		if self.dim ==  1:
			raise TypeError("Data is already one dimensional")
		self.tau = lag
		self.mean = data.mean(axis = 0)
		#first of all the data has to be mean free, so we simply subtract the mean
		self.X = data - self.mean
		#calculate the covariance matrix
		self.C0 = np.dot(np.transpose(self.X),self.X)
		#enforce symmetrie
		self.C0 = np.add(self.C0, np.transpose(self.C0)) /( 2 * (self.X.shape[0] - 1))
		#calculate the correlattion matrix. N-1 due to Bessels correction.
		self.C = np.dot(np.transpose(self.X[self.tau:, :]),self.X[: -self.tau,:])
		#enforce symmetrie
		self.C = np.add(self.C, np.transpose(self.C) ) / (2 * (self.X.shape[0] - 1 - self.tau))
		#solve the generalized eigenvalue problem: C Psi = sigma C0 Psi
		self.sigma, self.psi = eigh(self.C, b = self.C0, type = 1)
		#sort eigenvalue and -vectors descending
		idx = np.argsort(np.absolute(self.sigma))
		self.sigma = self.sigma[idx[::-1]]
		self.psi = self.psi[:,idx[::-1]]
		#calculate the transformed data
		self.Y = np.dot(self.X, self.psi)
		#calculate the cummulative autocorrelation
		self.s = np.cumsum(self.sigma) / np.sum(self.sigma)

	#Either specify new dimension by a cutoff cummulative variance or the desired dimension.
	def reduced_data(self, cutoff = 1., L = None):
		if L is not None:
			return self.Y[:,0:L]
		if cutoff == 1.:
			return self.Y
		return self.Y[:,np.where(self.s <= cutoff)[0]]

	@property
	def mean(self):
		return self.mean

	@property
	def cummulative_autocorrelation(self):
		return self.s

	@property
	def transformation_matrix(self):
		return self.psi

	@property
	def autocorrelation(self):
		return self.sigma

	@property
	def cross_correlation(self):
		return self.C

	@property
	def covariance(self):
		return self.C0

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