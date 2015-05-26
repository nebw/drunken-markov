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
    
    # constructs a new Clusters object from data and a membership map
    def __init__(self, data, membership, cluster_centers = None):
        self._data = data
        self._map = membership
        self._cluster_centers = cluster_centers
            
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
    
# Clusters a matrix of observations with a specified clustering algorithm.
# returns a Clusters object.
def cluster(data, algorithm=None):
    
    # by default, use the k-means algorithm for clustering
    if algorithm is None:
        from .clustering import kmeans
        algorithm = kmeans.KMeans(k = 3)
        
    clusters = algorithm.cluster(data)
    return clusters