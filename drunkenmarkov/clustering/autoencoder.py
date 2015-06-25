#!/usr/bin/python
from ..Clustering import Clusters

import numpy as np
import theanets
import copy

class AutoEncoder(object):
	
	# setup of the neural net's layers;
	# the layers must be symmetrical and the first and last layer must match the number of features.
	# the middle layer should be relatively small to keep the number of expected clusters low.
	# f.e. (15, 5, 3, 5, 15) with 15 features
	_layer_setup = None
	
	# for mapping the hidden layer's activation to a class
	_class_cutoffs = None
	_cutoff_mapping = None
	
	# the theano experiment;
	# this contains, f.e., the finished auto-encoder network
	_exp = None
	
	# the following attributes are accessible and usable debugging and visualization information:
	
	# after training, this will contain the error-while-learning over time
	learning_error_progress = None
	# after training, this contains the hidden layer's neurons' activation histograms
	activation_histograms = None
	# this contains the encoded original data - it can be used for visualization purposes
	meta_features = None
	
	def __init__(self, layer_setup = None):
		self._layer_setup = layer_setup
		self.learning_error_progress = []
		
	def cluster(self, data):
		# special case: the data might be a simple list - convert it to a standard feature matrix first!
		if (not isinstance(data, np.ndarray)) or len(data.shape) < 2:
			data = np.transpose(np.array([data]))
			
		# propose a default layer setup matching the data
		if self._layer_setup is None:
			features = int(data.shape[1])
			middle_layer = 4
			hidden = max(middle_layer, int(features / 2))
			self._layer_setup = (features, hidden, middle_layer, hidden, features)
		
		# set up an autoencoder matching the previously determined layer setup
		self._exp = theanets.Experiment(
			theanets.Autoencoder,
			layers = self._layer_setup,
			activation = 'tanh',
			tied_weights = True
		)
		
		# and train the network with the given data
		training_data_set = self._exp.create_dataset(data, name='valid')
		trainer = self._exp.create_trainer('layerwise') 
		for train, valid in trainer.itertrain(training_data_set, training_data_set):
			self.learning_error_progress.append(train["loss"])
		
		# now encode the training data and get the activation of the middle layer
		self.meta_features = self._exp.network.encode(data)
		
		# now find sensible cutoffs for the meta features (look for spikes)
		neuron_count = self.meta_features.shape[1]
		self._class_cutoffs = []
		self.activation_histograms = []
		
		for neuron in range(neuron_count):
			neuron_meta_feature = self.meta_features[:,neuron]
			
			# this finds local minima in the histogram of the activation distribution;
			# thus, it implicitly finds peaks
			def local_minima_detect():
				hist, bin_edges = np.histogram(neuron_meta_feature, bins=20)
				self.activation_histograms.append(hist)
				
				hist_len = len(hist)
				edges = []
				
				# a position is a local minimum if it's either surrounded by higher values or if it's a border value
				def is_local_minimum(index):
					left_ok = index == 0 or hist[index - 1] > hist[index]
					right_ok = index == hist_len - 1 or hist[index + 1] > hist[index]
					return left_ok and right_ok
					
				for i in range(0, hist_len):
					if is_local_minimum(i):
						edges.append(bin_edges[i])
				edges.append(1.0)
				
				# now subdivide the resulting edges to achieve a higher granularity
				rough_edges = edges
				edges = []
				
				# subdivide every edge, assuming a centered peak
				for i in range(len(rough_edges)):
					left = rough_edges[i - 1] if i > 0 else -1.0
					right = rough_edges[i]

					area = right - left
					step = area / 3.0
					edges.append(left)
					
					for j in range(1, 2 + 1):
						subedge = left + step * j
						edges.append(subedge)
					edges.append(right)
				
				return edges
				
			self._class_cutoffs.append(local_minima_detect())
			
			# Now we have cutoffs for every neuron and need a mapping to a cluster id;
			# this first expands a tree containing all different peak combinations and then
			# assigns unique IDs to the leaves.
			total_classes = 1
			cutoff_counts = []
			self._cutoff_mapping = [None]
			
			# recursively expand tree
			for cutoffs in self._class_cutoffs:
				total_classes = total_classes * len(cutoffs)
				cutoff_counts.append(len(cutoffs))
				new_mapping = []
				for cut in cutoffs:
					new_mapping.append(copy.deepcopy(self._cutoff_mapping))
				self._cutoff_mapping = new_mapping
			
			# and now (recursively) enumerate leaves of the deep map
			class current_class: # this encapsulated the counter for the inner function
				index = 0
			def assign_class(mapping):
				if mapping is None:
					return
				if len(mapping) == 1:
					current_class.index += 1
					mapping[0] = current_class.index
				else:
					for submap in mapping:
						assign_class(submap)
			assign_class(self._cutoff_mapping)
			assert(current_class.index > 0)
			
			self._cutoff_mapping = np.array(self._cutoff_mapping)
		
		# now we have the mapping tree and can happily map our previously calculated meta features
		all_classes = self.mapClassesToMetaFeatures(self.meta_features)
		
		return Clusters(data = data, 
			membership = all_classes, 
			cluster_centers = None, # we do not have any centers...
			algorithm = self)
	
	# map the activation of the hidden layer to a class
	def getClassForMetaFeature(self, feature):
		index_list = []
		for index, value in enumerate(feature):
			found = False
			for cutoff_index, cut in enumerate(self._class_cutoffs[index]):
				if value <= cut:
					index_list.append(cutoff_index)
					found = True
					break
			assert (found == True)
			
		return self._cutoff_mapping[tuple(reversed(index_list))][0]
	
	# takes a list of meta features and returns a cluster list
	def mapClassesToMetaFeatures(self, meta_feature_list):
		all_classes = []
		for meta in meta_feature_list:
			all_classes.append(self.getClassForMetaFeature(meta))
		all_classes = np.array(all_classes)
		return all_classes
		
	# discretizes data points with the previously trained ANN
	def discretise(self, data):
		# special case: the data might be a simple list - convert it to a standard feature matrix first!
		if (not isinstance(data, np.ndarray)) or len(data.shape) < 2:
			data = np.transpose(np.array([data]))
			
		# apply the trained network to the new data
		meta_features = self._exp.network.encode(data)
		all_classes = self.mapClassesToMetaFeatures(meta_features)
		return all_classes
