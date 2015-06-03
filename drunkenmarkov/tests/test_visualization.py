#!/usr/bin/python

import numpy as np
from unittest import TestCase

from drunkenmarkov.Analysis import MarkovStateModel
from drunkenmarkov.Visualization import get_graph


class GraphVisualizationTest(TestCase):
    T_single = np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.49, 0.01, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.01, 0.49, 0.5],
        [0.0, 0.0, 0.0, 0.5, 0.5]], dtype=np.float64)

    T_multiple = np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.51, 0.49, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.49, 0.51],
        [0.0, 0.0, 0.0, 0.5, 0.5]], dtype=np.float64)

    def test_graph_nodes(self):
        """
        Assert that generated graph has as many nodes as transition matrix
        """
        msm = MarkovStateModel(self.T_single)
        graph = get_graph(msm)

        self.assertTrue(graph.number_of_nodes() == 5)

    def test_graph_edges(self):
        """
        Assert that generated graph has as many edges as transitions that
        are greater than zero
        """
        msm = MarkovStateModel(self.T_single)
        graph = get_graph(msm)

        self.assertTrue(graph.number_of_edges() ==
                        np.count_nonzero(self.T_single))

    def test_graph_comm_classes(self):
        """
        Assert that number of subgraphs is equal to number of communication
        classes
        """
        msm = MarkovStateModel(self.T_single)
        graph = get_graph(msm, with_comm_classes=True)
        self.assertTrue(len(graph.subgraphs()) == 1)

        msm = MarkovStateModel(self.T_multiple)
        graph = get_graph(msm, with_comm_classes=True)
        self.assertTrue(len(graph.subgraphs()) == 3)
