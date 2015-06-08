#!/usr/bin/python

import numpy as np

def get_adjacent_nodes(msm, node, discard_self=True):
    adjacent_nodes = set(np.where(msm.T[node, :] > 0.)[0].tolist())
    if discard_self:
        adjacent_nodes.discard(node)
    return adjacent_nodes
    
def depth_first_search(msm, node, node_list, visited_nodes=None):
    visited_nodes = visited_nodes or set()
    visited_nodes.add(node)

    for recursive_node in get_adjacent_nodes(msm, node):
        if recursive_node not in visited_nodes:
            depth_first_search(msm, recursive_node, node_list, visited_nodes)

    if node not in node_list:
        node_list.append(node)
