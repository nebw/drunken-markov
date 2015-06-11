#!/usr/bin/python

import numpy as np
import pygraphviz as pgv

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

def gcd(a,b):
    m = np.minimum(np.absolute(a),np.absolute(b))
    M = np.maximum(np.absolute(a),np.absolute(b))
    if m == 0:
        return M
    u = 1
    while u != 0:
        u = M % m
        if u == 0:
            return m
        M = m
        m = u  

# fix for the pygraphviz graph constructor, which ignores the /strict/ argument on windows
# source: http://stackoverflow.com/questions/14374412/how-do-i-make-an-undirected-graph-in-pygraphviz
def AGraph(directed=False, strict=True, name='', **args):
    """Fixed AGraph constructor."""

    graph = '{0} {1} {2} {{}}'.format(
        'strict' if strict else '',
        'digraph' if directed else 'graph',
        name
    )

    return pgv.AGraph(graph, **args)
