#!/usr/bin/python

from io import BytesIO
from PIL import Image
import pygraphviz as pgv

from .Util import get_adjacent_nodes


def get_graph(msm, with_comm_classes=False):
    """Draw a graph representation of the chain using pygraphviz."""

    g = pgv.AGraph(strict=False, directed=True)

    g.graph_attr.update(size="7.75, 10.25")
    g.graph_attr.update(dpi="300")

    g.add_nodes_from(range(msm.num_nodes))

    if with_comm_classes:
        comm_classes = msm.communication_classes

        for (i, comm) in enumerate(comm_classes):
            g.add_subgraph(nbunch=comm, name='cluster%d' % i,
                           style='rounded, dotted',
                           color='lightgrey',
                           label='<<B>Communication class %d</B>>' % (i + 1))

    for from_node in range(msm.num_nodes):
        for to_node in get_adjacent_nodes(msm, from_node, discard_self=False):
            label = '%.2f' % msm.T[from_node, to_node]
            g.add_edge(from_node, to_node, label=label)

    return g


def draw_graph(msm, with_comm_classes=False):
    g = get_graph(msm, with_comm_classes)

    g.layout(prog='dot')
    data = g.draw(format='png')
    return Image.open(BytesIO(data))
