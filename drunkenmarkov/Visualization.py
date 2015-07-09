#!/usr/bin/python

import math
from io import BytesIO
from PIL import Image
import pygraphviz as pgv  # pgv not used, delete?

from .Util import get_adjacent_nodes, AGraph

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import matplotlib.cm

def get_graph(msm, with_comm_classes=False, edge_threshold=0.):
    """Draw a graph representation of the chain using pygraphviz."""

    g = AGraph(strict=False, directed=True)

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
            if msm.T[from_node, to_node] > edge_threshold:
                label = '%.2f' % msm.T[from_node, to_node]
                g.add_edge(from_node, to_node, label=label)

    return g


def draw_graph(msm, with_comm_classes=False, edge_threshold=0.):
    g = get_graph(msm, with_comm_classes, edge_threshold)

    g.layout(prog='dot')
    data = g.draw(format='png')
    return Image.open(BytesIO(data))


def interp_density_plot(centers, z, plot_pi_orig=True):
    """
    Plot a linearly interpolated function of the
    cluster centers.
    """

    import scipy.interpolate

    # get center coordinates
    if centers.shape[1] != 2:
        raise ValueError('Centers coordinates must be 2 dimensional')
    elif z.ndim != 1:
        raise ValueError('Stationary distribution must be a vector')

    x = centers[:, 0]
    y = centers[:, 1]

    img_ratio = (y.max() - y.min()) / (x.max() - x.min())
    pixels = 100**2

    # interpolate and meshgrid these coordinates
    xi, yi = np.linspace(x.min(), x.max(), int(np.sqrt(pixels / img_ratio))),\
        np.linspace(y.min(), y.max(), int(np.sqrt(pixels * img_ratio)))
    xi, yi = np.meshgrid(xi, yi)

    # interpolate the stationary distribution
    rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(xi, yi)

    # plot the interpolated distribution
    ax = plt.subplot(111)
    plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
               extent=[x.min(), x.max(), y.min(), y.max()])

    # add the not-interpolated distribution as scatter plot
    if plot_pi_orig:
        plt.scatter(x, y, c=z, s=60)

    # add colorbar
    plt.colorbar()

    return ax


def draw_stationary(centers, pi, plot_pi_orig=True, output_file=None):
    """
    Draw a density plot of the stationary distribution as a function
    of the cluster centers.

    Optional:
    Show the not-interpolated distribution (plot_pi_orig=True).
    Save the plot as file (output_file='path/to/file.png')
    """

    interp_density_plot(centers, pi, plot_pi_orig)
    plt.title('Stationary Distribution')

    # export
    if output_file:
        plt.savefig(output_file)
        print('Figure saved as ', output_file)
    else:
        plt.show()


def draw_spectrum(T):
    """
    Draw the biggest eigenvalues of the transition matrix for visualizing the
    spectral gap and showing the relation to the timescales of the Markov Chain.
    """      
    # eigenvalues of T sorted by the size
    W, _ = np.linalg.eig(T)
    length = min(len(W), 10) 
    a = sorted(W, reverse=True, key=lambda x: abs(x))[0:length]
    x = np.arange(1.0,11.0,1.0)

    # Define limits of the graph
    xmin = 0.7
    xmax = 10.1
    ymin = -0.1
    ymax = 1.1

    # Plot the ten biggest eigenvalues:
    ax1 = plt.subplot(111)
    plt.plot(x,a, 'yo')
    plt.vlines(x,0,a)
    plt.axhline(linewidth=1, color='k')
    plt.axhline(y=1, linewidth=1, color='r')
    plt.xlabel('Index i', fontsize=12)
    plt.ylabel(r'Eigenvalue $\lambda_i$', fontsize=12)
    ax1.xaxis.tick_bottom()
    ax1.yaxis.tick_left()
    plt.axis([xmin, xmax, ymin, ymax])

    # timescales on the right y-axis:
    ax2 = plt.twinx()
    plt.ylabel(r'Implied timescale $t_i$', fontsize=12)
    ax2.yaxis.tick_right()

    plt.title('Eigenvalues', fontsize=16)
    plt.show()  



def draw_free_energy(centers, pi, T=300, output_file=None):
    """
    Draw the free energy difference landscape at given temperature T
    (in Kelvin, default: T = 300K). Uses the definition
    A = - 1/kT *log(pi) and sets lowest point to zero.
    """

    kT = (8.314472 * T) / 1000.  # in kJ per mol
    A = - kT * np.log(pi)
    A = A - A.min()

    interp_density_plot(centers, A, plot_pi_orig=False)
    plt.title('Free Energy Landscape (kJ/mol) @ %d K' % T)

    # export
    if output_file:
        plt.savefig(output_file)
        print('Figure saved as ', output_file)
    else:
        plt.show()

def draw_clusters(clusters, plotter=None, colormap_name="jet"):
    """
    Visualize clustered data and cluster membership in a new plot or with an existing axis object.
    """
    plotter = plotter or plt

    # use PCA to be able to visualize the data in two dimensions
    all_data = clusters.getOriginalData()
    pca = PCA(all_data)

    # for nicer visualization
    data_length = len(all_data)
    alpha = 1.0 / (math.sqrt(data_length))
    if alpha < 0.05: alpha = 0.05
    elif alpha > 0.75: alpha = 0.75
    cluster_ids = clusters.getClusterIDs()
    colormap = matplotlib.cm.get_cmap(colormap_name, len(cluster_ids) + 1)
    for index, cluster in enumerate(cluster_ids):
        datapoints = all_data[clusters._map == cluster,:]
        datapoints_transformed = pca.project(datapoints)
        plotter.scatter(datapoints_transformed[:,0], datapoints_transformed[:,1], color=colormap(index), alpha=0.5)
        
        
        
        
def get_graph2(msm, with_comm_classes=False):
    """Draw a graph representation of the chain using pygraphviz. For dominant pathway"""

    g = AGraph(strict=False, directed=True, name='Eff. prob. current, dominant pathway')

    g.graph_attr.update(size="7.75, 10.25")
    g.graph_attr.update(dpi="300")
    for i in range(msm.num_nodes):
        if(i in msm.a):	
            g.add_node(i, color = 'blue')
        elif(i in msm.b):
            g.add_node(i, color = 'green')
        elif(i in msm.dominant_pathway):
            g.add_node(i, color= 'red')
        else:
            g.add_node(i)

    if with_comm_classes:
        comm_classes = msm.communication_classes

        for (i, comm) in enumerate(comm_classes):
            g.add_subgraph(nbunch=comm, name='cluster%d' % i,
                           style='rounded, dotted',
                           color='lightgrey',
                           label='<<B>Communication class %d</B>>' % (i + 1))

    for from_node in range(msm.num_nodes):
        for to_node in get_adjacent_nodes(msm, from_node, discard_self=False):
            if msm.effective_probability_current[from_node, to_node] != 0.0:
                label = '%.2E' % msm.effective_probability_current[from_node, to_node]
                if([from_node, to_node] in msm.dominant_pathway_format):
                    g.add_edge(from_node, to_node, color='red' , label=label)                    
                else:
                	g.add_edge(from_node, to_node, label=label)

    return g


def draw_graph_dominant(msm, with_comm_classes=False):
    g = get_graph2(msm, with_comm_classes)

    g.layout(prog='dot')
    data = g.draw(format='png')
    return Image.open(BytesIO(data))

def draw_pcca_memberships(original_data, pcca, discrete_trajectory, colormap_name="jet"):
    """
    Visualize the result of PCCA+ as colored plot of the PCA. 
    """
    pca = PCA(original_data)

    cluster_ids = range(0, pcca.shape[1])
    colormap = matplotlib.cm.get_cmap(colormap_name, len(cluster_ids) + 1)

    membership = pcca > 0.5
    pcca_traj = np.where(membership[discrete_trajectory])[1]


    for index, cluster in enumerate(cluster_ids):
        datapoints = original_data[np.where(pcca_traj == cluster)]
        print('points in cluster ', cluster, ': ', len(datapoints))
        datapoints_transformed = pca.project(datapoints)
        plt.scatter(datapoints_transformed[:,0], datapoints_transformed[:,1], color=colormap(index), alpha=0.5)
    plt.title('pcca')
