import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pygraphviz as pgv

# draw random smaples from a given discrete distribution
def tower_sample(distribution):
    cdf = np.cumsum(distribution)
    rnd = np.random.rand() * cdf[-1]
    ind = (cdf > rnd)
    idx = np.where(ind == True)
    return np.min(idx)

# compute a transition count matrix
def count_transitions(chain):
    n_markov_states = np.max(chain)+1
    count_matrix = np.zeros((n_markov_states,n_markov_states), dtype=np.intc)
    for i in xrange(1, chain.shape[0]):
        count_matrix[chain[i-1], chain[i]] += 1
    return count_matrix

def plot_transitions(transitions):
    plt.title('Transitions')
    plt.pcolor(transitions)
    set_cmap('gray')
    plt.colorbar()
    plt.show()
    
def plot_chain(chain):
    plt.title('State changes')
    plt.plot(chain + 1, '--o')
    
def get_adjacent_nodes(P, node, discard_self=True):
    adjacent_nodes = set(np.where(P[node, :] > 0.)[0].tolist())
    if discard_self:
        adjacent_nodes.discard(node)
    return adjacent_nodes
    
def depth_first_search(P, node, node_list, visited_nodes=None):
    visited_nodes = visited_nodes or set()
    visited_nodes.add(node)

    for recursive_node in get_adjacent_nodes(P, node):
        if recursive_node not in visited_nodes:
            depth_first_search(P, recursive_node, node_list, visited_nodes)   

    if node not in node_list:
        node_list.append(node)

class DiscreteMarkovChain:
    P = None
    
    def __init__(self, P):
        self.P = P
        
    @classmethod
    def with_random_transition_matrix(cls, num_states, sparsity = 0.7):
        """Generate a random Markov Chain."""
        P = np.random.rand(num_states, num_states)
        
        D = np.random.rand(num_states, num_states)
        P[D < sparsity] = 0.
        
        for rowi in range(0, P.shape[0]):
            if P[rowi, :].sum() == 0:
                P[rowi, rowi] = 1
                
            row_sum = P[rowi, :].sum()
            P[rowi, :] /= row_sum
            
        return cls(P)
    
    def get_num_nodes(self):
        return len(self.P[0, :])
    
    def plot_transition_matrix(self):
        plt.title('Transition Matrix')
        plt.pcolor(self.P)
        set_cmap('gray')
        plt.colorbar() 
        
    def plot_transition_image(self):
        plt.title('Transition Image')
        plt.imshow(self.P)
        set_cmap('jet')
        
    def plot_graph(self, with_comm_classes=True):
        """Draw a graph representation of the chain using pygraphviz."""
        
        g = pgv.AGraph(strict=False,directed=True)
        
        g.graph_attr.update(size="7.75, 10.25") 
        g.graph_attr.update(dpi="300") 
        
        g.add_nodes_from(range(self.get_num_nodes()))
                     
        if with_comm_classes:
            comm_classes = self.kosaraju()
            
            for (i, comm) in enumerate(comm_classes):
                cg = g.add_subgraph(nbunch=comm, name='cluster%d' % i,
                                    style='rounded, dotted',
                                    color='lightgrey',
                                    label='<<B>Communication class %d</B>>' % (i + 1))
                
        for from_node in range(self.get_num_nodes()):
            for to_node in get_adjacent_nodes(self.P, from_node, discard_self=False):
                label = '%.2f' % self.P[from_node, to_node]   
                edge = g.add_edge(from_node, to_node, label=label)
                
        g.layout(prog='dot')
        g.draw('model.png')
        
        img = mpimg.imread('model.png')
        fig = plt.figure(figsize=(10.25, 7.75), dpi=300)
        plt.imshow(img)     
        plt.axis('off')
        
    def evolve(self, start, length):
        """Evolve the chain starting from state `start` for `length` steps."""
        
        chain = np.zeros(length, dtype=np.intc)
        transitions = np.zeros(self.P.shape, dtype=np.intc)
        chain[0] = start
        for i in range(1, length):
            chain[i] = tower_sample(self.P[chain[i-1]])
            transitions[chain[i-1], chain[i]] += 1
        return chain, transitions

    def kosaraju(self):
        """Linear time algorithm to find the strongly connected components of a directed graph.
        
        Pseudocode: http://en.wikipedia.org/wiki/Kosaraju%27s_algorithm#The_algorithm
        """
        
        # Let P be a directed graph and node_list be an empty stack.
        node_list = []
        communication_classes = []
        
        # While node_list does not contain all vertices:
        while(len(node_list) < self.get_num_nodes()):
            # Choose an arbitrary vertex node not in node_list. 
            node = [node for node in range(0, self.get_num_nodes()) if node not in node_list][0]
            # Perform a depth-first search starting at node. 
            # Each time that depth-first search finishes expanding a vertex u, 
            # push u onto node_list.
            depth_first_search(self.P, node, node_list)
            
        # Reverse the directions of all arcs to obtain the transpose graph.
        reverse_graph = copy.deepcopy(np.transpose(self.P))
          
        # While node_list is nonempty: 
        while(len(node_list) > 0):
            # Pop the top vertex node from node_list.
            node = node_list.pop()
            
            # Perform a depth-first search starting at node in the transpose graph. 
            # The set of visited vertices will give the strongly connected component 
            # containing node.
            comm_class = []
            depth_first_search(reverse_graph, node, comm_class)        
            communication_classes.append(comm_class)
        
            # remove all these vertices from the graph and the stack node_list.
            for x in comm_class:
                reverse_graph[x, :] = 0.
                reverse_graph[:, x] = 0.
                
            node_list = [x for x in node_list if x not in comm_class]

        return communication_classes
