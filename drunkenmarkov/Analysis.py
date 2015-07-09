#!/usr/bin/python
import copy
import numpy as np

from .Util import get_adjacent_nodes, depth_first_search, gcd


class MarkovStateModel:
    def __init__(self, T, lagtime=1., k=5):
        self.T = T

        if not isinstance(T, np.ndarray):
            raise TypeError("T is no numpy array")
        if not self.is_transition_matrix:
            raise ValueError("T is not a transition matrix")
        # compute eigenvalues
        W, _ = np.linalg.eig(T)
        self.eigenv = sorted(W, reverse=True, key=lambda x: abs(x))[0:k]
        self.lagtime = lagtime
        # only compute the timescales if they are called explicitly.
        # This might not be necessary here, but can be useful at some
        # other point of the project.
        self._timescales = None
        self._stat_dist = None
        # also compute the communication classes lazily
        self._communication_classes = None

    @property
    def is_transition_matrix(self):
        """
        Check if the given matrix is a transition matrix (stochastic matrix)
        """
        # matrix should have exactly two dimensions
        if not len(self.T.shape) == 2:
            return False

        # matrix should be quadratic
        if not self.T.shape[0] == self.T.shape[1]:
            return False

        # all elements should be positive
        if not (self.T >= 0).all():
            return False

        # sum of each row should be 1.
        if not np.allclose([self.T[i, :].sum() for i in range(self.T.shape[0])], 1.):
            return False

        return True

    @property
    def is_reversible(self):
        """
        Check if the transition matrix fulfills the detailed balance condition
        pi(i) * T(i,j) == pi(j) * T(j, i)
        """
        pi = np.diag(self.stationary_distribution)
        return np.allclose(np.dot(pi, self.T), np.dot(np.transpose(self.T), pi))

    @property
    def num_nodes(self):
        """
        Return number of nodes
        """
        return self.T.shape[0]

    @property
    def is_connected(self):
        """
        Check if the given matrix is connected (=irreducible)
        """
        return len(self.communication_classes) == 1

    @property
    def stationary_distribution(self):
        """
        Compute the stationary distribution of a Markov Chain with transition matrix T
        """
        if self._stat_dist is None:
            if not self.is_connected:
                raise ValueError("T is not irreducible")
            else:
                left_eigenvalues, left_eigenvectors = np.linalg.eig(np.transpose(self.T))
                # Stationary distribution ist the eigenvector to the eigenvalue 1
                self._stat_dist = left_eigenvectors[:,np.where(np.isclose(left_eigenvalues, 1.))].reshape(self.T.shape[0])
                # Normalize stationary distribution s.t. the sum over all entries yields 1
                self._stat_dist = np.absolute(self._stat_dist / np.linalg.norm(self._stat_dist,1))
        return self._stat_dist

    @property
    def period(self):
        """
        Compute the period of a Markov Chain with transition matrix T
        (since only irreducible Markov Chains are considered all states have the same period,
        in particular the Markov Chain is either periodic or aperiodic)
        """
        if not self.is_connected:
            raise ValueError("T is not irreducible")
        else:
            d = 0
            n = self.T.shape[0]
            v = np.zeros(n)
            D = [0]
            E = []
            m = len(D)
            while (m > 0 and d != 1):
                i = D[0]
                D.remove(i)
                E.append(i)
                j = 0
                while (j < n):
                    if self.T[i,j] > 0:
                        if j in D or j in E:
                            d = gcd(d,v[i]+1-v[j])
                        else:
                            D.append(j)
                            v[j] = v[i] + 1
                    j = j+1
                m = len(D)
            
            if d == 1:
                print ("This Markov chain is aperiodic and converges to its stationary distribution")
            else:
                print ("This Markov chain is periodic with period" % d)
            return d

    @property
    def timescales(self):
        """
        Compute the time scales of a given transition matrix T.

        Keyword arguments:
        lagtime tau (default 1.0)
        """
        if self._timescales is None:
            # test for complex eigenvalues

            if np.any(np.imag(self.eigenv) > 0.):
                print('Complex eigenvalues found!')

            re_eigenv = np.real(self.eigenv)
            # continue with real part only
            self._timescales = np.zeros_like(re_eigenv)
            #find index corresponding to stationary distribution
            problematic_index = np.where(np.isclose(re_eigenv, 1., rtol=1e-20))
            #replace problematic value with one that behaves well
            re_eigenv[problematic_index] = 20.
            self._timescales = -self.lagtime / np.log(np.absolute(re_eigenv))
            #set stationary timescale to infinity
            self._timescales[problematic_index] = np.inf

        return self._timescales

    @property
    def communication_classes(self):
        """
        Finds and returns the communication classes from the transition matrix of this markov model.
        """
        if self._communication_classes is None:
            self._communication_classes = calculate_communication_classes(self.T)
        return self._communication_classes
            
    def pcca(self, m):
        """Use the pyemma pcca routine to calculate the matrix of membership probabilities
        for a detailed description of the function see http://pythonhosted.org/pyEMMA/api/generated/pyemma.msm.analysis.pcca.html
        """            
        from pyemma.msm.analysis import pcca_memberships as pyemma_pcca
        return pyemma_pcca(self.T, m)

    def reduce_matrix(self, m_pcca, min_pcca_memb_prob=0.5):
        """
        Reduce the transition matrix to the metastable states found by pcca+.
        T_{reduced} = \sum_{i,j} T_{i,j} P(i) / (\sum_i P(i) )
        """
        pc = self.pcca(m_pcca)
        membership = pc > min_pcca_memb_prob

        T_reduced = np.zeros((membership.shape[1], membership.shape[1]))

        # iterate over all states of reduced matrix
        for n in range(membership.shape[1]):
            for m in range(membership.shape[1]):
                # find all origin and destination states belonging to one pcca set (n,m)
                indx_n = np.where(membership[:, n]) 
                indx_m = np.where(membership[:, m])
                nominator = 0.
                denumerator = 0.
                # sum like stated in docstring
                for l in indx_n[0]:
                    denumerator += self.stationary_distribution[l]
                    for k in indx_m[0]:
                        nominator += self.T[l, k] * self.stationary_distribution[l]

                T_reduced[n, m] = nominator/denumerator

        return T_reduced
            

class TransitionPathTheory:
    def __init__(self, T, a, b):
        self.T = T
        self.a = a
        self.b = b

        if not isinstance(a, list) or not isinstance(b, list):
            raise TypeError("a and b must be lists")

        if len(set(a) - set(b)) != len(a):
            raise ValueError("sets a and b must be disjunct")

        if not isinstance(T, np.ndarray):
            raise TypeError("T is no numpy array")
        #if not MarkovStateModel.is_transition_matrix:
        #    raise ValueError("T is not a transition matrix")

        #self.lagtime = lagtime
        # only compute the timescales if they are called explicitly.
        # This might not be necessary here, but can be useful at some
        # other point of the project.
        self._fcom = None
        self._bcom = None
        self._probability_current = None
        self._stationary_distribution = None
        self._effective_probability_current = None
        self._flux = None
        self._transition_rate = None
        self._mean_first_passage_time = None
        self._dominant_pathway = None
    @property
    def fcom(self):
        """
        Compute the forward committor.
        """

        if self._fcom is None:
            from scipy.linalg import solve

            L = self.T - np.eye(len(self.T[0, :]))
            W = np.eye(len(L[0, :]))
            for i in range(len(W[:, 0])):
                if i not in self.a and i not in self.b:
                    W[i, :] = L[i, :]

            y = np.zeros_like(self.T[:, 0])
            for i in range(len(y)):
                if i in self.b:
                    y[i] = 1.

            self._fcom = solve(W, y)

        return self._fcom

    @property
    def bcom(self):
        """
        Compute the backward committor.
        """
        #print 'bcom ', self._bcom
        #print 'test ', not list(self._bcom)
        if self._bcom is None:
            from scipy.linalg import solve

            L = self.T - np.eye(len(self.T[0, :]))
            W = np.eye(len(L[0, :]))
            for i in range(len(W[:, 0])):
                if i not in self.a and i not in self.b:
                    W[i, :] = L[i, :]

            y = np.zeros_like(self.T[:, 0])
            for i in range(len(y)):
                if i in self.a:
                    y[i] = 1.

            self._bcom = solve(W, y)

        return self._bcom

    @property
    def probability_current(self):
        """
        Compute the probability current according to Script Lecture 4 p. 5. Note that vector operations are used for better performance.
        """
        if self._probability_current is None:
            self._probability_current = np.zeros_like(self.T)
            diagonal_zeros = -np.eye(self.T.shape[0])+ 1 
            MSM = MarkovStateModel(self.T)
            self._probability_current = np.kron(MSM.stationary_distribution * self.bcom, self.fcom).reshape(self.T.shape) * self.T * diagonal_zeros
        return self._probability_current

    @property
    def effective_probability_current(self):
        """
        Compute the effective probability current according to Script Lecture 4 p. 5. Note that still no vector operations are used for better performance. When do we have to use self.?
        """
        if self._effective_probability_current is None:
            self._effective_probability_current = np.zeros_like(self.T)
            for i in range(len(self.T[0])):
                for j in range(i, len(self.T[0])):
                    if(self.probability_current[i][j] > self.probability_current[j][i]):
                        self._effective_probability_current[i][j] = self.probability_current[i][j] - self.probability_current[j][i]
                        self._effective_probability_current[j][i] = 0.
                    else:
                        self._effective_probability_current[i][j] = 0.
                        self._effective_probability_current[j][i] = self.probability_current[j][i] - self.probability_current[i][j]
        return self._effective_probability_current

    @property
    def flux(self):
        """
        Compute the average total number of trajectories going from A to B per time unit. Note that vector operations are used for better performance. When do we have to use self.?
        """
        if self._flux is None:
            self._flux = 0.
            for i in self.a:
                    self._flux += sum(self.probability_current[i])
        return self._flux

    @property
    def transition_rate(self):
        """
        Compute the average fraction of reactive trajectories by the total number of trajectories that are going forward from state A. Note that vector operations are used for better performance. When do we have to use self.?
        """
        if self._transition_rate is None:
            MSM = MarkovStateModel(self.T)
            self._transition_rate = self.flux/(sum(MSM.stationary_distribution * self.bcom))
        return self._transition_rate

    @property
    def mean_first_passage_time(self):
        """
        Compute the mean first passage time. Note that vector operations are used for better performance. When do we have to use self.?
        """
        if self._mean_first_passage_time is None:
            self._mean_first_passage_time = 1/self.transition_rate
        return self._mean_first_passage_time

    @property
    def dominant_pathway(self):
        if self._dominant_pathway is None:
            paths = find_paths(self.effective_probability_current,self.a,self.b)
            current = get_current_of_paths(paths, self.effective_probability_current)
            pathnumber = range(len(paths))
            self._dominant_pathway = paths[dominant_pathway(current, pathnumber)]
        return self._dominant_pathway
        
# Dominant pathways are still missing. Test functions. We would need a fitting matrix for that.
    @property
    def num_nodes(self):
        """
        Return number of nodes
        """
        return self.T.shape[0]

    @property
    def dominant_pathway_format(self):
        """
        Makes dominant pathway to stepwise list
        """
        paths_format = [] 
        dominant = self.dominant_pathway
        for i in range(len(dominant)-1):
            paths_format.append([dominant[i],dominant[i+1]])        
        return paths_format


def find_paths(G, start, target): 
    """
    find all paths starting in list start ending in list target with direct matrix graph G initiate list of all paths
    """
    paths = []
    for a in start:
        paths.append([a])
    
    
    paths = expand(paths, G, target)
    
    
    return paths

def expand(paths, G, target): 
    """
    expand all paths in paths in all directions possible of G end in target
    """
    flag = True
    temp_paths = paths[:]
    for path in temp_paths:
        if path[-1] not in target and (G[path[-1]]).all == 0:#if it is not in target and there is an dead end 
            paths.remove(path)   #delete this path
        elif path[-1] not in target and not (G[path[-1]]).all == 0: #if it is not in target and not a dead end
            flag = False		#propagate one step in all possible directions
            paths.remove(path)
            for j in np.where(G[path[-1]]>0)[0]:
                path.append(j)
                temp_path = path[:]
                paths.append(temp_path)
                del path[-1]
    for i in paths[::-1]:
        if len(i) > G.shape[0]:
            paths.remove(i)
            flag = True
    if not flag: 		#if it was propagated start expand again
        paths = expand(paths, G, target)
                
            
    
    return paths		#if all dead ends are deleted and any path in target
    
def get_current_of_paths(paths, G): 
    """
    calculate out of all paths the effective current of all paths
    """
    effec_current_path = []
    for i in range(len(paths)):
        temp_list=[]
        for j in range(len(paths[i])-1):
            temp_list.append(G[paths[i][j]][paths[i][j+1]])
        
        effec_current_path.append(temp_list)
    
    return effec_current_path
    
def dominant_pathway(effec, pathnumber):
    mins = np.zeros(len(pathnumber))
    for i in range(len(pathnumber)):
        mins[i] = (min(effec[pathnumber[i]]))
    bottleneck_current = np.amax(mins)
    not_dominant = np.where(mins != bottleneck_current)
    if len(not_dominant[0]) < len(pathnumber) and len(not_dominant[0]) != 0:
        for i in not_dominant[0][::-1]:
            del pathnumber[i]
    for i in effec:
        if bottleneck_current in  i:
            i.remove(bottleneck_current)
    if [] in effec:
        return effec.index([])
    return dominant_pathway(effec, pathnumber)

def calculate_communication_classes(matrix):
    """Linear time algorithm to find the strongly connected components of
    a directed graph.
    
    Takes either a transition matrix or a count matrix.
    
    Pseudocode: http://en.wikipedia.org/wiki/Kosaraju%27s_algorithm#The_algorithm
    """

    # Let P be a directed graph and node_list be an empty stack.
    node_list = []
    communication_classes = []

    # While node_list does not contain all vertices:
    while(len(node_list) < matrix.shape[0]):
        # Choose an arbitrary vertex node not in node_list.
        node = [node for node in range(0, matrix.shape[0]) if node not in node_list][0]
        # Perform a depth-first search starting at node.
        # Each time that depth-first search finishes expanding a vertex u,
        # push u onto node_list.
        depth_first_search(matrix, node, node_list)

    # Reverse the directions of all arcs to obtain the transpose graph.
    reverse_graph = copy.deepcopy(np.transpose(matrix))

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

def get_connected_count_matrix(count_matrix):
    """
    Takes a count matrix and returns a count matrix where only the largest connected component remains.
    """
    communication_classes = calculate_communication_classes(count_matrix)
    if len(communication_classes) <= 1:
        return count_matrix
    communication_classes = sorted(communication_classes, key=lambda x: len(x), reverse=True)
    
    import numpy as np
    row_column_indices = sorted(np.array(communication_classes[0]))
    matrix = np.array(count_matrix)[row_column_indices, :][:, row_column_indices]
    return matrix

