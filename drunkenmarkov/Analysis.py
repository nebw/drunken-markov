#!/usr/bin/python
import copy
import numpy as np

from .Util import get_adjacent_nodes, depth_first_search, gcd


class MarkovStateModel:
    def __init__(self, T, lagtime=1.):
        self.T = T

        if not isinstance(T, np.ndarray):
            raise TypeError("T is no numpy array")
        if not self.is_transition_matrix:
            raise ValueError("T is not a transition matrix")

        # compute eigenvalues
        W, _ = np.linalg.eig(T)
        self.eigenv = sorted(W, reverse=True)
        self.lagtime = lagtime
        # only compute the timescales if they are called explicitly.
        # This might not be necessary here, but can be useful at some
        # other point of the project.
        self._timescales = None

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
        Compute the stationary distribution for the Markov Chain
        """
        if not self.is_connected:
            raise ValueError("T is not irreducible")
        else:
            ewp = np.linalg.eig(np.transpose(self.T))
            eigenvalues = ewp[0]
            eigenvectors = ewp[1]
            # Index b des Eigenwertes 1 finden:
            b = np.where(np.isclose(eigenvalues, 1.))
            stat_dist = np.zeros(len(self.T[0, :]))
            for i in range(0, len(self.T[0, :])):
                # im i-ten Array des Eigenvektor-Arrays den b-ten Eintrag auslesen
                # und in die i-te Zeile der stationaeren Verteilung stat_dist
                # schreiben
                stat_dist[i] = eigenvectors[i][b]
            # stationaere Verteilung normieren und positiv machen
            stat_dist_norm = np.linalg.norm(stat_dist,1)
            for i in range(0, len(self.T[0, :])):
                stat_dist[i] /= stat_dist_norm
                stat_dist[i] = np.absolute(stat_dist[i])
            return stat_dist

  

    @property
    def period(self):
        """
        Compute the period of a Markov Chain with transition matrix T
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
            ev_is_cmplx = np.where(np.imag(self.eigenv) > 0.)
            if sum(ev_is_cmplx) > 0:
                print('Complex eigenvalues found!')

            re_eigenv = np.real(self.eigenv)
            # continue with real part only
            self._timescales = np.zeros_like(re_eigenv)

            # take care of devision by zero (EV = 1) and compute time scales
            # for loop to be replaced by something faster
            for ii in range(len(re_eigenv)):
                if (re_eigenv[ii] - 1.)**2 < 1e-5:
                    self._timescales[ii] = np.inf
                else:
                    self._timescales[ii] = -self.lagtime / np.log(abs(re_eigenv[ii]))

        return self._timescales

    @property
    def communication_classes(self):
        """Linear time algorithm to find the strongly connected components of
        a directed graph.

        Pseudocode: http://en.wikipedia.org/wiki/Kosaraju%27s_algorithm#The_algorithm
        """

        # Let P be a directed graph and node_list be an empty stack.
        node_list = []
        communication_classes = []

        # While node_list does not contain all vertices:
        while(len(node_list) < self.num_nodes):
            # Choose an arbitrary vertex node not in node_list.
            node = [node for node in range(0, self.num_nodes) if node not in node_list][0]
            # Perform a depth-first search starting at node.
            # Each time that depth-first search finishes expanding a vertex u,
            # push u onto node_list.
            depth_first_search(self.T, node, node_list)

        # Reverse the directions of all arcs to obtain the transpose graph.
        reverse_graph = copy.deepcopy(np.transpose(self.T))

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

    def pcca(self, m):
        """Use the pyemma pcca routine to calculate the matrix of membership probabilities
        for a detailed description of the function see http://pythonhosted.org/pyEMMA/api/generated/pyemma.msm.analysis.pcca.html
        """			
        from pyemma.msm.analysis import pcca as pyemma_pcca
        return pyemma_pcca(self.T, m)
			

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

    @property
    def fcom(self):
        """
        Compute the forward committor.
        """

        if not self._fcom:
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

        if not self._bcom:
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
        if not self._probability_current:
            self._probability_current = np.zeros_like(self.T)
            diagonal_zeros = -np.eye(self.T.shape[0])+ 1 

            MSM = MarkovStateModel(self.T)
            self._probability_current = np.kron(MSM.stationary_distribution * self._bcom , self._fcom).reshape(self.T.shape) * self.T * diagonal_zeros #fehlen hier nicht ein paar unterstriche?
        return self._probability_current

    @property
    def effective_probability_current(self):
        """
        Compute the effective probability current according to Script Lecture 4 p. 5. Note that still no vector operations are used for better performance. When do we have to use self.?
        """
        if not self._effective_probability_current:
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
        if not self._flux:
            self._flux = 0.
            for i in self.a:
                    self._flux += sum(self._probability_current[i])
        return self._flux

    @property
    def transition_rate(self):
        """
        Compute the average fraction of reactive trajectories by the total number of trajectories that are going forward from state A. Note that vector operations are used for better performance. When do we have to use self.?
        """
        if not self._transition_rate:
            self._transition_rate = self._flux/(sum(self._stationary_distribution * self._bcom))
        return self._transition_rate

    @property
    def mean_first_passage_time(self):
        """
        Compute the mean first passage time. Note that vector operations are used for better performance. When do we have to use self.?
        """
        if not self._mean_first_passage_time:
            self._mean_first_passage_time = 1/self._transition_rate
        return self._mean_first_passage_time
# Dominant pathways are still missing. Test functions. We would need a fitting matrix for that.
