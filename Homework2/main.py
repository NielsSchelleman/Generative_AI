from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.sparse.csgraph import breadth_first_order as bfo
from scipy.special import logsumexp
import numpy as np
import itertools
import csv
import time
# these two just added for easy of use when reading files
import os
import sys


def prob_col(col, val, alpha):
    """
    Calcs P(X=x)
    :param col: data column
    :param val: 0 or 1
    :param alpha: laplace correction
    :return: P(X=x)
    """
    return np.log(2*alpha+len(col[col == val]))-np.log(4*alpha+len(col))


def joint_prob_col(both_cols, val1, val2, alpha):
    """
    Calcs P(X=x,Y=y)
    :param both_cols: 2 data columns in form (n,2)
    :param val1: value of column 1
    :param val2: value of column 2
    :param alpha: laplace correction
    :return: P(X=x,Y=y)
    """
    return np.log(alpha+len(both_cols[np.sum(both_cols == (val1, val2), axis=1) == 2]))-np.log(4*alpha+len(both_cols))


def cond_prob_col(both_cols, vals, alpha):
    """
    Calcs P(X=x|Y=y)
    :param both_cols: 2 data columns in form (n,2)
    :param vals: values of (col1,col2)
    :param alpha: laplace correction
    :return: P(X=x|Y=y)
    """
    return joint_prob_col(both_cols, vals[0], vals[1], alpha)-prob_col(both_cols[:, 1], vals[1], alpha)


def compute_partial_mi(cols, i, j, alpha):
    """
    computes the MI for 2 columns and 2 given values
    :param cols: 2 data columns in form (n,2)
    :param i: value of column 1
    :param j: value of column 2
    :param alpha: laplace correction
    :return: a value for the MI
    """
    logjoint = joint_prob_col(cols, i, j, alpha)
    logpx = prob_col(cols[:, 0], i, alpha)
    logpy = prob_col(cols[:, 1], j, alpha)
    return np.exp(logjoint)*(logjoint-(logpx+logpy))


def compute_mi(cols, alpha=0.01):
    """
    computes the Mutual information for a dataset
    :param cols: 2 columns
    :param alpha: laplace correction
    :return: the log of the mutual information
    """
    names, _ = np.unique(np.sum(cols, axis=1), return_counts=True)
    namelen = len(names)
    # if only 2 marginals, variables must be independent so mutual information = 0
    if namelen == 2:
        return 0

    # calculate the mi for all combinations of values
    mi = sum([compute_partial_mi(cols, i, j, alpha) for i in range(2) for j in range(2)])

    return np.log(mi)


class BinaryCLT:
    # all data imports need to be of type int, not floats (except for the marginals)
    def __init__(self, data, root: int = None, alpha: float = 0.01):
        self.cols = data.shape[1]
        self.data = data
        self.root = root
        self.alpha = alpha
        self.tree = self.get_tree()
        self.pmfs = self.get_log_params()

    def get_tree(self):
        """
        Creates the tree
        :return: 1D array of the tree where tree[0] gives the parent of node at position 0
        """
        # create the mutual information matrix
        mi_matrix = np.array(
            [[compute_mi(self.data[:, [i, j]], self.alpha) if j > i else 0 for j in range(self.cols)] for i in
             range(self.cols)])

        # use the inverse of the mutual information
        # to get the maximum spanning tree by calculating the minimum spanning tree of the inverse
        tree = mst(-mi_matrix)
        # add connections to the tree in both directions
        tree = tree.toarray().astype(float)
        tree = tree.T + tree
        # assign random root if not given
        if not type(self.root) == int:
            self.root = np.random.choice(range(0, self.cols))
        # get the breadth first ordering of the adjacency matrix for the maximum spanning tree
        clt = bfo(tree, i_start=self.root)
        clt = clt[1]
        # set the value of parent of the root to be -1 instead of -9999
        clt[clt == -9999] = -1
        return clt

    def get_log_params(self):
        """
        Computes the CPTs given the tree
        :return:
        """
        pmfs = []
        for i in range(self.cols):
            # root has no parent to take into account so is treated differently
            if self.tree[i] == -1:
                vals = [prob_col(self.data[:, i], j, self.alpha) for j in range(2)]
                pmfs.append([[vals[0], vals[1]], [vals[0], vals[1]]])
            else:
                vals = [cond_prob_col(self.data[:, [i, self.tree[i]]], j, self.alpha) for j in
                        [(0, 0), (1, 0), (0, 1), (1, 1)]]
                pmfs.append([[vals[0], vals[1]], [vals[2], vals[3]]])

        return np.array(pmfs)

    def log_prob(self, x, exhaustive=False):
        """
        Calculates the log probabilities of morginals
        :param x: a 2D array where each row is a marginal
        :param exhaustive: Whether to do the exhaustive method or not
        :return: List of log probabilities
        """
        if exhaustive:
            # check for each combination of RV's how often it occurs and put it into an array
            a, b = np.unique(self.data, axis=0, return_counts=True)
            total = np.log(sum(b))
            lookup = dict(zip(list(map(tuple, a)), list(np.log(b) - total)))
            lookup = np.array(list(lookup.items()), dtype='object')

            marginals = []
            for query in x:
                nans = np.isnan(query)
                # create all possible combinations for the unknown variables and add the known variables to the rows
                pos = np.array(list(itertools.product([0, 1], repeat=sum(nans))))
                insertions = np.argwhere(~nans).flatten()
                for i in insertions:
                    pos = np.insert(pos, i, query[i], axis=1)
                pos = list(map(tuple, pos))
                # add all combinations of RV's which have a log probability and can be created from the marginal
                marginals.append([logsumexp(lookup[[item in pos for item in lookup[:, 0]]][:, 1].astype(float))])

        else:
            # Do message passing
            marginals = []

            for query in x:
                insertions = np.argwhere(~np.isnan(query)).flatten()
                leaves = set(range(self.cols)) - set(self.tree)
                children = {node: set(np.argwhere(self.tree == node).flatten()) for node in set(range(self.cols))}
                l2 = list(leaves)

                temp_marginals = {}

                while len(l2) > 0:
                    point = l2.pop(0)
                    if (self.tree[point] != -1) and (self.tree[point] not in l2):
                        l2.append(self.tree[point])

                    # if we have already gotten messages from all children
                    if children[point] == children[point].intersection(set(temp_marginals.keys())):

                        child_margins = np.sum(np.array([temp_marginals[p] for p in children[point]]), axis=0)
                        # if no children, set this to not influence the outcome.
                        if len(child_margins.shape) < 1:
                            child_margins = np.array([0, 0])

                        if point in insertions:
                            qp = int(query[point])
                            temp_marginals[point] = np.array([self.pmfs[point][0, qp] + child_margins[qp],
                                                              self.pmfs[point][1, qp] + child_margins[qp]])
                        else:
                            temp_marginals[point] = np.array([logsumexp(self.pmfs[point][0] + child_margins),
                                                              logsumexp(self.pmfs[point][1] + child_margins)])
                    else:
                        l2.append(point)
                marginals.append([temp_marginals[self.root][0]])
        return np.array(marginals)

    def draw(self, parent: int, sample: list):
        """
        Sample recursively
        :param parent: some node with children
        :param sample: the current state of the sample
        :return: the sample
        """

        children = np.argwhere(self.tree == parent)
        for child in children:
            random = np.random.rand()
            # if random value larger than the p(X=0) then assing x=1.
            if random > np.exp(self.pmfs)[child[0]][sample[parent]][0]:
                sample[child[0]] = 1
            # do the same thing for all children
            sample = self.draw(parent=child[0], sample=sample)

        return sample

    def sample(self, nsamples: int):
        """
        draws some amount of i.i.d samples
        :param nsamples: number of samples
        :return: the samples as a 2D array of (rows,cols)
        """
        samples = []

        for i in range(nsamples):
            sample = [0 for _ in range(len(self.tree))]
            # Initialize for the root node
            random = np.random.rand()
            # root has no parent so treated as special case
            if random > np.exp(self.pmfs)[self.root][0][0]:
                sample[self.root] = 1
            parent = self.root
            # recursively construct sample
            sample = self.draw(parent=parent, sample=sample)
            samples.append(sample)
        return np.array(samples)

    def compute_ll(self, dataset):
        """
        Computes the average log likelihoods
        :param dataset: the data to analyze
        :return: the average of the log likelihoods of the data
        """
        likelihoods = []
        for row in dataset:
            pr = 0
            queue = list(np.argwhere(mytree.tree == -1).flatten())
            while len(queue) > 0:
                pos = queue.pop(0)
                parent = self.tree[pos]
                if parent == -1:
                    pr += self.pmfs[pos][0, row[pos]]

                else:
                    pr += self.pmfs[pos][row[parent], row[pos]]
                queue += list(np.argwhere(mytree.tree == pos).flatten())
            likelihoods.append(pr)
        # calculate the average in the log-domain
        return logsumexp(likelihoods)-np.log(len(likelihoods))


if __name__ == "__main__":
    with open(os.path.join(sys.path[0], 'nltcs.train.data'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        dataset = np.array(list(reader)).astype(int)
    with open(os.path.join(sys.path[0], 'nltcs_marginals.data'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        margs = np.array(list(reader)).astype(float)
    mytree = BinaryCLT(dataset, 0)

    start = time.time()
    print(mytree.log_prob(x=margs[0:10]))
    t1 = time.time()
    print(mytree.log_prob(x=margs[0:10], exhaustive=True))
    t2 = time.time()

    print(t1-start)
    print(t2-start)

    print(mytree.compute_ll(dataset))
    with open(os.path.join(sys.path[0], 'nltcs.test.data'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data_test = np.array(list(reader)).astype(int)
    print(mytree.compute_ll(data_test))

    print(mytree.compute_ll(mytree.sample(1000)))

    # s =  mytree.log_prob(x=np.array(list(itertools.product([0,1],repeat=16))))
    # print(logsumexp(s))
    # This adds up to 1

    # print(np.exp(mytree.log_prob(x=[[np.nan]*16],exhaustive=False)))
    # this also adds up to 1
