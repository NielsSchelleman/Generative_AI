from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.sparse.csgraph import breadth_first_order as bfo
from scipy.special import logsumexp
import numpy as np
import itertools
import csv
# these two just added for easy of use when reading files
import os
import sys


def prob_col(col, val, alpha):
    return np.log(2*alpha+len(col[col == val]))-np.log(4*alpha+len(col))


def joint_prob_col(both_cols, val1, val2, alpha):
    return np.log(alpha+len(both_cols[np.sum(both_cols == (val1, val2), axis=1) == 2]))-np.log(4*alpha+len(both_cols))


def cond_prob_col(both_cols, vals, alpha):
    return joint_prob_col(both_cols, vals[0], vals[1], alpha)-prob_col(both_cols[:, 1], vals[1], alpha)


def compute_partial_mi(cols, i, j, alpha):
    logjoint = joint_prob_col(cols, i, j, alpha)
    logpx = prob_col(cols[:, 0], i, alpha)
    logpy = prob_col(cols[:, 1], j, alpha)
    return np.exp(logjoint)*(logjoint-(logpx+logpy))


def compute_mi(cols, alpha):
    names, _ = np.unique(np.sum(cols, axis=1), return_counts=True)
    namelen = len(names)
    # if only 2 marginals, variables must be independent so mutual information = 0
    if namelen == 2:
        return 0

    # calculate the mi
    mi = sum([compute_partial_mi(cols, i, j, alpha) for i in range(2) for j in range(2)])

    return np.log(mi)


def find_block(me, tree, indices):
    if tree[me] == -1:
        return 1
    elif tree[me] in indices:
        return 0
    else:
        return find_block(tree[me], tree, indices)


class BinaryCLT:
    def __init__(self, data, root: int = None, alpha: float = 0.01):
        self.cols = data.shape[1]
        self.data = data
        self.root = root
        self.alpha = alpha
        self.tree = self.gettree()
        self.pmfs = self.getlogparams()

    def gettree(self):
        # create the mutual information matrix
        mi_matrix = np.array(
            [[compute_mi(self.data[:, [i, j]], self.alpha) if j > i else 0 for j in range(self.cols)] for i in
             range(self.cols)])
        # invert the mutual information
        # to get the maximum spanning tree by calculating the minimum spanning tree of the inverse

        tree = mst(-mi_matrix)
        # add connections to the tree in both directions
        tree = tree.toarray().astype(float)
        tree = tree.T + tree
        if not type(self.root) == int:
            self.root = np.random.choice(range(0, self.cols))

        clt = bfo(tree, i_start=self.root)
        clt = clt[1]
        clt[clt == -9999] = -1
        return clt

    def getlogparams(self):
        pmfs = []
        for i in range(self.cols):

            if self.tree[i] == -1:
                vals = [prob_col(self.data[:, i], j, self.alpha) for j in range(2)]
                pmfs.append([[vals[0], vals[1]], [vals[0], vals[1]]])
            else:
                vals = [cond_prob_col(self.data[:, [i, self.tree[i]]], j, self.alpha) for j in
                        [(0, 0), (1, 0), (0, 1), (1, 1)]]
                pmfs.append([[vals[0], vals[1]], [vals[2], vals[3]]])

        return np.array(pmfs)

    def logprob(self, x, exhaustive=False):
        if exhaustive:
            a, b = np.unique(self.data, axis=0, return_counts=True)
            total = sum(b)
            lookup = dict(zip(list(map(tuple, a)), list(b)))

            lookup = np.array(list(lookup.items()), dtype='object')

            marginals = []
            for query in x:
                nans = np.isnan(query)
                pos = np.array(list(itertools.product([0, 1], repeat=sum(nans))))
                insertions = np.argwhere(~nans)
                for i in insertions:
                    pos = np.insert(pos, i, query[i], axis=1)
                pos = list(map(tuple, pos))

                marginals.append(np.log(sum(lookup[:, 1] * [item in pos for item in lookup[:, 0]]) / total))

        else:
            # depths = [find_depth(point,self.tree) for point in range(len(self.tree))]
            marginals = []
            for query in x:
                insertions = np.argwhere(~np.isnan(query)).flatten()
                need_to_know = np.array(
                    [find_block(node, self.tree, insertions) for node in range(len(self.tree))]).astype('bool')

                needed_nodes = set(np.argwhere(need_to_know).flatten())
                not_needed = set(np.argwhere(~need_to_know).flatten())
                leaves = set(range(16)) - set(self.tree)

                l2 = leaves.union(set(insertions)) - not_needed

                non_leaves = needed_nodes - l2
                children = {node: set(np.argwhere(self.tree == node).flatten()) for node in non_leaves}
                l2 = list(l2)

                temp_marginals = {}

                while len(l2) > 0:
                    point = l2.pop(0)

                    if (self.tree[point] != -1) and (self.tree[point] not in l2):
                        l2.append(self.tree[point])
                    if point in insertions:
                        if query[point] == 0:
                            temp_marginals[point] = self.pmfs[point][0]
                        else:
                            temp_marginals[point] = self.pmfs[point][1]
                    elif point in leaves:
                        temp_marginals[point] = np.array([0, 0]).astype(float)
                    elif children[point] == children[point].intersection(set(temp_marginals.keys())):

                        child_margins = np.sum(np.array([temp_marginals[p] for p in children[point]]), axis=0)

                        temp_marginals[point] = np.array([logsumexp(self.pmfs[point][0] + child_margins),
                                                          logsumexp(self.pmfs[point][1] + child_margins)])

                    else:
                        l2.append(point)

                marginals.append(temp_marginals[self.root][0])

        return marginals

    # Function that takes a parent node and progresses recursively through a tree drawing samples for each child
    def draw(self, parent: int, sample: list):

        children = np.argwhere(self.tree == parent)

        for child in children:
            random = np.random.rand()
            if random <= np.exp(self.pmfs)[parent][sample[parent]][0]:
                sample[child[0]] = 1
            self.draw(parent=child[0], sample=sample)

        return sample

    # Function that draws an amount of i.i.d samples
    def sample(self, nsamples: int):
        samples = []

        for i in range(nsamples):
            sample = [0 for _ in range(len(self.tree))]
            # Initialize for the root node
            random = np.random.rand()

            if random <= np.exp(self.pmfs)[self.root][0][0]:
                sample[self.root] = 1
            parent = self.root

            self.draw(parent=parent, sample=sample)
            samples.append(sample)
        return np.array(samples)


if __name__ == "__main__":
    with open(os.path.join(sys.path[0], 'nltcs.train.data'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        dataset = np.array(list(reader)).astype(int)
    with open(os.path.join(sys.path[0], 'nltcs_marginals.data'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        margs = np.array(list(reader)).astype(float)
    mytree = BinaryCLT(dataset, 0)

    print(mytree.logprob(x=margs[0:10]))
    print(mytree.logprob(x=margs[0:10], exhaustive=True))
    print(mytree.sample(100))

