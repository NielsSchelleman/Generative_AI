from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.sparse.csgraph import breadth_first_order as bfo
from scipy.special import logsumexp
import numpy as np
import csv


def compute_mi(cols):
    cols[:, 0] *= 2
    names, occurs = np.unique(np.sum(cols, axis=1), return_counts=True)
    namelen = len(names)
    # if only 2 marginals, variables must be independent so mutual information = 0
    if namelen == 2:
        return 0

    # Use small value alpha to substitute the 0-values
    if namelen == 3:
        occurs = np.insert(occurs.astype('float'), np.argmin(np.isin([0, 1, 2, 3], names)), 0.01)

    probs = np.log(occurs / sum(occurs))

    px0 = logsumexp([probs[0], probs[1]])
    px1 = logsumexp([probs[2], probs[3]])
    py0 = logsumexp([probs[0], probs[2]])
    py1 = logsumexp([probs[1], probs[3]])

    mi = np.exp(probs[0]) * (probs[0] - (px0 + py0)) + \
         np.exp(probs[1]) * (probs[1] - (px0 + py1)) + \
         np.exp(probs[2]) * (probs[2] - (px1 + py0)) + \
         np.exp(probs[3]) * (probs[3] - (px1 + py1))

    return np.log(mi)


def prob_col(col, val, alpha):
    return np.log(2*alpha*len(col[col == val]))-np.log(4*alpha*len(col))


def joint_prob_col(both_cols, val1, val2, alpha):
    return np.log(alpha*len(both_cols[np.sum(both_cols == (val1, val2), axis=1) == 2]))-np.log(4*alpha*len(both_cols))


def cond_prob_col(both_cols, vals, alpha):
    return joint_prob_col(both_cols, vals[0], vals[1], alpha)-prob_col(both_cols[:, 0], vals[0], alpha)


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
            [[compute_mi(self.data[:, [i, j]]) if j > i else 0 for j in range(self.cols)] for i in range(self.cols)])
        # invert the mutual information
        # to get the maximum spanning tree by calculating the minimum spanning tree of the inverse
        mi_matrix_inv = -mi_matrix
        tree = mst(mi_matrix_inv)
        # add connections to the tree in both directions
        tree = tree.toarray().astype(float)
        tree = tree.T + tree
        if not self.root:
            self.root = np.random.choice(range(0, self.cols))
        clt = bfo(tree, self.root)
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
                        [(0, 0), (0, 1), (1, 0), (1, 1)]]
                pmfs.append([[vals[0], vals[1]], [vals[2], vals[3]]])
        # multiply by 2 so every column adds up to 1
        return pmfs + np.log(2)

    # def logprob(self, x, exhaustive: bool = False):

    # def sample(self, nsamples: int)


if __name__ == "__main__":
    with open('nltcs.train.data', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        dataset = np.array(list(reader)).astype(int)
    mytree = BinaryCLT(dataset, 2)
    print(mytree.pmfs)
    print(mytree.tree)
