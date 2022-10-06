from array import array
from unittest import skip
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.sparse.csgraph import breadth_first_order as bfo
from scipy.special import logsumexp
import numpy as np
import csv
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
    names, occurs = np.unique(np.sum(cols, axis=1), return_counts=True)
    namelen = len(names)
    # if only 2 marginals, variables must be independent so mutual information = 0
    if namelen == 2:
        return 0
    # need to add a 0 as it is not counted when there are none
    if namelen == 3:
        occurs = np.insert(occurs.astype('float'), np.argmin(np.isin([0, 1, 2, 3], names)), 0)

    # calculate the mi
    mi = sum([compute_partial_mi(cols, i, j, alpha) for i in range(2) for j in range(2)])

    return np.log(mi)



def sum_out(matrix, pair):
    #Dependant on which value is nan sum out the probability matrix
    if np.isnan(pair[1]):
        return matrix[0] + matrix[1]
    elif np.isnan(pair[0]):
        return matrix[:,0] + matrix[:,1]
    else:
        return matrix

def summed_pmfs(pmfs, pairs):
    #Build a tree out of all of the probability matrices
    sum_pmfs = []
    for i in range(len(pairs)):
        sum_pmfs.append(sum_out(pmfs[i], pairs[i]))
    return np.array(sum_pmfs)


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
            [[compute_mi(self.data[:, [i, j]], self.alpha) if j > i else 0 for j in range(self.cols)] for i in range(self.cols)])
        # invert the mutual information
        # to get the maximum spanning tree by calculating the minimum spanning tree of the inverse

        tree = mst(-mi_matrix)
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
                        [(0, 0), (1, 0), (0, 1), (1, 1)]]
                pmfs.append([[vals[0], vals[1]], [vals[2], vals[3]]])

        return np.array(pmfs)

    def logprob(self, x=np.array([]), exhaustive: bool = True):
        #The answers for this specific example are in HW2 therefore im using this pmfs and tree as data
        lp = []
        self.pmfs = np.array([[[ -1.204 , -0.357] , [-1.204 , -0.357]] ,
                    [[ -1.609 ,-0.223] , [ -0.511 , -0.916]] ,
                    [[ -0.916 , -0.511] , [ -2.303 , -0.105]] ,
                    [[ -0.223 , -1.609] , [ -0.693 , -0.693]] ,
                    [[ -0.105 , -2.303] , [ -0.916 , -0.511]]])

        self.tree = [ -1 , 0 , 4 , 4 , 0 ]
        
        if exhaustive:    
            for sample in x:

                #make [child value, parent value, index in pmfs]
                pairs = np.array([[sample[i], sample[self.tree[i]], i] for i in range(len(sample))], dtype=np.float64)

                #build a tree where nan values are summed out 
                new_tree = summed_pmfs(np.exp(self.pmfs), pairs[:,:2])
                print(new_tree)
                probs = []

                #for every pair take log of the correct value in the tree
                for pair in pairs:
                    #Probabilities of a fully nan matrix sum up to 2
                    if np.isnan(pair[0]) and np.isnan(pair[1]):
                        probs.append(np.log(2))
                    #If child is Nan probabilities sum up to 1 (see tree)
                    elif np.isnan(pair[0]):
                        probs.append(np.log(1))
                    #If parent is Nan use tree to get value
                    elif np.isnan(pair[1]):
                        probs.append(np.log(new_tree[int(pair[2])][int(pair[0])]))
                    #If neither is nan use full matrix 
                    else:
                        probs.append(np.log(new_tree[int(pair[2])][int(pair[1])][int(pair[0])]))
                    print(probs)
                lp.append(np.sum(probs))
            return lp
        
        
        elif ~exhaustive:
             



            return None


    # def sample(self, nsamples: int)


if __name__ == "__main__":
    with open(os.path.join(sys.path[0], 'nltcs.train.data'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        dataset = np.array(list(reader)).astype(int)
    with open(os.path.join(sys.path[0], 'nltcs_marginals.data'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        marginals = np.array(list(reader))
    mytree = BinaryCLT(dataset, 2)
    # print(mytree.pmfs)
    # print(mytree.tree)
    print(mytree.logprob(x=np.array([[np.nan,0,np.nan,np.nan,1]])))
