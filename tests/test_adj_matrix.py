
import pytest

from nbgwas.network import igraph_adj_matrix
import igraph as ig 
import numpy as np 

G = ig.Graph.Full(4)
weights = np.array([1, 2, 3, 4, 5, 6])
G.es['weight'] = weights


def test_adj_matrix():
    mat = igraph_adj_matrix(G, weighted=False)
    assert mat.shape == (4, 4)
    assert (mat.todense()[0] == np.array([0, 1, 1, 1])).all()
    assert (mat.todense()[1] == np.array([1, 0, 1, 1])).all()
    assert (mat.todense()[2] == np.array([1, 1, 0, 1])).all()
    assert (mat.todense()[3] == np.array([1, 1, 1, 0])).all()


def test_adj_matrix_weights(): 
    mat = igraph_adj_matrix(G, weighted='weight')
    print(weights)
    print(weights[0])
    assert mat.shape == (4, 4)
    assert (mat.todense()[0] == np.array([0, 1, 2, 3])).all()
    assert (mat.todense()[1] == np.array([1, 0, 4, 5])).all()
    assert (mat.todense()[2] == np.array([2, 4, 0, 6])).all()
    assert (mat.todense()[3] == np.array([3, 5, 6, 0])).all()