import pytest 
from nbgwas.network import igraph_adj_matrix
import igraph as ig 
import numpy as np 

G = ig.Graph.Full(4)
weights = np.random.rand(len(G.es))
G.es['weight'] = weights

def test_adj_matrix(): 
    mat = igraph_adj_matrix(G, weighted=False) 

    print(mat.todense()) 

    assert False

def test_adj_matrix_weights(): 
    mat = igraph_adj_matrix(G, weighted='weight') 


    print(mat.todense())

    assert False