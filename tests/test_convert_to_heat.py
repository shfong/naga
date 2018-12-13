import pytest 
import numpy as np 
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nbgwas import Nbgwas 
import igraph as ig 
import pandas as pd 

G = ig.Graph.Full(5) 
G.vs['name'] = list('ABCDE') 
p_values = pd.DataFrame(
    [
        ['A', 'B', 'C', 'D', 'E'],
        [0.5, 0.01, 0.0001, 0.2, 1],
    ], 
    index=['Genes', 'P-value'], 
    columns = [1,2,3,4,5]
).T

g = Nbgwas(
    network=G, 
    gene_level_summary=p_values,
    gene_col='Genes', 
    gene_pval_col='P-value'
)


def test_convert_to_heat(): 
    g.genes.convert_to_heat(threshold=0.05, name='heat1')

    assert_array_almost_equal(
        g.genes.table.loc[[0, 1, 2, 3, 4], 'heat1'].values.ravel(),
        np.array([np.nan, 0, 1, 1, 0])
    )

    g.genes.convert_to_heat(threshold=0.05, name='heat2')
    
    assert_array_almost_equal(
        g.genes.table.loc[[0, 1, 2, 3, 4], 'heat2'].values.ravel(),
        np.array([np.nan, 0, 1, 1, 0])
    )
    assert_array_equal(
        g.genes.table.loc[[1, 2, 3, 4, 5], 'Genes'].values.ravel(),
        np.array(['A', 'B', 'C', 'D', 'E'])
    )


def test_convert_to_heat_dup_node_nmaes(): 
    G.vs['name'] = list('ABADE') 

    g = Nbgwas(
        network=G, 
        gene_level_summary=p_values,
        gene_col='Genes', 
        gene_pval_col='P-value'
    )

    g.genes.convert_to_heat(threshold=0.05, name='heat1', fill_missing=-1)

    assert_array_almost_equal(
        g.genes.table.loc[[1, 2, 3, 4, 5], 'heat1'].values.ravel(),
        np.array([0, 1, 1, 0, 0])
    )

    g.genes.convert_to_heat(threshold=0.05, name='heat2')
    
    assert_array_almost_equal(
        g.genes.table.loc[[1, 2, 3, 4], 'heat2'].values.ravel(),
        np.array([0, 1, 1, 0])
    )

    assert_array_equal(
        g.genes.table.loc[[1, 2, 3, 4, 5], 'Genes'].values.ravel(),
        np.array(['A', 'B', 'C', 'D', 'E'])
    )
