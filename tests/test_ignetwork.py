import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from nbgwas.network import NxNetwork, IgNetwork 
import igraph as ig

G = ig.Graph.Full(4)
G.vs['name'] = ['1', '10', '100', '1000']

def test_IgNetwork_construction(): 
    IgNetwork()

# def test_ig_set_node_attributes(): 
#     dG = IgNetwork(network=G.copy())



def test_ig_get_node_attributes(): 
    dG = IgNetwork(network=G.copy())

    expected = {
        0: {"name": '1'}, 
        1: {"name": '10'}, 
        2: {"name": "100"}, 
        3: {"name": "1000"}
    }

    assert dG.get_node_attributes() == expected

def test_ig_set_node_attributes(): 
    dG = IgNetwork(network=G.copy())
    
    new_attr = {
        "new_attr": {
            0: 'a', 
            1: 'b', 
            2: 'c', 
            3: 'd'
        }
    }

    expected = {
        0: {"name": '1', "new_attr": "a"}, 
        1: {"name": '10', "new_attr": "b"}, 
        2: {"name": "100", "new_attr": "c"}, 
        3: {"name": "1000", "new_attr": "d"}
    }

    dG.set_node_attributes(new_attr, namespace="nodeids")

    assert dG.get_node_attributes() == expected

def test_convert_node_names(): 
    dG = IgNetwork(network=G.copy())
    dG.convert_node_names()

    expected = {
        0: {"name": '1', "symbol": "A1BG"}, 
        1: {"name": '10', "symbol": "NAT2"}, 
        2: {"name": "100", "symbol": "ADA"}, 
        3: {"name": "1000", "symbol": "CDH2"}
    }

    assert dG.get_node_attributes() == expected
