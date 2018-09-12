import networkx as nx
import numpy as np
import pandas as pd
import scipy
import time
import warnings

from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.sparse.linalg import expm_multiply


def random_walk_rst(
    F0, 
    A, 
    alpha, 
    normalize=True,  
    axis=1,
    threshold=1e-7,
    max_iter=100, 
    verbose=True
): 
    '''Random walk with restart
    
    Performs random walk with restart on a sparse matrix. If the 
    adjacency matrix is already normalized, this function will 
    work for dense numpy array matrices as well. (set normalize to False)
    
    TODO
    ----
    Update docstring to include normalize change

    Parameters
    ----------
    F0 : scipy.sparse
        Vector or matrix to propagate
    A : scipy.sparse
        Adjacency matrix to propagate over
    alpha : float 
        Restart probability
    threshold : float
        Threshold to consider the propagation has converged
    normalize : bool
        If normalize, the adjacency matrix will be row normalized 
        (divide by the degree)
    axis : int 
        0 or 1. Either row or column normalize
    max_iter: int
        Maximum number of iterations to perform the random walk
    verbose : bool (Deprecated)
        Prints progress (number of iterations and the tolerance)
    '''
    
    counter = 0
    tol = 10
    
    if not issparse(F0) and issparse(A): 
        warnings.warn("Forcing F0 to be sparse") 
        F0 = csr_matrix(F0)
        
 
    if normalize: 
        if issparse(A): 
            A = sparse_normalize(A, axis=axis)
        else: 
            A = dense_normalize(A, axis=axis)

    F_p = F0.copy()
    while tol > threshold: 
        F_t = (1 - alpha)*np.dot(F_p,A) + alpha*F0
        tol = frobenius_norm(F_t - F_p)
        
        F_p = F_t
        counter += 1
        
        if counter > max_iter: 
            warnings.warn('Max iteration reached. Did not converge!')
            
            break
        
    return F_t


def heat_diffusion(laplacian, heat, start=0, end=0.1): 
    """Heat diffusion 

    Iterative matrix multiplication between the graph laplacian and heat
    """

    out_vector=expm_multiply(
        -laplacian, 
        heat, 
        start=start, 
        stop=end, 
        endpoint=True
    )[-1]

    return out_vector


def frobenius_norm(sparse_mat): 
    '''Calculates the frobenius norm of a sparse matrix'''
    
    return np.sqrt(np.power(np.absolute(sparse_mat.data), 2).sum())


def get_common_indices(idx1, idx2):
    '''Gets a set of common index
    
    Take 2 lists and get the intersection of the 
    two lists. Also return the indices needed
    to rearrange each list to get the common index
    '''
    common_idx = np.intersect1d(idx1, idx2)
    
    map1 = dict(zip(list(idx1), range(len(idx1))))
    map2 = dict(zip(list(idx2), range(len(idx2))))
    
    new_idx1 = [map1[i] for i in common_idx]
    new_idx2 = [map2[i] for i in common_idx]
    
    return common_idx, new_idx1, new_idx2


def sparse_normalize(m, axis=0, inplace=False): 
    '''Normalize by one axis
    
    Divide row/column of a sparse matrix by the sum
    of row/column. This implementation does not require the 
    need to create a dense matrix and works directly at
    the coordinates and values of the non-zero elements.

    Parameters
    ----------
    sp_mat : scipy.sparse
        Sparse matrix
    axis : int 
        0/1 (row/column)
        
    Returns
    -------
    mat : scipy.sparse
        row/column normalized sparse matrix
    '''
    if inplace: 
        mat = m
    else:  
        mat = m.copy()
    
    #logger.debug(mat)
    #logger.debug(type(mat))

    row_index, col_index = mat.nonzero()
    data = mat.data
        
    #logger.debug(row_index)
    #logger.debug(col_index)

    marginals = np.array(mat.sum(axis=axis)).ravel()

    #logger.debug(marginals)

    data = data/marginals[row_index if axis else col_index]
    mat.data = data

    if inplace: 
        return None
    
    return mat


def dense_normalize(m, axis=0, inplace=False): 
    if inplace: 
        mat = m
    else: 
        mat = m.copy()
        
    marginals = np.array(mat.sum(axis=axis))
    marginals[marginals == 0] = 1
    
    mat = mat/marginals
    
    if inplace: 
        return None

    return mat


def calculate_alpha(network, m=-0.02935302, b=0.74842057):
    """Calculate optimal propagation coefficient

    Model from Huang and Carlin et al 2018
    """
    log_edge_count = np.log10(len(network.edges()))
    alpha_val = round(m*log_edge_count+b,3)
    
    if alpha_val <=0:
        # There should never be a case where Alpha >= 1, 
        # as avg node degree will never be negative

        raise ValueError('Alpha <= 0 - Network Edge Count is too high')

    else:
        return alpha_val