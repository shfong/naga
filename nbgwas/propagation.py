import networkx as nx
import numpy as np
import pandas as pd
import scipy
import time
import warnings

from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.sparse.linalg import expm_multiply

def heat_diffusion(network, diffusion_input, t=0.1):
    network_nodes = sorted(network.nodes())
    sparse_laplacian = csc_matrix(nx.laplacian_matrix(network))
    diffused_matrix = expm_multiply(-sparse_laplacian, diffusion_input, start=0, stop=t, endpoint=True)[-1]
    return diffused_matrix

##############################################

# Calculate optimal propagation coefficient
# Model from Huang and Carlin et al 2018
def calculate_alpha(network, m=-0.02935302, b=0.74842057):
    log_edge_count = np.log10(len(network.edges()))
    alpha_val = round(m*log_edge_count+b,3)
    if alpha_val <=0:
        raise ValueError('Alpha <= 0 - Network Edge Count is too high')
        # There should never be a case where Alpha >= 1, as avg node degree will never be negative
    else:
        return alpha_val

# Normalize network (or network subgraph) for random walk propagation
def normalize_network(network, symmetric_norm=False):
	adj_mat = nx.adjacency_matrix(network)
	adj_array = np.array(adj_mat.todense())
	if symmetric_norm:
		D = np.diag(1/np.sqrt(sum(adj_array)))
		adj_array_norm = np.dot(np.dot(D, adj_array), D)
	else:
		degree_norm_array = np.diag(1/sum(adj_array).astype(float))
		sparse_degree_norm_array = scipy.sparse.csr_matrix(degree_norm_array)
		adj_array_norm = sparse_degree_norm_array.dot(adj_mat).toarray()
	return adj_array_norm

# Closed form random-walk propagation (as seen in HotNet2) for each subgraph: Ft = (1-alpha)*Fo * (I-alpha*norm_adj_mat)^-1
# Concatenate to previous set of subgraphs
def fast_random_walk(alpha, binary_mat, subgraph_norm, prop_data):
	term1=(1-alpha)*binary_mat
	term2=np.identity(binary_mat.shape[1])-alpha*subgraph_norm
	term2_inv = np.linalg.inv(term2)
	subgraph_prop = np.dot(term1, term2_inv)
	return np.concatenate((prop_data, subgraph_prop), axis=1)

# Wrapper for random walk propagation of full network by subgraphs
def closed_form_network_propagation(network, binary_matrix, network_alpha, symmetric_norm=False,  verbose=False, save_path=None):
	starttime=time.time()
	if verbose:
		print('Alpha:', network_alpha)
	# Separate network into connected components and calculate propagation values of each sub-sample on each connected component
	subgraphs = list(nx.connected_component_subgraphs(network))
	# Initialize propagation results by propagating first subgraph
	subgraph = subgraphs[0]
	subgraph_nodes = list(subgraph.nodes())
	prop_data_node_order = list(subgraph_nodes)
	binary_matrix_filt = np.array(binary_matrix.T.ix[subgraph_nodes].fillna(0).T)
	subgraph_norm = normalize_network(subgraph, symmetric_norm=symmetric_norm)
	prop_data_empty = np.zeros((binary_matrix_filt.shape[0], 1))
	prop_data = fast_random_walk(network_alpha, binary_matrix_filt, subgraph_norm, prop_data_empty)
	# Get propagated results for remaining subgraphs
	for subgraph in subgraphs[1:]:
		subgraph_nodes = list(subgraph.nodes())
		prop_data_node_order = prop_data_node_order + subgraph_nodes
		binary_matrix_filt = np.array(binary_matrix.T.ix[subgraph_nodes].fillna(0).T)
		subgraph_norm = normalize_network(subgraph, symmetric_norm=symmetric_norm)
		prop_data = fast_random_walk(network_alpha, binary_matrix_filt, subgraph_norm, prop_data)
	# Return propagated result as dataframe
	prop_data_df = pd.DataFrame(data=prop_data[:,1:], index = binary_matrix.index, columns=prop_data_node_order)
	if save_path is None:
		if verbose:
			print('Network Propagation Complete:', time.time()-starttime, 'seconds')
		return prop_data_df
	else:
		prop_data_df.to_csv(save_path)
		if verbose:
			print('Network Propagation Complete:', time.time()-starttime, 'seconds')
		return prop_data_df

def random_walk_rst(F0, A, alpha, 
                    normalize=True,  
                    threshold=1e-7,
                    max_iter=100, 
                    verbose=True): 
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
            A = sparse_normalize(A, axis=1)
        else: 
            A = dense_normalize(A, axis=1)

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