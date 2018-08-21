"""Network boosted GWAS package

This package takes GWAS SNP level summary statistics and re-prioritizes 
using network data

Notes
-----
    This entire package is a WIP (and untested).

"""

from __future__ import print_function
import ndex2
import ndex2.client as nc
import networkx as nx
import numpy as np
import pandas as pd 
from scipy.sparse import coo_matrix,csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
import time

def _check_link(link): 
    if isinstance(link, str): 
        return Nbgwas._read_table(link) 
    elif isinstance(link, pd.DataFrame): 
        return link 
    else: 
        raise TypeError("Only strings (presumed to be file location) " + \
                            "or pandas DataFrame is allowed!")

def min_p(SNP_summary, gene_positions, window):
    """Assigns SNP p-value to gene
    
    This function assigns the SNP's p-value to a gene if the SNP is within a certain 
    window of the gene.

    Parameters
    ----------
    SNP_summary : pd.DataFrame 
        A pandas DataFrame that contains the SNP information
    gene_positions : pd.DataFrame
        A pandas DataFrame that lists each coding gene's start and end position on the
        chromosome
    window : float
        A number (in kilobases) that defines how large the window is.

    TODO 
    ---- 
    - Move this function to the Nbgwas class (or at least call this function)
    """

    starttime = time.time()
    dist = window*1000
    genelist = list(gene_positions.index)
    min_p_list = []
    SNP_summary['Chr']=SNP_summary['Chr'].astype(str)
    
    for gene in genelist:
        gene_info = gene_positions.ix[gene]
        chrom = str(gene_info['Chr'])
        start = gene_info['Start']
        stop = gene_info['End']

        # Get all SNPs on same chromosome
        SNP_summary_filt1 = SNP_summary[SNP_summary['Chr']==chrom]
        # Get all SNPs after window start position
        SNP_summary_filt2 = SNP_summary_filt1[SNP_summary_filt1['Pos'] >= (start-dist)]
        # Get all SNPs before window end position
        SNP_summary_filt3 = SNP_summary_filt2[SNP_summary_filt2['Pos'] <= (stop+dist)]
        
        # Get min_p statistics for this gene
        if len(SNP_summary_filt3) >= 1:
            min_p_data = SNP_summary_filt3.ix[SNP_summary_filt3['P-Value'].argmin()]
            min_p_list.append([gene, chrom, start, stop, SNP_summary_filt3.shape[0], min_p_data['Marker'], int(min_p_data['Pos']), min_p_data['P-Value']])
        else:
            min_p_list.append([gene, chrom, start, stop, 0, None, None, None])
    
    min_p_table = pd.DataFrame(min_p_list, columns = ['Gene', 'Chr', 'Gene Start', 'Gene End', 'nSNPs', 'TopSNP', 'TopSNP Pos', 'TopSNP P-Value'])
    min_p_table['SNP Distance'] = abs(min_p_table['TopSNP Pos'].subtract(min_p_table['Gene Start']))
    min_p_table = min_p_table.dropna().sort_values(by=['TopSNP P-Value', 'Chr', 'Gene Start'])
    
    print("P-Values assigned to genes:", time.time()-starttime, 'seconds')
    
    return min_p_table

# Load gene positions from file                                                                                            
def load_gene_pos(gene_pos_file, delimiter='\t', header=False, cols='0,1,2,3'):
    """Loads the gene position dataframe"""

    # Check for valid 'cols' parameter                                                                                 
    try:
        cols_idx = [int(c) for c in cols.split(',')]
    except:
        raise ValueError('Invalid column index string')
    # Load gene_pos_file                                                                                               
    if header:
        gene_positions = pd.read_csv(gene_pos_file, delimiter=delimiter)
    else:
        gene_positions = pd.read_csv(gene_pos_file, delimiter=delimiter, header=-1)
    # Check gene positions table format                                                                                
    if (gene_positions.shape[1] < 4) | (max(cols_idx) >  gene_positions.shape[1]-1):
        raise ValueError('Not enough columns in Gene Positions File')

    # Construct gene position table                                                                                    
    gene_positions = gene_positions[cols_idx]
    gene_positions.columns = ['Gene', 'Chr', 'Start', 'End']
    
    return gene_positions.set_index('Gene')

class Nbgwas(object): 
    """Interface to Network Boosted GWAS

    Parameters
    ----------
    snp_level_summary : str or pd.DataFrame
        A DataFrame object that holds the snp level summary or a file that points 
        to a text file
    gene_level_summary : str or pd.DataFrame
        A DataFrame object that holds the gene level summary or a file that points 
        to a text file     
    network : networkx object
        The network to propagate the p-value over. If None, PC-net (Huang, Cell Systems, 2018) 
        will be pulled from NDEx instead.

    Note
    ----
    Please be aware the interface is very unstable and will be changed. 

    TODO
    ----
    - Error handling when both summaries are presented (Should only just use one)  
    - Refactor out p-value assignment (with different methods) as different functions
    - Combines the heat diffusion code as one function (with a switch in behavior for kernel vs  no kernel)
    - Missing output code (to networkx subgraph, Upload to NDEx)
    - Missing utility functions (Manhanttan plots)   
    - Include logging
    """
    def __init__(self, 
                 snp_level_summary=None, 
                 gene_level_summary=None, 
                 network=None,
                 ): 
        
        if snp_level_summary is not None: 
            raise NotImplementedError("TBA")

        self.snp_level_summary = _check_link(snp_level_summary)
        self.gene_level_summary = _check_link(gene_level_summary) 

        if network is None: 
            self.network = Nbgwas._load_pcnet() 

        else: 
            if isinstance(network, nx.Graph): 
                self.network = network
            else: 
                raise TypeError("Network must be a networkx Graph object")

        self.node_names = [self.network.node[n]['name'] for n in self.network.nodes()]

    @staticmethod
    def _read_table(file):
        min_p_table = pd.read_csv(file, 
                                  sep='\t',
                                  usecols=[1,2,3,4,5,6,7,8,9])

        return min_p_table

    @staticmethod
    def _load_pcnet(): 
        anon_ndex = nc.Ndex2("http://public.ndexbio.org")
        network_niceCx = ndex2.create_nice_cx_from_server(server='public.ndexbio.org', 
                                                          uuid='f93f402c-86d4-11e7-a10d-0ac135e8bacf')

        return network_niceCx.to_networkx()

    def diffuse(self, threshold=5e-6, kernel=None): 
        """Runs random walk with pre-computed kernel 
        
        This propagation method relies on a pre-computed kernel. 

        Parameters
        ----------
        threshold : float
            Minimum p-value threshold to diffuse the p-value
        kernel : str
            Location of the kernel (expects to be in HDF5 format)        
        """
        if kernel is not None: 
            self.kernel = pd.read_hdf(kernel)
            network_genes = list(self.kernel.index)

        else: 
            raise NotImplementedError("Need to define the location to the kernel! TODO: Add non-pre-computed kernel code")


        if not hasattr(self, laplacian): 
            self.laplacian = csc_matrix(nx.laplacian_matrix(self.network))

        name='prop'
        threshold_genes = {}
        prop_vectors = []
            
        threshold_genes[name] = self.gene_level_summary[self.gene_level_summary['TopSNP P-Value'] < threshold]
        prop_vector = (self.gene_level_summary.set_index('Gene').loc[network_genes, 'TopSNP P-Value'] < threshold).astype(float)
        prop_vector.name = name
        prop_vectors.append(prop_vector)
        prop_vector_matrix = pd.concat(prop_vectors, axis=1).loc[network_genes].T

        #propagate with pre-computed kernel
        prop_val_matrix = np.dot(prop_vector_matrix, self.kernel)
        prop_val_table = pd.DataFrame(prop_val_matrix, index = prop_vector_matrix.index, columns = prop_vector_matrix.columns)
        
        self.boosted_pvalues = prop_val_table.T.sort_values(by='prop', ascending=False).head()

        return self

    def heat_diffusion(self, threshold=5e-6): 
        """Runs heat diffusion without a pre-computed kernel
        
        Parameters
        ----------
        threshold : float
            Minimum p-value to diffuse the p-value
        """
        if not hasattr(self, laplacian): 
            self.laplacian = csc_matrix(nx.laplacian_matrix(self.network))
            
        input_list = list(self.gene_level_summary[self.gene_level_summary['TopSNP P-Value'] < threshold]['Gene'])
        input_vector = np.array([n in input_list for n in self.node_names])
        out_vector=expm_multiply(-self.laplacian, input_vector, start=0, stop=0.1, endpoint=True)[-1]

        #out_dict= dict(zip(node_names, out_vector))
        out_dict= {'prop': out_vector,'Gene':self.node_names}
        heat_df=pd.DataFrame.from_dict(out_dict).set_index('Gene')

        self.boosted_pvalues = heat_df.sort_values(by='prop', ascending=False).head()

        return self
