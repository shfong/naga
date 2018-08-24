"""Network boosted GWAS package

This package takes GWAS SNP level summary statistics and re-prioritizes 
using network data

Notes
-----
    This entire package is a WIP (and untested).

"""

from __future__ import print_function
from collections import defaultdict, OrderedDict
import matplotlib as mpl 
import matplotlib.pyplot as plt
import ndex2
import ndex2.client as nc
import networkx as nx
import numpy as np
import pandas as pd 
from scipy.sparse import coo_matrix,csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
from .propagation import random_walk_rst, get_common_indices
import time
import warnings

def _read_link(link): 
    """Checks link to be string or pandas DataFrame"""

    if isinstance(link, str): 
        return Nbgwas._read_table(link) 
    elif isinstance(link, pd.DataFrame): 
        return link 
    elif link is None: 
        return link
    else: 
        raise TypeError("Only strings (presumed to be file location) " + \
                            "or pandas DataFrame is allowed!")

def _read_snp_file(file): 
    return pd.read_csv(file, header=0, index_col=None, sep='\s+')

def _read_pc_file(file): 
    return pd.read_csv(file, index_col=0, sep='\s+', 
                        names=['Chromosome', 'Start', 'End'])

def assign_snps_to_genes(snp, 
                         pc, 
                         window_size=0, 
                         to_table=False, 
                         agg_method='min', 
                         chrom_col='hg18chr', 
                         bp_col='bp', 
                         pval_col='pval'): 

    """Assigns SNP to genes

    Parameters
    ----------
    snp : pd.DataFrame
        pandas DataFrame of SNP summary statistics from GWAS. It must have the 
        following three columns
        - chromosome (chrom_col): the chromosome the SNP is on
        - basepair (bp_col): the base pair number 
        - p-value (pval_col): the GWAS associated p-value
    pc : pd.DataFrame
        pandas DataFrame of gene coding region
    window_size : int or float
        Move the start site of a gene back and move the end site forward
        by a fixed `window_size` amount. 
    agg_method : str or callable function
        Method to aggregate multiple p-values associated with a SNP
        - min : takes the minimum p-value
        - median : takes the median of all associated p-values
        - mean : takes the average of all assocaited p-values
        - <'callable' function> : a function that takes a list and output 
          a value. The output of this value will be used in the final dictionary.
    to_table : bool
        If to_table is true, the output is a pandas dataframe that augments the pc 
        dataframe with number of SNPs, top SNP P-value, and the position of the SNP 
        for each gene. Otherwise, a dictionary of gene to top SNP P-value is returned.
          
    Output
    ------
    assigned_pvals : dict or pd.Dataframe
        A dictionary of genes to p-value (see to_table above)

    TODO
    ----
    - Test me!
    - Expose column names for pc
    - Change pc to something more descriptive
    - Add an option for caching bin edges
    """
    
    if agg_method not in ['min', 'median', 'mean'] and not hasattr(agg_method, '__call__'): 
        raise ValueError('agg_method must be min, median, mean or a callable function!')
    
    window_size = int(window_size)
    
    assigned_pvals = defaultdict(lambda: [[], []])
    for chrom, df in snp.groupby(chrom_col): 
        bins, names = _get_bins(pc.loc[pc.iloc[:,0] == str(chrom)], window_size=window_size) #TODO: HARDCODE
        bps = df[bp_col].values
        binned = np.digitize(bps, bins)
        
        names = names[binned]
        pvals = df[pval_col].values
        
        index = np.array([ind for ind, i in enumerate(names) if i != []])
        
        for i in index: 
            for n in names[i]: 
                assigned_pvals[n][0].append(pvals[i])
                assigned_pvals[n][1].append(bps[i])
                
    # Aggregate p-values
    if agg_method == 'min': 
        f = np.argmin
    elif agg_method == 'median': 
        f = lambda x: np.argwhere(np.median(x)).ravel()
    elif agg_method == 'mean': 
        f = lambda x: np.argwhere(np.mean(x)).ravel()
    else: 
        f = agg_method    

    for i,j in assigned_pvals.items(): 
        if not to_table: 
            assigned_pvals[i] = j[0][f(j[0])]
        else: 
            p = f(j[0])
            assigned_pvals[i] = [len(j[0]), j[0][p], j[1][p]] #nSNPS, TopSNP-pvalue, TopSNP-pos

    if to_table: 
        assigned_df = pd.DataFrame(assigned_pvals, index=['nSNPS', 'TopSNP P-Value', 'TopSNP Position']).T
        assigned_df = pd.concat([pc, assigned_df], axis=1)
        assigned_df.index.name = 'Gene'
        assigned_df = assigned_df.reset_index()
    
        return assigned_df
    
    return assigned_pvals

def _get_bins(df, window_size=0, cols=[1,2]): 
    """Convert start and end sites to bin edges 
    
    Given the start and end site (defined by cols) in the dataframe, 
    a set of bin edges are defined which can be augmented by window_size. 
    Each bin is then annotated by a name (assumed to be in the index. 
    Note that each bin can have multiple names due to overlapping start 
    and end sites. If the name is empty, then that bin is not occupied by 
    a gene.
    """
    names = df.index.values
    
    arr = df.iloc[:, cols].values.astype(int)
    
    arr[:, 0] -= window_size
    arr[:, 1] += window_size
    
    bins = np.sort(arr.reshape(-1))
    
    mapped_names = [[] for _ in range(len(bins) + 1)] 
    
    for ind, (i,j) in enumerate(arr): 
        vals = np.argwhere((bins > i) & (bins <= j)).ravel()
        
        for v in vals: 
            mapped_names[v].append(names[ind])
            
    return bins, np.array(mapped_names)

# Load gene positions from file                                                                                            
def load_gene_pos(gene_pos_file, delimiter='\t', header=False, cols='0,1,2,3'):
    """Loads the gene position dataframe
    
    DEPRECATED
    """

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

def binarize(a, threshold=5e-6): 
    """Binarize array based on threshold"""

    if not isinstance(a, np.ndarray): 
        a = np.array(a)

    binned = np.zeros(a.shape)
    binned[a < threshold] = 1

    return binned

def neg_log_val(a, floor=None): 
    """Negative log of an array

    Parameters
    ----------
    a : `numpy ndarray` or list
        Array to be transformed
    floor : float
        Threshold after transformation. Below which, the 
        transformed array will be floored to 0.
    """

    if not isinstance(a, np.ndarray): 
        a = np.array(a)

    vals = -np.log(a)

    if floor is not None: 
        vals[vals < floor] = 0  

    return vals

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
        The network to propagate the p-value over.
    uuid : str
        The unique identifier that corresponds to an NDEx network
    protein_coding_table : str or pd.DataFrame
        A DataFrame object that defines the start and end position and chromosome number for 
        each coding gene. This mapping will be used for the snp to gene assignment
    

    Note
    ----
    Please be aware the interface is very unstable and will be changed. 

    TODO
    ----
    - Standardize SNP and gene level input and protein coding region (file format)
        - Document what columns are needed for each of the dataframe
    - Accept any UUID from NDEx for the load network
    - Factor out the numpy to pandas code after all diffusion functions
    - Combines the heat diffusion code as one function (with a switch in behavior for kernel vs  no kernel)
    - Missing output code (to networkx subgraph, Upload to NDEx)
    - Missing utility functions (Manhanttan plots)   
    - Include logging
    - Make `network` a property to factor out the nodes_name code
    """
    def __init__(self, 
                 snp_level_summary=None, 
                 gene_level_summary=None, 
                 network = None,
                 uuid = 'f93f402c-86d4-11e7-a10d-0ac135e8bacf', #pcnet
                 protein_coding_table=None, 
                 verbose=True
                 ): 
        
        #Note: I'm starting to think that it might be alright to save the input validation for the method calls
        #Validating files inputs
        # if snp_level_summary is None and gene_level_summary is None: 
        #     raise ValueError("Either snp_level_summary or gene_level_summary must be provided!")
        # elif snp_level_summary and protein_coding_table is None: 
        #     raise ValueError("If snp_level_summary is privided, protein_coding_table is also needed.")

        #Reading files
        if snp_level_summary is not None: 
            self.snp_level_summary = _read_snp_file(snp_level_summary) #TODO: BUG here, allow for pd.DataFrame input
            
        if protein_coding_table is not None:
             self.protein_coding_table = _read_pc_file(protein_coding_table) #TODO: ditto
            
        self._gene_level_summary = _read_link(gene_level_summary) 

        #Validating network inputs
        if isinstance(network, nx.Graph) or isinstance(network, nx.DiGraph): 
            self.network = network 
        elif isinstance(uuid, str): 
            print("Loading network from NDEx...") # Change to log message
            self.network = self.get_ndex_network(uuid)
        #else: 
        #    raise ValueError("Loading network failed! Make sure to provide either a networkx object in network or a valid UUID to uuid.")

        # self.chrom_col = chrom_col
        # self.bp_col = bp_col
        # self.pval_col = pval_col
        self.verbose = verbose

    @classmethod
    def from_files(cls, 
                   snp_level_summary_file=None, 
                   gene_level_summary_file=None, 
                   network=None, ): 

        pass

    @staticmethod
    def _read_table(file):
        min_p_table = pd.read_csv(file, 
                                  sep='\t',
                                  usecols=[1,2,3,4,5,6,7,8,9])

        return min_p_table

    def read_cx_file(self, file): 
        """Load CX file as network"""

        network = ndex2.create_nice_cx_from_file(file).to_networkx()
        self.network = network 
        self.node_names = [self.network.node[n]['name'] for n in self.network.nodes()]

        return self

    def read_nx_pickle_file(self, file): 
        """Read networkx pickle file as network"""

        network = nx.read_gpickle(file)
        self.network = network
        self.node_names = [self.network.node[n]['name'] for n in self.network.nodes()]

        return self

    def get_ndex_network(self, uuid): 
        anon_ndex = nc.Ndex2("http://public.ndexbio.org")
        network_niceCx = ndex2.create_nice_cx_from_server(server='public.ndexbio.org', 
                                                          uuid=uuid)

        self.network = network_niceCx.to_networkx()
        self.node_names = [self.network.node[n]['name'] for n in self.network.nodes()]

        return self

    def assign_pvalues(self, **kwargs): 
        """Wrapper for assign_snps_to_genes"""

        if self.protein_coding_table is None: 
            raise ValueError("protein_coding_table attribute must be provided!")

        if self.snp_level_summary is None: 
            raise ValueError("snp_level_summary attribute must be provided!")
        
        assign_pvalues = assign_snps_to_genes(self.snp_level_summary, 
                                              self.protein_coding_table,
                                              to_table=True,
                                              **kwargs)

        if self.gene_level_summary is not None: 
            warnings.warn("The existing gene_level_summary was overwritten!")

        self.gene_level_summary = assign_pvalues

        return self

    @property 
    def pvalues(self): 
        """OrderedDict: `str` to `float`: Dictionary that maps genes to p-values

        Requires gene_level_summary to be set (i.e. Does not run assign_pvalues 
        automatically. For now, pvalues cannot be reassigned.
        """

        if not hasattr(self, "_pvalues"): 
            try: 
                self._pvalues = self.gene_level_summary[['Gene', 'TopSNP P-Value']]
                self._pvalues = OrderedDict(self._pvalues.set_index('Gene').to_dict()['TopSNP P-Value'])
            except AttributeError: 
                raise AttributeError("No gene level summary found! Please use assign_pvalues to convert SNP to gene-level summary or input a gene-level summary!")
            except KeyError: 
                raise KeyError("'Gene' and 'TopSNP P-value' columns were not found in gene_level_summary!")
        
        return self._pvalues

    @pvalues.deleter
    def pvalues(self): 
        if hasattr(self, "_pvalues"):
            del self._pvalues

    @property 
    def gene_level_summary(self): 
        """pd.DataFrame : DataFrame that includes the gene_level_summary

        Handles error if no gene_level summary is given. If the gene_level_summary is
        overwritten, the DataFrame must include ['Gene', 'Top_SNP P-value'] and the 
        previously created pvalues would be destroyed.
        """

        try: 
            return self._gene_level_summary
        except AttributeError: 
            raise AttributeError("No gene level summary found! Please use assign_pvalues to convert SNP to gene-level summary or input a gene-level summary!")
    
    @gene_level_summary.setter
    def gene_level_summary(self, df): 
        if not isinstance(df, pd.DataFrame): 
            raise ValueError("gene_level_summary must be a pandas DataFrame")

        required_headers = set(['Gene', 'TopSNP P-Value']) 
        if len(required_headers.intersection(set(df.columns))) != len(required_headers): 
            raise ValueError("Input dataframe must include 'Gene' and 'TopSNP P-Value' in its columns")

        self._gene_level_summary = df
        del self.pvalues 

    def convert_to_heat(self, method='binarize', **kwargs):
        """Convert p-values to heat

        Parameters
        ----------
        method : str
            Must be in 'binarize' or 'neg_log'
            "binarize" uses the `binarize` function while "neg_log" uses the
            `neg_log_val` function. 
        kwargs 
            Any additional keyword arguments to be passed into the above functions

        TODO
        ----
        - Implement other methods to convert p-values to heat
        """

        allowed = ['binarize', "neg_log"] 
        if method not in allowed: 
            raise ValueError("Method must be in %s" % allowed)

        vals = np.array(list(self.pvalues.values()))
        if method == 'binarize': 
            heat = binarize(vals, threshold=kwargs.get('threshold', 5e-6))
        elif method == 'neg_log': 
            heat = neg_log_val(vals, floor=kwargs.get('floor', None))

        self.heat = pd.DataFrame(heat[...,np.newaxis], index = list(self.pvalues.keys()), columns=['Heat'])
        self.heat = self.heat.reindex(self.node_names).fillna(0)

        return self

    def diffuse(self, method="random_walk", name=None, replace=False, **kwargs): 
        """Wrapper for the various diffusion methods available 

        Parameters
        ----------
        method : str

        name : str 
            Column name of the result
        replace : bool
            If replace is True, any previous results are overwritten. If False, 
            the current data will be added to the previous dataframe. 
        kwargs 
            Any additional keyword arguments for each of the diffusion function. 
            See the individual function documentation. 

        TODO
        ----
        * Factor out various setup and tear-down code
        """
        allowed = ["random_walk", "random_walk_with_kernel", "heat_diffusion"]
        if method not in allowed: 
            raise ValueError("method must be one of the following: %s" % allowed) 

        if method == "random_walk": 
            df = self.random_walk(**kwargs) 
        
        elif method == "random_walk_with_restart": 
            df = self.random_walk_with_kernel()

        elif method == "heat_diffusion": 
            df = self.heat_diffusion()

        if replace or not hasattr(self, "boosted_pvalues"): 
            if name is None: 
                name = 0
            
            df.columns = [name]
            self.boosted_pvalues = df
        else: 
            if name is None: 
                name = self.boosted_pvalues.shape[1]

            self.boosted_pvalues[name] = df

        return self     

    def random_walk(self, alpha=0.5): 
        """Runs random walk iteratively 
        
        Parameters 
        ----------
        threshold: float
            Minimum p-value to diffuse the p-value
        alpha : float
            The restart probability
        """
               
        if not hasattr(self, "heat"): 
            warnings.warn("Attribute heat is not found. Generating using the binarize method.")
            self.convert_to_heat() 

        if not hasattr(self, "adjacency_matrix"): 
            self.adjacency_matrix = nx.adjacency_matrix(self.network)
        
        nodes = [self.network.node[i]['name'] for i in self.network.nodes()]
        common_indices, pc_ind, heat_ind = get_common_indices(nodes, self.heat.index)
        heat_mat = self.heat.values.T
        
        F0 = heat_mat[:, heat_ind]
        A = self.adjacency_matrix[:, pc_ind][pc_ind, :]

        out = random_walk_rst(F0, A, alpha)
        df = pd.DataFrame(list(zip(common_indices, np.array(out.todense()).ravel().tolist())), columns=['Genes', 'PROPVALUES'])
        df = df.set_index('Genes').sort_values(by='PROPVALUES', ascending=False)

        #self.boosted_pvalues = df

        #return self

        return df

    def random_walk_with_kernel(self, threshold=5e-6, kernel=None): 
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

        else: 
            warnings.warn("No kernel was given! Running random_walk instead")
            self.random_walk() 

            return self

        if not hasattr(self, "laplacian"): 
            self.laplacian = csc_matrix(nx.laplacian_matrix(self.network))

        if not hasattr(self, "heat"): 
            warnings.warn("Attribute heat is not found. Generating using the binarize method.")
            self.convert_to_heat() 

        network_genes = list(self.kernel.index)

        heat = self.heat.reindex(network_genes).fillna(0) #Not saving heat to object because the kernel index may not match network's

        #propagate with pre-computed kernel
        prop_val_matrix = np.dot(heat.values.T, self.kernel)
        prop_val_table = pd.DataFrame(prop_val_matrix, index = heat.columns, columns = heat.index)

        #self.boosted_pvalues = prop_val_table.T.sort_values(by='Heat', ascending=False)

        #return self

        return prop_val_table

    def heat_diffusion(self): 
        """Runs heat diffusion without a pre-computed kernel
        
        Parameters
        ----------
        threshold : float
            Minimum p-value to diffuse the p-value
        """
        
        if not hasattr(self, "laplacian"): 
            self.laplacian = csc_matrix(nx.laplacian_matrix(self.network))

        if not hasattr(self, "heat"): 
            warnings.warn("Attribute heat is not found. Generating using the binarize method.")
            self.convert_to_heat()

        out_vector=expm_multiply(-self.laplacian, self.heat.values.ravel(), start=0, stop=0.1, endpoint=True)[-1]
        out_dict= {'prop': out_vector,'Gene':self.node_names}
        heat_df=pd.DataFrame.from_dict(out_dict).set_index('Gene')

        #self.boosted_pvalues = heat_df.sort_values(by='prop', ascending=False)

        #return self

        return heat_df

    def annotate_network(self, values="Heat", inplace=False): 
        """Return a subgraph with node attributes

        Parameters
        ----------
        values : str
            Name of the column
        """

        #TODO: Refactor
        # if hasattr(self, "boosted_pvalues"):
        #     if values in self.boosted_pvalues.columns: 
        #         data = self.boosted_pvalues[values] 
        #     else: 
        #         data = pd.Series(self.pvalues)

        # else: 
        #     data = pd.Series(self.pvalues)

        # data = dict(zip(data.index, data.values))

        #values = "Heat"

        if values=="Heat": 
            data = self.heat.to_dict()[values]
        else: 
            data = self.boosted_pvalues.to_dict()[values]
        name_map = dict(zip(self.node_names, list(range(len(self.node_names)))))
        new_data = {name_map[k]:v for k,v in data.items()}

        if inplace: 
            G = self.network
        else: 
            G = self.network.copy() 

        nx.set_node_attributes(G, values, new_data)

        #if inplace: 
        #    self.network = G

        return G

    def view_subgraph(self, gene, neighbors=1, attributes="Heat"): 
        #nodes = set([center])
        #for i in range(neighbors): 
        #    nodes = nodes.union(set(self.network.neighbors()))

        name_map = dict(zip(self.node_names, list(range(len(self.node_names)))))
        center = name_map[gene]

        nodes = set([center]).union(set(self.network.neighbors(center)))
        G = self.network.subgraph(nodes)

        attr = nx.get_node_attributes(G, attributes)
        vals = [attr[i] for i in G.nodes()]

        cmap=plt.cm.Blues
        vmin = 0
        vmax = 1

        nx.draw(G, node_color=vals,
                labels=nx.get_node_attributes(G, "name"), 
                vmin=vmin, vmax=vmax, 
                cmap=cmap)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                                norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        plt.colorbar(sm)

        return G     

