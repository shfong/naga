from __future__ import print_function
from collections import defaultdict, OrderedDict, namedtuple
import networkx as nx 
import igraph as ig
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,csc_matrix
from scipy.stats import hypergeom
import time
import warnings

from .network import Network, NxNetwork, IgNetwork
from .tables import Genes, Snps
from .propagation import random_walk_rst, get_common_indices, heat_diffusion
from .utils import get_neighbors, binarize, neg_log_val, calculate_alpha


class Nbgwas(object):
    """Interface to Network Boosted GWAS

    Parameters
    ----------
    snp_level_summary : pd.DataFrame
        A DataFrame object that holds the snp level summary or a file that
        points to a text file
    gene_level_summary : pd.DataFrame
        A DataFrame object that holds the gene level summary or a file that
        points to a text file
    network : networkx object
        The network to propagate the p-value over.
    protein_coding_table : str or pd.DataFrame
        A DataFrame object that defines the start and end position and
        chromosome number for each coding gene. This mapping will be used for
        the snp to gene assignment


    Note
    ----
    Please be aware the interface is very unstable and will be changed.

    TODO
    ----
    - Standardize SNP and gene level input and protein coding region (file
        format)
        - Document what columns are needed for each of the dataframe
    - Factor out the numpy to pandas code after all diffusion functions
    - Missing utility functions (Manhanttan plots)
    - Include logging
    """

    def __init__(
        self,
        snp_level_summary=None,
        gene_level_summary=None,
        network = None,
        protein_coding_table=None,
        snp_chrom_col='hg18chr',
        bp_col='bp',
        snp_pval_col='pval',
        gene_pval_col='TopSNP P-Value',
        gene_col='Gene',
        pc_chrom_col='Chrom',
        start_col='Start',
        end_col='End',
        node_name="name",
        validate = True,
        verbose=True
    ):

        self.verbose = verbose
        self.validate = validate

        self.genes = Genes(
            gene_level_summary, 
            pval_col=gene_pval_col, 
            name_col=gene_col, 
        )

        self.snps = Snps(
            snp_level_summary, 
            protein_coding_table, 
            snp_chrom_col=snp_chrom_col,
            snp_bp_col=bp_col, 
            pval_col=snp_pval_col, 
            pc_chrom_col=pc_chrom_col, 
            start_col=start_col,
            end_col=end_col
        )

        self._node_name = node_name # The attribute contains the gene name
                                   # on the network

        self.network = network


    def __repr__(self): 
        contains = []
        if self.genes.table is not None: 
            contains.append('"genes table"') 

        if self.snps.snp_table is not None: 
            contains.append('"SNP table"')

        if self.network.network is not None: 
            contains.append('"network"')

        if not contains: 
            contains = 'Nothing'

        else: 
            contains = ', '.join(contains)

        return f'{self.__class__.__name__} object containing {contains}'


    @property
    def network(self):
        """networkx Graph object : Network object used for graph diffusion

        node_names attribute is automatically created if the network is a
        networkx object. If a node has a `self.node_name` attribute, that name
        is used for node_names. Otherwise, the node id itself is used as the
        name.
        """

        if self._network is None: 
            return None 

        return self._network


    @network.setter
    def network(self, network):
        if network is None: 
            self._network = NxNetwork(None, node_name=self._node_name) 

        elif isinstance(network, nx.Graph): 
            self._network = NxNetwork(network, node_name=self._node_name) 

        elif isinstance(network, ig.Graph): 
            self._network = IgNetwork(network, node_name=self._node_name)

        elif isinstance(network, Network): 
            self._network = network
        
        else: 
            raise ValueError("Graph type is not understood. Must be a networkx object or an igraph object")

        #TODO: Need to change were self.graphs point to (to Network maybe?)
        if not hasattr(self, "graphs"):
            self.graphs = {'full_network': network}


    def map_snps_to_genes(
        self, 
        window_size=0,
        agg_method='min',
    ): 
        """Maps SNP p-values to genes

        This is a convenience function for the functionality within the `Snps` 
        object. The output is forced to be a `Genes` object and is automatically 
        assigned to Nbgwas.genes. 

        See `Snps.assign_snps_to_genes` for documentation.
        """
        self.genes = self.snps.assign_snps_to_genes(
            window_size=window_size, 
            agg_method=agg_method, 
            to_Gene=True
        )

        return self


    def map_to_node_table(self, columns=None, update_node_attributes=False, fillna=0): 
        """Maps information from gene table to network

        Parameter
        ---------
        columns : str or list of str or None
            If None, all columns will be added
        """

        if columns is None: 
            columns = list(self.genes.table.columns)

        if isinstance(columns, str): 
            columns = [columns] 

        # Remove extra column merge seems to include
        remove=False
        if self.genes.name_col not in self.network.node_table.columns: 
            remove=True

        self.network.node_table = self.network.node_table.merge(
            self.genes.table[[self.genes.name_col] + columns], 
            left_on = self.network.node_name, 
            right_on = self.genes.name_col, 
            how='left'
        )

        if remove and self.genes.name_col in self.network.node_table.columns: 
            self.network.node_table.drop(columns=self.genes.name_col, inplace=True)

        self.network.node_table.fillna(fillna, inplace=True)

        if update_node_attributes: 
            self.network.set_node_attributes(tmp.to_dict(), namespace='nodenames')
            self.network.refresh_node_table()

        return self

    
    def map_to_gene_table(self, columns=None, fill_value=0): 
        """Maps columns from node_table to gene table"""

        def tmp_func(x): 
            if pd.notnull(x[0]): 
                return x[0]
            
            elif pd.notnull(x[1]): 
                return x[1]
            
            #else: 
            #    raise ValueError("Cannot both be Null")

        if isinstance(columns, str): 
            columns = [columns]
        elif columns is None: 
            columns = self.network.node_table.columns

        # Remove extra column merge seems to include
        remove=False
        if self.network.node_name not in self.genes.table.columns: 
            remove=True

        self.genes.table = self.genes.table.merge(
            self.network.node_table[[self.network.node_name] + columns],
            left_on=self.genes.name_col, 
            right_on=self.network.node_name, 
            how='outer'
        )

        self.genes.table[self.genes.name_col] = self.genes.table[
            [self.genes.name_col, self.network.node_name]
        ].agg(tmp_func, axis=1)

        if remove and self.network.node_name in self.genes.table.columns: 
            self.genes.table.drop(columns=self.network.node_name, inplace=True)

        return self


    def diffuse(
        self,
        method="random_walk",
        node_attribute="Heat",
        result_name="Diffused Heat",
        update_node_attributes=False,
        **kwargs
    ):

        """Wrapper for the various diffusion methods available

        Calls one of the three diffusion methods and add the results to the
        heat attribute.

        Parameters
        ----------
        method : str
            Must be one of the following: `random_walk`,
            `random_walk_with_kernel`, `heat_diffusion`. Each method calls the
            corresponding method.
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
            raise ValueError(
                "method must be one of the following: %s" % allowed
            )

        if self.network is None: 
            raise RuntimeError("Network was given!")

        if method == "random_walk":
            df = self.random_walk(
                node_attribute=node_attribute, 
                **kwargs
            )

        elif method == "random_walk_with_kernel":
            df = self.random_walk_with_kernel(
                node_attribute=node_attribute, 
                **kwargs
            )

        elif method == "heat_diffusion":
            df = self.heat_diffusion(
                node_attribute=node_attribute, 
                **kwargs
            )

        else:
            raise RuntimeError("Unexpected method name!")

        sorted_idx = self.network.node_table.index.sort_values()

        self.network.node_table.loc[sorted_idx, result_name] = df
        self.network.node_table.sort_values(
            by=result_name, 
            ascending=False, 
            inplace=True
        )

        if update_node_attributes: 
            self.network.refresh_node_attributes()

        return self


    def random_walk(
        self, 
        node_attribute='Heat', 
        alpha='optimal', 
        normalize=True, 
        axis=1, 
    ):

        """Runs random walk iteratively

        Parameters
        ----------
        alpha : float
            The restart probability
        normalize : bool
            If true, the adjacency matrix will be row or column normalized
            according to axis
        axis : int
            0 row normalize, 1 col normalize. (The integers are different
            than convention because the axis is the axis that is summed
            over)

        TODO
        ----
        * Allow for diffusing multiple heat columns
        """

        if not isinstance(node_attribute, list):
            node_attribute = [node_attribute]

        if isinstance(alpha, str): 
            alpha = calculate_alpha(len(self.network.edges()))

        sorted_idx = self.network.node_table.index.sort_values()
        F0 = self.network.node_table.loc[sorted_idx, node_attribute].values.T
        A = self.network.adjacency_matrix

        out = random_walk_rst(F0, A, alpha, normalize=normalize, axis=axis)

        return np.array(out.todense()).ravel()


    def heat_diffusion(self, node_attribute="Heat", t=0.1):
        """Runs heat diffusion without a pre-computed kernel

        Parameters
        ----------
        heat: str
            Indicate which column to use as starting heat for diffusion.
        t : float
            Total time of diffusion. t controls the amount of signal is allowed
            to diffuse over the network.
        """

        if not isinstance(node_attribute, list):
            node_attribute = [node_attribute]

        sorted_idx = self.network.node_table.index.sort_values()

        out_vector = heat_diffusion(
            self.network.laplacian_matrix,
            self.network.node_table.loc[sorted_idx, node_attribute].values.ravel(),
            start=0,
            end=t
        )

        return out_vector


    def hypergeom(
        self, 
        gold, 
        column,
        table='gene',
        top=100, 
        ngenes=20000,
        ascending=False
    ):
        """Run hypergemoetric test

        Parameters
        ----------
        gold : list
            An iterable of genes
        top : int
            The number of ranked genes to select.
        ngenes : int
            The number of genes to be considered as the global background.
        rank_col : str
            The name of the heat column to be determined for significance.
            If the rank_col is None, the p-value is used.
        """

        if table == 'gene': 
            table_df = self.genes.table 
            name_col = self.genes.name_col
        elif table == 'network': 
            table_df = self.network.node_table
            name_col = self.network.node_name

        sorted_genes = table_df.sort_values(by=column, ascending=ascending)
        sorted_genes = sorted_genes[name_col].values
        genes = sorted_genes[:top]

        intersect = set(genes).intersection(set(gold))
        score = len(intersect)
        M, n, N = ngenes, len(gold), top

        pvalue = 1 - hypergeom.cdf(score, M, n, N)
        Hypergeom = namedtuple('Hypergeom',
            ['pvalue', 'n_intersect', 'common_items']
        )

        return Hypergeom(pvalue, score, intersect)


    def check_significance(
        self, 
        gold, 
        column,
        table='gene',
        top=100, 
        threshold=0.05,
        ascending=False
    ):
        """Check if the top N genes are significant

        Parameters
        ----------
        gold : dict
            A gene to p-value dictionary. If a gene cannot be found in the
            dictionary, the default value is 1.
        top : int
            The number of ranked genes to select.
        threshold : float
            The p-value threshold to determine significance
        rank_col : str
            The name of the heat column to be determined for significance.
            If the rank_col is None, the p-value is used.
        """

        if table == 'gene': 
            table_df = self.genes.table 
            name_col = self.genes.name_col
        elif table == 'network': 
            table_df = self.network.node_table
            name_col = self.network.node_name

        sorted_genes =table_df.sort_values(by=column, ascending=ascending)
        sorted_genes = sorted_genes[name_col].values
        genes = sorted_genes[:top]

        return sum([gold.get(i, 1) < threshold for i in genes])
