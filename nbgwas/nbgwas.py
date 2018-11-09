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

from .assign_snps_to_genes import assign_snps_to_genes
from .network import NxNetwork, IgNetwork
from .tables import Genes, Snps
from .propagation import random_walk_rst, get_common_indices, heat_diffusion
from .utils import get_neighbors, binarize, neg_log_val


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

        else: 
            raise ValueError("Graph type is not understood. Must be a networkx object or an igraph object")

        #TODO: Need to change were self.graphs point to (to Network maybe?)
        if not hasattr(self, "graphs"):
            self.graphs = {'full_network': network}


    def map_to_node_table(self, columns=None): 
        """Maps information from gene table to network

        Parameter
        ---------
        columns : str or list of str or None
            If None, all columns will be added
        """

        if isinstance(columns, str): 
            columns = [columns] 
        elif columns is None: 
            columns = self.genes.table.columns

        idx, gs = [], []

        for ind, gene in self.genes.table[self.genes.name_col].items(): 
            if gene in self.network.node_names: 
                idx.append(ind)
                gs.append(gene)

        tmp = self.genes.table.loc[idx, columns]
        tmp.index = gs

        self.network.set_node_attributes(tmp.to_dict(), namespace='nodenames')
        self.network.refresh_node_table()

        return self

    
    def map_to_gene_table(self, columns, fill_value=0): 
        """Maps columns from node_table to gene table"""

        df = self.network.node_table.copy()
        df = df.set_index(self.network.node_name)
        df = df[columns]

        gtable = self.genes.table.copy()
        gtable.set_index('Gene') 

        ind = gtable.index

        self.genes.table = pd.concat(
            [gtable, df.reindex(ind, fill_value=fill_value)],
            axis=1, 
            sort=False,
        )

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
        alpha=0.5, 
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

        sorted_idx = self.network.node_table.index.sort_values()
        F0 = self.network.node_table.loc[sorted_idx, node_attribute].values.T
        A = self.network.adjacency_matrix

        out = random_walk_rst(F0, A, alpha, normalize=normalize, axis=axis)

        return np.array(out.todense()).ravel().tolist()


    def random_walk_with_kernel(self, node_attribute="Heat", kernel=None):
        """Runs random walk with pre-computed kernel

        This propagation method relies on a pre-computed kernel.

        Parameters
        ----------
        kernel : str
            Location of the kernel (expects to be in HDF5 format)
        """

        raise NotImplementedError

        #TODO: Fix this 

        # if not isinstance(node_attribute, list):
        #     heat = [node_attribute]

        # if isinstance(kernel, str):
        #     self.kernel = pd.read_hdf(kernel)

        # elif isinstance(kernel, pd.DataFrame):
        #     self.kernel = kernel

        # else:
        #     raise ValueError("A kernel must be provided!")


        # # if not hasattr(self, "heat"):
        # #     warnings.warn(
        # #         "Attribute heat is not found. Generating using the binarize method."
        # #     )

        # #     self.convert_to_heat()

        # network_genes = list(self.kernel.index)

        # # Not saving heat to object because the kernel index may not
        # # match network's
        # heat = self.heat[heat].reindex(network_genes).fillna(0)

        # # Propagate with pre-computed kernel
        # prop_val_matrix = np.dot(heat.values.T, self.kernel)
        # prop_val_table = pd.DataFrame(
        #     prop_val_matrix,
        #     index = heat.columns,
        #     columns = heat.index
        # ).T

        # return prop_val_table


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

    # def get_rank(self):
    #     """Gets the ranking of each heat and pvalues"""

    #     def convert_to_rank(series, name):
    #         series = series.sort_values(ascending=True if name == "P-values" else False)

    #         return pd.DataFrame(
    #             np.arange(1, len(series) + 1),
    #             index=series.index,
    #             columns=[name]
    #         )

    #     ranks = []
    #     for col in self.heat.columns:
    #         ranks.append(convert_to_rank(self.heat[col], col))

    #     ranks.append(convert_to_rank(self.pvalues[self.gene_cols['gene_pval_col']], "P-values"))

    #     self.ranks = pd.concat(ranks, axis=1, sort=False)

    #     return self

    def hypergeom(
        self, 
        gold, 
        column,
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

        sorted_genes = self.genes.table.sort_values(by=column, ascending=ascending)
        sorted_genes = sorted_genes[self.genes.name_col].values
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

        sorted_genes = self.genes.table.sort_values(by=column, ascending=ascending)
        sorted_genes = sorted_genes[self.genes.name_col].values
        genes = sorted_genes[:top]

        return sum([gold.get(i, 1) < threshold for i in genes])
