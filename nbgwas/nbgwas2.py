from __future__ import print_function
from collections import defaultdict, OrderedDict, namedtuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx 
import igraph as ig
import ndex2
import ndex2.client as nc
import numpy as np
import pandas as pd
from py2cytoscape.data.cyrest_client import CyRestClient
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

        self.network = network

        #self._node_name = node_name # The attribute contains the gene name
                                   # on the network


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

        return self._network.network


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

    
    def diffuse(
        self,
        method="random_walk",
        heat="Heat",
        result_name="Diffused Heat",
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
            df = self.random_walk(heat=heat, **kwargs)

        elif method == "random_walk_with_kernel":
            df = self.random_walk_with_kernel(heat=heat, **kwargs)

        elif method == "heat_diffusion":
            df = self.heat_diffusion(heat=heat, **kwargs)

        else:
            raise RuntimeError("Unexpected method name!")

        #result_name = avoid_overwrite(result_name, self.heat.columns)
        df.columns = [result_name]

        #TODO: FIX ME!
        self.heat = pd.concat([self.heat, df], axis=1)
        self.heat.index.name = "Node IDs"
        self.heat.sort_values(by=result_name, ascending=False, inplace=True)

        return self


    def random_walk(self, heat='Heat', alpha=0.5, normalize=True, axis=1):
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

        if not isinstance(heat, list):
            heat = [heat]

        if not hasattr(self, "heat"):
            warnings.warn(
                "Attribute heat is not found. Generating using the binarize method."
            )

            self.convert_to_heat()

        if not hasattr(self, "adjacency_matrix"):
            self.adjacency_matrix = nx.adjacency_matrix(self.network)

        common_indices, pc_ind, heat_ind = get_common_indices(
            self._network.node_ids,
            self.heat.index
        )

        heat_mat = self.heat[heat].values.T

        F0 = heat_mat[:, heat_ind]
        A = self.adjacency_matrix[:, pc_ind][pc_ind, :]

        out = random_walk_rst(F0, A, alpha, normalize=normalize, axis=axis)

        df = pd.DataFrame(
            list(zip(common_indices, np.array(out.todense()).ravel().tolist())),
            columns=['Genes', 'PROPVALUES']
        )

        df = df.set_index('Genes').sort_values(by='PROPVALUES', ascending=False)
        df.index = df.index.astype(int)

        return df


    def random_walk_with_kernel(self, heat="Heat", kernel=None):
        """Runs random walk with pre-computed kernel

        This propagation method relies on a pre-computed kernel.

        Parameters
        ----------
        kernel : str
            Location of the kernel (expects to be in HDF5 format)
        """
        if not isinstance(heat, list):
            heat = [heat]

        if isinstance(kernel, str):
            self.kernel = pd.read_hdf(kernel)

        elif isinstance(kernel, pd.DataFrame):
            self.kernel = kernel

        else:
            raise ValueError("A kernel must be provided!")


        if not hasattr(self, "heat"):
            warnings.warn(
                "Attribute heat is not found. Generating using the binarize method."
            )

            self.convert_to_heat()

        network_genes = list(self.kernel.index)

        # Not saving heat to object because the kernel index may not
        # match network's
        heat = self.heat[heat].reindex(network_genes).fillna(0)

        # Propagate with pre-computed kernel
        prop_val_matrix = np.dot(heat.values.T, self.kernel)
        prop_val_table = pd.DataFrame(
            prop_val_matrix,
            index = heat.columns,
            columns = heat.index
        ).T

        return prop_val_table


    def heat_diffusion(self, heat="Heat", t=0.1):
        """Runs heat diffusion without a pre-computed kernel

        Parameters
        ----------
        heat: str
            Indicate which column to use as starting heat for diffusion.
        t : float
            Total time of diffusion. t controls the amount of signal is allowed
            to diffuse over the network.
        """

        if not isinstance(heat, list):
            heat = [heat]

        if not hasattr(self, "laplacian"):
            self.laplacian = csc_matrix(nx.laplacian_matrix(self.network))

        if not hasattr(self, "heat"):
            warnings.warn(
                "Attribute heat is not found. Generating using the binarize method."
            )

            self.convert_to_heat()

        out_vector = heat_diffusion(
            self.laplacian,
            self.heat.loc[self._network.node_ids, heat].values.ravel(),
            start=0,
            end=t
        )

        out_dict= {'prop': out_vector,'Gene':self._network.node_ids}
        heat_df=pd.DataFrame.from_dict(out_dict).set_index('Gene')

        return heat_df
