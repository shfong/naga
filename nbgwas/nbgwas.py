"""Network boosted GWAS package

This package takes GWAS SNP level summary statistics and re-prioritizes
using network data

Notes
-----
    This entire package is a WIP (and untested).

"""

from __future__ import print_function
from collections import defaultdict, OrderedDict, namedtuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import ndex2
import ndex2.client as nc
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
import time
import warnings

from .assign_snps_to_genes import assign_snps_to_genes
from .propagation import random_walk_rst, get_common_indices, heat_diffusion
from .utils import get_neighbors, binarize, neg_log_val


def _validate_dataframe(df, require_columns, var_name="df"):
    """Checks to see if a dataframe has the require columns
    
    The dataframe is allowed to be None or a pandas DataFrame. 
    """

    if df is None:
        return None

    if not isinstance(df, pd.DataFrame):
        raise ValueError("%s must be a pandas DataFrame!" % var_name)
    else:
        if not set(df.columns).issuperset(set(require_columns.values())):
            missing_columns = set(require_columns.values()).difference(set(df.columns))
            
            raise ValueError("%s must include %s. The following columns are missing from %s: %s" % ( 
                var_name, 
                ",".join(require_columns.keys()), 
                var_name,
                ",".join(missing_columns)
            ))

def avoid_overwrite(name, iterable): 
    if name not in iterable: 
        return name 

    else: 
        count = sum([1 for i in iterable if name in i]) 
        return name + " (%s)" % str(count + 1)


class Nbgwas(object):
    """Interface to Network Boosted GWAS

    Parameters
    ----------
    snp_level_summary : pd.DataFrame
        A DataFrame object that holds the snp level summary or a file that points
        to a text file
    gene_level_summary : pd.DataFrame
        A DataFrame object that holds the gene level summary or a file that points
        to a text file
    network : networkx object
        The network to propagate the p-value over.
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
        validate = True,
        verbose=True
    ):

        self.verbose = verbose
        self.validate = validate

        self.snp_cols = {
            'snp_chrom_col': snp_chrom_col, 
            'bp_col' : bp_col, 
            'snp_pval_col' : snp_pval_col
        }

        self.gene_cols = {
            'gene_pval_col' : gene_pval_col, 
            'gene_col' : gene_col
        }

        self.pc_cols = {
            'pc_chrom_col' : pc_chrom_col, 
            'start_col' : start_col, 
            'end_col' : end_col
        }

        self.snp_level_summary = snp_level_summary
        self.gene_level_summary = gene_level_summary
        self.protein_coding_table = protein_coding_table
        self.network = network


    @property
    def snp_level_summary(self):
        return self._snp_level_summary


    @snp_level_summary.setter
    def snp_level_summary(self, df):
        if self.validate: 
            _validate_dataframe(
                df,
                self.snp_cols,
                var_name="snp_level_summary"
            )
        self._snp_level_summary = df


    @property
    def gene_level_summary(self):
        """pd.DataFrame : DataFrame that includes the gene_level_summary

        If the gene_level_summary is overwritten, the DataFrame must include
        ['Gene', 'Top_SNP P-value'] and the previously created pvalues
        would be destroyed.
        """

        return self._gene_level_summary


    @gene_level_summary.setter
    def gene_level_summary(self, df):
        if self.validate: 
            _validate_dataframe(
                df,
                self.gene_cols,
                var_name="gene_level_summary"
            )

        self._gene_level_summary = df

        if hasattr(self, "pvalues"):
            del self.pvalues


    @property
    def protein_coding_table(self):
        return self._protein_coding_table


    @protein_coding_table.setter
    def protein_coding_table(self, df):
        if self.validate: 
            _validate_dataframe(
                df,
                self.pc_cols,
                var_name="protein_coding_table"
            )

        self._protein_coding_table = df


    def read_snp_table(self, file, bp_col='bp', snp_pval_col='pval', snp_chrom_col='hg18chr'):
        self.snp_cols = {
            'snp_chrom_col': snp_chrom_col, 
            'bp_col' : bp_col, 
            'snp_pval_col' : snp_pval_col
        }

        self.snp_level_summary = pd.read_csv(
            file, header=0, index_col=None, sep='\s+'
        )

        return self


    def read_gene_table(self, file, gene_col='Gene', gene_pval_col='TopSNP P-Value'):
        self.gene_cols = {
            'gene_pval_col' : gene_pval_col, 
            'gene_col' : gene_col
        }

        self.gene_level_summary = pd.read_csv(
            file, sep='\t', usecols=[1,2,3,4,5,6,7,8,9]
        )

        return self


    def read_protein_coding_table(self, file, pc_chrom_col="Chromosome", start_col="Start", end_col="End"):
        self.pc_cols = {
            'pc_chrom_col' : pc_chrom_col, 
            'start_col' : start_col, 
            'end_col' : end_col
        }

        self.protein_coding_table = pd.read_csv(
            file, index_col=0, sep='\s+',
            names=[pc_chrom_col, start_col, end_col]
        )

        return self


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
        self.graphs['full_network'] = network

        return self


    def get_network_from_ndex(
        self,
        uuid="f93f402c-86d4-11e7-a10d-0ac135e8bacf", #PCNet
    ):
        anon_ndex = nc.Ndex2("http://public.ndexbio.org")
        network_niceCx = ndex2.create_nice_cx_from_server(server='public.ndexbio.org',
                                                          uuid=uuid)

        self.network = network_niceCx.to_networkx()

        return self


    def assign_pvalues(self, window_size=0, agg_method='min'):
        """Wrapper for assign_snps_to_genes"""

        if self.protein_coding_table is None:
            raise ValueError("protein_coding_table attribute must be provided!")

        if self.snp_level_summary is None:
            raise ValueError("snp_level_summary attribute must be provided!")

        assigned_pvalues = assign_snps_to_genes(
            self.snp_level_summary,
            self.protein_coding_table,
            window_size=window_size,
            agg_method=agg_method,
            to_table=True,
            snp_chrom_col=self.snp_cols['snp_chrom_col'],
            bp_col=self.snp_cols['bp_col'],
            pval_col=self.snp_cols['snp_pval_col'],
            pc_chrom_col=self.pc_cols['pc_chrom_col'],
            start_col=self.pc_cols['start_col'],
            end_col=self.pc_cols['end_col'],
        )

        if self.gene_level_summary is not None:
            warnings.warn("The existing gene_level_summary was overwritten!")

        self.gene_level_summary = assigned_pvalues

        return self


    @property
    def pvalues(self):
        """OrderedDict: `str` to `float`: Dictionary that maps genes to p-values

        Requires gene_level_summary to be set (i.e. Does not run assign_pvalues
        automatically. For now, pvalues cannot be reassigned.
        """

        if self.gene_level_summary is not None:
            if not hasattr(self, "_pvalues"):
                try:
                    self._pvalues = self.gene_level_summary[
                        [self.gene_cols['gene_col'], self.gene_cols['gene_pval_col']]
                    ]

                    self._pvalues = OrderedDict(
                        self._pvalues.\
                            set_index(self.gene_cols['gene_col']).\
                            to_dict()[self.gene_cols['gene_pval_col']]
                    )

                except AttributeError:
                    raise AttributeError("No gene level summary found! Please use assign_pvalues to convert SNP to gene-level summary or input a gene-level summary!")

        else:
            self._pvalues = None

        return self._pvalues


    @pvalues.deleter
    def pvalues(self):
        if hasattr(self, "_pvalues"):
            del self._pvalues


    @property
    def network(self):
        """networkx Graph object : Network object used for graph diffusion

        node_names attribute is automatically created if the network is a
        networkx object. If a node has a "name" attribute, that name is used
        for node_names. Otherwise, the node id itself is used as the name.
        """
        return self._network


    @network.setter
    def network(self, network):
        if network is not None and (not isinstance(network, nx.Graph) and not isinstance(network, nx.DiGraph)):
            raise ValueError("Network must be a networks Graph of DiGraph object!")

        self._network = network 

        if not hasattr(self, "graphs"): 
            self.graphs = {'full_network': network}
    
        if network is not None: 
            self.node_names = [
                self._network.node[n].get('name', n) for n in self.network.nodes()
            ]

            self.node_2_name = dict(zip(self.network.node.keys(), self.node_names))
            self.name_2_node = dict(zip(self.node_names, self.network.node.keys()))

        else: 
            self.node_names = None   


    def cache_network_data(self):
        if not hasattr(self, "adjacency_matrix"):
            self.adjacency_matrix = nx.adjacency_matrix(self.network)

        if not hasattr(self, "laplacian"):
            self.laplacian = csc_matrix(nx.laplacian_matrix(self.network))

        return self


    def convert_to_heat(
        self, 
        method='binarize', 
        replace=False, 
        fill_missing=0,
        name='Heat', 
        **kwargs
    ):
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

        heat = pd.DataFrame(
            heat[...,np.newaxis], 
            index = list(self.pvalues.keys()), 
            columns=[name]
        )
        heat = heat.reindex(self.node_names).fillna(fill_missing)

        if not hasattr(self, "heat"): 
            self.heat = heat

        else: 
            name = avoid_overwrite(name, self.heat.columns)
            self.heat.loc[:, name] = heat

        self.heat.sort_values(name, ascending=False, inplace=True)

        return self

    
    def reset_cache(self, mode="results"): 
        if mode == "results": 
            del self.heat

        elif mode == "all": 
            essentials = ['verbose', 'validate', 'snp_cols', 'gene_cols', 
                'pc_cols', '_snp_level_summary', '_gene_level_summary', 
                '_protein_coding_table', '_network', 'node_names', 
                'node_2_name', 'name_2_node']

            keys_to_delete = []
            for i in self.__dict__.keys(): 
                if i not in essentials: 
                    keys_to_delete.append(i)

            for key in keys_to_delete: 
                del self.__dict__[key]
     


    def diffuse(self, method="random_walk", heat="Heat", result_name="Diffused Heat", **kwargs):
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
            df = self.random_walk(heat=heat, **kwargs)

        elif method == "random_walk_with_restart":
            df = self.random_walk_with_kernel(heat=heat, **kwargs)

        elif method == "heat_diffusion":
            df = self.heat_diffusion(heat=heat, **kwargs)

        result_name = avoid_overwrite(result_name, self.heat.columns)
        self.heat.loc[:, result_name] = df
        self.heat.sort_values(by=result_name, ascending=False, inplace=True)

        return self


    def random_walk(self, heat='Heat', alpha=0.5):
        """Runs random walk iteratively

        Parameters
        ----------
        threshold: float
            Minimum p-value to diffuse the p-value
        alpha : float
            The restart probability

        TODO
        ----
        * Allow for diffusing multiple heat columns
        """
        if not isinstance(heat, list): 
            heat = [heat]

        if not hasattr(self, "heat"):
            warnings.warn("Attribute heat is not found. Generating using the binarize method.")
            self.convert_to_heat()

        if not hasattr(self, "adjacency_matrix"):
            self.adjacency_matrix = nx.adjacency_matrix(self.network)

        nodes = [self.network.node[i]['name'] for i in self.network.nodes()]
        common_indices, pc_ind, heat_ind = get_common_indices(nodes, self.heat.index)
        heat_mat = self.heat[heat].values.T

        F0 = heat_mat[:, heat_ind]
        A = self.adjacency_matrix[:, pc_ind][pc_ind, :]

        out = random_walk_rst(F0, A, alpha)

        df = pd.DataFrame(list(zip(common_indices, np.array(out.todense()).ravel().tolist())), columns=['Genes', 'PROPVALUES'])
        df = df.set_index('Genes').sort_values(by='PROPVALUES', ascending=False)

        return df


    def random_walk_with_kernel(self, heat="Heat", threshold=5e-6, kernel=None):
        """Runs random walk with pre-computed kernel

        This propagation method relies on a pre-computed kernel.

        Parameters
        ----------
        threshold : float
            Minimum p-value threshold to diffuse the p-value
        kernel : str
            Location of the kernel (expects to be in HDF5 format)
        """
        if not isinstance(heat, list): 
            heat = [heat]

        if kernel is not None:
            self.kernel = pd.read_hdf(kernel)

        else:
            warnings.warn("No kernel was given! Running random_walk instead")
            self.random_walk()

            return self

        if not hasattr(self, "heat"):
            warnings.warn("Attribute heat is not found. Generating using the binarize method.")
            self.convert_to_heat()

        network_genes = list(self.kernel.index)

        heat = self.heat[heat].reindex(network_genes).fillna(0) #Not saving heat to object because the kernel index may not match network's

        #propagate with pre-computed kernel
        prop_val_matrix = np.dot(heat.values.T, self.kernel)
        prop_val_table = pd.DataFrame(prop_val_matrix, index = heat.columns, columns = heat.index)

        return prop_val_table


    def heat_diffusion(self, heat="Heat", t=0.1):
        """Runs heat diffusion without a pre-computed kernel

        Parameters
        ----------
        t : float
            Total time of diffusion. t controls the amount of signal is allowed 
            to diffuse over the network.
        """

        if not isinstance(heat, list): 
            heat = [heat]

        if not hasattr(self, "laplacian"):
            self.laplacian = csc_matrix(nx.laplacian_matrix(self.network))

        if not hasattr(self, "heat"):
            warnings.warn("Attribute heat is not found. Generating using the binarize method.")
            self.convert_to_heat()

        out_vector = heat_diffusion(
            self.laplacian, 
            self.heat[heat].values.ravel(), 
            start=0, 
            end=t
        )

        out_dict= {'prop': out_vector,'Gene':self.node_names}
        heat_df=pd.DataFrame.from_dict(out_dict).set_index('Gene')

        return heat_df


    def annotate_network(self, values="all"):
        """Return a subgraph with node attributes

        Parameters
        ----------
        values : str
            Name of the column
        """

        if values=="all":
            data = self.heat.to_dict()
        else:
            data = self.heat[values].to_dict()

            if isinstance(values, str): 
                data = {values:data}

        for key, d in data.items(): 
            d = {self.name_2_node[k]:v for k,v in d.items()}
            nx.set_node_attributes(self.network, key, d)

        return self


    def get_subgraph(self, gene, neighbors=1, name="subgraph"): 
        """Gets a subgraph center on a node

        Parameter
        ---------
        gene : str
            The name of the node (not the ID) 
        neighbors : int 
            The number of 
        """

        center = self.name_2_node[gene]
        nodes = get_neighbors(self.network, neighbors, center)
        G = self.network.subgraph(nodes)

        self.graphs[name] = G

        return self


    def view(self, 
        name="subgraph", 
        attributes="Heat", 
        vmin=0, 
        vmax=1, 
        cmap=plt.cm.Blues
    ): 
        """Plot the subgraph"""

        try: 
            G = self.graphs[name]
        except KeyError: 
            raise KeyError("%s is not in the self.graphs dictionary!" % name)

        fig, ax = plt.subplots()

        attr = nx.get_node_attributes(G, attributes)

        try: 
            vals = [attr[i] for i in G.nodes()]
        except KeyError: 
            warnings.warn("The specified graph does not have the attribute %s. Replacing values with 0.")
            vals = [0 for _ in G.nodes()]

        nx.draw(
            self.graphs[name], 
            ax=ax,
            node_color=vals,
            labels=nx.get_node_attributes(G, "name"), #TODO: "name" may not be in node attributes
            vmin=vmin, 
            vmax=vmax, 
            cmap=cmap
        )

        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        )

        sm._A = []
        plt.colorbar(sm)

        return fig, ax


    def to_ndex(
        self, 
        name="subgraph", 
        server="http://test.ndexbio.org", 
        username="scratch2", 
        password="scratch2"
    ):

        try: 
            g = ndex2.create_nice_cx_from_networkx(self.graphs[name])
        except KeyError: 
            raise KeyError("%s is not in self.graphs dictionary!" % name)

        uuid = g.upload_to(
            server=server,
            username=username,
            password=password 
        )

        warnings.warn("Upload to ndex currently fails. Not sure why..")

        return uuid
        
