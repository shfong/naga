"""Network boosted GWAS package

This package takes GWAS SNP level summary statistics and re-prioritizes
using network data
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
from scipy.stats import hypergeom
import time
import warnings

from py2cytoscape.data.cyrest_client import CyRestClient

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
        req_col_vals = set(require_columns.values())
        in_cols = set(df.columns)
        if not in_cols.issuperset(req_col_vals):
            missing_columns = req_col_vals.difference(in_cols)

            raise ValueError("%s must include %s. " % (
                var_name, ",".join(require_columns.keys())
            ), "" "The following columns are missing from %s: %s" % (
                   var_name,
                   ",".join(missing_columns)
               )
            )

def avoid_overwrite(name, iterable):
    """Take a list to and see if the string is in the iterable.

    If it is, then the string is returned with the current count. Otherwise,
    the string itself is returned.
    """

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

        self.node_name = node_name # The attribute contains the gene name
                                   # on the network

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


    def read_snp_table(
        self,
        file,
        bp_col='bp',
        snp_pval_col='pval',
        snp_chrom_col='hg18chr'
    ):

        self.snp_cols = {
            'snp_chrom_col': snp_chrom_col,
            'bp_col' : bp_col,
            'snp_pval_col' : snp_pval_col
        }

        self.snp_level_summary = pd.read_csv(
            file, header=0, index_col=None, sep='\s+'
        )

        return self


    def read_gene_table(
        self,
        file,
        gene_col='Gene',
        gene_pval_col='TopSNP P-Value'
    ):

        self.gene_cols = {
            'gene_pval_col' : gene_pval_col,
            'gene_col' : gene_col
        }

        self.gene_level_summary = pd.read_csv(
            file, sep='\t', usecols=[1,2,3,4,5,6,7,8,9]
        )

        return self


    def read_protein_coding_table(
        self,
        file,
        pc_chrom_col="Chromosome",
        start_col="Start",
        end_col="End"
    ):

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


    def read_cx_file(self, file, node_name="name"):
        """Load CX file as network"""

        self.node_name = node_name
        network = ndex2.create_nice_cx_from_file(file).to_networkx()
        self.network = network

        return self


    def read_nx_pickle_file(self, file, node_name="name"):
        """Read networkx pickle file as network"""

        self.node_name = node_name

        network = nx.read_gpickle(file)
        self.network = network
        self.graphs['full_network'] = network

        return self


    def get_network_from_ndex(
        self,
        uuid="f93f402c-86d4-11e7-a10d-0ac135e8bacf", #PCNet
        node_name="name",
    ):

        self.node_name = node_name

        #anon_ndex = nc.Ndex2("http://public.ndexbio.org")
        network_niceCx = ndex2.create_nice_cx_from_server(
            server='public.ndexbio.org',
            uuid=uuid
        )

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

        gene_col = self.gene_cols['gene_col']
        pval_col = self.gene_cols['gene_pval_col']

        if self.gene_level_summary is not None:
            if not hasattr(self, "_pvalues"):
                self._pvalues = self.gene_level_summary[
                    [gene_col, pval_col]
                ]

                self._pvalues = self._pvalues.set_index(gene_col)
                self._pvalues = self._pvalues.sort_values(
                    by=pval_col,
                    ascending=True
                )

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
        networkx object. If a node has a `self.node_name` attribute, that name
        is used for node_names. Otherwise, the node id itself is used as the
        name.
        """

        return self._network


    @network.setter
    def network(self, network):
        if network is not None and not isinstance(network, nx.Graph):
            raise ValueError(
                "Network must be a networks Graph of DiGraph object!"
            )

        self._network = network

        if not hasattr(self, "graphs"):
            self.graphs = {'full_network': network}

        if network is not None:
            self.node_names = [
                self._network.node[n].get(self.node_name, n) \
                    for n in self.network.nodes()
            ]

            self.node_2_name = dict(zip(
                self.network.node.keys(), self.node_names
            ))

            self.name_2_node = dict(zip(
                self.node_names, self.network.node.keys()
            ))

        else:
            self.node_names = None


    def cache_network_data(self):
        if not hasattr(self, "adjacency_matrix"):
            self.adjacency_matrix = nx.adjacency_matrix(self.network)

        if not hasattr(self, "laplacian"):
            self.laplacian = csc_matrix(nx.laplacian_matrix(self.network))

        return self


    def extract_network_attributes(self, pvalues=None, heat=None, genes="name"):
        """Build internal dataframes from network attributes"""

        df = pd.DataFrame.from_dict(self.network.node, orient="index")

        self.gene_cols = {"gene_col": genes, "gene_pval_col": pvalues}
        self.gene_level_summary = df.loc[:, [genes, pvalues]]

        if not isinstance(heat, list):
            heat = [heat]

        self.heat = df.loc[:, heat]
        self.heat.index = [self.node_2_name[i] for i in self.heat.index]

        return self


    def convert_to_heat(
        self,
        method='binarize',
        fill_missing=0,
        name='Heat',
        normalize=None,
        **kwargs
    ):
        """Convert p-values to heat

        Parameters
        ----------
        method : str
            Must be in 'binarize' or 'neg_log'
            "binarize" uses the `binarize` function while "neg_log" uses the
            `neg_log_val` function. `binarize` places a heat of 1 if the
            p-value is < threshold otherwise 0. `neg_log` scales the p-value
            using the following function, $$f(x) = -log(x)$$
        name : str
            The column name that will for the self.heat dataframe
        fill_missing : float
            A value to give the heat if a node is available in the network,
            but not in the p-values
        normalize : float
            If provided, the total amount of input heat is scaled to the 
            specified value. Otherwise, no transformation will be done.
        kwargs
            Any additional keyword arguments to be passed into the above
            functions

            For binarize:
            * threshold : float
                Default to 5*10^-6.

            For neg_log:
            * floor : float
                Default to None. If floor is provided, any converted value
                below the floor is dropped to 0. If None, no additional
                transformation is done.

        TODO
        ----
        - Implement other methods to convert p-values to heat
        """

        allowed = ['binarize', "neg_log"]
        if method not in allowed:
            raise ValueError("Method must be in %s" % allowed)

        vals = self.pvalues.values.ravel()
        if method == 'binarize':
            heat = binarize(vals, threshold=kwargs.get('threshold', 5e-6))
        elif method == 'neg_log':
            heat = neg_log_val(vals, floor=kwargs.get('floor', None),
                                ceiling=kwargs.get('ceiling', 10.0))

        heat = pd.DataFrame(
            heat[...,np.newaxis],
            index = self.pvalues.index,
            columns=[name]
        )
        heat = heat.reindex(self.node_names).fillna(fill_missing)

        if normalize is not None: 
            heat = (heat/heat.sum())*normalize

        if not hasattr(self, "heat"):
            self.heat = heat

        else:
            name = avoid_overwrite(name, self.heat.columns)
            self.heat.loc[:, name] = heat

        self.heat.sort_values(name, ascending=False, inplace=True)

        return self


    def get_rank(self):
        """Gets the ranking of each heat and pvalues"""

        def convert_to_rank(series, name):
            series = series.sort_values(ascending=True if name == "P-values" else False)

            return pd.DataFrame(
                np.arange(1, len(series) + 1),
                index=series.index,
                columns=[name]
            )

        ranks = []
        for col in self.heat.columns:
            ranks.append(convert_to_rank(self.heat[col], col))

        ranks.append(convert_to_rank(self.pvalues[self.gene_cols['gene_pval_col']], "P-values"))

        self.ranks = pd.concat(ranks, axis=1, sort=False)

        return self


    def reset_cache(self, mode="results"):
        """Deletes the intermediate states of the object

        Parameters
        ----------
        mode : str
            Either "results" or "all". If `mode` is `results`, only the heat
            attribute will be deleted. `convert_2_heat` will need to run again.
            If `mode` is `all`, any generated intermediates will be deleted.
        """

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

        result_name = avoid_overwrite(result_name, self.heat.columns)
        self.heat.loc[:, result_name] = df
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
                "Attribute heat is not found. ",
                "Generating using the binarize method."
            )

            self.convert_to_heat()

        if not hasattr(self, "adjacency_matrix"):
            self.adjacency_matrix = nx.adjacency_matrix(self.network)

        common_indices, pc_ind, heat_ind = get_common_indices(
            self.node_names,
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
                "Attribute heat is not found. ",
                "Generating using the binarize method."
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
                "Attribute heat is not found. ",
                "Generating using the binarize method."
            )

            self.convert_to_heat()

        out_vector = heat_diffusion(
            self.laplacian,
            self.heat.loc[self.node_names, heat].values.ravel(),
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
            data.update(self.pvalues.to_dict())

        elif values == "p-values":
            data = self.pvalues.to_dict()

        else:
            data = self.heat[values].to_dict()

            if isinstance(values, str):
                data = {values:data}

        for key, d in data.items():
            d = {self.name_2_node[k]:v for k,v in d.items() \
                if k in self.name_2_node}

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
        """Plot the subgraph

        Parameters
        ----------
        name : str
            The key in self.graphs that contains the graph
        attributes : str
            The node attributes on the network (in self.graphs) to be
            visualized. Note that this means, `annotate_network` should
            be used first to add the node attributes
        vmin : float
            The lower end of the colorbar
        vmax : float
            The upper end of the colorbar
        cmap :
            Matplotlib colormap object
        """

        try:
            G = self.graphs[name]
        except KeyError:
            raise KeyError("%s is not in the self.graphs dictionary!" % name)

        fig, ax = plt.subplots()

        attr = nx.get_node_attributes(G, attributes)

        try:
            vals = [attr[i] for i in G.nodes()]
        except KeyError:
            warnings.warn(
                "The specified graph does not have the attribute %s. ",
                "Replacing values with 0."
            )
            vals = [0 for _ in G.nodes()]

        nx.draw(
            self.graphs[name],
            ax=ax,
            node_color=vals,
            labels=nx.get_node_attributes(G, self.node_name),
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


    def view_in_cytoscape(self, name="subgraph"):
        """Ports subgraph to Cytoscape"""

        if not hasattr(self, "cyrest"):
            self.cyrest = CyRestClient()

        hdl = self.cyrest.network.create_from_networkx(self.graphs[name])
        self.cyrest.layout.apply(name='degree-circle', network=hdl)

        return self


    def to_ndex(
        self,
        name="subgraph",
        server="http://test.ndexbio.org",
        username="scratch2",
        password="scratch2"
    ):

        """Uploads graph to NDEx

        Parameters
        ----------
        name : str
            The key in self.graphs that contains the graph
        server: str
            The NDEx server hyperlink
        username : str
            Username of the NDEx account
        password : str
            Password of the NDEx account
        """

        try:
            g = ndex2.create_nice_cx_from_networkx(self.graphs[name])
        except KeyError:
            raise KeyError("%s is not in self.graphs dictionary!" % name)

        uuid = g.upload_to(
            server=server,
            username=username,
            password=password
        )

        return uuid


    def hypergeom(self, gold, top=100, ngenes=20000, rank_col=None):
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

        if rank_col is None:
            genes = self.pvalues.sort_values(by=self.gene_cols['gene_pval_col'])
            genes = genes.iloc[:top].index

        else:
            genes = self.heat.sort_values(by=rank_col, ascending=False)
            genes = genes.iloc[:top].index

        intersect = set(genes).intersection(set(gold))
        score = len(intersect)
        M, n, N = ngenes, len(gold), top

        pvalue = 1 - hypergeom.cdf(score, M, n, N)
        Hypergeom = namedtuple('Hypergeom',
            ['pvalue', 'n_intersect', 'common_items']
        )

        return Hypergeom(pvalue, score, intersect)

    def check_significance(self, gold, top=100, threshold=0.05, rank_col=None):
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

        if rank_col is None:
            genes = self.pvalues.sort_values(by=self.gene_cols['gene_pval_col'])
            genes = genes.iloc[:top].index

        else:
            genes = self.heat.sort_values(by=rank_col, ascending=False)
            genes = genes.iloc[:top].index

        return sum([gold.get(i, 1) < threshold for i in genes])