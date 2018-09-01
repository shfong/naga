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
from .utils import get_neighbors
import time
import warnings

def assign_snps_to_genes(snp,
                         pc,
                         window_size=0,
                         to_table=False,
                         agg_method='min',
                         snp_chrom_col='hg18chr',
                         bp_col='bp',
                         pval_col='pval',
                         pc_chrom_col='Chrom',
                         start_col='Start',
                         end_col='End', ):

    """Assigns SNP to genes

    Parameters
    ----------
    snp : pd.DataFrame
        pandas DataFrame of SNP summary statistics from GWAS. It must have the
        following three columns
        - chromosome (chrom_col): the chromosome the SNP is on (str)
        - basepair (bp_col): the base pair number (Position)
        - p-value (pval_col): the GWAS associated p-value
    pc : pd.DataFrame
        pandas DataFrame of gene coding region. It must have the following 3 columns
        and index
        - Chromosome: Chromosome Name (str). The chromosome name must be consistent
            with the ones defined in snp[chrom_col]. This columns is expected to be a
            superset of the snp chromosome column.
        - Start (int)
        - End (int)
    window_size : int or float
        Move the start site of a gene back and move the end site forward
        by a fixed `window_size` amount.
    agg_method : str or callable function
        Method to aggregate multiple p-values associated with a SNP. If min is selected,
        the position of the SNP that corresponds to the min p-value is also returned.
        Otherwise, the position column is filled with NaN.
        - min : takes the minimum p-value
        - median : takes the median of all associated p-values
        - mean : takes the average of all assocaited p-values
        - <'callable' function> : a function that takes a list and output
          a value. The output of this value will be used in the final dictionary.
    to_table : bool
        If to_table is true, the output is a pandas dataframe that augments the pc
        dataframe with number of SNPs, top SNP P-value, and the position of the SNP
        for each gene. Otherwise, a dictionary of gene to top SNP P-value is returned.
        *Note*: The current behavior for the output table is that if a coding gene is
        duplicated, only the first row will be kept.

    Output
    ------
    assigned_pvals : dict or pd.Dataframe
        A dictionary of genes to p-value (see to_table above)

    TODO
    ----
    - Change pc to something more descriptive
    - Add an option for caching bin edges
    - Change output format to include additional information about multiple coding regions for a gene
    """

    """Input validation and Type Enforcement"""

    window_size = int(window_size)

    if agg_method not in ['min', 'median', 'mean'] and not hasattr(agg_method, '__call__'):
        raise ValueError('agg_method must be min, median, mean or a callable function!')

    try:
        snp[snp_chrom_col] = snp[snp_chrom_col].astype(str)
        snp[bp_col] = snp[bp_col].astype(int)
    except ValueError:
        raise ValueError("Column bp_col from `snp` cannot be coerced into int!")

    try:
        snp[pval_col] = snp[pval_col].astype(float)
    except ValueError:
        raise ValueError("Column pval_col from `snp` cannot be coerced into float!")

    try:
        pc[pc_chrom_col] = pc[pc_chrom_col].astype(str)
        pc[[start_col, end_col]] = pc[[start_col, end_col]].astype(int)
    except ValueError:
        raise ValueError("Columns start and end from `pc` cannot be coerced into int!")

    #PC validation code here
    if not set(pc[pc_chrom_col]).issuperset(set(snp[snp_chrom_col])):
        raise ValueError("pc_chrom_col column from pc is expected to be a superset of snp_chrom_col from snp!")


    """Real Code"""

    assigned_pvals = defaultdict(lambda: [[], []])
    for chrom, df in snp.groupby(snp_chrom_col):
        pc_i = pc.loc[pc[pc_chrom_col] == str(chrom)]

        if pc_i.shape[0] == 0:
            raise RuntimeError("No proteins found for this chromosome!")

        bins, names = _get_bins(pc_i, window_size=window_size)
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
        f = np.median
    elif agg_method == 'mean':
        f = np.mean
    else:
        f = agg_method

    for i,j in assigned_pvals.items():
        if agg_method == 'min':
            pos = j[1][f(j[0])]
            p = j[0][f(j[0])]

        else:
            pos = np.nan
            p = f(j[0])

        if not to_table:
            assigned_pvals[i] = p
        else:
            assigned_pvals[i] = [len(j[0]), p, pos] #nSNPS, TopSNP-pvalue, TopSNP-pos

    if to_table:
        assigned_df = pd.DataFrame(assigned_pvals, index=['nSNPS', 'TopSNP P-Value', 'TopSNP Position']).T

        pc =  pc[~pc.index.duplicated(keep='first')] # TODO: Change this

        assigned_df = pd.concat([pc, assigned_df], axis=1, sort=True)
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

    mapped_names = [set([]) for _ in range(len(bins) + 1)]

    for ind, (i,j) in enumerate(arr):
        vals = np.argwhere((bins > i) & (bins <= j)).ravel()

        for v in vals:
            mapped_names[v] = mapped_names[v].union([names[ind]])

    return bins, np.array(mapped_names)


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


def _validate_dataframe(df, require_columns, var_name="df"):
    if df is None:
        return None

    if not isinstance(df, pd.DataFrame):
        raise ValueError("%s must be a pandas DataFrame!" % var_name)
    else:
        if not set(df.columns).issuperset(set(require_columns.values())):
            raise ValueError("%s must include %s" % ( #TODO: This needs to be better (see github issue #2)
                var_name, ",".join(require_columns.keys())
            ))


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
        verbose=True
    ):

        self.snp_chrom_col = snp_chrom_col
        self.bp_col = bp_col
        self.snp_pval_col = snp_pval_col

        self.gene_col = gene_col
        self.gene_pval_col = gene_pval_col

        self.pc_chrom_col = pc_chrom_col
        self.start_col = start_col
        self.end_col = end_col

        self.snp_level_summary = snp_level_summary
        self.gene_level_summary = gene_level_summary
        self.protein_coding_table = protein_coding_table
        self.network = network

        self.verbose = verbose


    @property
    def snp_level_summary(self):
        return self._snp_level_summary


    @snp_level_summary.setter
    def snp_level_summary(self, df):
        _validate_dataframe(
            df,
            {
                'snp_chrom_col': self.snp_chrom_col,
                'bp_col': self.bp_col,
                'snp_pval_col': self.snp_pval_col
            },
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
        _validate_dataframe(
            df,
            {
                'gene_col': self.gene_col,
                'pval_col': self.gene_pval_col
            },
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
        _validate_dataframe(
            df,
            {
                'pc_chrom_col': self.pc_chrom_col,
                'start_col': self.start_col,
                'end_col': self.end_col
            },
            var_name="protein_coding_table"
        )

        self._protein_coding_table = df


    def read_snp_table(self, file, bp_col='bp', snp_pval_col='pval', snp_chrom_col='hg18chr'):
        self.bp_col = bp_col
        self.snp_pval_col = snp_pval_col
        self.snp_chrom_col = snp_chrom_col

        self.snp_level_summary = pd.read_csv(
            file, header=0, index_col=None, sep='\s+'
        )

        return self


    def read_gene_table(self, file, gene_col='Gene', gene_pval_col='TopSNP P-Value'):
        self.gene_col = gene_col
        self.gene_pval_col = gene_pval_col

        self.gene_level_summary = pd.read_csv(
            file, sep='\t', usecols=[1,2,3,4,5,6,7,8,9]
        )

        return self


    def read_protein_coding_table(self, file, pc_chrom_col="Chromosome", start_col="Start", end_col="End"):
        self.pc_chrom_col = pc_chrom_col
        self.start_col = start_col
        self.end_col = end_col

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
            snp_chrom_col=self.snp_chrom_col,
            bp_col=self.bp_col,
            pval_col=self.snp_pval_col,
            pc_chrom_col=self.pc_chrom_col,
            start_col=self.start_col,
            end_col=self.end_col,
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
                    self._pvalues = self.gene_level_summary[[self.gene_col, self.gene_pval_col]]
                    self._pvalues = OrderedDict(self._pvalues.set_index(self.gene_col).to_dict()[self.gene_pval_col])
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


    def convert_to_heat(self, method='binarize', replace=False, fill_missing=0,name='Initial Heat', **kwargs):
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
            columns=['Heat']
        )
        heat = heat.reindex(self.node_names).fillna(fill_missing)
        heat = heat.sort_values('Heat', ascending=False)

        if not hasattr(self, "heat"): 
            self.heat = heat

        else: 
            self.heat.loc[:, name] = heat

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
            df = self.heat_diffusion(**kwargs)

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


    def heat_diffusion(self, t=0.1):
        """Runs heat diffusion without a pre-computed kernel

        Parameters
        ----------
        t : float
            Total time of diffusion. t controls the amount of signal is allowed to diffuse over the network.
        """

        if not hasattr(self, "laplacian"):
            self.laplacian = csc_matrix(nx.laplacian_matrix(self.network))

        if not hasattr(self, "heat"):
            warnings.warn("Attribute heat is not found. Generating using the binarize method.")
            self.convert_to_heat()

        out_vector=expm_multiply(-self.laplacian, self.heat.values.ravel(), start=0, stop=t, endpoint=True)[-1]
        out_dict= {'prop': out_vector,'Gene':self.node_names}
        heat_df=pd.DataFrame.from_dict(out_dict).set_index('Gene')

        self.boosted_pvalues = heat_df.sort_values(by='prop', ascending=False)

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


    def get_subgraph(self, gene, neighbors=1, attributes="Heat", name="subgraph"): 
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


    def view(self, name="subgraph", attributes="Heat", vmin=0, vmax=1, cmap=plt.cm.Blues): 
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
        
