"""Tables needed for Nbgwas"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .utils import get_neighbors, binarize, neg_log_val

class Genes(object): 
    def __init__(
        self, 
        table, 
        pval_col='TopSNP P-Value', 
        name_col='Gene', 
        use_index=False
    ): 
        """Stores the gene information

        Parameters
        ----------
        table : pd.DataFrame 
            Pandas DataFrame that store all gene information
        pval_col : str 
            Column name that corresponds to p-values
        name_col : str 
            Column name that corresponds to the gene names
        use_index : bool
            If true, the index will be used as the gene names
        """

        self.table = table 
        self.pval_col = pval_col 
        self.name_col = name_col 
        self.use_index = use_index 

        self._validate()


    def __repr__(self): 
        if self.table is not None: 
            length = self.table.shape[0]

        else: 
            length = 0

        return f'<{self.__class__.__name__}> object containing {length} genes'


    def from_file(
        self, 
        path, 
        pval_col='TopSNP P-Value', 
        name_col='Gene', 
        use_index=False, 
        **kwargs
    ):

        self.table = pd.read_csv(path, **kwargs)
        self.pval_col = pval_col 
        self.name_col = name_col 
        self.use_index = use_index

        return self        


    def _validate(self): 
        #TODO
        
        pass


    @property 
    def names(self): 
        if self.use_index: 
            return self.table.index.values 
        
        else: 
            return self.table[self.name_col].values.ravel()


    def set_names(self, name_col=None, names=None, overwrite=False): 
        #TODO
        pass 


    @property 
    def pvalues(self): 
        return self.table[self.pval_col].values.ravel()


    def convert_to_heat(
        self,
        method='binarize',
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
        """

        allowed = ['binarize', "neg_log"]
        if method not in allowed:
            raise ValueError("Method must be in %s" % allowed)

        if method == 'binarize':
            heat = binarize(self.pvalues, threshold=kwargs.get('threshold', 5e-6))
        elif method == 'neg_log':
            heat = neg_log_val(
                self.pvalues, 
                floor=kwargs.get('floor', None), 
                ceiling=kwargs.get('ceiling', None)
            )
            
            #TODO: Why is this not ```heat = neg_log_val(self.pvalues, **kwargs)```

        if normalize is not None: 
            heat = (heat/heat.sum())*normalize

        self.table[name] = heat
        self.table.sort_values(name, ascending=False, inplace=True)

        return self

    def normalize_by_gene_length(
        self, 
        column, 
        out_name="Normalized", 
        gene_lengths=None, 
        gene_length_col=None
    ):

        """Normalize a column by the gene length
        
        Parameters
        ----------
        column : str 
            Column name in self.table 
        out_name : str
            Column name of the result to be added to self.table
        gene_lengths : list
            List of gene lengths corresponding to each row of the table
        gene_length_col : str
            Column name of that corresponds to gene_length         
        """

        if gene_lengths is not None and gene_length_col is not None: 
            raise ValueError("Only either gene_lengths or gene_length_col can be supplied.")

        if gene_length_col is not None: 
            gene_size = self.table[gene_length_col].values
        else: 
            gene_size = np.array(gene_lengths).ravel()

        init_heat = self.table[column].values

        regression_model = LinearRegression()
        regression_model.fit(gene_size, init_heat)

        y_pred = regression_model.predict(gene_size)
        residual = init_heat - y_pred
        y_adjusted = residual + np.abs(residual.min())
        
        self.table[out_name] = y_adjusted 

        return self        


class Snps(object): 
    def __init__(
        self, 
        snp_table, 
        protein_coding_table,
        snp_chrom_col='hg18chr', 
        snp_bp_col='bp', 
        pval_col='pval',
        pc_chrom_col='Chrom',
        start_col='Start', 
        end_col='End',
    ): 

        """Stores snp information

        Parameters
        ----------
        snp_table: pd.DataFrame
            Pandas DataFrame that stores the SNP pvalues and locations
        chrom_col : str
            Column name that corresponds to the chromosome of the SNP is on
        bp_col : str
            Column name that corresponds to the basepair the SNP is located 
        pval_col : str
            Column name that corresponds to the basepair the pvalues are in 
        """

        self.snp_table = snp_table 
        self.snp_chrom_col = snp_chrom_col 
        self.snp_bp_col = snp_bp_col 
        self.pval_col = pval_col 

        self.protein_coding_table = protein_coding_table
        self.pc_chrom_col = pc_chrom_col
        self.start_col = start_col 
        self.end_col = end_col

        self._validate()
        
    def __repr__(self): 
        return f'<{self.__class__.__name__}> object'
        

    def from_files(
        self, 
        snp_file, 
        protein_coding_file, 
        snp_chrom_col='hg18chr', 
        snp_bp_col='bp', 
        pval_col='pval',
        pc_chrom_col='Chrom',
        start_col='Start', 
        end_col='End',
        snp_kwargs={}, 
        pc_kwargs={}, 
    ):
        snp_table = pd.read_csv(snp_file, **snp_kwargs)
        protein_coding_table = pd.read_csv(protein_coding_file, **pc_kwargs) 

        self.snp_table = snp_table
        self.protein_coding_table = protein_coding_table
        self.snp_chrom_col = snp_chrom_col
        self.snp_bp_col = snp_bp_col
        self.pval_col = pval_col
        self.pc_chrom_col = pc_chrom_col
        self.start_col = start_col
        self.end_col = end_col

        return self


    def _validate(self): 
        #TODO

        pass


    @property 
    def pvalues(self): 
        return self.snp_table[self.pval_col].values.ravel()


    def assign_snps_to_genes(
        self,
        window_size=0,
        agg_method='min',
        to_table=False,
        to_Gene=True,
    ):

        """Assigns SNP to genes

        Parameters
        ----------
        window_size : int or float
            Move the start site of a gene back and move the end site forward
            by a fixed `window_size` amount.
        agg_method : str or callable function
            Method to aggregate multiple p-values associated with a SNP. If min 
            is selected, the position of the SNP that corresponds to the min 
            p-value is also returned. Otherwise, the position column is filled 
            with NaN.
            - min : takes the minimum p-value
            - median : takes the median of all associated p-values
            - mean : takes the average of all assocaited p-values
            - <'callable' function> : a function that takes a list and output
            a value. The output of this value will be used in the final 
            dictionary.
        to_table : bool
            If to_table is true, the output is a pandas dataframe that augments the 
            pc dataframe with number of SNPs, top SNP P-value, and the position of 
            the SNP for each gene. Otherwise, a dictionary of gene to top SNP 
            P-value is returned. *Note*: The current behavior for the output table 
            is that if a coding gene is duplicated, only the first row will be kept.

        Output
        ------
        assigned_pvals : dict or pd.Dataframe
            A dictionary of genes to p-value (see to_table above)

        TODO
        ----
        - Change pc to something more descriptive
        - Add an option for caching bin edges
        - Change output format to include additional information about multiple 
            coding regions for a gene
        """

        """Input validation and Type Enforcement"""

        window_size = int(window_size)

        if agg_method not in ['min', 'median', 'mean'] and not hasattr(agg_method, '__call__'):
            raise ValueError('agg_method must be min, median, mean or a callable function!')

        try:
            self.snp_table[self.snp_chrom_col] = self.snp_table[self.snp_chrom_col].astype(str)
            self.snp_table[self.snp_bp_col] = self.snp_table[self.snp_bp_col].astype(int)
        except ValueError:
            raise ValueError("Column bp_col from `snp` cannot be coerced into int!")

        try:
            self.snp_table[self.pval_col] = self.snp_table[self.pval_col].astype(float)
        except ValueError:
            raise ValueError("Column pval_col from `snp` cannot be coerced into float!")

        try:
            self.protein_coding_table[self.pc_chrom_col] = self.protein_coding_table[self.pc_chrom_col].astype(str)
            self.protein_coding_table[[self.start_col, self.end_col]] = self.protein_coding_table[[self.start_col, self.end_col]].astype(int)
        except ValueError:
            raise ValueError("Columns start and end from `pc` cannot be coerced into int!")

        #PC validation code here
        if not set(self.protein_coding_table[self.pc_chrom_col]).issuperset(set(self.snp_table[self.snp_chrom_col])):
            raise ValueError(
                "pc_chrom_col column from pc is expected to be a superset ", 
                "of snp_chrom_col from snp!"
            )

        """Real Code"""

        assigned_pvals = defaultdict(lambda: [[], []])
        for chrom, df in self.snp_table.groupby(self.snp_chrom_col):
            pc_i = self.protein_coding_table.loc[self.protein_coding_table[self.pc_chrom_col] == str(chrom)]

            if pc_i.shape[0] == 0:
                raise RuntimeError("No proteins found for this chromosome!")

            bins, names = _get_bins(
                pc_i, 
                window_size=window_size, 
                cols=[self.start_col, self.end_col]
            )

            bps = df[self.snp_bp_col].values
            binned = np.digitize(bps, bins)

            names = names[binned]
            pvals = df[self.pval_col].values

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

            if not to_table and not to_Gene:
                assigned_pvals[i] = p
            else:
                assigned_pvals[i] = [len(j[0]), p, pos] #nSNPS, TopSNP-pvalue, TopSNP-pos

        if to_table or to_Gene:
            assigned_df = pd.DataFrame(
                assigned_pvals, 
                index=['nSNPS', 'TopSNP P-Value', 'TopSNP Position']).T

            gene_lengths = []
            for val, df in self.protein_coding_table.groupby(self.protein_coding_table.index): 
                tmp = df[[self.start_col, self.end_col]].astype(str).values
                gene_lengths.append((val, df[self.pc_chrom_col].values[0], ','.join(['-'.join(i) for i in tmp.tolist()])))
                
            gene_lengths_df = pd.DataFrame(gene_lengths, columns=['Gene', 'Chrom', 'Start-End'])
            gene_lengths_df = gene_lengths_df.set_index('Gene')

            assigned_df = pd.concat([gene_lengths_df, assigned_df], axis=1, sort=True)

            assigned_df.index.name = 'Gene'
            assigned_df = assigned_df.reset_index()

            assigned_df = assigned_df.loc[pd.notnull(assigned_df.iloc[:, -1])] #Remove genes that do not have 
                                                                            #p-values

            if to_Gene: 
                return Genes(assigned_df, pval_col='TopSNP P-Value', name_col='Gene')

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

    If cols contain strings, the dataframe is indexed by column names. 
    Otherwise, the dataframe will be extracted by column number.
    """
    names = df.index.values

    if isinstance(cols[0], str): 
        arr = df.loc[:, cols]
    
    else: 
        arr = df.iloc[:, cols] 

    arr = arr.values.astype(int)

    arr[:, 0] -= window_size
    arr[:, 1] += window_size

    bins = np.sort(arr.reshape(-1))

    mapped_names = [set([]) for _ in range(len(bins) + 1)]

    for ind, (i,j) in enumerate(arr):
        vals = np.argwhere((bins > i) & (bins <= j)).ravel()

        for v in vals:
            mapped_names[v] = mapped_names[v].union([names[ind]])

    return bins, np.array(mapped_names)

