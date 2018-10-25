from collections import defaultdict
import numpy as np
import pandas as pd

def assign_snps_to_genes(
    snp,
    pc,
    window_size=0,
    to_table=False,
    agg_method='min',
    snp_chrom_col='hg18chr',
    bp_col='bp',
    pval_col='pval',
    pc_chrom_col='Chrom',
    start_col='Start',
    end_col='End'
):

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
        pandas DataFrame of gene coding region. It must have the following 3 
        columns and index
        - Chromosome: Chromosome Name (str). The chromosome name must be 
            consistent with the ones defined in snp[chrom_col]. This columns is 
            expected to be a superset of the snp chromosome column.
        - Start (int)
        - End (int)
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
        raise ValueError(
            "pc_chrom_col column from pc is expected to be a superset ", 
            "of snp_chrom_col from snp!"
        )


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
        assigned_df = pd.DataFrame(
            assigned_pvals, 
            index=['nSNPS', 'TopSNP P-Value', 'TopSNP Position']).T

        gene_lengths = []
        for val, df in pc.groupby(pc.index): 
            tmp = df[['Start', 'End']].astype(str).values
            gene_lengths.append((val, df['Chrom'].values[0], ','.join(['-'.join(i) for i in tmp.tolist()])))
            
        gene_lengths_df = pd.DataFrame(gene_lengths, columns=['Gene', 'Chrom', 'Start-End'])
        gene_lengths_df = gene_lengths_df.set_index('Gene')

        assigned_df = pd.concat([gene_lengths_df, assigned_df], axis=1, sort=True)

        assigned_df.index.name = 'Gene'
        assigned_df = assigned_df.reset_index()

        assigned_df = assigned_df.loc[pd.notnull(assigned_df.iloc[:, -1])] #Remove genes that do not have 
                                                                           #p-values

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