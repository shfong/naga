import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from nbgwas import assign_snps_to_genes
from nbgwas.nbgwas import _get_bins

"""Tests for assign_snps_to_genes"""

def test_assign_snps_to_genes_nonoverlapping():
    snp, pc = _setup_nonoverlapping_inputs()

    out = assign_snps_to_genes(snp, pc)

    assert dict(out) == {'A': 0.2, 'C': 0.3, 'D':0.4}

def test_assign_snps_to_genes_nonoverlapping_with_window_size():
    snp, pc = _setup_nonoverlapping_inputs()

    out = assign_snps_to_genes(snp, pc, window_size=10)

    assert dict(out) == {'A': 0.1, 'C': 0.3, 'D':0.4}

def test_assign_snps_to_genes_overlapping():
    snp, pc = _setup_nonoverlapping_inputs()

    out = assign_snps_to_genes(snp, pc, window_size=10)

    assert dict(out) == {'A': 0.1, 'C': 0.3, 'D':0.4}

def test_assign_snps_to_genes_overlapping_with_window_size():
    snp, pc = _setup_nonoverlapping_inputs() #Start with non-overlapping
                                           #Use window size to create overlapping case

    out = assign_snps_to_genes(snp, pc, window_size=60)

    assert dict(out) == {'A': 0.1, 'C': 0.3, 'D':0.4}

def test_assign_snps_to_genes_multiple_inputs():
    snp, pc = _setup_multiple_inputs()

    out = assign_snps_to_genes(snp, pc)
    assert dict(out) == {'A':0.2, 'B':0.01, 'D':0.3, 'E':0.4}

    out = assign_snps_to_genes(snp, pc, to_table=True)

    assert_equal(out['nSNPS'].values.ravel(), np.array([1, 1, 1, 1]))
    assert_equal(out['TopSNP P-Value'].values.ravel(), np.array([0.2, 0.01, 0.3, 0.4]))
    assert_equal(out['TopSNP Position'].values.ravel(), np.array([59, 400, 100, 1300]))

def test_assign_snps_to_genes_agg_median():
    snp, pc = _setup_nonoverlapping_inputs()
    snp.loc[5] = [1, 60, 0.3]
    snp[['hg18chr', 'bp']] = snp[['hg18chr', 'bp']].astype(int)

    out = assign_snps_to_genes(snp, pc, window_size=10, agg_method='median')

    expected = {'A': 0.2, 'C': 0.3, 'D':0.4}

    for k in out.keys():
        assert np.isclose(out[k], expected[k])

def test_assign_snps_to_genes_agg_mean():
    snp, pc = _setup_nonoverlapping_inputs()

    out = assign_snps_to_genes(snp, pc, window_size=10, agg_method='mean')

    expected = {'A': 0.15, 'C': 0.3, 'D':0.4}

    for k in out.keys():
        assert np.isclose(out[k], expected[k])

def test_assign_snps_to_genes_agg_error():
    snp, pc = _setup_nonoverlapping_inputs()

    with pytest.raises(ValueError):
        out = assign_snps_to_genes(snp, pc, agg_method='random')
        out = assign_snps_to_genes(snp, pc, agg_method={})

def test_assign_snps_to_genes_input_error():
    snp, pc = _setup_nonoverlapping_inputs()
    snp['hg18chr'] = snp['hg18chr'].apply(lambda x: 'Chr%s' % x)

    with pytest.raises(ValueError):
        out = assign_snps_to_genes(snp, pc)

def test_assign_snps_to_genes_to_table():
    snp, pc = _setup_nonoverlapping_inputs()
    out = assign_snps_to_genes(snp, pc, window_size=10, to_table=True)

    for i, row in out.iterrows():
        for n in ['Chrom', 'Start', 'End']:
            assert row[n] == pc.loc[row['Gene'], n]

    assert_equal(out['nSNPS'].values.ravel(), np.array([2., np.NaN, 1., 1.]))
    assert_equal(out['TopSNP P-Value'].values.ravel(), np.array([0.1, np.NaN, 0.3, 0.4]))
    assert_equal(out['TopSNP Position'].values.ravel(), np.array([40, np.NaN, 100, 1300]))

def test_assign_snps_to_genes_to_table():
    snp, pc = _setup_nonoverlapping_inputs()

    snp.iloc[4, 0] = 5
    with pytest.raises(ValueError):
        out = assign_snps_to_genes(snp, pc)

"""Tests for _get_bins()"""

def test_get_bins_nonoverlapping():
    _, pc = _setup_nonoverlapping_inputs()
    pc = pc.iloc[:2]

    bins, names = _get_bins(pc)

    expected_names = [[], ['A'], [], ['B'], []]
    expected_names = [set(i) for i in expected_names]

    assert np.array_equal(bins, np.array([50, 100, 150, 200]))
    assert np.array_equal(names, np.array(expected_names))

def test_get_bins_nonoverlapping_genenames():
    _, pc = _setup_nonoverlapping_genename_inputs()
    pc = pc.iloc[:2]

    bins, names = _get_bins(pc)

    expected_names = [[], ['Kdm6a'], [], ['Ddx3'], []]
    expected_names = [set(i) for i in expected_names]

    assert np.array_equal(bins, np.array([50, 100, 150, 200]))
    assert np.array_equal(names, np.array(expected_names))

def test_get_bins_nonoverlapping_with_window_size():
    _, pc = _setup_nonoverlapping_inputs()
    pc = pc.iloc[:2]

    bins, names = _get_bins(pc, window_size=10)

    expected_names = [[], ['A'], [], ['B'], []]
    expected_names = [set(i) for i in expected_names]

    assert np.array_equal(bins, np.array([40, 110, 140, 210]))
    assert np.array_equal(names, np.array(expected_names))

def test_get_bins_overlapping():
    _, pc = _setup_overlapping_inputs()
    pc = pc.iloc[:3]

    bins, names = _get_bins(pc)
    expected_names = [[], ['A'], [], ['B'], ['B', 'C'], ['C'], []]
    expected_names = [set(i) for i in expected_names]

    assert np.array_equal(bins, np.array([50, 100, 150, 175, 200, 500]))
    assert np.array_equal(names, np.array(expected_names))

def test_get_bins_overlapping_with_window_size():
    _, pc = _setup_nonoverlapping_inputs() #Start with non-overlapping
                                           #Use window size to create overlapping case
    pc = pc.iloc[:2]

    bins, names = _get_bins(pc, window_size=60)
    expected_names = [[], ['A'], ['A', 'B'], ['B'], []]
    expected_names = [set(i) for i in expected_names]

    assert np.array_equal(bins, np.array([-10, 90, 160, 260]))
    assert np.array_equal(names, np.array(expected_names))

def test_get_bins_multiple_inputs():
    _, pc = _setup_multiple_inputs()

    pc = pc.iloc[:3]

    bins, names = _get_bins(pc)
    expected_names = [[], ['A'], [], ['B'], ['B'], ['B'], []]
    expected_names = [set(i) for i in expected_names]

    assert_equal(names, np.array(expected_names))

    pc.index = list('ABA')
    bins, names = _get_bins(pc)
    expected_names = [[], ['A'], [], ['B'], ['A', 'B'], ['A'], []]
    expected_names = [set(i) for i in expected_names]

    assert_equal(names, np.array(expected_names))

"""Set up code"""
def _setup_nonoverlapping_inputs():
    snp = pd.DataFrame([
        [1, 40, 0.1],
        [1, 59, 0.2],
        [1, 1000, 0.01],
        [2, 100, 0.3],
        [2, 1300, 0.4],
    ], columns=['hg18chr', 'bp', 'pval'])

    pc = pd.DataFrame([
        ['1', 50, 100],
        ['1', 150, 200],
        ['2', 10, 1000],
        ['2', 1200, 1500]
    ], columns=['Chrom', 'Start', 'End'], index=list('ABCD'))

    return snp, pc

def _setup_nonoverlapping_genename_inputs():
    snp = pd.DataFrame([
        [1, 40, 0.1],
        [1, 59, 0.2],
        [1, 1000, 0.01],
        [2, 100, 0.3],
        [2, 1300, 0.4],
    ], columns=['hg18chr', 'bp', 'pval'])

    pc = pd.DataFrame([
        ['1', 50, 100],
        ['1', 150, 200],
        ['2', 10, 1000],
        ['2', 1200, 1500]
    ], columns=['Chrom', 'Start', 'End'], index=['Kdm6a', 'Ddx3', 'Vom2r5', 'LOC679873'])

    return snp, pc

def _setup_overlapping_inputs():
    snp = pd.DataFrame([
        [1, 40, 0.1],
        [1, 59, 0.2],
        [1, 1000, 0.01],
        [2, 100, 0.3],
        [2, 1300, 0.4],
    ], columns=['hg18chr', 'bp', 'pval'])

    pc = pd.DataFrame([
        ['1', 50, 100],
        ['1', 150, 200],
        ['1', 175, 500],
        ['2', 10, 1000],
        ['2', 1200, 1500]
    ], columns=['Chrom', 'Start', 'End'], index=list('ABCDE'))

    return snp, pc

def _setup_multiple_inputs():
    snp = pd.DataFrame([
        [1, 40, 0.1],
        [1, 59, 0.2],
        [1, 400, 0.01],
        [1, 1000, 0.01],
        [2, 100, 0.3],
        [2, 1300, 0.4],
    ], columns=['hg18chr', 'bp', 'pval'])

    pc = pd.DataFrame([
        ['1', 50, 100],
        ['1', 150, 200],
        ['1', 175, 500],
        ['2', 10, 1000],
        ['2', 1200, 1500]
    ], columns=['Chrom', 'Start', 'End'], index=list('ABBDE'))

    return snp, pc
