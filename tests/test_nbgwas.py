from nbgwas import Nbgwas
import pytest
import pandas as pd

def test_init(): 
    g = Nbgwas() 
    
    assert g.snp_level_summary is None 
    assert g.gene_level_summary is None 
    assert g.network is None 
    assert g.protein_coding_table is None 

    empty = pd.DataFrame()
    snp_df = pd.DataFrame([], columns=list('ABC'))
    pc_df = pd.DataFrame([], columns=list('ABC')) 
    gene_df = pd.DataFrame([], columns=list('AB'))

    with pytest.raises(ValueError): 
        Nbgwas(snp_level_summary="my_file_link") 
        Nbgwas(gene_level_summary="my_file_link") 
        Nbgwas(network="my_uuid") 
        Nbgwas(protein_coding_table="my_file_link") 
    
        Nbgwas(snp_level_summary=empty)
        Nbgwas(snp_level_summary=snp_df) 
        Nbgwas(protein_coding_table=pc_df) 
        Nbgwas(gene_level_summary=gene_df) 

    Nbgwas(
        snp_level_summary=snp_df,
        snp_chrom_col='A', 
        bp_col='B', 
        snp_pval_col='C'
    ) 

    Nbgwas(
        gene_level_summary=gene_df,
        gene_col='A', 
        gene_pval_col='B', 
    ) 

    Nbgwas(
        protein_coding_table=pc_df,
        pc_chrom_col='A', 
        start_col='B', 
        end_col='C',
    ) 

def test_assign_pvalues(): 
    snp_df = pd.DataFrame([], columns=list('ABC'))
    pc_df = pd.DataFrame([], columns=list('ABC')) 
    
    with pytest.raises(ValueError): 
        Nbgwas().assign_pvalues()
        Nbgwas(snp_level_summary=snp_df).assgin_pvalues()
        Nbgwas(protein_coding_table=pc_df).assign_pvalues()