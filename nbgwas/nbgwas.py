import pandas as pd 
import ndex2
import ndex2.client as nc
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix,csc_matrix
from scipy.sparse.linalg import expm, expm_multiply

def _check_link(link): 
    if isinstance(link, str): 
        return _Nbgwas._read_table(link) 
    elif isinstance(link, pd.DataFrame): 
        return link 
    else: 
        raise TypeError("Only strings (presumed to be file location) " + \
                            "or pandas DataFrame is allowed!")

class Nbgwas(object): 
    def __init__(self, 
                 snp_level_summary=None, 
                 gene_level_summary=None, 
                 network=None,
                 ): 
        
        if snp_level_summary is not None: 
            raise NotImplementedError("TBA")

        self.snp_level_summary = _check_link(snp_level_summary)
        self.gene_level_summary = _check_link(gene_level_summary) 

        if network is None: 
            self.network = Nbgwas._load_pcnet() 

        else: 
            if isinstance(network, nx.Graph): 
                self.network = network
            else: 
                raise TypeError("Network must be a networkx Graph object")

        self.node_names = [self.network.node[n]['name'] for n in self.network.nodes()]

    @staticmethod
    def _read_table(file):
        min_p_table = pd.read_csv(file, 
                                  sep='\t',
                                  usecols=[1,2,3,4,5,6,7,8,9])

        return min_p_table

    @staticmethod
    def _load_pcnet(): 
        anon_ndex = nc.Ndex2("http://public.ndexbio.org")
        network_niceCx = ndex2.create_nice_cx_from_server(server='public.ndexbio.org', 
                                                          uuid='f93f402c-86d4-11e7-a10d-0ac135e8bacf')

        return network_niceCx.to_networkx()

    def diffuse(self, threshold=5e-6, kernel=None): 
        if kernel is not None: 
            self.kernel = pd.read_hdf(kernel)
            network_genes = list(self.kernel.index)

        else: 
            raise NotImplementedError("TODO: Add non-pre-computed kernel code")


        if not hasattr(self, laplacian): 
            self.laplacian = csc_matrix(nx.laplacian_matrix(self.network))

        name='prop'
        threshold_genes = {}
        prop_vectors = []
            
        threshold_genes[name] = self.gene_level_summary[self.gene_level_summary['TopSNP P-Value'] < threshold]
        prop_vector = (self.gene_level_summary.set_index('Gene').loc[network_genes, 'TopSNP P-Value'] < threshold).astype(float)
        prop_vector.name = name
        prop_vectors.append(prop_vector)
        prop_vector_matrix = pd.concat(prop_vectors, axis=1).loc[network_genes].T

        #propagate with pre-computed kernel
        prop_val_matrix = np.dot(prop_vector_matrix, self.kernel)
        prop_val_table = pd.DataFrame(prop_val_matrix, index = prop_vector_matrix.index, columns = prop_vector_matrix.columns)
        
        self.boosted_pvalues = prop_val_table.T.sort_values(by='prop', ascending=False).head()

        return self

    def heat_diffusion(self, threshold=5e-6): 
        if not hasattr(self, laplacian): 
            self.laplacian = csc_matrix(nx.laplacian_matrix(self.network))
            
        input_list = list(self.gene_level_summary[self.gene_level_summary['TopSNP P-Value'] < threshold]['Gene'])
        input_vector = np.array([n in input_list for n in self.node_names])
        out_vector=expm_multiply(-self.laplacian, input_vector, start=0, stop=0.1, endpoint=True)[-1]

        #out_dict= dict(zip(node_names, out_vector))
        out_dict= {'prop': out_vector,'Gene':self.node_names}
        heat_df=pd.DataFrame.from_dict(out_dict).set_index('Gene')

        self.boosted_pvalues = heat_df.sort_values(by='prop', ascending=False).head()

        return self
