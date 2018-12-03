from abc import ABC, abstractmethod
import networkx as nx
import ndex2
import numpy as np
import igraph as ig
from scipy.sparse import diags, coo_matrix, csr_matrix
import pandas as pd
import mygene
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from py2cytoscape.data.cyrest_client import CyRestClient
import warnings
from .utils import get_neighbors

def igraph_adj_matrix(G, weighted=False): 

    source, target, weights = zip(*[(i.source, i.target, i[weighted] if weighted else 1) for i in G.es])

    n_nodes = len(G.vs)
    adj = csr_matrix(coo_matrix((weights, (source, target)), shape=(n_nodes, n_nodes)))

    if not G.is_directed(): 
       adj += adj.T

    return adj


class Network(ABC): 
    """Base class for all network classes to inherit from
    
    This base class defines interfacing functionalities that 
    Nbgwas expects. 
    """

    def __init__(self, network=None, node_name = "name"): 
        self.network = network 
        self.node_name = node_name 

        super().__init__() 

    @property
    @abstractmethod
    def adjacency_matrix(self): 
        pass


    @property
    @abstractmethod
    def laplacian_matrix(self): 
        pass


    @abstractmethod
    def add_adjacency_matrix(self): 
        pass 

    @abstractmethod
    def add_laplacian_matrix(self): 
        pass 

    @abstractmethod
    def nodes(self): 
        pass 

    @abstractmethod
    def edges(self): 
        pass

    @abstractmethod
    def subgraph(self):
        pass 

    @abstractmethod
    def get_node_attributes(self): 
        pass

    @abstractmethod 
    def set_node_attributes(self, attr_map, namespace="nodeids"): 
        """set node attributes 

        attr_map is a dictionary of dictionaries

        TODO
        ----
        - Need to make sure the required attributes are created (there are 3 of them)

        """

        pass 

    @property
    @abstractmethod
    def node_ids(self): 
        pass

    @abstractmethod 
    def set_node_names(self, attr=None): 
        pass

    # @property 
    # @abstractmethod 
    # def node_names(self): 
    #     pass

    def convert_node_names(
        self, 
        attribute="name", 
        current="entrezgene", 
        to="symbol", 
        rerun_query=True,
        use_key_for_missing=False, 
        write_to_node_table=True, 
        **kwargs,
    ): 

        """Maps network node names using mygene.info"""

        mg = mygene.MyGeneInfo()
        node_attributes = self.get_node_attributes()

        attr = [v[attribute] for k,v in node_attributes.items()]

        query_result = mg.querymany(
            attr, 
            scopes=current, 
            field=to,
            as_dataframe=True, 
            returnall=True, 
            **kwargs, 
        )

        gene_map = query_result['out'][to].to_dict()

        missing = query_result['missing']
        if missing: 
            if rerun_query: 
                sec_query_df = mg.getgenes(
                    query_result['missing'], 
                    fields='%s,%s' % (current, to),
                    as_dataframe=True
                )

                missing = sec_query_df.loc[sec_query_df['notfound'] == True].index

                gene_map.update(sec_query_df[to].to_dict())

            if len(missing) != 0: 
                warnings.warn('%s nodes cannot be converted. Their original name will be kept!' % len(missing))

                for i in missing: 
                    gene_map[i] = i

        if query_result['dup']: 
            warnings.warn("Gene name conversion contains duplicate mappings!")

        change_to = {}
        for k,v in node_attributes.items(): 
            change_to[k] = gene_map.get(v[attribute], k if use_key_for_missing else None)

        self.set_node_attributes({to:change_to}, namespace="nodeids")

        if write_to_node_table: 
            self.refresh_node_table()

        return self

    def map_attr_data(self, data, store=False): 
        """
        Parameter
        ---------
        data : dict
        """
        
        values = [data.get(node, None) for node in self.node_ids]

        if store: 
            self.set_node_attributes({store: dict(zip(self.node_ids, values))})
        else: 
            return values

    @property
    def node_table(self): 
        if not hasattr(self, "_node_table"): 
            self._node_table = pd.DataFrame.from_dict(self.get_node_attributes()).T
            self._node_table = self._node_table.fillna(0)

        return self._node_table

    @node_table.setter 
    def node_table(self, node_table): 
        #TODO: Add Validation code here

        self._node_table = node_table

    @node_table.deleter
    def node_table(self): 
        if hasattr(self, "_node_table"): 
            del self._node_table

    def refresh_node_table(self): 
        del self.node_table

        self.node_table

        return self

    def refresh_node_attributes(self): 
        self.set_node_attributes(self.node_table.to_dict(), namespace="nodeids")

        return self

    def __getstate__(self): 
        return self.__dict__

    def __setstate__(self, state): 
        self.__dict__.update(state)        

    def to_pickle(self, filename): 
        with open(filename, 'wb') as f: 
            pickle.dump(self, f) 


    @classmethod
    def from_pickle(cls, filename): 
        with open(filename, 'rb') as f: 
            obj = pickle.load(f) 

        return obj

class NxNetwork(Network): 
    """Internal object to expose networkx functionalities"""

    def __init__(self, network=None, node_name="name"): 
        super().__init__(network, node_name=node_name)

        if network is not None:
            self.set_node_names(attr=node_name)

        else: 
            self.node_names = None


    @property 
    def node_ids(self): 
        return self.network.nodes()


    def set_node_names(self, attr=None): 
        if attr is None: 
            attr = self.node_name

        self.node_name=attr

        nodes = self.network.node.keys() 

        self.node_names = [
            str(self.network.node[n].get(self.node_name, n)) \
                for n in self.network.nodes()
        ]

        self.node_2_name = dict(zip(nodes, self.node_names))
        self.name_2_node = dict(zip(self.node_names, nodes))

        return self

    
    @property
    def adjacency_matrix(self): 
        if not hasattr(self, "_adjacency_matrix"): 
            self.add_adjacency_matrix()

        return self._adjacency_matrix

    @property
    def laplacian_matrix(self): 
        if not hasattr(self, "_laplacian_matrix"): 
            self.add_laplacian_matrix()

        return self._laplacian_matrix


    def add_adjacency_matrix(self, weights=None): 
        self._adjacency_matrix = nx.adjacency_matrix(
            self.network, weight=weights
        )

        return self


    def add_laplacian_matrix(self, weights=None): 
        self._laplacian_matrix = nx.laplacian_matrix(self.network, weight=weights)

        return self  


    def nodes(self): 
        return self.network.nodes()

    
    def edges(self): 
        return self.network.edges()


    def subgraph(self, node_ids=None, node_names=None): 
        if node_names is not None and node_ids is not None: 
            raise ValueError("Expected either node_names or node_ids. Both given.")

        elif node_names is not None: 
            node_ids = [self.name_2_node[n] for n in node_names]

        return NxNetwork(
            network=self.network.subgraph(node_ids), 
            node_name=self.node_name
        )


    def get_node_attributes(self): 
        return self.network.node


    def set_node_attributes(self, attr_map, namespace="nodenames"):
        for attr_name, d in attr_map.items():  
            if namespace == "nodenames": 
                d = {self.name_2_node[k]:v for k, v in d.items() if k in self.name_2_node}

            nx.set_node_attributes(
                self.network, 
                attr_name, 
                d
            )

        return self


    def from_cx(self, file, node_name="name"):
        """Load CX file as network"""

        del self.__dict__

        network = ndex2.create_nice_cx_from_file(file).to_networkx()

        self.__init__(
            network=network, 
            node_name=node_name
        )
        
        return self


    def from_pickle(self, file, node_name="name"):
        """Read networkx pickle file as network"""

        del self.__dict__
        
        self.__init__(
            network = nx.read_gpickle(file),
            node_name = node_name,
        )

        return self


    def from_ndex(
        self,
        uuid="f93f402c-86d4-11e7-a10d-0ac135e8bacf", #PCNet
        node_name="name",
    ):

        del self.__dict__

        network_niceCx = ndex2.create_nice_cx_from_server(
            server='public.ndexbio.org',
            uuid=uuid
        )

        network = network_niceCx.to_networkx()

        self.__init__(
            network=network, 
            node_name=node_name
        )

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
            g = ndex2.create_nice_cx_from_networkx(self.network)
        except KeyError:
            raise KeyError("%s is not in self.graphs dictionary!" % name)

        uuid = g.upload_to(
            server=server,
            username=username,
            password=password
        )

        return uuid


    def view(
        self,
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

        fig, ax = plt.subplots()

        attr = nx.get_node_attributes(self.network, attributes)

        #TODO: replace missing except behavior with default fall back value
        try:
            vals = [attr[i] for i in self.network.nodes()]
        except KeyError:
            warnings.warn(
                "The specified graph does not have the attribute %s. Replacing values with 0." % attributes
            )
            vals = [0 for _ in self.network.nodes()]

        nx.draw(
            self.network,
            ax=ax,
            node_color=vals,
            labels=self.node_2_name,
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


    def view_in_cytoscape(self):
        """Ports graph to Cytoscape"""

        if not hasattr(self, "cyrest"):
            self.cyrest = CyRestClient()

        hdl = self.cyrest.network.create_from_networkx(self.network)
        self.cyrest.layout.apply(name='degree-circle', network=hdl)

        return self


    def local_neighborhood(self, center_name=None, center_id=None, neighbors=1): 
        if center_name is not None and center_id is not None: 
            raise ValueError("Either center_name or center_id can be supplied.")

        if center_name is not None: 
            center_id = self.name_2_node[center_name] 


        nodes = get_neighbors(self.network, neighbors, center_id) 

        return self.subgraph(node_ids = nodes)


class IgNetwork(Network): 
    """Internal object to expose igraph functionalities"""

    def __init__(self, network=None, node_name="name"): 
        super().__init__(network, node_name=node_name) 

        if network is not None: 
            self.set_node_names(attr=node_name)

        else: 
            self.node_names = None

    @property
    def node_ids(self): 
        return [v.index for v in self.network.vs]

    def set_node_names(self, attr=None): 
        if attr is None: 
            attr = self.node_name 

        self.node_name = attr

        nodes = [v.index for v in self.network.vs]
        if self.node_name in self.network.vs.attributes(): 
            self.node_names = self.network.vs[self.node_name]
        else: 
            self.node_names = nodes

        self.node_names = [str(i) for i in self.node_names]
        
        self.node_2_name = dict(zip(nodes, self.node_names))
        self.name_2_node = dict(zip(self.node_names, nodes))

        return self

    @property
    def adjacency_matrix(self): 
        if not hasattr(self, "_adjacency_matrix"): 
            self.add_adjacency_matrix()

        return self._adjacency_matrix

    @property
    def laplacian_matrix(self): 
        if not hasattr(self, "_laplacian_matrix"): 
            self.add_laplacian_matrix()

        return self._laplacian_matrix

    def add_adjacency_matrix(self, weights=None): 
        self._adjacency_matrix = igraph_adj_matrix(
            self.network, weighted=weights)

        return self

    def add_laplacian_matrix(self, weights=None): 
        if not hasattr(self, "adjacency_matrix"): 
            self.add_adjacency_matrix(weights=weights)

        D = diags(self.adjacency_matrix.sum(axis=1))
        
        #TODO: Need to test this functionality against networkx
        self._laplacian_matrix = D - self.adjacency_matrix

        return self

    def nodes(self): 
        return self.network.vs

    def edges(self): 
        return self.network.es
    
    def subgraph(self, node_ids=None, node_names=None): 
        if node_names is not None and node_ids is not None: 
            raise ValueError("Expected either node_names or node_ids. Both given.")

        elif node_names is not None: 
            node_ids = [self.node_2_name[n] for n in node_names]

        return self.network.subgraph(node_ids)

    def get_node_attributes(self): 
        attr = {}
        for v in self.network.vs:
            #attr[a] = dict([(i.index, i.attributes()) for i in self.network.vs])
            attr[v.index] = v.attributes()

        return attr

    def set_node_attributes(self, attr_map, namespace="nodenames"): 
        for attr_name, d in attr_map.items():
            attr = [None]*len(self.network.vs)
            for ind, v in enumerate(self.network.vs): 
                if namespace == "nodenames": 
                    if v[self.node_name] in d: 
                        attr[ind] = d[v[self.node_name]]

                elif namespace == "nodeids": 
                    if v.index in d: 
                        attr[ind] = d[v.index] 

                else: 
                    raise ValueError("namespace must be nodenames or nodeids")

            self.network.vs[attr_name] = attr

        return self

    # def collapse_duplicate_nodes(self, attribute, inplace=False): 
    #     """Collapse any nodes with the same attribute into a single node"""
        
    #     if not inplace: 
    #         g = self.network.copy()
    #     else:
    #         g = self.network
        
    #     duplicated_nodes_table = self.node_table.loc[
    #         self.node_table[attribute].duplicated(keep=False)
    #     ]
        
    #     counter = len(g.vs)
    #     nodemap = {}
    #     for a, df in duplicated_nodes_table.groupby(attribute): 
    #         for ind, i in enumerate(df.index): 
    #             if ind == 0: 
    #                 g.add_vertex(**g.vs[i].attributes())
                    
    #             nodemap[i] = counter
                
    #         counter += 1
            
    #     nodeslist = list(nodemap.keys())

    #     source = set([e.tuple for e in g.es.select(_source_in=nodeslist)])
    #     target = set([e.tuple for e in g.es.select(_target_in=nodeslist)])
    #     affected_edges = source.union(target)
        
    #     new_edges = [(nodemap.get(i,i), nodemap.get(j,j)) for i,j in affected_edges]
    #     new_edges = [(i,j) for i,j in new_edges if i != j]

    #     return new_edges, affected_edges, nodeslist
        
    #     g.add_edges(new_edges)       
        
    #     g.delete_edges(affected_edges)
    #     g.delete_vertices(nodeslist)
            
    #     if inplace: 
    #         self.network = g
    #         self.refresh_node_table()

    #         return self
        
    #     else: 
    #         return g
