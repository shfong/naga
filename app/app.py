import nbgwas
from nbgwas import Nbgwas
from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
import pandas as pd
import networkx as nx
import logging

from call_biggim import get_table_from_biggim

app = Flask(__name__)
api = Api(app)

def create_gene_level_summary(genes, seeds): 
    # Add heat to gene_level_summary

    genes_level_summary = pd.DataFrame([genes], index=['Genes']).T
    genes_level_summary['p-value'] = 1
    genes_level_summary.loc[genes_level_summary['Genes'].isin(seeds), 'p-value'] = 0
    
    return genes_level_summary

class nbgwasapp(Resource): 

    def post(self): 

        logging.info("Begin!")

        #Setting alpha
        alpha = float(request.values.get("alpha", 0.5))
        
        #Getting network
        if "network" in request.files: 
            logging.info("Reading Network File")
            
            network_file = request.files['network']
            network_df = pd.read_csv(
                network_file.stream, 
                sep='\t', 
                names=['Gene1', 'Gene2', 'Val']
            )
        elif "column" in request.values: 
            logging.info("Getting file from BigGIM")
            network_df = get_table_from_biggim(request.values['column'], 0.8)
            network_df = network_df.astype(str)
        
        elif "ndex" in request.values: 
            ndex_uuid = request.values['ndex']

        else: 
            return "failed"
        
        logging.info("Finished getting network_df")
        
        #Making networkx object
        dG = nx.from_pandas_dataframe(network_df, 'Gene1', 'Gene2')

        logging.info("Finished making network")

        #Parsing seeds
        seeds = request.values['seeds']
        seeds = seeds.split(',')

        logging.info("Finished converting seeds")
        
        #Making sure the seeds are in the network
        for i in seeds: 
            if i not in dG.nodes(): 
                seeds.remove(i) 
                logging.info(f"{i} not in nodes")
            
        if len(seeds) == 0:
            logging.info("No seeds left!")
            return "failed"

        gene_level_summary = create_gene_level_summary(dG.nodes(), seeds)

        logging.info("Finished gene level Summary")

        g = Nbgwas(
            gene_level_summary=gene_level_summary,
            gene_col = 'Genes',
            gene_pval_col = 'p-value', 
            network=dG,
        )

        g.convert_to_heat()
        g.diffuse(method='random_walk', alpha=alpha)

        logging.info("Done!")

        return g.heat.iloc[:, -1].to_json() + '\n'

        

# api.add_resource(HelloWorld, '/')
api.add_resource(nbgwasapp, '/nbgwas', endpoint='nbgwas')
#api.add_resource(job, '/nbgwas/<jobid>')

if __name__ == '__main__':
    app.run(debug=True) #Change this in production