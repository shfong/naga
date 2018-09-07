.. automodule:: nbgwas

Nbgwas class
============
.. autoclass:: Nbgwas

Entry points
============
Nbgwas was designed to be very flexible in how users can interact with the code.

Table reading methods
---------------------
.. automethod:: Nbgwas.read_snp_table
.. automethod:: Nbgwas.read_gene_table
.. automethod:: Nbgwas.read_protein_coding_table

Network input methods
---------------------
.. automethod:: Nbgwas.read_cx_file
.. automethod:: Nbgwas.read_nx_pickle_file
.. automethod:: Nbgwas.get_network_from_ndex

Properties
==========
.. automethod:: Nbgwas.snp_level_summary
.. automethod:: Nbgwas.gene_level_summary 
.. automethod:: Nbgwas.protein_coding_table
.. automethod:: Nbgwas.network 
.. automethod:: Nbgwas.pvalues


Assigning Heat
==============
.. automethod:: Nbgwas.convert_to_heat

Diffusion methods
=================
.. automethod:: Nbgwas.diffuse
.. automethod:: Nbgwas.random_walk
.. automethod:: Nbgwas.random_walk_with_kernel
.. automethod:: Nbgwas.heat_diffusion

Visualization methods
=====================
.. automethod:: Nbgwas.annotate_network
.. automethod:: Nbgwas.get_subgraph
.. automethod:: Nbgwas.view 

.. automethod:: Nbgwas.to_ndex

Utilities
=========
.. automethod:: Nbgwas.cache_network_data
.. automethod:: Nbgwas.reset_cache
.. automethod:: Nbgwas.extract_network_attributes