Network Assisted Genomic Analysis (NAGA)
========================================

Network Assisted Genomic Analysis(NAGA) re-prioritizes significant single
nucleotide polymorphisms (SNPs) to genes using network diffusion methods
including random walk and heat diffusion. 

A companion website and REST API can be found at http://nbgwas.ucsd.edu/.

Documentation
=============

A readthedocs page will be coming soon! In the mean time, you can view
the documentations by building the sphinx documentation in the docs
directory. Simply run the following in the docs folder

.. code:: bash

    make docs

and open the index.html in the docs/build/html directory.

Installation
============

It is recommended that NAGA be run under Anaconda_ with python-igraph_ manually installed using **conda**
and to create a new conda environment


To create a new **conda** environment and activate it:

.. code:: bash

   conda create -n nagaenv
   source activate nagaenv

To install python-igraph_ via **conda**:

.. code:: bash

   conda install -c conda-forge python-igraph


To install NAGA via pip:

.. code:: bash
    
    pip install naga-gwas


Tutorial
========

`notebooks/tutorial.ipynb <https://github.com/shfong/naga/blob/master/notebooks/tutorial.ipynb>`_ demonstrates how to use this package once it
is installed.

Citing NAGA
=============

Publication to come...

.. _Anaconda: https://anaconda.org
.. _python-igraph: https://anaconda.org/conda-forge/python-igraph