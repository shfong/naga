.. nbgwas documentation master file, created by
   sphinx-quickstart on Tue Aug 21 15:58:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nbgwas's documentation!
==================================
NBGWAS is a python package to perform network-boosted Genome-Wide Association
study. While this package was originally designed for boosting GWAS results,
this package is a generalizable to any network diffusion problems.

How it works
============
.. image:: https://www.agri.ohio.gov/wps/wcm/connect/gov/2049f01b-0952-4808-be75-2c2637d58118/Canine2.jpg?MOD=AJPERES&CACHEID=ROOTWORKSPACE.Z18_M1HGGIK0N0JO00QO9DDDDM3000-2049f01b-0952-4808-be75-2c2637d58118-mi7LBo7
    :alt: A more appropriate figure will be added soon!

**TODO: Add pipeline figure**

Installation
============
This package will soon be pip installable. In the mean time, you can install the
package by pulling it from `github <https://github.com/shfong/nbgwas>`_ by
doing the following: 

    .. code-block:: bash

        git clone https://github.com/shfong/nbgwas.git

followed by the following command: 
    .. code-block:: bash 

        python setup.py install

Alternative Access
==================
In addition to this python package, NBGWAS will also be available as a
gene-pattern notebook and will be available on the gene-pattern server. From
there, you will be able to run your analysis without installing anything.

In addition, the code may also be available via REST API, but that is still TBD.

Documentation 
=============
.. toctree::
   :maxdepth: 2

   nbgwas
   propagation
   utils
