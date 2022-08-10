.. Gawa documentation master file, created by
   sphinx-quickstart on Tue Aug  9 11:48:36 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GAWA's documentation!
================================

Scientific use of GAWA: 
-----------------------

| - The user should define a survey / 2 filters / 1 color

::

      survey: 'MOCK_DES'  # DES
      ref_bfilter: 'filter_g'
      ref_rfilter: 'filter_r'
      ref_color: 'gr'

| - Installation of a new survey : 

   | in gawa.cfg under starcat > footprint > isochrone_masks, duplicate the description of a given survey and create a new one from a photometric system.    
   
| - GAWA detection parameters

::

   + gawa_cfg / detection_mode => galaxy OR cluster
   + gawa_cfg / dslices => define the set of distances to perform detection.  

   # Warning: the code should be updated once the CMD masks are properly set and strategy defined
   
| - Current Isochrone

   | The mask used by gawa was taken from an isochrone with 13.5 and [Fe/H]=-2, with small deviations towards bluer and redder limits. The mask encompasses binaries and possible small shifts in age/metalicity/reddening/magnitude measurements.
   | Each isochronal mask is related to a specific photometric system.
   

What is new in GAWA version 2:
------------------------------

| - the tiling is done in healpix pixels so that it can operate at any RA-Dec
| - the divisision of the N tiles in P cores is done to optimize the distribution of the area to be analyzed (as equal area as possible). 
| - SNR has a new definition => decreases Nr false positives
| - cylinders are stored 
| - identification of clusters in cylinders is performed
| - method for filtering of possible duplicates has improved in the tiles and between the tiles. 
| - several steps of the codes were rewritten to decrease CPU time. Some steps were improved by a factor > 10. 
| - there are 3 levels of verbose. With verbose = 0 no intermediate file is written on disc except those necessary for the code.
| - there are several re-entry points with the generation of numpy files (.npx). But this can be switched off if necessary


What should be improved / dev: 
------------------------------

| - currently the code provides detections at discrete distances corresponding to those provided by the gawa/compute_dslices function. We should add a fct to refine this first distance estimate. 
| - Update the SPARSE2D package which is currently a C++ code => python version to avoid current i/o's 
| - BKG is computed in each tile. If a tile is too incomplete it may lead to a poor estimate of the BKG => bad SNRs.   This has to be checked. 
| - Filtering based on 3 pass-bands (2 CMDs)

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   usage/quickstart
   usage/api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
