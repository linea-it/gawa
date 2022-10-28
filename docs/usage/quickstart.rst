
QuickStart
==========

| To install the code, you have to clone the repository and follow instructions to install dependencies.

| The code is intended to run on Linux distributions.


Code and dependencies
=====================

| The code is written and runs in Python 3.9, but it is compatible to python 3.X. The following libraries are mandatory to run the code:

* `conda <https://docs.conda.io/en/latest/>`_
* `astropy <https:/www.astropy.org/>`_
* `healpy <https:/healpy.readthedocs.io/en/latest>`_
* `matplotlib <https:/matplotlib.org/>`_
* `cfitsio=3.430 <https://anaconda.org/conda-forge/cfitsio>`_
* `sparse2d <https://anaconda.org/cta-observatory/sparse2d>`_
* `scikit-image <https://scikit-image.org/docs/stable/install.html>`_
* `parsl <https://parsl.readthedocs.io/en/stable/>`_


Installation
============

| Clone the repository and create an environment with Conda:

::

	git clone https://github.com/linea-it/gawa && cd gawa
	conda create -n gawa python=3.9

| Install packages needed to your environment (for example):

::

    conda activate gawa
    conda install -c conda-forge cfitsio=3.430
    conda install -c cta-observatory sparse2d
    conda install jupyterlab
    conda install ipykernel
    pip install scikit-image
    pip install astropy
    pip install healpy
    pip install parsl
    ipython kernel install --user --name=gawa

| If you have error messages from missing libraries, install it in a similar manner as packages installed above.


| Copy gawa.cfg and env.sh

::

    cp gawa.cfg.template gawa.cfg
    cp gawa.sh.template gawa.sh   # You need to edit it if you want to run with Parsl in cluster.


Running
=======

| In jupyterlab, run the following command:

::

    jupyter-lab gawazpy.ipynb


| In the terminal, run the following command:

::

    python -W ignore gawa_main.py gawa.cfg

Warnings
--------

| To run again the code, please remove the following folders with the command (being in the root folder):

::

    rm -r tiles footprint isochrone_masks

and files:

::

    rm tiles_specs.fits cluster.fits cluster0.fits

Running with Parsl (Pilot Jobs)
============================================

| Edit gawa.sh, adding the path to Conda (CONDAPATH) and the path to this repository (GAWA_ROOT):

::

    export CONDAPATH=<conda path>
    export GAWA_ROOT=<gawa repository path>
    export PYTHONPATH=$PYTHONPATH:$GAWA_ROOT
    export GAWA_LOG_LEVEL=info
    source $CONDAPATH/activate
    conda activate gawa
    python -m ipykernel install --user --name=gawa

| Choose the 'executor' option in gawa.cfg and run:

::

    source gawa.sh
    python -W ignore gawa_main.py gawa.cfg