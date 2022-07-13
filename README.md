# GAWA Version 2.0

## GAWA workflow

This is a code intended to identify stellar clusters using wavelet transformation in a catalog of stars filtered by isochronal masks.

### Instalation

Clone the repository and create an environment with Conda:
```bash
git clone https://github.com/linea-it/gawa && cd gawa 
conda create -n gawa python=3.9
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
```

Copy gawa.cfg and env.sh
```bash
cp gawa.cfg.template gawa.cfg
cp env.sh.template env.sh # You need to edit it if you want to run with Parsl in cluster.
```

### Running with Parsl(Pilot Jobs - Remote jobs)
Edit env.sh, adding the path to Conda (CONDAPATH) and the path to this repository (GAWA_ROOT):
```bash
export CONDAPATH=<conda path>
export GAWA_ROOT=<gawa repository path>
export PYTHONPATH=$PYTHONPATH:$GAWA_ROOT
export GAWA_LOG_LEVEL=info

source $CONDAPATH/activate
conda activate gawa
python -m ipykernel install --user --name=gawa
```
Choose the 'executor' option in gawa.cfg and run:
```bash
python -W ignore gawa_main.py gawa.cfg
```

### Running

```bash
jupyter-lab gawazpy.ipynb
```

### Warnings

To run again the code, please remove the following folders with the command (being in the root folder):

```bash
rm -r tiles footprint isochrone_masks
```

and files:

```bash
rm tiles_specs.fits cluster.fits cluster0.fits
```

If you run with max_threads > 1, the kernel may die while running the main cell. We think that this is related to the memory required by the jobs. Set max_threads = 1 and run again.
