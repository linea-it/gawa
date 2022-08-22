# GAWA Version 2.0

## GAWA workflow

This is a code intended to identify stellar clusters using wavelet transformation in a catalog of stars filtered by isochronal masks.

### Instalation

Clone the repository and create an environment with Conda:

```bash
git clone https://github.com/linea-it/gawa && cd gawa
conda create -n gawa python=3.8
conda activate gawa
conda install -c conda-forge cfitsio=3.430
conda install -c cta-observatory sparse2d
conda install jupyterlab
conda install ipykernel
pip install scikit-image
pip install astropy
pip install healpy
ipython kernel install --user --name=gawa
```

### Running

```bash
jupyter-lab gawazpy.ipynb
```

### Warnings

To run again the code, please remove the following folders with the command (being in the root folder):

```bash
rm -r output/tiles output/footprint output/isochrone_masks
```

and files:

```bash
rm output/tiles_specs.fits output/clusters.fits output/clusters0.fits
```

If you run with max_threads > 1, the kernel may die while running the main cell. We think that this is related to the memory required by the jobs. Set max_threads = 1 and run again.
