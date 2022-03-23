# GAWA Version 2.0

## GAWA workflow

This is a code intended to identify stellar clusters using wavelet transformation in a catalog of stars filtered by isochronal masks.

### Instalation

Clone the repository and create an environment with Conda:
```bash
git clone https://github.com/linea-it/gawa && cd gawa 
conda create -n gawa python=3.6
conda update -n base -c defaults conda
conda activate gawa
conda install -c conda-forge cfitsio=3.430
conda install -c cta-observatory sparse2d
conda install pip
pip install -r requirements.txt
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
