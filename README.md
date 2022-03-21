# GAWA Version 2.0

## GAWA workflow

This is a code intended to identify stellar clusters using wavelet transformation in a catalog of stars filtered by isochronal masks.

### Instalation

Clone the repository and create an environment with Conda:
```bash
git clone https://github.com/linea-it/gawa && cd gawa 
conda create -n gawa python=3.6
conda activate gawa
conda install ipykernel
conda install -c conda-forge cfitsio=3.430
conda install -c cta-observatory sparse2d
conda install pip
pip install -r requirements.txt
conda install numpy
ipython kernel install --user --name=gawa
```

### Running

```bash
jupyter-lab gawazpy.ipynb
```
And select the `gawa` kernel.
