import numpy as np
import yaml
import os
import astropy.io.fits as fits
import healpy as hp
from astropy.table import Table
import sys
from utils import tile_radius, filter_hpx_tile, concatenate_clusters
from utils import read_mosaicFitsCat_in_disc, create_directory 
from utils import read_FitsFootprint, read_FitsCat, concatenate_fits
from utils import read_mosaicFootprint_in_disc
from utils import add_clusters_unique_id
from gawa import create_gawa_directories, gawa_tile, tile_dir_name
from gawa import cl_duplicates_filtering

# read config file as online argument 
config = sys.argv[1]

# read config file
with open(config) as fstream:
  param_data = yaml.load(fstream)
  globals().update(param_data)

# concatenate all tiles 
all_tiles = read_FitsCat(os.path.join(out_paths['workdir'], admin['tiling']['tiles_filename']))
list_results = []
for it in range(0, len(all_tiles)):
  tile_dir = tile_dir_name(out_paths['workdir'], int(all_tiles['id'][it]) )
  list_results.append(os.path.join(tile_dir, out_paths['gawa']['results']))
concatenate_clusters(list_results, os.path.join(out_paths['workdir'],'clusters0.fits')) 

# final filtering 
data_clusters0 = read_FitsCat(os.path.join(out_paths['workdir'],'clusters0.fits'))
data_clusters0f = cl_duplicates_filtering(data_clusters0, gawa_cfg, 'survey')
# create unique index with decreasing SNR 
data_clusters = add_clusters_unique_id(data_clusters0f, gawa_cfg['clkeys'])
data_clusters.write(os.path.join(out_paths['workdir'],'clusters.fits'), overwrite=True)
