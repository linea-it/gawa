import numpy as np
import yaml
import os
import astropy.io.fits as fits
import healpy as hp
from astropy.table import Table
import sys
from utils import tile_radius, filter_hpx_tile, create_tile_specs
from utils import read_mosaicFitsCat_in_disc, create_directory 
from utils import read_FitsFootprint, read_FitsCat
from utils import read_mosaicFootprint_in_disc
from gawa import create_gawa_directories, gawa_tile, tile_dir_name

# read config file as online argument 
config = sys.argv[1]

# read config file
with open(config) as fstream:
  param_data = yaml.load(fstream)
  globals().update(param_data)

workdir = out_paths['workdir']
thread_id = int(sys.argv[2])  #  #os.environ.get('OAR_ARRAY_INDEX') #2

####
all_tiles = read_FitsCat(os.path.join(workdir, admin['tiling']['tiles_filename']))
tiles = all_tiles[(all_tiles['thread_id']==int(thread_id))]    
print ('THREAD ', int(thread_id))

for it in range(0, len(tiles)):
  tile_dir = tile_dir_name(workdir, int(tiles['id'][it]) )
  print ('..... Tile ', int(tiles['id'][it]))

  create_directory(tile_dir)
  create_gawa_directories(tile_dir, out_paths['gawa'])
  out_paths['tile_dir'] = tile_dir # local update 

  tile_radius_deg = tile_radius(admin['tiling'])
  tile_specs = create_tile_specs(tiles[it], tile_radius_deg, admin)
  data_star_tile = read_mosaicFitsCat_in_disc(starcat[survey], tiles[it], tile_radius_deg)   
  data_fp_tile   = read_mosaicFootprint_in_disc (footprint[survey], tiles[it], tile_radius_deg)

  if verbose>=2:
    t = Table (data_star_tile)#, names=names)
    t.write(os.path.join(tile_dir, out_paths['gawa']['files'],"starcat.fits"),overwrite=True)
    t = Table (data_fp_tile)#, names=names)
    t.write(os.path.join(tile_dir, out_paths['gawa']['files'], "footprint.fits"),overwrite=True)
  
  gawa_tile(tile_specs, isochrone_masks[survey], data_star_tile, starcat[survey], \
            data_fp_tile, footprint[survey], gawa_cfg, admin, out_paths, verbose)

