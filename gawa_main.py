import numpy as np
import yaml
import os
import astropy.io.fits as fits
import healpy as hp
from astropy.table import Table
import sys
import json

from lib.multithread import split_equal_area_in_threads
from lib.utils import hpx_split_survey
from lib.utils import read_FitsCat, concatenate_fits, create_directory
from lib.utils import create_mosaic_footprint, create_survey_footprint_from_mosaic
from lib.utils import add_key_to_fits
from lib.gawa import compute_cmd_masks, compute_dslices, gawa_concatenate
from lib.gawa import tiles_with_clusters, run_gawa_tile

# read config file as online argument 
config = sys.argv[1]

pypath = '/home/benoist/python-scripts/code_magyc/cogito/venv3/bin/python'
scr = '/home/benoist/python-scripts/ga_wazpy_v2'

# read config file
with open(config) as fstream:
    param_data = yaml.safe_load(fstream)
globals().update(param_data)

# Working & output directories 
workdir = out_paths['workdir']
create_directory(workdir)
create_directory(os.path.join(workdir, 'tiles'))
print ('Survey : ', survey)
print ('workdir : ', workdir)
tiles_filename = os.path.join(workdir, admin['tiling']['tiles_filename'])

# create required data structure if not exist and update config 
if not input_data_structure[survey]['footprint_hpx_mosaic']:
    create_mosaic_footprint(
        footprint[survey], os.path.join(workdir, 'footprint')
    )
    param_data['footprint'][survey]['mosaic']['dir'] = os.path.join(
        workdir, 'footprint'
    )

# update parameters with selected filters in config 
param_data['starcat'][survey]['keys']['key_mag_blue'] = starcat[survey]\
                                                        ['keys']['key_mag']\
                                                        [ref_bfilter]
param_data['starcat'][survey]['keys']['key_mag_red'] = starcat[survey]\
                                                       ['keys']['key_mag']\
                                                       [ref_rfilter]
param_data['isochrone_masks'][survey]['magerr_blue_file'] = isochrone_masks\
                                                            [survey]\
                                                            ['magerr_file']\
                                                            [ref_bfilter]
param_data['isochrone_masks'][survey]['magerr_red_file'] = isochrone_masks\
                                                           [survey]\
                                                           ['magerr_file']\
                                                           [ref_rfilter]
param_data['isochrone_masks'][survey]['model_file'] = isochrone_masks\
                                                      [survey]['model_file']\
                                                      [ref_color]
param_data['isochrone_masks'][survey]['mask_color_min'] = isochrone_masks\
                                                          [survey]\
                                                          ['mask_color_min']\
                                                          [ref_color]
param_data['isochrone_masks'][survey]['mask_color_max'] = isochrone_masks\
                                                          [survey]\
                                                          ['mask_color_max']\
                                                          [ref_color]
param_data['isochrone_masks'][survey]['mask_mag_min'] = isochrone_masks\
                                                        [survey]\
                                                        ['mask_mag_min']\
                                                        [ref_bfilter]
param_data['isochrone_masks'][survey]['mask_mag_max'] = isochrone_masks\
                                                        [survey]\
                                                        ['mask_mag_max']\
                                                        [ref_bfilter]

# store config file in workdir
with open(os.path.join(workdir, 'gawa.cfg'), 'w') as outfile:
    json.dump(param_data, outfile)
config = os.path.join(workdir, 'gawa.cfg')    

# split_area:
if input_data_structure[survey]['footprint_hpx_mosaic']: 
    survey_footprint = os.path.join(workdir, 'survey_footprint.fits')
    if not os.path.isfile(survey_footprint):
        create_survey_footprint_from_mosaic(
            footprint[survey], survey_footprint
        )
else:
    survey_footprint = footprint[survey]['survey_footprint']

if not os.path.isfile(tiles_filename):
    ntiles = hpx_split_survey(
        survey_footprint, footprint[survey], admin['tiling'], tiles_filename
    )
    n_threads, thread_ids = split_equal_area_in_threads(
        admin['nthreads_max'], tiles_filename
    )
    add_key_to_fits(tiles_filename, thread_ids, 'thread_id', 'int')
    all_tiles = read_FitsCat(tiles_filename)
else:
    all_tiles = read_FitsCat(tiles_filename)
    ntiles, n_threads = len(all_tiles), np.amax(all_tiles['thread_id']) 
    thread_ids = all_tiles['thread_id']
print ('Ntiles / Nthreads = ', ntiles, ' / ', n_threads)

# prepare dslices 
compute_dslices(isochrone_masks[survey], gawa_cfg['dslices'], workdir)

# compute cmd_masks 
print ('Compute CMD masks')
compute_cmd_masks(isochrone_masks[survey], out_paths, gawa_cfg)

# detect clusters on all tiles 
for ith in np.unique(thread_ids): 
    run_gawa_tile(config, ith)

# concatenate
# tiles with clusters 
eff_tiles = tiles_with_clusters(out_paths, all_tiles)
data_clusters = gawa_concatenate(eff_tiles, gawa_cfg, out_paths)
data_clusters.write(
    os.path.join(out_paths['workdir'],'clusters.fits'), overwrite=True
)

print ('all done folks !')
print ('results in ', workdir)
