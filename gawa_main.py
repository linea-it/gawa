import numpy as np
import yaml
import os, sys, json

from lib.multithread import split_equal_area_in_threads
from lib.utils import hpx_split_survey
from lib.utils import read_FitsCat, concatenate_fits, create_directory
from lib.utils import create_mosaic_footprint, create_survey_footprint_from_mosaic
from lib.utils import add_key_to_fits, update_hpx_parameters
from lib.gawa import update_filters_in_params
from lib.gawa import compute_cmd_masks, compute_dslices, gawa_concatenate
from lib.gawa import tiles_with_clusters, run_gawa_tile
from multiprocessing import Pool

# read config file as online argument 
config = sys.argv[1]

# read config file
with open(config) as fstream:
    param = yaml.safe_load(fstream)
globals().update(param)

# Working & output directories 
workdir = param['out_paths']['workdir']
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

# update parameters with hpx params + selected filters in config 
param_data = update_hpx_parameters(
    param_data, survey, input_data_structure
)
param_data = update_filters_in_params(
    param_data, survey, ref_bfilter, ref_rfilter, ref_color
)

# store config file in workdir
with open(os.path.join(workdir, 'gawa.cfg'), 'w') as outfile:
    json.dump(param_data, outfile)
config = os.path.join(workdir, 'gawa.cfg')    

# split_area:
if not os.path.isfile(tiles_filename):
    ntiles = hpx_split_survey(
        footprint[survey], admin['tiling'], tiles_filename
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
with Pool(3) as p:
    p.map(run_gawa_tile, [(config, ith) for ith in np.unique(thread_ids)])

# concatenate
# tiles with clusters 
eff_tiles = tiles_with_clusters(out_paths, all_tiles)
data_clusters = gawa_concatenate(eff_tiles, gawa_cfg, out_paths)
data_clusters.write(
    os.path.join(out_paths['workdir'],'clusters.fits'), overwrite=True
)

print ('all done folks !')
print ('results in ', workdir)
