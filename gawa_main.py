import time
start = time.time()
from dask.distributed import Client, progress, LocalCluster
import dask.array as da
import yaml
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')

from lib.multithread import split_equal_area_in_threads
from lib.utils import hpx_split_survey
from lib.utils import read_FitsCat, create_directory
from lib.utils import create_mosaic_footprint, create_survey_footprint_from_mosaic
from lib.utils import add_key_to_fits
from lib.gawa import compute_cmd_masks, compute_dslices, gawa_concatenate
from lib.gawa import tiles_with_clusters, run_gawa_tile
from multiprocessing import Pool

# read config file as online argument 
config = sys.argv[1]

# read config file
with open(config) as fstream:
    param = yaml.safe_load(fstream)
    fstream.close()
globals().update(param)

# Working & output directories 
workdir = param['out_paths']['workdir']
create_directory(workdir)
create_directory(os.path.join(workdir, 'tiles'))
survey = param['survey']
print ('Survey : ', survey)
print ('workdir : ', workdir)
tiles_filename = os.path.join(workdir, param['admin']['tiling']['tiles_filename'])

# create required data structure if not exist and update config 
if not param['input_data_structure'][survey]['footprint_hpx_mosaic']:
    create_mosaic_footprint(
        param['footprint'][survey], os.path.join(workdir, 'footprint')
    )
    param['footprint'][survey]['mosaic']['dir'] = os.path.join(
        workdir, 'footprint'
    )

ref_bfilter = param['ref_bfilter']
ref_rfilter = param['ref_rfilter']
ref_color = param['ref_color']
isochrone_masks = param['isochrone_masks']

# update parameters with selected filters in config 
param['starcat'][survey]['keys'][
    'key_mag_blue'
] = param['starcat'][survey]['keys']['key_mag'][ref_bfilter]
param['starcat'][survey]['keys'][
    'key_mag_red'
] = param['starcat'][survey]['keys']['key_mag'][ref_rfilter]
isochrone_masks[survey]['magerr_blue_file'] = isochrone_masks[survey]['magerr_file'][ref_bfilter]
isochrone_masks[survey]['magerr_red_file'] = isochrone_masks[survey]['magerr_file'][ref_rfilter]
isochrone_masks[survey]['model_file'] = isochrone_masks[survey]['model_file'][ref_color]
isochrone_masks[survey]['mask_color_min'] = isochrone_masks[survey]['mask_color_min'][ref_color]
isochrone_masks[survey]['mask_color_max'] = isochrone_masks[survey]['mask_color_max'][ref_color]
isochrone_masks[survey]['mask_mag_min'] = isochrone_masks[survey]['mask_mag_min'][ref_bfilter]
isochrone_masks[survey]['mask_mag_max'] = isochrone_masks[survey]['mask_mag_max'][ref_bfilter]

# store config file in workdir
with open(os.path.join(workdir, 'gawa.cfg'), 'w') as outfile:
    json.dump(param, outfile)

config = os.path.join(workdir, 'gawa.cfg')    

input_data_structure = param['input_data_structure']
footprint = param['footprint']

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
        survey_footprint, footprint[survey], param['admin']['tiling'], tiles_filename
    )
    n_threads, thread_ids = split_equal_area_in_threads(
        param['admin']['nthreads_max'], tiles_filename
    )
    add_key_to_fits(tiles_filename, thread_ids, 'thread_id', 'int')
    all_tiles = read_FitsCat(tiles_filename)
else:
    all_tiles = read_FitsCat(tiles_filename)
    ntiles, n_threads = len(all_tiles), da.max(all_tiles['thread_id']).compute() 
    thread_ids = all_tiles['thread_id']
print ('Ntiles / Nthreads = ', ntiles, ' / ', n_threads)

gawa_cfg = param['gawa_cfg']

# prepare dslices 
compute_dslices(isochrone_masks[survey], gawa_cfg['dslices'], workdir)

# compute cmd_masks 
print ('Compute CMD masks')

out_paths = param['out_paths']
compute_cmd_masks(isochrone_masks[survey], out_paths, gawa_cfg)

# detect clusters on all tiles 
# with Pool(3) as p:
#     p.map(run_gawa_tile, [(config, ith) for ith in np.unique(thread_ids)])
with LocalCluster(processes=False, threads_per_worker=2,
            n_workers=4, memory_limit='20GB') as cluster:
    with Client(cluster) as client:
        client.restart()
        futures = client.map(run_gawa_tile, [(config, ith) for ith in da.unique(thread_ids).compute()])
        for future in futures:
            progress(future)

# concatenate
# tiles with clusters 
eff_tiles = tiles_with_clusters(out_paths, all_tiles)
data_clusters = gawa_concatenate(eff_tiles, gawa_cfg, out_paths)
data_clusters.write(
    os.path.join(out_paths['workdir'],'clusters.fits'), overwrite=True
)

print ('all done folks !')
end = time.time()
print(f"elapsed time: {end-start}")