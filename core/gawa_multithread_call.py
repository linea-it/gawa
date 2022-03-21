import numpy as np
import yaml
import os
import astropy.io.fits as fits
import healpy as hp
from astropy.table import Table
import sys
import json

from multithread import split_equal_area_in_threads
from utils import split_survey_from_hpx, hpx_split_survey
from utils import read_FitsCat, concatenate_fits, create_directory
from utils import create_mosaic_footprint, create_survey_footprint_from_mosaic
from utils import add_key_to_fits
from gawa import compute_cmd_masks, compute_dslices

# read config file as online argument 
config = sys.argv[1]

pypath = '/home/benoist/python-scripts/code_magyc/cogito/venv3/bin/python'
scr = '/home/benoist/python-scripts/ga_wazpy_v2'

# read config file
with open(config) as fstream:
    param_data = yaml.load(fstream)
globals().update(param_data)

# Working & output directories 
workdir = out_paths['workdir']
create_directory(workdir)
create_directory(os.path.join(workdir, 'tiles'))
print ('Survey : ', survey)
print ('workdir : ', workdir)
tiles_filename = os.path.join(workdir, admin['tiling']['tiles_filename'])

# store config file in workdir
with open(os.path.join(workdir, 'gawa.cfg'), 'w') as outfile:
    json.dump(param_data, outfile)

# create required data structure if not exist and update config 
if not input_data_structure[survey]['footprint_hpx_mosaic']:
    create_mosaic_footprint(footprint[survey], os.path.join(workdir, 'footprint'))
    param_data['footprint'][survey]['mosaic']['dir'] = os.path.join(workdir, 'footprint')
with open(os.path.join(workdir, 'gawa.cfg'), 'w') as outfile:
    json.dump(param_data, outfile)
config = os.path.join(workdir, 'gawa.cfg')    

# split_area:
if input_data_structure[survey]['footprint_hpx_mosaic']: 
    survey_footprint = os.path.join(workdir, 'survey_footprint.fits')
    if not os.path.isfile(survey_footprint):
        create_survey_footprint_from_mosaic(footprint[survey],   survey_footprint)
else:
    survey_footprint = footprint[survey]['survey_footprint']

if not os.path.isfile(tiles_filename):
    ntiles = hpx_split_survey (survey_footprint, footprint[survey], admin['tiling'], tiles_filename)
    n_threads, thread_ids = split_equal_area_in_threads(admin['nthreads_max'], tiles_filename)
    add_key_to_fits(tiles_filename, thread_ids, 'thread_id', 'int')
else:
    dat = read_FitsCat(tiles_filename)
    ntiles, n_threads = len(dat), np.amax(dat['thread_id']) 
    thread_ids = dat['thread_id']
print ('Ntiles / Nthreads = ', ntiles, ' / ', n_threads)

# prepare dslices 
compute_dslices(isochrone_masks[survey], gawa_cfg['dslices'], workdir)

# compute cmd_masks 
print ('Compute CMD masks')
compute_cmd_masks(isochrone_masks[survey], out_paths, gawa_cfg)


#os.system ("oarsub --array "+str(n_threads)+" -n gawa -l /core=1,walltime=20:00:00 -S '"+pypath+\
#           " -W ignore "+scr+"gawa_thread_call.py "+config)

# 
for i in np.unique(thread_ids): 
    os.system  (pypath+" -W ignore "+os.path.join(scr, "gawa_thread_call.py")+" "+config+" "+str(i))


# concatenate
os.system  (pypath+" -W ignore "+os.path.join(scr, "gawa_concatenate.py")+" "+config)

#os.system ('oarsub $(oarstat -u benoist | grep pmem | awk \'{printf(\" -a %d \", $1)}\') -n pmem_end -l /core=1,walltime=05:00:00 -S \''+pypath+' -W ignore '+scr+'pmem_multithread_concatenate.py '+config+' '+str(n_threads)+'\'')


print ('all done folks !')
