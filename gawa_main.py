import numpy as np
import yaml
import os
import json
import matplotlib
matplotlib.use('Agg')

from lib.multithread import split_equal_area_in_threads
from lib.utils import (
    hpx_split_survey, get_logger, read_FitsCat,
    create_directory, create_mosaic_footprint,
    create_survey_footprint_from_mosaic, add_key_to_fits
)
from lib.gawa import (compute_dslices, gawa_concatenate, tiles_with_clusters)
from lib.parsl_config import get_executor
from lib.apps import run_gawa_tile_job, compute_cmd_masks_job
import argparse
import logging, time
import parsl

LEVEL = os.environ.get('GAWA_LOG_LEVEL', 'info')


def run(param, parsl_conf):
    """ Run GAWA workflow

    Args:
        param (dict): workflow config
        parsl_conf (object): Config Parsl instance 
    """
    workdir = param['out_paths']['workdir']
    create_directory(workdir)

    # Save the configuration used in the output directory
    with open(f'{workdir}/gawa.ini', 'w') as _file:
        yaml.dump(param, _file)

    logger = get_logger(
        name=__name__, level=LEVEL,
        stdout=os.path.join(workdir, 'pipeline.log')
    )

    handler_cons = logging.StreamHandler()
    logger.addHandler(handler_cons)

    start_time_full = time.time()

    # Changing run directory to sandbox in "child jobs".
    parsl_conf.run_dir = os.path.join(workdir, "runinfo")

    # Settings Parsl configurations
    parsl.clear()
    parsl.load(parsl_conf)

    # Working & output directories 
    create_directory(os.path.join(workdir, 'tiles'))
    survey = param['survey']
    logger.info(f'Survey: {survey}')
    logger.info(f'Workdir: {workdir}')
    tiles_filename = os.path.join(
        workdir, param['admin']['tiling']['tiles_filename']
    )

    # create required data structure if not exist and update config 
    if not param['input_data_structure'][survey]['footprint_hpx_mosaic']:
        start_time = time.time()
        logger.info(f'> Creating footprint mosaic for {survey}')
        create_mosaic_footprint(
            param['footprint'][survey], os.path.join(workdir, 'footprint')
        )
        param['footprint'][survey]['mosaic']['dir'] = os.path.join(
            workdir, 'footprint'
        )
        logger.info(f'...Done in {time.time() - start_time} seconds')

    ref_bfilter = param['ref_bfilter']
    ref_rfilter = param['ref_rfilter']
    ref_color = param['ref_color']
    isochrone_masks = param['isochrone_masks']

    # update parameters with selected filters in config 
    param['starcat'][survey]['keys']['key_mag_blue'] = param['starcat'][survey]['keys']['key_mag'][ref_bfilter]
    param['starcat'][survey]['keys']['key_mag_red'] = param['starcat'][survey]['keys']['key_mag'][ref_rfilter]
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
            start_time = time.time()
            logger.info(f'> Creating survey_footprint from mosaic')
            create_survey_footprint_from_mosaic(
                footprint[survey], survey_footprint
            )
            logger.info(f'...Done in {time.time() - start_time} seconds')
    else:
        survey_footprint = footprint[survey]['survey_footprint']

    if not os.path.isfile(tiles_filename):
        start_time = time.time()
        logger.info(f'> Creating tiles')
        ntiles = hpx_split_survey(
            survey_footprint, footprint[survey], param['admin']['tiling'], tiles_filename
        )
        n_threads, thread_ids = split_equal_area_in_threads(
            param['admin']['nthreads_max'], tiles_filename
        )
        add_key_to_fits(tiles_filename, thread_ids, 'thread_id', 'int')
        all_tiles = read_FitsCat(tiles_filename)
        logger.info(f'...Done in {time.time() - start_time} seconds')
    else:
        all_tiles = read_FitsCat(tiles_filename)
        ntiles, n_threads = len(all_tiles), np.amax(all_tiles['thread_id']) 
        thread_ids = all_tiles['thread_id']
    logger.info(f'Ntiles / Nthreads = {ntiles}/{n_threads}')

    gawa_cfg = param['gawa_cfg']

    # prepare dslices 
    start_time = time.time()
    logger.info(f'> Preparing dslices')
    compute_dslices(isochrone_masks[survey], gawa_cfg['dslices'], workdir)
    logger.info(f'...Done in {time.time() - start_time} seconds')

    # compute cmd_masks 
    start_time = time.time()
    logger.info('> Compute CMD masks')

    out_paths = param['out_paths']
    proc_masks = compute_cmd_masks_job(isochrone_masks[survey], out_paths, gawa_cfg)
    # proc_masks.result()
    logger.info(f'...Done in {time.time() - start_time} seconds')

    logger.info(f'> Compute Gawa per tiles')
    start_time = time.time()
    procs = list()
    for tile in all_tiles:
        procs.append(run_gawa_tile_job((tile, config))) 

    for proc in procs:
        proc.result()
    logger.info(f'...Done in {time.time() - start_time} seconds')

    # concatenate
    # tiles with clusters 
    start_time = time.time()
    logger.info(f'> Concatenating tiles with clusters')
    eff_tiles = tiles_with_clusters(out_paths, all_tiles)
    data_clusters = gawa_concatenate(eff_tiles, gawa_cfg, out_paths)
    data_clusters.write(
        os.path.join(out_paths['workdir'],'clusters.fits'), overwrite=True
    )
    logger.info(f'...All done folks: {time.time() - start_time} seconds')
    logger.info(f'Results in {workdir}')
    logger.info(f'Time elapsed: {time.time() - start_time_full}')
    parsl.clear()


if __name__ == '__main__':
    working_dir = os.getcwd()

    # Create the parser and add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config_path', help="yaml config path")

    args = parser.parse_args()
    config_path = args.config_path

    # Loading Lephare configurations
    with open(config_path) as _file:
        gawa_config = yaml.load(_file, Loader=yaml.FullLoader)

    parsl_config = get_executor(gawa_config['executor'], gawa_config['executors'])

    gawa_root = os.getenv('GAWA_ROOT', '.')
    os.chdir(gawa_root)

    # Run GAWA
    run(gawa_config, parsl_config)
