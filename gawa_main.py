from lib.multithread import split_equal_area_in_threads
from lib.utils import (
    hpx_split_survey,
    get_logger,
    read_FitsCat,
    create_directory,
    create_survey_footprint_from_mosaic,
    add_key_to_fits,
)
from lib.gawa import compute_dslices, gawa_concatenate, tiles_with_clusters
from lib.parsl_config import get_parsl_config
from lib.apps import (
    run_gawa_tile_job,
    compute_cmd_masks_job,
    create_mosaic_footprint_job,
    initialize,
)
import argparse
import logging
import time
import parsl
import matplotlib
import numpy as np
import yaml
import os
import json

matplotlib.use("Agg")

LEVEL = os.environ.get("GAWA_LOG_LEVEL", "info")


class Gawa(object):
    config = {}
    parsl_cfg = None
    workdir = "."
    step_time = {}
    logger = logging.getLogger(__name__)

    def __init__(self, config: dict):
        self.config = config
        self.parsl_cfg = get_parsl_config(self.config)
        self.workdir = self.config["out_paths"]["workdir"]

        # creating working & output directories
        create_directory(self.workdir)
        create_directory(os.path.join(self.workdir, "tiles"))

        # save the configuration used in the output directory
        with open(f"{self.workdir}/gawa.ini", "w") as _file:
            yaml.dump(config, _file)

        self.logger = self._get_parsl_log()

        # changing run directory to sandbox in "child jobs".
        self.parsl_cfg.run_dir = os.path.join(self.workdir, "runinfo")

    def run(self):
        """Run GAWA workflow"""

        # settings Parsl configurations
        parsl.clear()
        parsl.load(self.parsl_cfg)

        survey = self.config.get("survey")
        self.logger.info(f"Survey: {survey}")
        self.logger.info(f"Workdir: {self.workdir}")
        self.logger.info(f"Executor: {self.config.get('executor')}")

        # step 1: creating footprint mosaic if not exist and updating config
        if not self.config["input_data_structure"][survey]["footprint_hpx_mosaic"]:
            footprint_path = self.create_mosaic_footprint()
            self.config["footprint"][survey]["mosaic"]["dir"] = footprint_path

        ref_bfilter = self.config["ref_bfilter"]
        ref_rfilter = self.config["ref_rfilter"]
        ref_color = self.config["ref_color"]
        isoc_masks = self.config["isochrone_masks"][survey]

        # update parameters with selected filters in config
        keys = self.config["starcat"][survey]["keys"]
        keys["key_mag_blue"] = keys["key_mag"][ref_bfilter]
        keys["key_mag_red"] = keys["key_mag"][ref_rfilter]
        isoc_masks["magerr_blue_file"] = isoc_masks["magerr_file"][ref_bfilter]
        isoc_masks["magerr_red_file"] = isoc_masks["magerr_file"][ref_rfilter]
        isoc_masks["model_file"] = isoc_masks["model_file"][ref_color]
        isoc_masks["mask_color_min"] = isoc_masks["mask_color_min"][ref_color]
        isoc_masks["mask_color_max"] = isoc_masks["mask_color_max"][ref_color]
        isoc_masks["mask_mag_min"] = isoc_masks["mask_mag_min"][ref_bfilter]
        isoc_masks["mask_mag_max"] = isoc_masks["mask_mag_max"][ref_bfilter]

        # store config file in workdir
        with open(os.path.join(self.workdir, "gawa.cfg"), "w") as outfile:
            json.dump(self.config, outfile)

        survey_footprint = self.create_survey_footprint_from_mosaic()
        footprint = self.config["footprint"]
        tiles_filename = os.path.join(
            self.workdir, self.config["admin"]["tiling"]["tiles_filename"]
        )

        if not os.path.isfile(tiles_filename):
            start_time = time.time()
            self.logger.info("> Creating tiles")
            ntiles = hpx_split_survey(
                survey_footprint,
                footprint[survey],
                self.config["admin"]["tiling"],
                tiles_filename,
            )
            n_threads, thread_ids = split_equal_area_in_threads(
                self.config["admin"]["nthreads_max"], tiles_filename
            )
            add_key_to_fits(tiles_filename, thread_ids, "thread_id", "int")
            all_tiles = read_FitsCat(tiles_filename)
            _time = time.time() - start_time
            self.step_time.update({"create_tiles": _time})
            self.logger.info(f"...Done in {_time} seconds")
        else:
            all_tiles = read_FitsCat(tiles_filename)
            ntiles, n_threads = len(all_tiles), np.amax(all_tiles["thread_id"])
            thread_ids = all_tiles["thread_id"]

        self.logger.info(f"Ntiles / Nthreads = {ntiles}/{n_threads}")

        gawa_cfg = self.config["gawa_cfg"]

        # prepare dslices
        start_time = time.time()
        self.logger.info("> Preparing dslices")
        compute_dslices(isoc_masks, gawa_cfg["dslices"], self.workdir)
        _time = time.time() - start_time
        self.step_time.update({"prepare_dslices": _time})
        self.logger.info(f"...Done in {_time} seconds")

        # waiting for resources availability
        self.logger.info("> Waiting for resources availability...")
        start_wait_time = time.time()
        future = initialize()
        future.result()
        wait_time = time.time() - start_wait_time
        self.logger.info(f"...Done in {wait_time} seconds")

        # compute cmd_masks
        start_time = time.time()
        self.logger.info("> Compute CMD masks")
        out_paths = self.config["out_paths"]
        proc_masks = compute_cmd_masks_job(isoc_masks, out_paths, gawa_cfg)
        proc_masks.result()
        _time = time.time() - start_time
        self.step_time.update({"compute_cmd_masks": _time})
        self.logger.info(f"...Done in {_time} seconds")

        # compute gawa per tiles
        self.logger.info("> Compute Gawa per tiles")
        start_time = time.time()
        procs = list()
        config = os.path.join(self.workdir, "gawa.cfg")

        for tile in all_tiles:
            procs.append(run_gawa_tile_job((tile, config)))

        for proc in procs:
            proc.result()

        _time = time.time() - start_time
        self.step_time.update({"compute_gawa_per_tiles": _time})
        self.logger.info(f"...Done in {_time} seconds")

        self.concatenate_tiles(all_tiles)

        self.logger.info(f"Results in {self.workdir}")
        self.logger.info(f"Time elapsed: {self.fulltime}")
        parsl.clear()

    @property
    def fulltime(self):
        """Sum the execution time of all steps"""

        fulltime = 0.0

        for key, value in self.step_time.items():
            fulltime += value

        return fulltime

    def concatenate_tiles(self, all_tiles):
        """Concatenate tiles with clusters

        Args:
            all_tiles (astropy.io.fits.fitsrec.FITS_rec): all tiles
        """

        out_paths = self.config["out_paths"]
        gawa_cfg = self.config["gawa_cfg"]

        # concatenate tiles with clusters
        start_time = time.time()
        self.logger.info("> Concatenating tiles with clusters")
        eff_tiles = tiles_with_clusters(out_paths, all_tiles)
        data_clusters = gawa_concatenate(eff_tiles, gawa_cfg, out_paths)
        data_clusters.write(
            os.path.join(out_paths["workdir"], "clusters.fits"), overwrite=True
        )
        _time = time.time() - start_time
        self.logger.info(f"...Done in {_time} seconds")
        self.step_time.update({"concatenate_tiles": _time})

    def create_survey_footprint_from_mosaic(self):
        """Create survey footprint from mosaic"""

        start_time = time.time()
        survey = self.config.get("survey")
        input_data_structure = self.config["input_data_structure"]
        footprint = self.config["footprint"]

        if input_data_structure[survey]["footprint_hpx_mosaic"]:
            survey_footprint = os.path.join(self.workdir, "survey_footprint.fits")
            if not os.path.isfile(survey_footprint):
                self.logger.info("> Creating survey_footprint from mosaic")
                create_survey_footprint_from_mosaic(footprint[survey], survey_footprint)
                _time = time.time() - start_time
                self.step_time.update({"create_survey_footprint_from_mosaic": _time})
                self.logger.info(f"...Done in {_time} seconds")
        else:
            survey_footprint = footprint[survey]["survey_footprint"]

        return survey_footprint

    def create_mosaic_footprint(self):
        """Create mosaic footprint"""

        start_time = time.time()
        survey = self.config.get("survey")
        self.logger.info(f"> Creating footprint mosaic for {survey}")
        footprint_path = os.path.join(self.workdir, "footprint")
        process = create_mosaic_footprint_job(
            self.config["footprint"][survey], footprint_path
        )
        process.result()
        _time = time.time() - start_time
        self.step_time.update({"create_mosaic_footprint": _time})
        self.logger.info(f"...Done in {_time} seconds")
        return footprint_path

    def _get_parsl_log(self):
        """Get logging instance

        Returns:
            logging.Logger: logging instance
        """

        logger = get_logger(
            name=__name__,
            level=LEVEL,
            stdout=os.path.join(self.workdir, "pipeline.log"),
        )
        logger.addHandler(logging.StreamHandler())
        return logger


if __name__ == "__main__":
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="config_path", help="yaml config path")

    args = parser.parse_args()
    config_path = args.config_path

    # Loading configurations
    with open(config_path) as _file:
        gawa_config = yaml.load(_file, Loader=yaml.FullLoader)

    gawa_root = os.getenv("GAWA_ROOT", ".")
    os.chdir(gawa_root)

    # Run GAWA
    gawa_wf = Gawa(gawa_config)
    gawa_wf.run()
