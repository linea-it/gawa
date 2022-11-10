from parsl import python_app, bash_app


@python_app
def run_gawa_tile_job(args):
    """_summary_

    Args:
        args (tuple): _description_
    """

    import time
    import yaml
    import os
    from astropy.table import Table
    from lib.utils import get_logger

    from lib.utils import (
        create_directory,
        create_tile_specs,
        read_mosaicFitsCat_in_disc,
        read_mosaicFootprint_in_disc,
    )

    from lib.gawa import create_gawa_directories, gawa_tile

    gawa_root = os.environ.get("GAWA_ROOT", ".")
    level = os.environ.get("GAWA_LOG_LEVEL", "info")
    os.chdir(gawa_root)

    start_time_full = time.time()
    tile, config = args

    # read config file
    with open(config) as fstream:
        params = yaml.load(fstream, Loader=yaml.FullLoader)

    survey = params["survey"]
    starcat = params["starcat"][survey]
    out_paths = params["out_paths"]
    admin = params["admin"]
    footprint = params["footprint"][survey]
    isochrone_masks = params["isochrone_masks"][survey]
    gawa_cfg = params["gawa_cfg"]
    workdir = out_paths["workdir"]

    logger = get_logger(
        name=f'gawa-{tile["id"]}',
        level=level,
        stdout=os.path.join(workdir, f'gawa-{tile["id"]}.log'),
    )

    tile_dir = os.path.join(workdir, "tiles", "tile_" + str(int(tile["id"])).zfill(3))
    logger.info(f'.....Tile {int(tile["id"])}')
    create_directory(tile_dir)
    create_gawa_directories(tile_dir, out_paths["gawa"])
    out_paths["workdir_loc"] = tile_dir  # local update
    tile_specs = create_tile_specs(tile, admin)
    tile_radius_deg = tile_specs.get("radius_tile_deg")
    logger.info("tile_radius_deg: {}".format(tile_radius_deg))
    data_star_tile = read_mosaicFitsCat_in_disc(starcat, tile, tile_radius_deg)
    data_fp_tile = read_mosaicFootprint_in_disc(footprint, tile, tile_radius_deg)
    logger.info("starcat: {}".format(starcat))

    if params["verbose"] >= 2:
        t = Table(data_star_tile)
        t.write(os.path.join(tile_dir, "starcat.fits"), overwrite=True)
        t = Table(data_fp_tile)
        t.write(os.path.join(tile_dir, "footprint.fits"), overwrite=True)

    if not os.path.isfile(
        os.path.join(tile_dir, out_paths["gawa"]["results"], "clusters.fits")
    ):
        data_clusters, tile_info = gawa_tile(
            tile_specs,
            isochrone_masks,
            data_star_tile,
            starcat,
            data_fp_tile,
            footprint,
            gawa_cfg,
            admin,
            out_paths,
            params["verbose"],
            logger=logger,
        )

        if data_clusters is not None:
            t = Table(data_clusters)
            t.write(
                os.path.join(tile_dir, out_paths["gawa"]["results"], "clusters.fits"),
                overwrite=True,
            )
        tile_info = Table(tile_info)
        tile_info.write(
            os.path.join(
                out_paths["workdir_loc"], out_paths["gawa"]["results"], "tile_info.fits"
            ),
            overwrite=True,
        )

    logger.info(f'gawa-{tile["id"]} ~ time elapsed: {time.time() - start_time_full}')
    return tile


@python_app
def create_mosaic_footprint_job(footprint, fpath):
    """_summary_

    Args:
        footprint (_type_): _description_
        fpath (_type_): _description_
    """

    import os
    from lib.utils import read_FitsFootprint, hpix2radec, radec2hpix
    from astropy.io import fits
    import numpy as np

    # from a survey footprint create a mosaic of footprints at lower resol.
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    hpix0, frac0 = read_FitsFootprint(footprint["survey_footprint"], footprint)
    ra0, dec0 = hpix2radec(hpix0, footprint["Nside"], footprint["nest"])
    hpix = radec2hpix(
        ra0, dec0, footprint["mosaic"]["Nside"], footprint["mosaic"]["nest"]
    )
    for hp in np.unique(hpix):
        all_cols = fits.ColDefs(
            [
                fits.Column(
                    name=footprint["key_pixel"],
                    format="K",
                    array=hpix0[np.isin(hpix, hp)],
                ),
                fits.Column(name="ra", format="E", array=ra0[np.isin(hpix, hp)]),
                fits.Column(name="dec", format="E", array=dec0[np.isin(hpix, hp)]),
                fits.Column(
                    name=footprint["key_frac"],
                    format="K",
                    array=frac0[np.isin(hpix, hp)],
                ),
            ]
        )
        hdu = fits.BinTableHDU.from_columns(all_cols)
        hdu.writeto(os.path.join(fpath, str(hp) + "_footprint.fits"), overwrite=True)
    return


@python_app
def compute_cmd_masks_job(isochrone_masks, out_paths, gawa_cfg):
    from lib.gawa import compute_cmd_masks
    from lib.utils import get_logger
    import os
    import time

    gawa_root = os.environ.get("GAWA_ROOT", ".")
    level = os.environ.get("GAWA_LOG_LEVEL", "info")
    os.chdir(gawa_root)

    start_time_full = time.time()

    workdir = out_paths["workdir"]

    logger = get_logger(
        name="compute_masks",
        level=level,
        stdout=os.path.join(workdir, "compute_masks.log"),
    )

    logger.info("> Computing cmd masks")
    compute_cmd_masks(isochrone_masks, out_paths, gawa_cfg, logger)
    logger.info(f"...time elapsed: {time.time() - start_time_full}")
    return


@bash_app
def initialize(stderr="init.out", stdout="init.out"):
    return 'echo "$(date): Initialing processing..."'
