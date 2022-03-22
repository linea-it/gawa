from astropy.io import fits
from astropy import wcs
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import os
from astropy import units as u
import math
import astropy.coordinates
import healpy as hp
from astropy.table import Table
from skimage.feature import peak_local_max
from astropy.coordinates import SkyCoord
from matplotlib import path
import subprocess

from .utils import create_directory, read_FitsCat
from .utils import tile_radius, sub_hpix, radec2hpix, hpix2radec, dist_ang
from .utils import area_ann_deg2, hpx_in_annulus, join_struct_arrays
from .utils import cond_in_disc


def tile_dir_name(workdir, tile_nr):
    """_summary_

    Args:
        workdir (_type_): _description_
        tile_nr (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.join(workdir, "tiles", "tile_" + str(tile_nr).zfill(3))


def create_gawa_directories(root, path):
    """creates the relevant directories for writing results/plots

    Args:
        root (_type_): _description_
        path (_type_): _description_
    """

    if not os.path.exists(os.path.join(root, path["results"])):
        os.mkdir(os.path.join(root, path["results"]))
    if not os.path.exists(os.path.join(root, path["plots"])):
        os.mkdir(os.path.join(root, path["plots"]))
    if not os.path.exists(os.path.join(root, path["files"])):
        os.mkdir(os.path.join(root, path["files"]))
    return


def pix2xy(ix, iy, xmin, xmax, ymin, ymax, nx, ny):
    """_summary_

    Args:
        ix (_type_): _description_
        iy (_type_): _description_
        xmin (_type_): _description_
        xmax (_type_): _description_
        ymin (_type_): _description_
        ymax (_type_): _description_
        nx (_type_): _description_
        ny (_type_): _description_

    Returns:
        _type_: _description_
    """
    xstep, ystep = (xmax - xmin) / float(nx), (ymax - ymin) / float(ny)
    x, y = (
        xmin + (ix.astype(float) + 0.5) * xstep,
        ymin + (iy.astype(float) + 0.5) * ystep,
    )  # noqa
    return x, y


def xy2pix(x, y, xmin, xmax, ymin, ymax, nx, ny):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        xmin (_type_): _description_
        xmax (_type_): _description_
        ymin (_type_): _description_
        ymax (_type_): _description_
        nx (_type_): _description_
        ny (_type_): _description_

    Returns:
        _type_: _description_
    """
    xstep, ystep = (xmax - xmin) / float(nx), (ymax - ymin) / float(ny)
    ix, iy = (x - xmin) / xstep, (y - ymin) / ystep
    ix[(ix >= float(nx))], iy[(iy >= float(ny))] = nx - 1, ny - 1
    return ix.astype(int), iy.astype(int)


def x2pix(x, xmin, xmax, nx):
    """_summary_

    Args:
        x (_type_): _description_
        xmin (_type_): _description_
        xmax (_type_): _description_
        nx (_type_): _description_

    Returns:
        _type_: _description_
    """
    xstep = (xmax - xmin) / float(nx)
    ix = (x - xmin) / xstep
    ix[(ix >= float(nx))] = nx - 1
    return ix.astype(int)


def effective_area_framed_tile(tile, data_fp, footprint, admin):
    """_summary_

    Args:
        tile (_type_): _description_
        data_fp (_type_): _description_
        footprint (_type_): _description_
        admin (_type_): _description_

    Returns:
        _type_: _description_
    """

    Nside, nest = footprint["Nside"], footprint["nest"]
    hpix_map, frac_map = (
        data_fp[footprint["key_pixel"]],
        data_fp[footprint["key_frac"]],
    )  # noqa
    radius_deg = tile_radius(admin)
    pixels_in_disc = hp.query_disc(
        nside=Nside,
        nest=nest,
        vec=hp.pixelfunc.ang2vec(
            np.radians(90.0 - tile["dec"]), np.radians(tile["ra"])
        ),  # noqa
        radius=np.radians(radius_deg),
        inclusive=False,
    )  # noqa
    framed_eff_area_deg2 = np.sum(
        frac_map[np.isin(hpix_map, pixels_in_disc)]
    ) * hp.pixelfunc.nside2pixarea(
        Nside, degrees=True
    )  # noqa
    return framed_eff_area_deg2


def plot_cmd(xall, yall, xar_in, yar_in, isochrone_masks, file_png):
    """_summary_

    Args:
        xall (_type_): _description_
        yall (_type_): _description_
        xar_in (_type_): _description_
        yar_in (_type_): _description_
        isochrone_masks (bool): _description_
        file_png (_type_): _description_
    """
    xmin, xmax = (
        isochrone_masks["mask_color_min"],
        isochrone_masks["mask_color_max"],
    )  # noqa
    ymin, ymax = (
        isochrone_masks["mask_mag_min"],
        isochrone_masks["mask_mag_max"],
    )  # noqa
    plt.clf()
    plt.scatter(xar_in, yar_in, s=2, c="red", alpha=1, zorder=1)
    plt.scatter(xall, yall, s=2, c="green", alpha=1, zorder=0)
    plt.xlabel("g-r", fontsize=20)
    plt.ylabel("g", fontsize=20)
    plt.axis((xmin, xmax, ymax, ymin))
    plt.tight_layout()
    plt.savefig(file_png)
    return


def cmd_filter(magg, magr, cmd_mask, isochrone_masks):
    """_summary_

    Args:
        magg (_type_): _description_
        magr (_type_): _description_
        cmd_mask (_type_): _description_
        isochrone_masks (bool): _description_

    Returns:
        _type_: _description_
    """
    xmin, xmax = isochrone_masks["mask_color_min"], isochrone_masks["mask_color_max"]
    ymin, ymax = isochrone_masks["mask_mag_min"], isochrone_masks["mask_mag_max"]
    nx = isochrone_masks["mask_resolution"]
    mag, color = magg, magg - magr
    # clip objects falling outside the frame
    cond_frame = (color < xmax) & (color > xmin) & (mag < ymax) & (mag > ymin)
    inframe = np.argwhere(cond_frame).T[0]

    # mask grid
    step = (xmax - xmin) / float(nx)
    ny = int((ymax - ymin) / step)
    ixdata, iydata = xy2pix(
        color[cond_frame], mag[cond_frame], xmin, xmax, ymin, ymax, nx, ny
    )
    flag_vis = cmd_mask[ixdata, iydata]
    cond_in = np.zeros(len(mag)).astype(bool)
    cond_in[inframe] = flag_vis.astype(bool)

    return cond_in


def cmd_mask(dslice, isochrone_masks, out_paths):
    """_summary_

    Args:
        dslice (_type_): _description_
        isochrone_masks (bool): _description_
        out_paths (_type_): _description_

    Returns:
        _type_: _description_
    """

    nsig = 2.0
    xmin, xmax = isochrone_masks["mask_color_min"], isochrone_masks["mask_color_max"]
    ymin, ymax = isochrone_masks["mask_mag_min"], isochrone_masks["mask_mag_max"]
    model_file = isochrone_masks["model_file"]
    gerr_file, rerr_file = (
        isochrone_masks["magerr_g_file"],
        isochrone_masks["magerr_r_file"],
    )
    nx = isochrone_masks["mask_resolution"]

    # mask grid
    step = (xmax - xmin) / float(nx)
    ny = int((ymax - ymin) / step)
    ix, iy = np.linspace(0, nx - 1, nx).astype(int), np.linspace(0, ny - 1, ny).astype(
        int
    )
    x, y = np.meshgrid(ix, iy)

    ixar, iyar = np.ravel(x), np.ravel(y)  # all pixels
    xar, yar = pix2xy(ixar, iyar, xmin, xmax, ymin, ymax, nx, ny)
    points = np.dstack((xar, yar)).reshape(len(xar), 2)

    # read error files
    gm, gm_err = np.loadtxt(gerr_file, usecols=(0, 1), unpack=True)
    rm, rm_err = np.loadtxt(rerr_file, usecols=(0, 1), unpack=True)

    # get the polygon in color mag for a given cldistance
    gr, g0 = np.loadtxt(model_file, usecols=(0, 1), unpack=True)
    g = g0 + 5.0 * np.log10(dslice / 10.0)
    vertices = np.dstack((gr, g)).reshape(len(g), 2)

    p = path.Path(vertices)
    points_in = p.contains_points(points)
    xar_in, yar_in = xar[points_in], yar[points_in]

    g_in, r_in = yar_in, yar_in - xar_in
    gerr_in, rerr_in = np.interp(g_in, gm, gm_err), np.interp(r_in, rm, rm_err)
    xerr_in, yerr_in = (gerr_in**2 + rerr_in**2) ** 0.5, gerr_in

    for i in range(0, len(xar_in)):
        xarmin, xarmax = xar_in[i] - nsig * xerr_in[i], xar_in[i] + nsig * xerr_in[i]
        yarmin, yarmax = yar_in[i] - nsig * yerr_in[i], yar_in[i] + nsig * yerr_in[i]
        xran = (xarmax - xarmin) * np.random.random_sample(20) + xarmin
        yran = (yarmax - yarmin) * np.random.random_sample(20) + yarmin
        if i == 0:
            xall, yall = np.copy(xran), np.copy(yran)
        else:
            xall, yall = np.hstack((xall, xran)), np.hstack((yall, yran))

    ixall, iyall = xy2pix(xall, yall, xmin, xmax, ymin, ymax, nx, ny)

    mask = np.zeros((nx, ny))
    mask[ixall, iyall] = 1.0

    # remove remaining holes in the mask
    iy_HB = max(
        x2pix(np.array([gmag_HB(isochrone_masks, dslice) + 1.0]), ymin, ymax, ny)[0], 0
    )
    ixar, iyar = np.argwhere(mask == 1).T[0], np.argwhere(mask == 1).T[1]
    ixarn, iyarn = ixar[(iyar < iy_HB)], iyar[(iyar < iy_HB)]
    for iy in range(iy_HB, ny):
        xmask = mask[:, iy]
        mask[
            np.argmin(xmask[xmask == 1.0]): np.argmax(xmask[xmask == 1.0]) + 1, iy
        ] = 1.0
        ixarn = np.hstack(
            (ixarn, np.arange(np.amin(ixar[iyar == iy]), np.amax(ixar[iyar == iy]) + 1))
        )
        iyarn = np.hstack(
            (
                iyarn,
                iy * np.ones(
                    np.amax(ixar[iyar == iy]) - np.amin(ixar[iyar == iy]) + 1
                ).astype(int),
            )
        )
    xalln, yalln = pix2xy(ixarn, iyarn, xmin, xmax, ymin, ymax, nx, ny)

    # check cmd plot
    plot_filename = os.path.join(
        out_paths["workdir"],
        out_paths["masks"],
        "cmd_D" + str(dslice / 1000) + "_err.png",
    )
    plot_cmd(xalln, yalln, xar_in, yar_in, isochrone_masks, plot_filename)

    return mask


def compute_cmd_masks(isochrone_masks, out_paths, gawa_cfg):
    """_summary_

    Args:
        isochrone_masks (bool): _description_
        out_paths (_type_): _description_
        gawa_cfg (_type_): _description_
    """

    dslices = read_FitsCat(
        os.path.join(out_paths["workdir"], gawa_cfg["dslices"]["dslices_filename"])
    )["dist_pc"]
    create_directory(os.path.join(out_paths["workdir"], out_paths["masks"]))
    for i in range(0, len(dslices)):
        print(".....Mask distance (kpc) = ", np.round(dslices[i] / 1000.0, 2))
        if not os.path.isfile(
            os.path.join(
                out_paths["workdir"],
                out_paths["masks"],
                "cmd_mask_D" + str(dslices[i] / 1000) + ".npy",
            )
        ):
            mask = cmd_mask(dslices[i], isochrone_masks, out_paths)
            np.save(
                os.path.join(
                    out_paths["workdir"],
                    out_paths["masks"],
                    "cmd_mask_D" + str(dslices[i] / 1000) + ".npy",
                ),
                mask,
            )
    return


def select_stars_in_slice(data_star, starcat, gawa_cfg, mask, isochrone_masks):
    """_summary_

    Args:
        data_star (_type_): _description_
        starcat (_type_): _description_
        gawa_cfg (_type_): _description_
        mask (_type_): _description_
        isochrone_masks (bool): _description_

    Returns:
        _type_: _description_
    """
    magg, magr = (
        data_star[starcat["keys"]["key_mag_g"]],
        data_star[starcat["keys"]["key_mag_r"]],
    )
    cond = cmd_filter(magg, magr, mask, isochrone_masks)
    # ra, dec = data_star[cond][starcat['keys']['key_ra']], data_star[cond][starcat['keys']['key_dec']]
    return data_star[cond]


def pixelized_radec(ra_map, dec_map, weight_map, w, nxy):
    """_summary_

    Args:
        ra_map (_type_): _description_
        dec_map (_type_): _description_
        weight_map (_type_): _description_
        w (_type_): _description_
        nxy (_type_): _description_

    Returns:
        _type_: _description_
    """
    xmap = w.wcs_world2pix(ra_map, dec_map, 0)[0]
    ymap = w.wcs_world2pix(ra_map, dec_map, 0)[1]
    xycat, xedges, yedges = np.histogram2d(
        xmap,
        ymap,
        bins=nxy,
        range=((0, nxy), (0, nxy)),
        normed=None,
        weights=weight_map,
    )
    return xycat


def pixelized_colmag(color, mag, weight, isochrone_masks, out):
    """_summary_

    Args:
        color (_type_): _description_
        mag (_type_): _description_
        weight (_type_): _description_
        isochrone_masks (bool): _description_
        out (_type_): _description_

    Returns:
        _type_: _description_
    """
    xmin, xmax = isochrone_masks["mask_color_min"], isochrone_masks["mask_color_max"]
    ymin, ymax = isochrone_masks["mask_mag_min"], isochrone_masks["mask_mag_max"]
    step = 0.05
    nx = int((xmax - xmin) / step)
    ny = int((ymax - ymin) / step)
    xmap, ymap = xy2pix(color, mag, xmin, xmax, ymin, ymax, nx, ny)
    colmag_pix, cedges, medges = np.histogram2d(
        xmap, ymap, bins=(nx, ny), range=((0, nx), (0, ny)), normed=None, weights=weight
    )

    if out is not None:
        xbins = np.linspace(xmin, xmax, nx)
        ybins = np.linspace(ymin, ymax, ny)
        plt.clf()
        plt.hist2d(color, mag, bins=(xbins, ybins), density=True, cmap=plt.cm.jet)
        plt.xlabel("g-r", fontsize=20)
        plt.ylabel("g", fontsize=20)
        plt.axis((xmin, xmax, ymax, ymin))
        plt.tight_layout()
        plt.savefig(out)

    return colmag_pix


def plot_pixelized_colmag(
    color_aper,
    gmag_aper,
    weight_aper,
    color_slaper,
    gmag_slaper,
    weight_slaper,
    color,
    mag,
    weight,
    isochrone_masks,
    out,
):
    """_summary_

    Args:
        color_aper (_type_): _description_
        gmag_aper (_type_): _description_
        weight_aper (_type_): _description_
        color_slaper (_type_): _description_
        gmag_slaper (_type_): _description_
        weight_slaper (_type_): _description_
        color (_type_): _description_
        mag (_type_): _description_
        weight (_type_): _description_
        isochrone_masks (bool): _description_
        out (_type_): _description_
    """
    xmin, xmax = isochrone_masks["mask_color_min"], isochrone_masks["mask_color_max"]
    ymin, ymax = isochrone_masks["mask_mag_min"], isochrone_masks["mask_mag_max"]
    step = 0.05
    nx = int((xmax - xmin) / step)
    ny = int((ymax - ymin) / step)

    xbins = np.linspace(xmin, xmax, nx)
    ybins = np.linspace(ymin, ymax, ny)
    plt.clf()
    plt.hist2d(color, mag, bins=(xbins, ybins), density=True, cmap=plt.cm.jet)
    plt.xlabel("g-r", fontsize=20)
    plt.ylabel("g", fontsize=20)

    plt.scatter(color_aper, gmag_aper, s=15, c="yellow", alpha=1, label="stars in aper")
    plt.scatter(
        color_slaper, gmag_slaper, s=15, c="red", alpha=1, label="stars in aper + mask"
    )

    plt.axis((xmin, xmax, ymax, ymin))
    plt.tight_layout()
    plt.legend()
    plt.savefig(out)

    return


def create_wcs(gawa_cfg, tile_specs):
    """_summary_

    Args:
        gawa_cfg (_type_): _description_
        tile_specs (_type_): _description_

    Returns:
        _type_: _description_
    """

    racen, deccen = tile_specs["ra"], tile_specs["dec"]
    pix_deg = 1.0 / (60.0 * float(gawa_cfg["map_resolution"]))
    nxy = int(2.0 * tile_specs["radius_tile_deg"] / pix_deg) + 1
    if (nxy % 2) == 0:
        nxy += 1
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [nxy / 2.0, nxy / 2.0]
    w.wcs.cdelt = np.array([-pix_deg, pix_deg])
    w.wcs.crval = [racen, deccen]
    w.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
    return w, nxy


def compute_catimage(ra_map, dec_map, weight_map, gawa_cfg, tile_specs):
    """_summary_

    Args:
        ra_map (_type_): _description_
        dec_map (_type_): _description_
        weight_map (_type_): _description_
        gawa_cfg (_type_): _description_
        tile_specs (_type_): _description_

    Returns:
        _type_: _description_
    """
    w, nxy = create_wcs(gawa_cfg, tile_specs)
    xycat = pixelized_radec(ra_map, dec_map, weight_map, w, nxy)
    return xycat


def randoms_in_spherical_cap(tile, bkg_arcmin2):
    """_summary_

    Args:
        tile (_type_): _description_
        bkg_arcmin2 (_type_): _description_

    Returns:
        _type_: _description_
    """

    area_sph = 3600.0 * 4 * np.pi * (np.degrees(1.0)) ** 2  # arcmin2
    for i in range(0, 20):
        nside = hp.order2nside(i)
        pdens = float(hp.nside2npix(nside)) / area_sph
        if pdens >= bkg_arcmin2:
            Nside_samp = nside  # *2
            break
    pixels_in_disc = hp.query_disc(
        nside=Nside_samp,
        nest=False,
        vec=hp.pixelfunc.ang2vec(
            np.radians(90.0 - tile["dec"]), np.radians(tile["ra"])
        ),
        radius=np.radians(tile["radius_tile_deg"]),
        inclusive=False,
    )
    area = 3600.0 * area_ann_deg2(0.0, tile["radius_tile_deg"])
    nsamp = int(bkg_arcmin2 * area)
    pix_samp = np.random.choice(pixels_in_disc, nsamp, replace=False)

    return hpix2radec(pix_samp, Nside_samp, False)


def compute_filled_catimage(
    ra_map, dec_map, weight_map, gawa_cfg, tile, data_fp, footprint, bkg_arcmin2
):
    """_summary_

    Args:
        ra_map (_type_): _description_
        dec_map (_type_): _description_
        weight_map (_type_): _description_
        gawa_cfg (_type_): _description_
        tile (_type_): _description_
        data_fp (_type_): _description_
        footprint (_type_): _description_
        bkg_arcmin2 (_type_): _description_

    Returns:
        _type_: _description_
    """

    # find edge pixels of the footprint
    nlist = hp.pixelfunc.get_all_neighbours(
        footprint["Nside"], data_fp[footprint["key_pixel"]], None, footprint["nest"]
    )
    mask = np.isin(nlist, data_fp[footprint["key_pixel"]])
    edge_pixels = data_fp[footprint["key_pixel"]][(np.sum(mask, axis=0) < 8)]

    # find empty footprint pixels at resolution Nside*2
    sub_edge_hpix = sub_hpix(edge_pixels, footprint["Nside"], footprint["nest"])
    shpix_filled = np.unique(
        radec2hpix(ra_map, dec_map, footprint["Nside"] * 2, footprint["nest"])
    )

    # hpix_filled = np.unique(radec2hpix(ra_map, dec_map, footprint['Nside'], footprint['nest']))
    # empty_edge_hpix = edge_pixels[np.isin(edge_pixels, hpix_filled, invert=True)]

    empty_sub_edge_hpix = sub_edge_hpix[
        np.isin(sub_edge_hpix, shpix_filled, invert=True)
    ]
    # keep only sub pixels not neighbouring the pixels with actual galaxies
    nlist_esep = hp.pixelfunc.get_all_neighbours(
        2 * footprint["Nside"], empty_sub_edge_hpix, None, footprint["nest"]
    )
    mask_esep = np.isin(nlist_esep, shpix_filled, invert=True)
    empty_sub_edge_hpix = empty_sub_edge_hpix[(np.sum(mask_esep, axis=0) == 8)]

    # build uniform randoms in spherical cap with proper density
    ra_ran, dec_ran = randoms_in_spherical_cap(tile, bkg_arcmin2)

    # keep randoms outside of the footprint or in empty sub-edge pixels
    hpx_ran1 = radec2hpix(ra_ran, dec_ran, footprint["Nside"], footprint["nest"])
    ra_ranf1 = ra_ran[np.isin(hpx_ran1, data_fp[footprint["key_pixel"]], invert=True)]
    dec_ranf1 = dec_ran[np.isin(hpx_ran1, data_fp[footprint["key_pixel"]], invert=True)]

    hpx_ran2 = radec2hpix(ra_ran, dec_ran, footprint["Nside"] * 2, footprint["nest"])
    ra_ranf2 = ra_ran[np.isin(hpx_ran2, empty_sub_edge_hpix)]
    dec_ranf2 = dec_ran[np.isin(hpx_ran2, empty_sub_edge_hpix)]

    # stack galaxies + randoms
    ra_ranf1, dec_ranf1 = np.hstack((ra_ranf1, ra_ranf2)), np.hstack(
        (dec_ranf1, dec_ranf2)
    )
    ra_all, dec_all = np.hstack((ra_ranf1, ra_map)), np.hstack((dec_ranf1, dec_map))
    weight_all = np.hstack((np.ones(len(ra_ranf1)), weight_map))

    # build catalogue image
    w, nxy = create_wcs(gawa_cfg, tile)
    xycat = pixelized_radec(ra_all, dec_all, weight_all, w, nxy)
    return xycat


def map2fits(imap, gawa_cfg, tile, fitsname):
    """_summary_

    Args:
        imap (_type_): _description_
        gawa_cfg (_type_): _description_
        tile (_type_): _description_
        fitsname (_type_): _description_
    """
    w, nxy = create_wcs(gawa_cfg, tile)
    # write xycat_fi to a file for mr_filter
    header = w.to_header()
    hdu = fits.PrimaryHDU(header=header)
    hdu.data = imap.T
    hdu.writeto(fitsname, overwrite=True)
    return


def fits2map(wmap):
    """_summary_

    Args:
        wmap (_type_): _description_

    Returns:
        _type_: _description_
    """
    hdulist = fits.open(wmap, ignore_missing_end=True)
    hdu = hdulist[0]
    wmap_t = hdulist[0].data
    w = wcs.WCS(hdu.header)
    wmap_data = wmap_t.T
    return wmap_data


def run_mr_filter(filled_catimage, wmap, gawa_cfg):
    """_summary_

    Args:
        filled_catimage (_type_): _description_
        wmap (_type_): _description_
        gawa_cfg (_type_): _description_
    """

    # run mr_filter to build wavelet map
    scale_min_pix = gawa_cfg["wavelet_specs"][gawa_cfg["detection_mode"]][
        "scale_min_arcmin"
    ] * float(gawa_cfg["map_resolution"])
    scale_max_pix = gawa_cfg["wavelet_specs"][gawa_cfg["detection_mode"]][
        "scale_max_arcmin"
    ] * float(gawa_cfg["map_resolution"])
    smin = int(round(math.log10(scale_min_pix) / math.log10(2.0)))
    smax = int(round(math.log10(scale_max_pix) / math.log10(2.0)))
    if smin == 0:
        # os.system(
        subprocess.call((
            "mr_filter -m 10 -i 3 -s 3.,3. -n " + str(smax + 1) + " -f 3 -K -C 2 -p -e0 -A " + filled_catimage + " " + wmap
        ), shell=True)
    if smin == 1:
        # os.system(
        subprocess.call((
            "mr_filter -m 10 -i 3 -s 10.,3.,3. -n " + str(smax + 1) + " -f 3 -K -C 2 -p -e0 -A " + filled_catimage + " " + wmap
        ), shell=True)
    if smin == 2:
        # os.system(
        subprocess.call((
            "mr_filter -m 10 -i 3 -s 10.,10.,3.,3. -n " + str(smax + 1) + " -f 3 -K -C 2 -p -e0 -A " + filled_catimage + " " + wmap
        ), shell=True)
    if smin == 3:
        # os.system(
        subprocess.call((
            "mr_filter -m 10 -i 3 -s 10.,10.,10.,3.,3. -n " + str(smax + 1) + " -f 3 -K -C 2 -p -e0 -A " + filled_catimage + " " + wmap
        ), shell=True)
    return


def wmap2peaks(wmap, wazp_specs, tile_specs):
    """_summary_

    Args:
        wmap (_type_): _description_
        wazp_specs (_type_): _description_
        tile_specs (_type_): _description_

    Returns:
        _type_: _description_
    """
    wmap_thresh = wazp_specs["wmap_thresh"]
    wmap_data = fits2map(wmap)
    w, nxy = create_wcs(wazp_specs, tile_specs)
    # peak detection on wmap
    local_maxi = peak_local_max(
        wmap_data, min_distance=8
    )  # this has to be reviewed and configured !
    # npeaks0 = len(local_maxi)  # nr of peaks in the filled_map
    iobj0, jobj0 = (
        local_maxi[:, 0] + 1.0,
        local_maxi[:, 1] + 1.0,
    )  # pixel coordinates of the peaks
    iobj = iobj0[(wmap_data[iobj0.astype(int), jobj0.astype(int)] > wmap_thresh)]
    jobj = jobj0[(wmap_data[iobj0.astype(int), jobj0.astype(int)] > wmap_thresh)]
    ra_peak, dec_peak = (
        w.all_pix2world(iobj, jobj, 1)[0],
        w.all_pix2world(iobj, jobj, 1)[1],
    )
    return ra_peak, dec_peak, iobj, jobj


def wave_radius(wmap_data, ip, jp, gawa_cfg):
    """_summary_

    Args:
        wmap_data (_type_): _description_
        ip (_type_): _description_
        jp (_type_): _description_
        gawa_cfg (_type_): _description_

    Returns:
        _type_: _description_
    """

    dwmap = ndi.distance_transform_edt(
        (wmap_data > gawa_cfg["wmap_thresh"]) * wmap_data
    )
    radius_arcmin = dwmap[ip, jp] / float(gawa_cfg["map_resolution"])
    return radius_arcmin


def filter_peaks(tile, map_resolution, ra0, dec0, ip0, jp0):
    """_summary_

    Args:
        tile (_type_): _description_
        map_resolution (_type_): _description_
        ra0 (_type_): _description_
        dec0 (_type_): _description_
        ip0 (_type_): _description_
        jp0 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if tile["hpix"] > 0:  # not target mode
        err_deg = 2.0 / (60.0 * float(map_resolution))
        dx, dy = err_deg * np.cos(np.radians(dec0)), err_deg
        ghpx = radec2hpix(ra0, dec0, tile["Nside"], tile["nest"])
        ghpx1 = radec2hpix(ra0 - dx, dec0 - dy, tile["Nside"], tile["nest"])
        ghpx2 = radec2hpix(ra0 - dx, dec0 + dy, tile["Nside"], tile["nest"])
        ghpx3 = radec2hpix(ra0 + dx, dec0 + dy, tile["Nside"], tile["nest"])
        ghpx4 = radec2hpix(ra0 + dx, dec0 - dy, tile["Nside"], tile["nest"])
        cond = (
            (np.isin(ghpx, tile["hpix"])) | (np.isin(ghpx1, tile["hpix"])) | (np.isin(ghpx2, tile["hpix"])) | (np.isin(ghpx3, tile["hpix"])) | (np.isin(ghpx4, tile["hpix"]))
        )
        ra, dec = ra0[cond], dec0[cond]
        ip, jp = ip0[cond], jp0[cond]
    else:
        ra = ra0[
            np.degrees(dist_ang(ra0, tile["ra"], dec0, tile["dec"])) <= tile["radius"]
        ]
        dec = dec0[
            np.degrees(dist_ang(ra0, tile["ra"], dec0, tile["dec"])) <= tile["radius"]
        ]
        ip = ip0[
            np.degrees(dist_ang(ra0, tile["ra"], dec0, tile["dec"])) <= tile["radius"]
        ]
        jp = jp0[
            np.degrees(dist_ang(ra0, tile["ra"], dec0, tile["dec"])) <= tile["radius"]
        ]
    return ra, dec, ip, jp


def coverfrac_disc(ra, dec, data_footprint, footprint, radius_deg):
    """_summary_

    Args:
        ra (_type_): _description_
        dec (_type_): _description_
        data_footprint (_type_): _description_
        footprint (_type_): _description_
        radius_deg (_type_): _description_

    Returns:
        _type_: _description_
    """
    coverfrac = np.zeros(len(ra))
    for i in range(0, len(ra)):
        hpx_in_ann, frac_in_ann, area_deg2, coverfrac[i] = hpx_in_annulus(
            ra[i], dec[i], 0.0, radius_deg, data_footprint, footprint, False
        )
    return coverfrac


def init_peaks_table(
    ra_peaks,
    dec_peaks,
    iobj,
    jobj,
    coverfrac,
    wradius,
    dslice,
    isochrone_masks,
    gawa_cfg,
):
    """
    initialize the table of peaks
    """
    Derr, Dmin, Dmax = Dist_err(isochrone_masks, dslice)

    data_peaks = np.zeros(
        (len(ra_peaks)),
        dtype={
            "names": (
                "peak_id",
                "ra",
                "dec",
                "iobj",
                "jobj",
                "dist_init_kpc",
                "dist_err_kpc",
                "dist_min_kpc",
                "dist_max_kpc",
                "coverfrac",
                gawa_cfg["clkeys"]["key_radius"],
            ),
            "formats": (
                "i8",
                "f8",
                "f8",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
            ),
        },
    )
    data_peaks["peak_id"] = np.arange(len(ra_peaks))
    data_peaks["ra"] = ra_peaks
    data_peaks["dec"] = dec_peaks
    data_peaks["iobj"] = iobj
    data_peaks["jobj"] = jobj
    data_peaks["coverfrac"] = coverfrac
    data_peaks[gawa_cfg["clkeys"]["key_radius"]] = wradius
    data_peaks["dist_init_kpc"] = (dslice / 1000.0) * np.ones(len(ra_peaks))
    data_peaks["dist_err_kpc"] = Derr * np.ones(len(ra_peaks))
    data_peaks["dist_min_kpc"] = Dmin * np.ones(len(ra_peaks))
    data_peaks["dist_max_kpc"] = Dmax * np.ones(len(ra_peaks))

    return Table(data_peaks)


def select_stars_in_disc(racen, deccen, data_stars, starcat, aper):
    """_summary_

    Args:
        racen (_type_): _description_
        deccen (_type_): _description_
        data_stars (_type_): _description_
        starcat (_type_): _description_
        aper (_type_): _description_

    Returns:
        _type_: _description_
    """
    ras, decs = (
        data_stars[starcat["keys"]["key_ra"]],
        data_stars[starcat["keys"]["key_dec"]],
    )
    cond = np.degrees(dist_ang(ras, decs, racen, deccen)) <= aper
    return data_stars[cond]


def compute_flux_aper(rap, decp, hpix, weight, aper, Nside, nest):
    """_summary_

    Args:
        rap (_type_): _description_
        decp (_type_): _description_
        hpix (_type_): _description_
        weight (_type_): _description_
        aper (_type_): _description_
        Nside (_type_): _description_
        nest (_type_): _description_

    Returns:
        _type_: _description_
    """
    pixels_in_disc = hp.query_disc(
        nside=Nside,
        nest=nest,
        vec=hp.pixelfunc.ang2vec(np.radians(90.0 - decp), np.radians(rap)),
        radius=np.radians(aper),
        inclusive=False,
    )

    Nraw = np.sum(weight[np.isin(hpix, pixels_in_disc)])
    pixelized_area = float(len(pixels_in_disc)) * hp.pixelfunc.nside2pixarea(
        Nside, degrees=True
    )
    return Nraw * area_ann_deg2(0.0, aper) / pixelized_area


def compute_flux_aper_vec(rap, decp, aper, ras, decs, weights):
    """_summary_

    Args:
        rap (_type_): _description_
        decp (_type_): _description_
        aper (_type_): _description_
        ras (_type_): _description_
        decs (_type_): _description_
        weights (_type_): _description_

    Returns:
        _type_: _description_
    """

    Nside = 16384
    nest = False
    hpix = radec2hpix(ras, decs, Nside, nest)
    Naper = np.zeros(len(rap))
    for i in range(0, len(rap)):
        Naper[i] = compute_flux_aper(rap[i], decp[i], hpix, weights, aper, Nside, nest)
    return Naper


def gmag_HB(isochrone_masks, dslice):
    """_summary_

    Args:
        isochrone_masks (bool): _description_
        dslice (_type_): _description_

    Returns:
        _type_: _description_
    """
    model_file = isochrone_masks["model_file"]
    gr, g0 = np.loadtxt(model_file, usecols=(0, 1), unpack=True)
    g0_HB_min = np.amin(g0[(gr < 0.05) & (gr > -0.05)])
    g0_HB_max = np.amax(g0[(gr < 0.05) & (gr > -0.05)])
    g0_HB = (g0_HB_min + g0_HB_max) / 2.0
    g_HB = g0_HB + 5.0 * np.log10(dslice / 10.0)
    return g_HB


def Dist_err(isochrone_masks, dslice):
    """_summary_

    Args:
        isochrone_masks (bool): _description_
        dslice (_type_): _description_

    Returns:
        _type_: _description_
    """
    # read model
    model_file = isochrone_masks["model_file"]
    gr, g0 = np.loadtxt(model_file, usecols=(0, 1), unpack=True)
    # read error files
    # gerr_file, rerr_file = isochrone_masks["magerr_g_file"], isochrone_masks["magerr_r_file"]
    gerr_file = isochrone_masks["magerr_g_file"]
    gm, gm_err = np.loadtxt(gerr_file, usecols=(0, 1), unpack=True)

    g = g0 + 5.0 * np.log10(dslice / 10.0)
    gmin = np.amin(g[(gr < 0.05) & (gr > -0.05)])
    gmax = np.amax(g[(gr < 0.05) & (gr > -0.05)])
    gmin = gmin - np.interp(gmin, gm, gm_err)
    gmax = gmax + np.interp(gmax, gm, gm_err)
    g_HB = (gmin + gmax) / 2.0
    dg = gmax - g_HB
    Dmin = (dslice / 10.0) / 10 ** (dg / 10.0)
    Dmax = (dslice / 10.0) * 10 ** (dg / 10.0)
    delta_D = Dmax - Dmin
    return delta_D, Dmin, Dmax


def compute_dslices(isochrone_masks, dslices_specs, workdir):
    """_summary_

    Args:
        isochrone_masks (bool): _description_
        dslices_specs (_type_): _description_
        workdir (_type_): _description_
    """
    model_file = isochrone_masks["model_file"]
    gr, g0 = np.loadtxt(model_file, usecols=(0, 1), unpack=True)

    # read error files
    gerr_file, rerr_file = (
        isochrone_masks["magerr_g_file"],
        isochrone_masks["magerr_r_file"],
    )
    gm, gm_err = np.loadtxt(gerr_file, usecols=(0, 1), unpack=True)

    dstep = 10.0
    dslices = [dslices_specs["dmin"]]
    dist = dslices[0]
    nstep = 2 * int((dslices_specs["dmax"] - dslices_specs["dmin"]) / dstep) + 1
    g = g0 + 5.0 * np.log10(dist / 10.0)
    gmin, gmax = np.amin(g[(gr < 0.05) & (gr > -0.05)]), np.amax(
        g[(gr < 0.05) & (gr > -0.05)]
    )
    gmin, gmax = gmin - np.interp(gmin, gm, gm_err), gmax + np.interp(gmax, gm, gm_err)

    for j in range(0, nstep):
        dist += dstep
        g = g0 + 5.0 * np.log10(dist / 10.0)
        gmin1 = np.amin(g[(gr < 0.05) & (gr > -0.05)])
        gmax1 = np.amax(g[(gr < 0.05) & (gr > -0.05)])
        gmin1 = gmin1 - np.interp(gmin1, gm, gm_err)
        gmax1 = gmax1 + np.interp(gmax1, gm, gm_err)
        if gmin1 >= gmax:
            dslices.append(dist)
            gmin = gmin1
            gmax = gmax1
            if dist > dslices_specs["dmax"]:
                break

    dslices = np.array(dslices)
    dslices = dslices[(dslices < 125000.0) & (dslices > 80000.0)]
    data = np.zeros(
        (len(dslices)), dtype={"names": ("id", "dist_pc"), "formats": ("i8", "f8")}
    )
    data["id"] = np.arange(len(dslices))
    data["dist_pc"] = dslices
    t = Table(data)  # , names=names)
    t.write(os.path.join(workdir, dslices_specs["dslices_filename"]), overwrite=True)

    return


def gmag_weight(gmag, dslice, isochrone_masks, gawa_cfg):
    """_summary_

    Args:
        gmag (_type_): _description_
        dslice (_type_): _description_
        isochrone_masks (bool): _description_
        gawa_cfg (_type_): _description_

    Returns:
        _type_: _description_
    """
    # weight stars according to their gmag : brighter => w larger
    g_HB = gmag_HB(isochrone_masks, dslice)
    gmag_w1 = g_HB + gawa_cfg["dgmag_w1"]
    weight = 10 ** (-(gmag - gmag_w1) / gawa_cfg["dgmag_w1"])
    weight[weight >= gawa_cfg["wmax"]] = gawa_cfg["wmax"]
    weight[weight <= gawa_cfg["wmin"]] = gawa_cfg["wmin"]
    return weight


def gmag_weight_map(dslice, isochrone_masks, gawa_cfg):
    """_summary_

    Args:
        dslice (_type_): _description_
        isochrone_masks (bool): _description_
        gawa_cfg (_type_): _description_

    Returns:
        _type_: _description_
    """
    xmin, xmax = isochrone_masks["mask_color_min"], isochrone_masks["mask_color_max"]
    ymin, ymax = isochrone_masks["mask_mag_min"], isochrone_masks["mask_mag_max"]
    step = 0.05
    nx = int((xmax - xmin) / step)
    ny = int((ymax - ymin) / step)
    gmag = np.zeros(ny)
    for j in range(0, ny):
        gmag[j] = ymin + float(j) * step
    weight = gmag_weight(gmag, dslice, isochrone_masks, gawa_cfg)
    return np.tile(weight, (nx, 1))


def excess_cmd_flux(
    rap,
    decp,
    data_stars,
    starcat,
    isochrone_masks,
    aper,
    colmag_pix_tile,
    dslice,
    gawa_cfg,
):
    """_summary_

    Args:
        rap (_type_): _description_
        decp (_type_): _description_
        data_stars (_type_): _description_
        starcat (_type_): _description_
        isochrone_masks (bool): _description_
        aper (_type_): _description_
        colmag_pix_tile (_type_): _description_
        dslice (_type_): _description_
        gawa_cfg (_type_): _description_

    Returns:
        _type_: _description_
    """
    weight_map = gmag_weight_map(dslice, isochrone_masks, gawa_cfg)
    # colmag_pix_tile in /arcmin2
    # aper : radius in arcmin
    aper_area = np.pi * aper**2
    Naper, Naperw = np.zeros(len(rap)), np.zeros(len(rap))
    for i in range(0, len(rap)):
        data_star_aper = select_stars_in_disc(
            rap[i], decp[i], data_stars, starcat, aper / 60.0
        )
        gmag = data_star_aper[starcat["keys"]["key_mag_g"]]
        rmag = data_star_aper[starcat["keys"]["key_mag_r"]]
        color = gmag - rmag
        colmag_pix = pixelized_colmag(
            color, gmag, np.ones(len(rmag)), isochrone_masks, None
        )
        excess = colmag_pix - aper_area * colmag_pix_tile
        wexcess = excess * weight_map
        Naper[i] = np.sum(excess[excess > 0.0])
        Naperw[i] = np.sum(wexcess[wexcess > 0.0])
    return Naper, Naperw


def add_peaks_attributes(
    data_peaks,
    data_stars,
    starcat,
    bkg_arcmin2,
    dslice,
    isochrone_masks,
    tile,
    gawa_cfg,
):
    """_summary_

    Args:
        data_peaks (_type_): _description_
        data_stars (_type_): _description_
        starcat (_type_): _description_
        bkg_arcmin2 (_type_): _description_
        dslice (_type_): _description_
        isochrone_masks (bool): _description_
        tile (_type_): _description_
        gawa_cfg (_type_): _description_
    """

    clkeys = gawa_cfg["clkeys"]
    ras, decs = (
        data_stars[starcat["keys"]["key_ra"]],
        data_stars[starcat["keys"]["key_dec"]],
    )
    gmag = data_stars[starcat["keys"]["key_mag_g"]]
    # rmag = data_stars[starcat["keys"]["key_mag_r"]]
    # color = gmag - rmag
    weights = gmag_weight(gmag, dslice, isochrone_masks, gawa_cfg)

    Nraw_snr = compute_flux_aper_vec(
        data_peaks["ra"],
        data_peaks["dec"],
        gawa_cfg["radius_aper_arcmin"] / 60.0,
        ras,
        decs,
        np.ones(len(ras)),
    )

    Naper_weighted = compute_flux_aper_vec(
        data_peaks["ra"],
        data_peaks["dec"],
        gawa_cfg["radius_aper_arcmin"] / 60.0,
        ras,
        decs,
        weights,
    )
    Nbkg = int(np.pi * bkg_arcmin2 * gawa_cfg["radius_aper_arcmin"] ** 2) + 1.0
    Nsnr = Nraw_snr - Nbkg
    if Nbkg < 9.0:
        sig_N = 2.0 * Nbkg**0.5 - Nbkg / 3.0
    else:
        sig_N = Nbkg**0.5
    snr = np.maximum(Nsnr / sig_N, np.zeros(len(Nsnr)))

    # attempt to use cmd map to derive richnesses - similar results..
    # colmag_pix_tile = pixelized_colmag(color, gmag, np.ones(len(gmag)), isochrone_masks, None)/(tile['eff_area_framed_deg2']*3600.)
    # Ncmd, Ncmdw = np.zeros(len(data_peaks)), np.zeros(len(data_peaks))
    # snrcond = (snr>gawa_cfg['snr_min'])
    # Ncmd[snrcond], Ncmdw[snrcond] = excess_cmd_flux(data_peaks[snrcond]['ra'], data_peaks[snrcond]['dec'], \
    #                                data_stars, starcat, isochrone_masks,\
    #                                gawa_cfg['radius_aper_arcmin'], colmag_pix_tile, dslice, gawa_cfg)

    # add to Table
    data_peaks[clkeys["key_snr"]] = snr
    data_peaks[clkeys["key_Naper"]] = Nsnr
    data_peaks[clkeys["key_Napertot"]] = Nraw_snr
    data_peaks[clkeys["key_Napertot_weighted"]] = Naper_weighted
    # data_peaks['Ncmd'] = Ncmd
    data_peaks[clkeys["key_Nbkg"]] = (
        gawa_cfg["radius_aper_arcmin"] * bkg_arcmin2 * np.ones(len(Nsnr))
    )
    return


def key_cylinder(key_cyl, key_1, length, i0, i1, isl, nslices, type):
    """_summary_

    Args:
        key_cyl (_type_): _description_
        key_1 (_type_): _description_
        length (_type_): _description_
        i0 (_type_): _description_
        i1 (_type_): _description_
        isl (bool): _description_
        nslices (_type_): _description_
        type (_type_): _description_

    Returns:
        _type_: _description_
    """

    if type == "int":
        key_match = -np.ones(length).astype(int)
    else:
        key_match = -np.ones(length)
    key_match[i0] = key_1[i1]
    key_cyl[:, isl] = key_match

    # append peaks that were not matched
    key_new = key_1[~np.isin(np.arange(key_1.size), i1)]
    if type == "int":
        key_cyl_new = -np.ones((len(key_new), nslices)).astype(int)
    else:
        key_cyl_new = -np.ones((len(key_new), nslices))
    key_cyl_new[:, isl] = key_new
    key_cyl = np.concatenate((key_cyl, key_cyl_new))

    return key_cyl


def init_cylinders(keyrank, peak_ids, wazp_specs):
    """_summary_

    Args:
        keyrank (_type_): _description_
        peak_ids (_type_): _description_
        wazp_specs (_type_): _description_

    Returns:
        _type_: _description_
    """

    ncyl = keyrank.shape[0]
    nsl = keyrank.shape[1]

    isl_mode = np.zeros(ncyl).astype(int)
    ip = np.zeros(ncyl).astype(int)
    isl_min = np.zeros(ncyl).astype(int)
    isl_max = (nsl - 1) * np.ones(ncyl).astype(int)

    for i in range(0, ncyl):
        keyrank_i = keyrank[
            i,
        ]
        isl_mode[i] = np.argmax(keyrank_i)
        ip[i] = peak_ids[i, isl_mode[i]]
        if isl_mode[i] < nsl - 1:
            for isl in range(isl_mode[i] + 1, nsl):
                if keyrank_i[isl] < 0.0:
                    isl_max[i] = isl - 1
                    break
        else:
            isl_max[i] = isl_mode[i]

        if isl_mode[i] > 0:
            for isl in range(isl_mode[i] - 1, -1, -1):
                if keyrank_i[isl] < 0.0:
                    isl_min[i] = isl + 1
                    break
        else:
            isl_min[i] = isl_mode[i]

    icyl = np.arange(ncyl)

    data_cylinders = np.zeros(
        (ncyl),
        dtype={
            "names": (
                "id_cyl",
                "peak_id",
                "cyl_length",
                "cyl_isl_min",
                "cyl_isl_max",
                "cyl_isl_mode",
            ),
            "formats": ("i4", "i4", "i4", "i4", "i4", "i4"),
        },
    )
    data_cylinders["id_cyl"] = icyl
    data_cylinders["peak_id"] = ip
    data_cylinders["cyl_length"] = isl_max - isl_min + 1
    data_cylinders["cyl_isl_min"] = isl_min
    data_cylinders["cyl_isl_max"] = isl_max
    data_cylinders["cyl_isl_mode"] = isl_mode

    return data_cylinders


def append_peaks_infos_to_cylinders(
    data_cylinders_init, peaks_list, dslices, ip_cyl, ra_cyl, dec_cyl, rank_cyl, snr_cyl
):
    """_summary_

    Args:
        data_cylinders_init (_type_): _description_
        peaks_list (_type_): _description_
        dslices (_type_): _description_
        ip_cyl (_type_): _description_
        ra_cyl (_type_): _description_
        dec_cyl (_type_): _description_
        rank_cyl (_type_): _description_
        snr_cyl (_type_): _description_

    Returns:
        _type_: _description_
    """
    dist_init = dslices[data_cylinders_init["cyl_isl_mode"]] / 1000.0
    nsl = len(dslices)
    ncl = len(data_cylinders_init)

    snr = np.zeros(ncl)
    ra = np.zeros(ncl)
    dec = np.zeros(ncl)

    for icl in range(0, ncl):
        isl = data_cylinders_init["cyl_isl_mode"][icl]
        ip = data_cylinders_init["peak_id"][icl]
        snr[icl] = peaks_list[isl]["snr"][ip]
        ra[icl] = peaks_list[isl]["ra"][ip]
        dec[icl] = peaks_list[isl]["dec"][ip]

    data_extra = np.zeros(
        (ncl),
        dtype={
            "names": (
                "ra",
                "dec",
                "dist_init_kpc",
                "snr",
                "ip_cyl",
                "ra_cyl",
                "dec_cyl",
                "rank_cyl",
                "snr_cyl",
            ),
            "formats": (
                "f8",
                "f8",
                "f8",
                "f8",
                str(nsl) + "i8",
                str(nsl) + "f8",
                str(nsl) + "f8",
                str(nsl) + "f8",
                str(nsl) + "f8",
            ),
        },
    )

    data_extra["dist_init_kpc"] = dist_init
    data_extra["snr"] = snr
    data_extra["ra"] = ra
    data_extra["dec"] = dec
    data_extra["ip_cyl"] = ip_cyl
    data_extra["ra_cyl"] = ra_cyl
    data_extra["dec_cyl"] = dec_cyl
    data_extra["rank_cyl"] = np.around(rank_cyl, 3)
    data_extra["snr_cyl"] = np.around(snr_cyl, 3)

    arrays = [data_cylinders_init, data_extra]
    data_cylinders = join_struct_arrays(arrays)

    return data_cylinders


def make_cylinders(peaks_list, dslices, gawa_cfg):
    """_summary_

    Args:
        peaks_list (_type_): _description_
        dslices (_type_): _description_
        gawa_cfg (_type_): _description_

    Returns:
        _type_: _description_
    """
    rad_deg = gawa_cfg["merge_specs"]["match_radius"] / 60.0
    flag_min = 0
    npeaks_all = 0
    nsl = len(peaks_list)

    key_snr = gawa_cfg["clkeys"]["key_snr"]
    key_rank = gawa_cfg["merge_specs"]["key_rank"]

    for isl in range(0, nsl):
        npeaks_all += len(peaks_list[isl])

    for isl in range(0, nsl):
        if flag_min == 0:
            if len(peaks_list[isl]) > 0:
                islmin = isl
                flag_min = 1

        if flag_min == 1:
            if len(peaks_list[isl]) == 0:  # no detection in intermediate slice
                continue

            if isl == islmin:  # all peaks are new cylinders
                dat = peaks_list[isl]
                np0 = len(dat)
                # initialize output
                ip_cyl = -np.ones((np0, nsl)).astype(int)
                ip_cyl[:, islmin] = np.arange(np0)
                rank_cyl, snr_cyl = np.ones((np0, nsl)) * (-1), np.ones((np0, nsl)) * (
                    -1
                )
                rank_cyl[:, islmin], snr_cyl[:, islmin] = dat[key_rank], dat[key_snr]
                ra_cyl, dec_cyl = np.ones((np0, nsl)) * (-1), np.ones((np0, nsl)) * (-1)
                ra_cyl[:, islmin], dec_cyl[:, islmin] = dat["ra"], dat["dec"]
            else:
                if len(peaks_list[isl - 1]) == 0:  # no detection in intermediate slice
                    dat = peaks_list[isl]
                    np0 = len(dat)
                    ip_cyl_new = -np.ones((np0, nsl)).astype(int)
                    ip_cyl_new[:, isl] = np.arange(np0)
                    rank_cyl_new, snr_cyl_new = np.ones((np0, nsl)) * (-1), np.ones(
                        (np0, nsl)
                    ) * (-1)
                    ra_cyl_new, dec_cyl_new = np.ones((np0, nsl)) * (-1), np.ones(
                        (np0, nsl)
                    ) * (-1)
                    rank_cyl_new[:, isl], snr_cyl_new[:, isl] = (
                        dat[key_rank],
                        dat[key_snr],
                    )
                    ra_cyl_new[:, isl], dec_cyl_new[:, isl] = dat["ra"], dat["dec"]

                    ip_cyl = np.concatenate((ip_cyl, ip_cyl_new))
                    rank_cyl, snr_cyl = np.concatenate(
                        (rank_cyl, rank_cyl_new)
                    ), np.concatenate((snr_cyl, snr_cyl_new))
                    ra_cyl, dec_cyl = np.concatenate(
                        (ra_cyl, ra_cyl_new)
                    ), np.concatenate((dec_cyl, dec_cyl_new))
                else:
                    rank_0, snr_0 = rank_cyl[:, isl - 1], snr_cyl[:, isl - 1]
                    ra_0, dec_0 = ra_cyl[:, isl - 1], dec_cyl[:, isl - 1]
                    np0 = len(ra_0)

                    dat = peaks_list[isl]  # open next slice
                    ra_1, dec_1 = dat["ra"], dat["dec"]
                    rank_1, snr_1 = dat[key_rank], dat[key_snr]
                    np1 = len(dat)
                    id_1 = np.arange(np1)

                    c0 = SkyCoord(ra=ra_0 * u.degree, dec=dec_0 * u.degree)
                    c1 = SkyCoord(ra=ra_1 * u.degree, dec=dec_1 * u.degree)
                    i0, i1, sep2d, dist3d = astropy.coordinates.search_around_sky(
                        c0, c1, rad_deg * u.degree, storekdtree="kdtree_sky"
                    )

                    ip_cyl = key_cylinder(
                        ip_cyl, id_1, np0, i0, i1, isl, nsl, type="int"
                    )
                    rank_cyl = key_cylinder(
                        rank_cyl, rank_1, np0, i0, i1, isl, nsl, type="float"
                    )
                    snr_cyl = key_cylinder(
                        snr_cyl, snr_1, np0, i0, i1, isl, nsl, type="float"
                    )
                    ra_cyl = key_cylinder(
                        ra_cyl, ra_1, np0, i0, i1, isl, nsl, type="float"
                    )
                    dec_cyl = key_cylinder(
                        dec_cyl, dec_1, np0, i0, i1, isl, nsl, type="float"
                    )

    if npeaks_all > 0:
        data_init = init_cylinders(rank_cyl, ip_cyl, gawa_cfg)
        data_cylinders = append_peaks_infos_to_cylinders(
            data_init, peaks_list, dslices, ip_cyl, ra_cyl, dec_cyl, rank_cyl, snr_cyl
        )
        ncyl = len(data_cylinders)
        print("")
        print("..........Number of cylinders : " + str(ncyl))
        print(
            "..........Ratio npeaks / ncyl : " + str(
                np.round(float(npeaks_all) / float(ncyl), 2)
            )
        )
    else:
        data_cylinders = None

    return data_cylinders


def gawa_tile_slice(
    tile,
    dslice,
    isochrone_masks,
    data_star,
    starcat,
    data_fp,
    footprint,
    gawa_cfg,
    out_paths,
    verbose,
):
    """_summary_

    Args:
        tile (_type_): _description_
        dslice (_type_): _description_
        isochrone_masks (bool): _description_
        data_star (_type_): _description_
        starcat (_type_): _description_
        data_fp (_type_): _description_
        footprint (_type_): _description_
        gawa_cfg (_type_): _description_
        out_paths (_type_): _description_
        verbose (_type_): _description_

    Returns:
        _type_: _description_
    """
    # select objects for computing density maps
    cmd_mask = np.load(
        os.path.join(
            out_paths["workdir"],
            out_paths["masks"],
            "cmd_mask_D" + str(dslice / 1000) + ".npy",
        )
    )
    data_star_slice = select_stars_in_slice(
        data_star, starcat, gawa_cfg, cmd_mask, isochrone_masks
    )
    ras, decs = (
        data_star_slice[starcat["keys"]["key_ra"]],
        data_star_slice[starcat["keys"]["key_dec"]],
    )
    gmag = data_star_slice[starcat["keys"]["key_mag_g"]]
    rmag = data_star_slice[starcat["keys"]["key_mag_r"]]
    cmd_histogram = os.path.join(
        out_paths["tile_dir"],
        out_paths["gawa"]["plots"],
        "cmd_hist_D" + str(dslice / 1000) + ".png",
    )
    pixelized_colmag(
        gmag - rmag, gmag, np.ones(len(gmag)), isochrone_masks, cmd_histogram
    )

    xycat = compute_catimage(ras, decs, np.ones(len(ras)), gawa_cfg, tile)
    bkg_arcmin2 = float(len(ras)) / (tile["framed_eff_area_deg2"] * 3600.0)
    xycat_fi = compute_filled_catimage(
        ras, decs, np.ones(len(ras)), gawa_cfg, tile, data_fp, footprint, bkg_arcmin2
    )

    # build density map /  extract peaks /compute attributes and filter
    xycat_fi_fitsname = os.path.join(
        out_paths["tile_dir"],
        out_paths["gawa"]["files"],
        "xycat_fi_D" + str(dslice / 1000) + ".fits",
    )
    wmap_fitsname = os.path.join(
        out_paths["tile_dir"],
        out_paths["gawa"]["files"],
        "wmap_D" + str(dslice / 1000) + ".fits",
    )
    if not os.path.isfile(wmap_fitsname):
        map2fits(xycat_fi, gawa_cfg, tile, xycat_fi_fitsname)
        run_mr_filter(xycat_fi_fitsname, wmap_fitsname, gawa_cfg)
    wmap_data = fits2map(wmap_fitsname)
    rap0, decp0, ip0, jp0 = wmap2peaks(wmap_fitsname, gawa_cfg, tile)
    rap, decp, ip, jp = filter_peaks(
        tile, gawa_cfg["map_resolution"], rap0, decp0, ip0, jp0
    )  # to keep inner tile peaks
    wradius_arcmin = wave_radius(wmap_data, ip.astype(int), jp.astype(int), gawa_cfg)

    coverfrac = coverfrac_disc(
        rap, decp, data_fp, footprint, gawa_cfg["radius_aper_arcmin"] / 60.0
    )
    pcond = coverfrac > 0.3
    data_peaks = init_peaks_table(
        rap[pcond],
        decp[pcond],
        ip[pcond],
        jp[pcond],
        coverfrac[pcond],
        wradius_arcmin[pcond],
        dslice,
        isochrone_masks,
        gawa_cfg,
    )

    # compute fluxes in given aper
    add_peaks_attributes(
        data_peaks,
        data_star_slice,
        starcat,
        bkg_arcmin2,
        dslice,
        isochrone_masks,
        tile,
        gawa_cfg,
    )

    # peaks to be kept
    condf = (data_peaks["snr"] >= gawa_cfg["snr_min"]) & (
        data_peaks["Naper"] >= gawa_cfg["N_min"]
    )

    # plot pixelized CMD diagrams bkg + stars in aper
    if verbose >= 2:
        for i in range(0, len(data_peaks[condf])):
            data_star_slice_aper = select_stars_in_disc(
                data_peaks[condf][i]["ra"],
                data_peaks[condf][i]["dec"],
                data_star_slice,
                starcat,
                gawa_cfg["radius_aper_arcmin"] / 60.0,
            )
            data_star_aper = select_stars_in_disc(
                data_peaks[condf][i]["ra"],
                data_peaks[condf][i]["dec"],
                data_star,
                starcat,
                gawa_cfg["radius_aper_arcmin"] / 60.0,
            )
            gmag_slaper, rmag_slaper = (
                data_star_slice_aper[starcat["keys"]["key_mag_g"]],
                data_star_slice_aper[starcat["keys"]["key_mag_r"]],
            )
            color_slaper = gmag_slaper - rmag_slaper
            gmag_aper, rmag_aper = (
                data_star_aper[starcat["keys"]["key_mag_g"]],
                data_star_aper[starcat["keys"]["key_mag_r"]],
            )
            color_aper = gmag_aper - rmag_aper
            cmd_histogram = os.path.join(
                out_paths["tile_dir"], out_paths["gawa"]["plots"], "cmd_hist_D" + str(dslice / 1000) + "_radec" + str(np.round(data_peaks[condf][i]["ra"], 3)) + "_" + str(np.round(data_peaks[condf][i]["dec"], 3)) + ".png",
            )
            plot_pixelized_colmag(
                color_aper,
                gmag_aper,
                np.ones(len(gmag_aper)),
                color_slaper,
                gmag_slaper,
                np.ones(len(gmag_slaper)),
                gmag - rmag,
                gmag,
                np.ones(len(gmag)),
                isochrone_masks,
                cmd_histogram,
            )

    if verbose >= 1:
        print("..............peaks filtering inner tile in / out ", len(rap0), len(rap))
        print(
            "..............peaks filtering coverfrac  in / out ",
            len(rap),
            len(rap[pcond]),
        )
        print(
            "..............peaks filtering SNR/N      in / out ",
            len(data_peaks),
            len(data_peaks[condf]),
        )
        print(
            "         distance / density/arcmin2 = ",
            np.round(dslice / 1000.0, 2),
            " / ",
            np.round(bkg_arcmin2, 3),
        )

    if verbose >= 2:
        xycat_fitsname = os.path.join(
            out_paths["tile_dir"],
            out_paths["gawa"]["files"],
            "xycat_D" + str(dslice / 1000) + ".fits",
        )
        map2fits(xycat, gawa_cfg, tile, xycat_fitsname)
        weights = gmag_weight(
            data_star_slice[starcat["keys"]["key_mag_g"]],
            dslice,
            isochrone_masks,
            gawa_cfg,
        )
        t = Table(data_star_slice)
        t["weights"] = weights
        t.write(
            os.path.join(
                out_paths["tile_dir"],
                out_paths["gawa"]["files"],
                "starcat_D" + str(dslice / 1000) + ".fits",
            ),
            overwrite=True,
        )
    return data_peaks[condf]


def gawa_tile(
    tile_specs,
    isochrone_masks,
    data_star,
    starcat,
    data_fp,
    footprint,
    gawa_cfg,
    admin,
    out_paths,
    verbose,
):
    """_summary_

    Args:
        tile_specs (_type_): _description_
        isochrone_masks (bool): _description_
        data_star (_type_): _description_
        starcat (_type_): _description_
        data_fp (_type_): _description_
        footprint (_type_): _description_
        gawa_cfg (_type_): _description_
        admin (_type_): _description_
        out_paths (_type_): _description_
        verbose (_type_): _description_
    """
    dslices = read_FitsCat(
        os.path.join(out_paths["workdir"], gawa_cfg["dslices"]["dslices_filename"])
    )["dist_pc"]

    print("..........Start gawa tile catalog construction")

    if not os.path.isfile(
        os.path.join(
            out_paths["tile_dir"], out_paths["gawa"]["results"], "clusters0.fits"
        )
    ):
        peaks_list = []
        for isl in range(0, len(dslices)):
            if not os.path.isfile(
                os.path.join(
                    out_paths["tile_dir"],
                    out_paths["gawa"]["files"],
                    "peaks_" + str(isl) + ".npy",
                )
            ):
                print(".............. Detection in slice ", isl)
                data_peaks = gawa_tile_slice(
                    tile_specs,
                    dslices[isl],
                    isochrone_masks,
                    data_star,
                    starcat,
                    data_fp,
                    footprint,
                    gawa_cfg,
                    out_paths,
                    verbose,
                )
                np.save(
                    os.path.join(
                        out_paths["tile_dir"],
                        out_paths["gawa"]["files"],
                        "peaks_" + str(isl) + ".npy",
                    ),
                    data_peaks,
                )
            else:
                print(".............. Use existing detections in slice ", isl)
                data_peaks = np.load(
                    os.path.join(
                        out_paths["tile_dir"],
                        out_paths["gawa"]["files"],
                        "peaks_" + str(isl) + ".npy",
                    )
                )
            peaks_list.append(data_peaks)

        print("..........Start cylinders")
        data_cylinders = make_cylinders(peaks_list, dslices, gawa_cfg)
        if verbose >= 1 and data_cylinders is not None:
            t = Table(data_cylinders)  # , names=names)
            t.write(
                os.path.join(
                    out_paths["tile_dir"],
                    out_paths["gawa"]["results"],
                    "cylinders.fits",
                ),
                overwrite=True,
            )

        if data_cylinders is not None:
            print("..........Start cylinders_2_clusters")
            data_clusters0 = cylinders2clusters(
                data_cylinders,
                peaks_list,
                tile_specs,
                dslices,
                out_paths,
                gawa_cfg["clkeys"],
            )
            # data_clusters = cylinders2clusters(data_cylinders)
            Nclusters = len(data_clusters0)
        else:
            Nclusters = 0

        if Nclusters > 0:
            t = Table(data_clusters0)  # , names=names)
            t.write(
                os.path.join(
                    out_paths["tile_dir"],
                    out_paths["gawa"]["results"],
                    "clusters0.fits",
                ),
                overwrite=True,
            )

    else:
        data_clusters0 = read_FitsCat(
            os.path.join(
                out_paths["tile_dir"], out_paths["gawa"]["results"], "clusters0.fits"
            )
        )
        Nclusters = len(data_clusters0)

    if Nclusters > 0:
        data_clusters = cl_duplicates_filtering(data_clusters0, gawa_cfg, "tile")
        t = Table(data_clusters0)
        t.write(
            os.path.join(
                out_paths["tile_dir"], out_paths["gawa"]["results"], "clusters.fits"
            ),
            overwrite=True,
        )

    # write final tile recap for final concatenation of clusters
    tile_info = np.zeros(
        1,
        dtype={
            "names": ("id", "eff_area_deg2", "Nclusters"),
            "formats": ("i8", "f8", "i8"),
        },
    )
    tile_info["id"] = tile_specs["id"]
    tile_info["eff_area_deg2"] = tile_specs["eff_area_deg2"]
    tile_info["Nclusters"] = Nclusters
    tile_info = Table(tile_info)
    tile_info.write(
        os.path.join(
            out_paths["tile_dir"], out_paths["gawa"]["results"], "tile_info.fits"
        ),
        overwrite=True,
    )

    return


def cylinders2clusters(data_cylinders, peaks_list, tile, dslices, out_paths, clkeys):
    """_summary_

    Args:
        data_cylinders (_type_): _description_
        peaks_list (_type_): _description_
        tile (_type_): _description_
        dslices (_type_): _description_
        out_paths (_type_): _description_
        clkeys (_type_): _description_

    Returns:
        _type_: _description_
    """
    i = 0
    for isl in range(0, len(dslices)):
        if len(peaks_list[isl]) > 0:
            data_peaks = peaks_list[isl]
            ip = data_cylinders["peak_id"][data_cylinders["cyl_isl_mode"] == isl]
            data_peaks_ext = Table(data_peaks[ip])
            data_peaks_ext["icyl"] = data_cylinders["id_cyl"][
                data_cylinders["cyl_isl_mode"] == isl
            ]
            data_peaks_ext["tile"] = tile["id"] * np.ones(len(ip)).astype(int)
            data_peaks_ext["slice"] = isl * np.ones(len(ip)).astype(int)

            if i == 0:  # initate cluster Table
                data_clusters = np.copy(data_peaks_ext)
                i = 1
            else:
                data_clusters = np.hstack((data_clusters, data_peaks_ext))

    id_in_tile = np.arange(len(data_clusters))
    snr = data_clusters[clkeys["key_snr"]]
    data_clusters = data_clusters[np.argsort(-snr)]
    t = Table(data_clusters)  # , names=names)
    t["id_in_tile"] = id_in_tile
    return t


def cl_duplicates_filtering(data_clusters_in, gawa_cfg, mode):
    """_summary_

    Args:
        data_clusters_in (_type_): _description_
        gawa_cfg (_type_): _description_
        mode (_type_): _description_

    Returns:
        _type_: _description_
    """
    # mode can be tile or survey
    # if mode = survey => search for duplicates coming from diff tiles

    clkeys = gawa_cfg["clkeys"]
    min_delta_distkpc = gawa_cfg["min_delta_distkpc"]
    min_dist_arcmin = gawa_cfg["min_dist_arcmin"]

    idecr = np.argsort(-data_clusters_in[clkeys["key_snr"]])
    data_cl = data_clusters_in[idecr]

    distpc = data_cl["dist_init_kpc"]
    ra, dec = data_cl[clkeys["key_ra"]], data_cl[clkeys["key_dec"]]
    clid = data_cl["id_in_tile"]
    tile_id = data_cl["tile"]
    flagdp = np.zeros(len(data_cl)).astype(int)

    Nside_tmp, nest_tmp = gawa_cfg["Nside_tmp"], gawa_cfg["nest_tmp"]
    clhpx = radec2hpix(ra, dec, Nside_tmp, nest_tmp)

    for i in range(0, len(data_cl)):
        if flagdp[i] == 0:
            radius_deg = min_dist_arcmin / 60.0
            if mode == "tile":
                cond = (abs(distpc[i] - distpc) <= min_delta_distkpc) & (
                    clid[i] != clid
                )
            if mode == "survey":
                cond = (abs(distpc[i] - distpc) <= min_delta_distkpc) & (
                    tile_id[i] != tile_id
                )
            in_cone = cond_in_disc(
                ra[cond],
                dec[cond],
                clhpx[cond],
                Nside_tmp,
                nest_tmp,
                ra[i],
                dec[i],
                radius_deg,
            )
            cond[np.argwhere(cond).T[0]] = in_cone
            flagdp[cond] = 1

    data_clusters_out = data_clusters_in[flagdp == 0]
    print("              Nr. of duplicates = " + str(len(data_cl) - len(data_clusters_out)) + " / " + str(len(data_cl)))
    print("              Final Nr of clusters : " + str(len(data_clusters_out)))

    return data_clusters_out
