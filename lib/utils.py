import numpy as np
import astropy.io.fits as fits
import os
import healpy as hp
from astropy.table import Table


def create_directory(dir):
    """_summary_

    Args:
        dir (_type_): _description_
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
    return


def read_FitsCat(cat):
    """_summary_

    Args:
        cat (_type_): _description_

    Returns:
        _type_: _description_
    """
    hdulist = fits.open(cat)
    dat = hdulist[1].data
    hdulist.close()
    return dat


def read_FitsFootprint(hpx_footprint, hpx_meta):
    """_summary_

    Args:
        hpx_footprint (_type_): _description_
        hpx_meta (_type_): _description_

    Returns:
        _type_: _description_
    """
    hdulist = fits.open(hpx_footprint)
    dat = hdulist[1].data
    hdulist.close()
    hpix_map = dat[hpx_meta["key_pixel"]].astype(int)
    if hpx_meta["key_frac"] == "none":
        frac_map = np.ones(len(hpix_map)).astype(float)
    else:
        frac_map = dat[hpx_meta["key_frac"]]
    return hpix_map, frac_map


def read_mosaicFitsCat_in_disc(galcat, tile, radius_deg):
    """From a list of galcat files, selects objects in a cone centered on racen, deccen
    Output is a structured array

    Args:
        galcat (_type_): _description_
        tile (_type_): _description_
        radius_deg (_type_): _description_

    Returns:
        _type_: _description_
    """

    # tile
    racen, deccen = tile["ra"], tile["dec"]
    # list of available galcats => healpix pixels
    gdir = galcat["mosaic"]["dir"]
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array([os.path.splitext(x)[0] for x in raw_list]).astype(int)
    extension = os.path.splitext(raw_list[0])[1]

    # find list of fits intersection cluster field
    Nside_fits, nest_fits = galcat["mosaic"]["Nside"], galcat["mosaic"]["nest"]
    fits_pixels_in_disc = hp.query_disc(
        nside=Nside_fits,
        nest=nest_fits,
        vec=hp.pixelfunc.ang2vec(np.radians(90.0 - deccen), np.radians(racen)),
        radius=np.radians(radius_deg),
        inclusive=True,
    )

    relevant_fits_pixels = fits_pixels_in_disc[np.isin(fits_pixels_in_disc, hpix_fits)]

    if len(relevant_fits_pixels) > 0:
        # merge intersecting fits
        for i in range(0, len(relevant_fits_pixels)):
            dat_disc = read_FitsCat(
                os.path.join(gdir, str(relevant_fits_pixels[i]) + extension)
            )
            dcen = np.degrees(
                dist_ang(
                    racen,
                    deccen,
                    dat_disc[galcat["keys"]["key_ra"]],
                    dat_disc[galcat["keys"]["key_dec"]],
                )
            )
            if i == 0:
                data_gal_disc = np.copy(dat_disc[dcen < radius_deg])
            else:
                data_gal_disc = np.append(data_gal_disc, dat_disc[dcen < radius_deg])
    else:
        data_gal_disc = np.zeros(0)
    return data_gal_disc


def read_mosaicFootprint_in_disc(footprint, tile, radius_deg):
    """From a list of galcat files, selects objects in a cone centered on racen, deccen
    Output is a structured array

    Args:
        footprint (_type_): _description_
        tile (_type_): _description_
        radius_deg (_type_): _description_

    Returns:
        _type_: _description_
    """

    # tile
    racen, deccen = tile["ra"], tile["dec"]
    # list of available galcats => healpix pixels
    gdir = footprint["mosaic"]["dir"]
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array(
        [os.path.splitext(x)[0].replace("_footprint", "") for x in raw_list]
    ).astype(int)
    extension = os.path.splitext(raw_list[0])[1]
    # find list of fits intersection cluster field
    Nside_fits, nest_fits = footprint["mosaic"]["Nside"], footprint["mosaic"]["nest"]
    fits_pixels_in_disc = hp.query_disc(
        nside=Nside_fits,
        nest=nest_fits,
        vec=hp.pixelfunc.ang2vec(np.radians(90.0 - deccen), np.radians(racen)),
        radius=np.radians(radius_deg),
        inclusive=True,
    )
    relevant_fits_pixels = fits_pixels_in_disc[np.isin(fits_pixels_in_disc, hpix_fits)]
    if len(relevant_fits_pixels) > 0:
        # merge intersecting fits
        for i in range(0, len(relevant_fits_pixels)):
            dat_disc = read_FitsCat(
                os.path.join(
                    gdir, str(relevant_fits_pixels[i]) + "_footprint" + extension
                )
            )
            ra, dec = hpix2radec(
                dat_disc[footprint["key_pixel"]], footprint["Nside"], footprint["nest"]
            )
            dcen = np.degrees(dist_ang(racen, deccen, ra, dec))
            if i == 0:
                data_fp_disc = np.copy(dat_disc[dcen < radius_deg])
            else:
                data_fp_disc = np.append(data_fp_disc, dat_disc[dcen < radius_deg])
    else:
        data_fp_disc = np.zeros(0)

    return data_fp_disc


def read_mosaicFitsCat_in_hpix(galcat, hpix_tile, Nside_tile, nest_tile):
    # list of available galcats => healpix pixels
    gdir = galcat["mosaic"]["dir"]
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array([os.path.splitext(x)[0] for x in raw_list]).astype(int)
    extension = os.path.splitext(raw_list[0])[1]
    Nside_fits, nest_fits = galcat["mosaic"]["Nside"], galcat["mosaic"]["nest"]

    # warning we assume Nside_tile > Nside_fits !!
    ra_fits, dec_fits = hpix2radec(hpix_fits, Nside_fits, nest_fits)
    hpix_fits_tile = radec2hpix(ra_fits, dec_fits, Nside_tile, nest_tile)

    relevant_fits_pixels = np.unique(hpix_fits[np.isin(hpix_fits_tile, hpix_tile)])

    if len(relevant_fits_pixels) > 0:
        # merge intersecting fits
        for i in range(0, len(relevant_fits_pixels)):
            dat = read_FitsCat(
                os.path.join(gdir, str(relevant_fits_pixels[i]) + extension)
            )
            if i == 0:
                data_gal_hpix = np.copy(dat)
            else:
                data_gal_hpix = np.append(data_gal_hpix, dat)
    else:
        data_gal_hpix = np.zeros(0)
    return data_gal_hpix


def read_mosaicFootprint_in_hpix(footprint, hpix_tile, Nside_tile, nest_tile):
    """_summary_

    Args:
        footprint (_type_): _description_
        hpix_tile (_type_): _description_
        Nside_tile (_type_): _description_
        nest_tile (_type_): _description_

    Returns:
        _type_: _description_
    """

    # list of available footprints => healpix pixels
    gdir = footprint["mosaic"]["dir"]
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array(
        [os.path.splitext(x)[0].replace("_footprint", "") for x in raw_list]
    ).astype(int)
    extension = os.path.splitext(raw_list[0])[1]
    Nside_fits, nest_fits = footprint["mosaic"]["Nside"], footprint["mosaic"]["nest"]

    # warning we assume Nside_tile > Nside_fits !!
    ra_fits, dec_fits = hpix2radec(hpix_fits, Nside_fits, nest_fits)
    hpix_fits_tile = radec2hpix(ra_fits, dec_fits, Nside_tile, nest_tile)

    relevant_fits_pixels = np.unique(hpix_fits[np.isin(hpix_fits_tile, hpix_tile)])

    if len(relevant_fits_pixels) > 0:
        # merge intersecting fits
        for i in range(0, len(relevant_fits_pixels)):
            dat = read_FitsCat(
                os.path.join(
                    gdir, str(relevant_fits_pixels[i]) + "_footprint" + extension
                )
            )
            if i == 0:
                data_fp_hpix = np.copy(dat)
            else:
                data_fp_hpix = np.append(data_fp_hpix, dat)
    else:
        data_fp_hpix = np.zeros(0)
    return data_fp_hpix


def create_survey_footprint_from_mosaic(footprint, survey_footprint):
    all_files = np.array(os.listdir(footprint["mosaic"]["dir"]))
    flist = [os.path.join(footprint["mosaic"]["dir"], f) for f in all_files]
    concatenate_fits(flist, survey_footprint)
    return


def create_mosaic_footprint(footprint, fpath):
    """_summary_

    Args:
        footprint (_type_): _description_
        fpath (_type_): _description_
    """
    # from a survey footprint create a mosaic of footprints at lower resol.
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    hpix0, frac0 = read_FitsFootprint(footprint["survey_footprint"], footprint)
    ra0, dec0 = hpix2radec(hpix0, footprint["Nside"], footprint["nest"])
    hpix = radec2hpix(
        ra0, dec0, footprint["mosaic"]["Nside"], footprint["mosaic"]["nest"]
    )
    for hpx in np.unique(hpix):
        all_cols = fits.ColDefs(
            [
                fits.Column(
                    name=footprint["key_pixel"],
                    format="K",
                    array=hpix0[np.isin(hpix, hpx)],
                ),
                fits.Column(name="ra", format="E", array=ra0[np.isin(hpix, hpx)]),
                fits.Column(name="dec", format="E", array=dec0[np.isin(hpix, hpx)]),
                fits.Column(
                    name=footprint["key_frac"],
                    format="K",
                    array=frac0[np.isin(hpix, hpx)],
                ),
            ]
        )
        hdu = fits.BinTableHDU.from_columns(all_cols)
        hdu.writeto(os.path.join(fpath, str(hpx) + "_footprint.fits"), overwrite=True)
    return


def concatenate_fits(flist, output):
    """_summary_

    Args:
        flist (_type_): _description_
        output (_type_): _description_

    Returns:
        _type_: _description_
    """

    for i in range(0, len(flist)):
        dat = read_FitsCat(flist[i])
        if i == 0:
            cdat = np.copy(dat)
        else:
            cdat = np.append(cdat, dat)

    t = Table(cdat)  # , names=names)
    t.write(output, overwrite=True)
    return cdat


def concatenate_fits_with_label(flist, label_name, label, output):
    """_summary_

    Args:
        flist (_type_): _description_
        label_name (_type_): _description_
        label (_type_): _description_
        output (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in range(0, len(flist)):
        dat = read_FitsCat(flist[i])
        datL = Table(dat)
        datL[label_name] = int(label[i]) * np.ones(len(dat)).astype(int)
        if i == 0:
            cdat = np.copy(datL)
        else:
            cdat = np.append(cdat, datL)

    t = Table(cdat)  # , names=names)
    t.write(output, overwrite=True)
    return cdat


def add_key_to_fits(fitsfile, key_val, key_name, key_type):
    """_summary_

    Args:
        fitsfile (_type_): _description_
        key_val (_type_): _description_
        key_name (_type_): _description_
        key_type (_type_): _description_
    """
    hdulist = fits.open(fitsfile)
    dat = hdulist[1].data
    hdulist.close()

    orig_cols = dat.columns

    if key_type == "float":
        new_col = fits.ColDefs([fits.Column(name=key_name, format="E", array=key_val)])
    if key_type == "int":
        new_col = fits.ColDefs([fits.Column(name=key_name, format="J", array=key_val)])

    hdu = fits.BinTableHDU.from_columns(orig_cols + new_col)
    hdu.writeto(fitsfile, overwrite=True)
    return


def add_hpx_to_cat(data_gal, ra, dec, Nside_tmp, nest_tmp, keyname):
    """_summary_

    Args:
        data_gal (_type_): _description_
        ra (_type_): _description_
        dec (_type_): _description_
        Nside_tmp (_type_): _description_
        nest_tmp (_type_): _description_
        keyname (_type_): _description_

    Returns:
        _type_: _description_
    """
    ghpx = radec2hpix(ra, dec, Nside_tmp, nest_tmp)
    t = Table(data_gal)
    t[keyname] = ghpx
    return t


def mad(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    xmad = 1.4826 * np.median(abs(x))
    return xmad


def gaussian(x, mu, sig):
    """_summary_

    Args:
        x (_type_): _description_
        mu (_type_): _description_
        sig (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.exp(-((x - mu) ** 2) / (2.0 * sig**2)) / (sig * np.sqrt(2.0 * np.pi))


def dist_ang(ra1, dec1, ra_ref, dec_ref):
    """_summary_

    Args:
        ra1 (_type_): _description_
        dec1 (_type_): _description_
        ra_ref (_type_): _description_
        dec_ref (_type_): _description_

    Returns:
        _type_: _description_
    """

    # angular distance between (ra1, dec1) and (ra_ref, dec_ref)
    # ra-dec in degrees
    # ra1-dec1 can be arrays
    # ra_ref-dec_ref are scalars
    # output is in radian

    costheta = np.sin(np.radians(dec_ref)) * np.sin(np.radians(dec1)) + np.cos(
        np.radians(dec_ref)
    ) * np.cos(np.radians(dec1)) * np.cos(np.radians(ra1 - ra_ref))
    dist_ang = np.arccos(costheta)

    return dist_ang  # radian


def area_ann_deg2(theta_1, theta_2):
    """_summary_

    Args:
        theta_1 (_type_): _description_
        theta_2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (
        2.0 * np.pi * (np.cos(np.radians(theta_1)) - np.cos(np.radians(theta_2))) * (180.0 / np.pi) ** 2
    )


def _mstar_(mstar_filename, zin):
    """From a given (z, mstar) ascii file
    interpolate to provide the mstar at a given z_in
    """

    zst, mst = np.loadtxt(mstar_filename, usecols=(0, 1), unpack=True)
    return np.interp(zin, zst, mst)


def join_struct_arrays(arrays):
    """_summary_

    Args:
        arrays (_type_): _description_

    Returns:
        _type_: _description_
    """
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = np.empty((n, offsets[-1]), dtype=np.uint8)
    for a, size, offset in zip(arrays, sizes, offsets):
        # print ('a ',a)
        # print (a.view(np.uint8))#.reshape(n,size))
        joint[:, offset: offset + size] = a.view(np.uint8).reshape(n, size)
        # print ('desc ', a.dtype.descr)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)


def radec_window_area(ramin, ramax, decmin, decmax):
    """_summary_

    Args:
        ramin (_type_): _description_
        ramax (_type_): _description_
        decmin (_type_): _description_
        decmax (_type_): _description_

    Returns:
        _type_: _description_
    """
    nstep = int((decmax - decmin) / 0.1) + 1
    step = (decmax - decmin) / float(nstep)
    decmini = np.arange(decmin, decmax, step)
    decmaxi = decmini + step
    decceni = (decmini + decmaxi) / 2.0
    darea = (ramax - ramin) * np.cos(np.pi * decceni / 180.0) * (decmaxi - decmini)
    return np.sum(darea)


# healpix functions
def radec2phitheta(ra, dec):
    """
    Turn (ra,dec) [deg] in (phi, theta) [rad] used by healpix
    """
    phi, theta = np.radians(ra), np.radians(90.0 - dec)
    return phi, theta


def phitheta2radec(phi, theta):
    """_summary_

    Args:
        phi (_type_): _description_
        theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.degrees(phi), 90.0 - np.degrees(theta)


def radec2hpix(ra, dec, Nside, nest):
    """
    From a list of ra-dec's (deg) compute the list of associated healpix index
    """
    phi, theta = radec2phitheta(ra, dec)  # np.radians(ra), np.radians(90.-dec)
    return hp.ang2pix(Nside, theta, phi, nest)


def hpix2radec(hpix, nside, nest):
    """_summary_

    Args:
        hpix (_type_): _description_
        nside (_type_): _description_
        nest (_type_): _description_

    Returns:
        _type_: _description_
    """
    theta, phi = hp.pix2ang(nside, hpix, nest)
    # return ra, dec :
    return phitheta2radec(phi, theta)


def sub_hpix(hpix, Nside, nest):
    """_summary_

    Args:
        hpix (_type_): _description_
        Nside (_type_): _description_
        nest (_type_): _description_

    Returns:
        _type_: _description_
    """
    # from a list of pixels at resolution Nside
    # get the corresponding list at resolution Nside*2
    rac, decc = np.zeros(4 * len(hpix)), np.zeros(4 * len(hpix))
    i = 0
    for p in hpix:
        theta, phi = hp.vec2ang(hp.boundaries(Nside, p, 1, nest).T)
        ra, dec = phitheta2radec(phi, theta)
        racen, deccen = hpix2radec(p, Nside, nest)
        for j in range(0, 4):
            rac[i], decc[i] = (ra[j] + racen) / 2.0, (dec[j] + deccen) / 2.0
            i += 1

    return radec2hpix(rac, decc, Nside * 2, nest)


def makeHealpixMap(ra, dec, weights=None, nside=1024, nest=False):
    """_summary_

    Args:
        ra (_type_): _description_
        dec (_type_): _description_
        weights (_type_, optional): _description_. Defaults to None.
        nside (int, optional): _description_. Defaults to 1024.
        nest (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # convert a ra/dec catalog into healpix map with counts pe_r cell
    ipix = hp.ang2pix(nside, (90 - dec) / 180 * np.pi, ra / 180 * np.pi, nest=nest)
    return np.bincount(ipix, weights=weights, minlength=hp.nside2npix(nside))


def all_hpx_in_annulus(ra, dec, radius_in_deg, radius_out_deg, hpx_meta, inclusive):
    """
    Get the list of all healpix pixels falling in an annulus around ra-dec (deg)
    the radii that define the annulus are in degrees
    pixels are inclusive on radius_out but not radius_in
    """
    Nside, nest = hpx_meta["Nside"], hpx_meta["nest"]
    pixels_in_disc = hp.query_disc(
        nside=Nside,
        nest=nest,
        vec=hp.pixelfunc.ang2vec(np.radians(90.0 - dec), np.radians(ra)),
        radius=np.radians(radius_out_deg),
        inclusive=inclusive,
    )
    if radius_in_deg > 0.0:
        pixels_in_disc_in = hp.query_disc(
            nside=Nside,
            nest=nest,
            vec=hp.pixelfunc.ang2vec(np.radians(90.0 - dec), np.radians(ra)),
            radius=np.radians(radius_in_deg),
            inclusive=inclusive,
        )

        id_annulus = np.isin(pixels_in_disc, pixels_in_disc_in, invert=True)
        pixels_in_ann = pixels_in_disc[id_annulus]
    else:
        pixels_in_ann = np.copy(pixels_in_disc)

    return pixels_in_ann


def hpx_in_annulus(
    ra, dec, radius_in_deg, radius_out_deg, data_fp, hpx_meta, inclusive
):
    """
    Given an array of healpix pixels (hpix, frac) where frac is the covered fraction of each hpix pixel,
    computes the sub list of these pixels falling in an annulus around position ra-dec (deg)
    the radii that define the annulus are in degrees
    hpx pixels are inclusive on radius_out but not radius_in
    """
    # Nside, nest = hpx_meta["Nside"], hpx_meta["nest"]
    Nside = hpx_meta["Nside"]
    hpix, frac = data_fp[hpx_meta["key_pixel"]], data_fp[hpx_meta["key_frac"]]

    area_pix = hp.pixelfunc.nside2pixarea(Nside, degrees=True)
    pixels_in_ann = all_hpx_in_annulus(
        ra, dec, radius_in_deg, radius_out_deg, hpx_meta, inclusive
    )
    npix_all = len(pixels_in_ann)
    area_deg2 = 0.0
    coverfrac = 0.0
    hpx_in_ann, frac_in_ann = [], []

    if npix_all > 0:
        idx = np.isin(hpix, pixels_in_ann)
        hpx_in_ann = hpix[idx]  # visible pixels
        frac_in_ann = frac[idx]
        npix = len(hpx_in_ann)
        if npix > 0:
            area_deg2 = np.sum(frac_in_ann) * area_pix
            coverfrac = np.sum(frac_in_ann) / float(npix_all)
    # print ('nnn ', npix_all, np.sum(frac_in_ann), Nside, radius_out_deg)

    return hpx_in_ann, frac_in_ann, area_deg2, coverfrac


# FCT to split surveys
def survey_ra_minmax(ra):
    """_summary_

    Args:
        ra (_type_): _description_

    Returns:
        _type_: _description_
    """
    ramin, ramax = np.amin(ra), np.amax(ra)
    if ramin < 0.5 and ramax > 359.5:
        nbins = 360
        hist, bin_edges = np.histogram(ra, bins=nbins, range=(0.0, 360))
        ramin_empty = bin_edges[np.amin(np.argwhere(hist == 0))]
        ramax_empty = bin_edges[np.amax(np.argwhere(hist == 0))]

        ra1 = ra[(ra < ramin_empty + 1.0)]
        ra2 = ra[(ra > ramax_empty - 1.0)] - 360.0
        ra_new = np.hstack((ra1, ra2))
        ramin, ramax = np.amin(ra_new), np.amax(ra_new)
    return ramin, ramax


def split_survey_from_hpx(
    hpx_footprint, hpx_meta, eff_tile_width, tile_overlap, output
):
    """_summary_

    Args:
        hpx_footprint (_type_): _description_
        hpx_meta (_type_): _description_
        eff_tile_width (_type_): _description_
        tile_overlap (_type_): _description_
        output (_type_): _description_

    Returns:
        _type_: _description_
    """
    # starting from a given survey footprint split survey in roughly equal
    # squared area tiles

    hdulist = fits.open(hpx_footprint)
    dat = hdulist[1].data
    hdulist.close()
    hpix = dat[hpx_meta["key_pixel"]].astype(int)
    Nside, nest = hpx_meta["Nside"], hpx_meta["nest"]
    ra, dec = hpix2radec(hpix, Nside, nest)

    size_pix_deg = hp.pixelfunc.nside2resol(Nside, arcmin=True) / 60.0
    ra_1, ra_2 = ra - size_pix_deg / np.cos(
        np.radians(dec)
    ), ra + size_pix_deg / np.cos(np.radians(dec))
    dec_1, dec_2 = dec - size_pix_deg, dec + size_pix_deg

    ra_all, dec_all = np.hstack((ra_1, ra_2)), np.hstack((dec_1, dec_2))
    decmin, decmax = np.amin(dec), np.amax(dec)
    ramin, ramax = survey_ra_minmax(ra)

    # tiling along dec
    dec_i = np.arange(decmin, decmax, eff_tile_width)
    ny0 = len(dec_i)

    if decmax - decmin < eff_tile_width:
        ystep = decmax - decmin
        ny = ny0
    else:
        if decmax - dec_i[ny0 - 1] >= eff_tile_width / 2.0:
            dec_i = np.linspace(decmin, decmax, ny0 + 1)
            ystep = dec_i[1] - dec_i[0]
            dec_i = dec_i[0:ny0]
            ny = ny0
        else:
            dec_i = np.linspace(decmin, decmax, ny0)
            ystep = dec_i[1] - dec_i[0]
            dec_i = dec_i[0: ny0 - 1]
            ny = ny0 - 1

    decmin_i, decmax_i = dec_i, dec_i + ystep
    deccen_i = (decmin_i + decmax_i) / 2.0
    cdec_frame0 = np.maximum(
        np.cos(decmin_i * np.pi / 180.0), np.cos(decmax_i * np.pi / 180.0)
    )
    cdec0 = np.cos(deccen_i * np.pi / 180.0)

    # init
    nx = np.zeros(ny).astype(int)
    racen, deccen, cdec, cdec_frame = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    ramin_tile, ramax_tile, decmin_tile, decmax_tile = (
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
    )
    ra_id, dec_id = np.zeros(1), np.zeros(1)
    size_ra, size_dec = np.zeros(1), np.zeros(1)
    ramin_sl, ramax_sl = np.zeros(ny), np.zeros(ny)

    k = 0
    for i in range(0, ny):
        # start tiling along RA
        ramin_sl[i], ramax_sl[i] = survey_ra_minmax(
            ra[(dec < decmax_i[i]) & (dec > decmin_i[i])]
        )
        ra_j = np.arange(ramin_sl[i], ramax_sl[i], eff_tile_width / cdec0[i])
        nx0 = len(ra_j)
        if ramax_sl[i] - ramin_sl[i] < eff_tile_width:
            xstep = ramax_sl[i] - ramin_sl[i]
            nx[i] = nx0
        else:
            if ramax_sl[i] - ra_j[nx0 - 1] >= eff_tile_width / 2.0:
                ra_j = np.linspace(ramin_sl[i], ramax_sl[i], nx0 + 1)
                xstep = ra_j[1] - ra_j[0]
                ra_j = ra_j[0:nx0]
                nx[i] = nx0
            else:
                ra_j = np.linspace(ramin_sl[i], ramax_sl[i], nx0)
                xstep = ra_j[1] - ra_j[0]
                ra_j = ra_j[0: nx0 - 1]
                nx[i] = nx0 - 1

        ramin_i, ramax_i = ra_j, ra_j + xstep
        racen_i = (ramin_i + ramax_i) / 2.0

        for j in range(0, nx[i]):
            if k == 0:
                racen[k], deccen[k] = racen_i[j], deccen_i[i]
                cdec[k], cdec_frame[k] = cdec0[i], cdec_frame0[i]
                ramin_tile[k], ramax_tile[k] = ramin_i[j], ramax_i[j]
                decmin_tile[k], decmax_tile[k] = decmin_i[i], decmax_i[i]
                ra_id[k], dec_id[k] = j, i
                size_ra[k], size_dec[k] = xstep * cdec0[i], ystep
            else:
                racen = np.vstack((racen, np.array(racen_i[j])))
                deccen = np.vstack((deccen, np.array(deccen_i[i])))
                cdec = np.vstack((cdec, np.array(cdec0[i])))
                cdec_frame = np.vstack((cdec_frame, np.array(cdec_frame0[i])))
                ramin_tile = np.vstack((ramin_tile, np.array(ramin_i[j])))
                ramax_tile = np.vstack((ramax_tile, np.array(ramax_i[j])))
                decmin_tile = np.vstack((decmin_tile, np.array(decmin_i[i])))
                decmax_tile = np.vstack((decmax_tile, np.array(decmax_i[i])))
                ra_id = np.vstack((ra_id, np.array(j)))
                dec_id = np.vstack((dec_id, np.array(i)))
                size_ra = np.vstack((size_ra, np.array(xstep * cdec0[i])))
                size_dec = np.vstack((size_dec, np.array(ystep)))
            k += 1

    nxmax = np.amax(nx)
    ntiles = np.sum(nx)
    tile_id = np.arange(0, ntiles, 1).astype(int)
    size_frame_ra = (size_ra + 2.0 * tile_overlap) * cdec_frame / cdec
    size_frame_dec = size_dec + 2.0 * tile_overlap
    size_frame_i = np.maximum(size_frame_ra, size_frame_dec)
    size_frame = np.amax(size_frame_i)

    ramin_frame, ramax_frame = racen - size_frame / (2.0 * cdec), racen + size_frame / (
        2.0 * cdec
    )
    decmin_frame, decmax_frame = deccen - size_frame / 2.0, deccen + size_frame / 2.0
    size_frame_arr = np.ones(ntiles) * size_frame

    # TBD merge tiles with too small ngals

    all_cols = fits.ColDefs(
        [
            fits.Column(name="tile_id", format="J", array=tile_id),
            fits.Column(name="dec_id", format="J", array=dec_id),
            fits.Column(name="ra_id", format="J", array=ra_id),
            fits.Column(name="racen", format="E", array=np.around(racen, 4)),
            fits.Column(name="deccen", format="E", array=np.around(deccen, 4)),
            fits.Column(name="ramin_tile", format="E", array=np.around(ramin_tile, 4)),
            fits.Column(name="ramax_tile", format="E", array=np.around(ramax_tile, 4)),
            fits.Column(
                name="decmin_tile", format="E", array=np.around(decmin_tile, 4)
            ),
            fits.Column(
                name="decmax_tile", format="E", array=np.around(decmax_tile, 4)
            ),
            fits.Column(
                name="ramin_frame", format="E", array=np.around(ramin_frame, 4)
            ),
            fits.Column(
                name="ramax_frame", format="E", array=np.around(ramax_frame, 4)
            ),
            fits.Column(
                name="decmin_frame", format="E", array=np.around(decmin_frame, 4)
            ),
            fits.Column(
                name="decmax_frame", format="E", array=np.around(decmax_frame, 4)
            ),
            fits.Column(name="tile_size_ra", format="E", array=np.around(size_ra, 4)),
            fits.Column(name="tile_size_dec", format="E", array=np.around(size_dec, 4)),
            fits.Column(
                name="frame_size", format="E", array=np.around(size_frame_arr, 4)
            ),
        ]
    )

    hdu = fits.BinTableHDU.from_columns(all_cols)
    hdu.writeto(output, overwrite=True)

    return ntiles


def filter_tile(dat, galcat_keys_dict, tiles_specs):
    """_summary_

    Args:
        dat (_type_): _description_
        galcat_keys_dict (_type_): _description_
        tiles_specs (_type_): _description_

    Returns:
        _type_: _description_
    """
    # read tiles
    ramin_frame, ramax_frame = tiles_specs["ramin_frame"], tiles_specs["ramax_frame"]
    decmin_frame, decmax_frame = (
        tiles_specs["decmin_frame"],
        tiles_specs["decmax_frame"],
    )
    tile_id = tiles_specs["tile_id"]

    ra, dec = dat[galcat_keys_dict["key_ra"]], dat[galcat_keys_dict["key_dec"]]

    if ramin_frame < 0.0:
        ra2 = ra - 360.0
        dat_tile1 = dat[
            (ra <= ramax_frame) & (ra >= ramin_frame) & (dec <= decmax_frame) & (dec >= decmin_frame)
        ]
        dat_tile2 = dat[
            (ra2 <= ramax_frame) & (ra2 >= ramin_frame) & (dec <= decmax_frame) & (dec >= decmin_frame)
        ]
        dat_tile = np.append(dat_tile1, dat_tile2)
    else:
        dat_tile = dat[
            (ra <= ramax_frame) & (ra >= ramin_frame) & (dec <= decmax_frame) & (dec >= decmin_frame)
        ]

    return dat_tile


def hpx_degrade(pix_in, nside_in, nest_in, nside_out, nest_out):
    """_summary_

    Args:
        pix_in (_type_): _description_
        nside_in (_type_): _description_
        nest_in (_type_): _description_
        nside_out (_type_): _description_
        nest_out (_type_): _description_

    Returns:
        _type_: _description_
    """
    ra, dec = hpix2radec(pix_in, nside_in, nest_in)
    pix_out0 = radec2hpix(ra, dec, nside_out, nest_out)
    pix_out, counts = np.unique(pix_out0, return_counts=True)
    nsamp = (float(nside_in) / float(nside_out)) ** 2
    return pix_out, counts.astype(float) / nsamp


def hpx_split_survey(footprint_file, footprint, admin, output):
    """_summary_

    Args:
        footprint_file (_type_): _description_
        footprint (_type_): _description_
        admin (_type_): _description_
        output (_type_): _description_

    Returns:
        _type_: _description_
    """

    Nside_fp, nest_fp = footprint["Nside"], footprint["nest"]
    Nside_tile, nest_tile = admin["Nside"], admin["nest"]

    dat = read_FitsCat(footprint_file)
    hpix_map = dat[footprint["key_pixel"]]
    frac_map = dat[footprint["key_frac"]]

    """
    # inner tile
    hmap = np.arange(hp.nside2npix(Nside_fp))
    pixel0 = np.zeros(len(hmap))
    pixel0[hpix_map]=1
    frac0 = hp.pixelfunc.ud_grade(pixel0, Nside_tile)
    hpix_tile = np.argwhere(frac0>0).T[0]
    frac_tile = frac0[(frac0>0)]
    """
    hpix_tile, frac_tile = hpx_degrade(
        hpix_map, Nside_fp, nest_fp, Nside_tile, nest_tile
    )

    racen, deccen = hpix2radec(hpix_tile, Nside_tile, nest_tile)
    area_tile = hp.pixelfunc.nside2pixarea(Nside_tile, degrees=True)

    # tiles
    radius_deg = tile_radius(admin)
    framed_eff_area_deg2 = np.zeros(len(racen))
    for i in range(0, len(racen)):
        pixels_in_disc = hp.query_disc(
            nside=Nside_fp,
            nest=nest_fp,
            vec=hp.pixelfunc.ang2vec(
                np.radians(90.0 - deccen[i]), np.radians(racen[i])
            ),
            radius=np.radians(radius_deg),
            inclusive=False,
        )
        framed_eff_area_deg2[i] = np.sum(
            frac_map[np.isin(hpix_map, pixels_in_disc)]
        ) * hp.pixelfunc.nside2pixarea(Nside_fp, degrees=True)

    data_tiles = np.zeros(
        (len(hpix_tile)),
        dtype={
            "names": (
                "id",
                "hpix",
                "ra",
                "dec",
                "area_deg2",
                "eff_area_deg2",
                "framed_eff_area_deg2",
                "radius_tile_deg",
                "Nside",
                "nest",
            ),
            "formats": ("i8", "i8", "f8", "f8", "f8", "f8", "f8", "f8", "i8", "b"),
        },
    )
    data_tiles["id"] = np.arange(len(hpix_tile))
    data_tiles["hpix"] = hpix_tile
    data_tiles["ra"], data_tiles["dec"] = np.around(racen, 4), np.around(deccen, 4)
    data_tiles["area_deg2"] = np.around(area_tile, 4)
    data_tiles["eff_area_deg2"] = np.around(frac_tile * area_tile, 4)
    data_tiles["framed_eff_area_deg2"] = np.around(framed_eff_area_deg2, 4)
    data_tiles["radius_tile_deg"] = np.ones(len(hpix_tile)) * tile_radius(admin)
    data_tiles["Nside"] = Nside_tile * np.ones(len(hpix_tile)).astype(int)
    data_tiles["nest"] = len(hpix_tile) * [nest_tile]

    t = Table(data_tiles)
    t.write(output, overwrite=True)
    print(
        ".....tile area (deg2) = ",
        np.round(hp.pixelfunc.nside2pixarea(Nside_tile, degrees=True), 2),
    )
    print(
        ".....effective survey area (deg2) = ",
        np.around(np.sum(frac_tile) * area_tile, 4),
    )

    return len(data_tiles)


def tile_radius(tiling):
    """_summary_

    Args:
        tiling (_type_): _description_

    Returns:
        _type_: _description_
    """
    Nside_tile = tiling["Nside"]
    frame_deg = tiling["overlap_deg"]
    tile_radius = (
        2.0 * hp.pixelfunc.nside2pixarea(Nside_tile, degrees=True)
    ) ** 0.5 / 2.0
    return tile_radius + frame_deg


def create_tile_specs(tile, tile_radius_deg, admin):
    """_summary_

    Args:
        tile (_type_): _description_
        tile_radius_deg (_type_): _description_
        admin (_type_): _description_

    Returns:
        _type_: _description_
    """
    if admin["target_mode"]:
        hpix = -1
        Nside, nest = -1, None
        area_deg2 = -1.0
        eff_area_deg2 = -1.0
        framed_eff_area_deg2 = -1.0
    else:
        hpix = tile["hpix"]
        Nside, nest = admin["tiling"]["Nside"], admin["tiling"]["nest"]
        area_deg2 = tile["area_deg2"]
        eff_area_deg2 = tile["eff_area_deg2"]
        framed_eff_area_deg2 = tile["framed_eff_area_deg2"]

    tile_specs = {
        "id": tile["id"],
        "ra": tile["ra"],
        "dec": tile["dec"],
        "hpix": hpix,
        "Nside": Nside,
        "nest": nest,
        "area_deg2": area_deg2,
        "eff_area_deg2": eff_area_deg2,
        "framed_eff_area_deg2": framed_eff_area_deg2,
        "radius_tile_deg": tile_radius_deg,
        "radius_filter_deg": -1.0,  # active only if target_mode=True
        "target_mode": admin["target_mode"],
    }
    return tile_specs


def cond_in_disc(rag, decg, hpxg, Nside, nest, racen, deccen, rad_deg):
    """_summary_

    Args:
        rag (_type_): _description_
        decg (_type_): _description_
        hpxg (_type_): _description_
        Nside (_type_): _description_
        nest (_type_): _description_
        racen (_type_): _description_
        deccen (_type_): _description_
        rad_deg (_type_): _description_

    Returns:
        _type_: _description_
    """
    pixels_in_disc_strict = hp.query_disc(
        nside=Nside,
        nest=nest,
        vec=hp.pixelfunc.ang2vec(np.radians(90.0 - deccen), np.radians(racen)),
        radius=np.radians(rad_deg),
        inclusive=False,
    )
    pixels_in_disc = hp.query_disc(
        nside=Nside,
        nest=nest,
        vec=hp.pixelfunc.ang2vec(np.radians(90.0 - deccen), np.radians(racen)),
        radius=np.radians(rad_deg),
        inclusive=True,
    )

    pixels_edge = pixels_in_disc[np.isin(pixels_in_disc, pixels_in_disc_strict)]
    cond_strict = np.isin(hpxg, pixels_in_disc_strict)
    cond_edge = np.isin(hpxg, pixels_edge)

    dist2cl = np.ones(len(rag)) * 2.0 * rad_deg
    dist2cl[cond_strict] = 0.0
    dist2cl[cond_edge] = np.degrees(
        dist_ang(rag[cond_edge], decg[cond_edge], racen, deccen)
    )
    return dist2cl < rad_deg


def cond_in_hpx_disc(hpxg, Nside, nest, racen, deccen, rad_deg):
    """_summary_

    Args:
        hpxg (_type_): _description_
        Nside (_type_): _description_
        nest (_type_): _description_
        racen (_type_): _description_
        deccen (_type_): _description_
        rad_deg (_type_): _description_

    Returns:
        _type_: _description_
    """
    pixels_in_disc_strict = hp.query_disc(
        nside=Nside,
        nest=nest,
        vec=hp.pixelfunc.ang2vec(np.radians(90.0 - deccen), np.radians(racen)),
        radius=np.radians(rad_deg),
        inclusive=False,
    )

    cond_strict = np.isin(hpxg, pixels_in_disc_strict)
    return cond_strict


def concatenate_clusters(tiles_dir, clusters_outfile):
    """_summary_

    Args:
        tiles_dir (_type_): _description_
        clusters_outfile (_type_): _description_
    """
    # assumes that clusters are called 'clusters.fits'
    # and the existence of 'tile_info.fits'

    print("")
    print("Concatenation starts")

    clist = []
    for tile_dir in tiles_dir:
        if os.path.isfile(os.path.join(tile_dir, "tile_info.fits")):
            if (
                read_FitsCat(os.path.join(tile_dir, "tile_info.fits"))[0]["Nclusters"] > 0
            ):
                clist.append(os.path.join(tile_dir, "clusters.fits"))
            else:
                print("warning : no detection in tile ", tile_dir)
        else:
            print("warning : missing tile ", tile_dir)
    concatenate_fits(clist, clusters_outfile)

    return


def add_clusters_unique_id(data_clusters, clkeys):
    """_summary_

    Args:
        data_clusters (_type_): _description_
        clkeys (_type_): _description_

    Returns:
        _type_: _description_
    """
    # create a unique id
    id_in_survey = np.arange(len(data_clusters))
    snr = data_clusters[clkeys["key_snr"]]
    data_clusters = data_clusters[np.argsort(-snr)]
    t = Table(data_clusters)  # , names=names)
    t["id"] = id_in_survey
    return t
