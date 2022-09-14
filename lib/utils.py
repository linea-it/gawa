import numpy as np
import astropy.io.fits as fits
import os
import healpy as hp
from astropy.table import Table
import logging


def get_logger(name=None, stdout=True, level='info'):
    """
    Returns a logger object
    
    Args:
        name (string, optional): logger name. Defaults to None.
        stdout (boolean, optional): print to stdout. Defaults to True.
        level (string, optional): logger level. Defaults to 'info'.
    
    Returns:
        logger: logger object
    """

    log_debug = {
        'debug': logging.DEBUG, 'info': logging.INFO,
        'warning': logging.WARNING, 'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    if not name:
        name = __name__

    logger = logging.getLogger(name)

    if stdout:
        handler = logging.FileHandler(stdout)
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(log_debug[level])

    return logger


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
    hdulist=fits.open(cat)
    dat=hdulist[1].data
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

    hdulist=fits.open(hpx_footprint)
    dat = hdulist[1].data
    hdulist.close()
    hpix_map = dat[hpx_meta['key_pixel']].astype(int)
    if hpx_meta['key_frac'] == 'none':
        frac_map = np.ones(len(hpix_map0)).astype(float)
    else:
        frac_map = dat[hpx_meta['key_frac']]
    return  hpix_map, frac_map


def read_mosaicFitsCat_in_disc (galcat, tile, radius_deg):
    """From a list of galcat files, selects objects in a cone centered 
    on racen, deccen Output is a structured array

    Args:
        galcat (_type_): _description_
        tile (_type_): _description_
        radius_deg (_type_): _description_

    Returns:
        _type_: _description_
    """

    # tile 
    racen, deccen = tile['ra'], tile['dec']
    # list of available galcats => healpix pixels 
    gdir = galcat['mosaic']['dir']
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array(
        [os.path.splitext(x)[0] for x in raw_list]
    ).astype(int)
    extension = os.path.splitext(raw_list[0])[1]

    # find list of fits intersection cluster field
    Nside_fits, nest_fits = galcat['mosaic']['Nside'],\
                            galcat['mosaic']['nest']
    fits_pixels_in_disc = hp.query_disc(
        nside=Nside_fits, nest=nest_fits, 
        vec=hp.pixelfunc.ang2vec(
            np.radians(90.-deccen), np.radians(racen)
        ),
        radius = np.radians(radius_deg), inclusive=True
    )
    relevant_fits_pixels = fits_pixels_in_disc\
                           [np.isin(
                               fits_pixels_in_disc, 
                               hpix_fits, 
                               assume_unique=True
                           )]

    if len(relevant_fits_pixels):
        # merge intersecting fits 
        for i in range (0, len(relevant_fits_pixels)):
            dat_disc = read_FitsCat(
                os.path.join(gdir, str(relevant_fits_pixels[i])+extension)
            )
            dcen = np.degrees( 
                dist_ang(
                    dat_disc[galcat['keys']['key_ra']], 
                    dat_disc[galcat['keys']['key_dec']],
                    racen, deccen
                )
            )
            if i == 0:
                data_gal_disc = np.copy(dat_disc[dcen<radius_deg])
            else:
                data_gal_disc = np.append(
                    data_gal_disc, 
                    dat_disc[dcen<radius_deg]
                )
    else:
        data_gal_disc = None
    return data_gal_disc


def read_mosaicFootprint_in_disc (footprint, tile, radius_deg):
    """From a list of galcat files, selects objects in a cone 
    centered on racen, deccen
    Output is a structured array

    Args:
        footprint (_type_): _description_
        tile (_type_): _description_
        radius_deg (_type_): _description_

    Returns:
        _type_: _description_
    """

    # tile 
    racen, deccen = tile['ra'], tile['dec']
    # list of available galcats => healpix pixels 
    gdir = footprint['mosaic']['dir']
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array(
        [os.path.splitext(x)[0].replace('_footprint','') for x in raw_list]
    ).astype(int)
    extension = os.path.splitext(raw_list[0])[1]
    # find list of fits intersection cluster field
    Nside_fits, nest_fits = footprint['mosaic']['Nside'],\
                            footprint['mosaic']['nest']
    fits_pixels_in_disc = hp.query_disc(
        nside=Nside_fits, nest=nest_fits, 
        vec=hp.pixelfunc.ang2vec(
            np.radians(90.-deccen), np.radians(racen)
        ),
        radius = np.radians(radius_deg), 
        inclusive=True
    )
    relevant_fits_pixels = fits_pixels_in_disc\
                           [np.isin(
                               fits_pixels_in_disc, 
                               hpix_fits, 
                               assume_unique=True
                           )]
    if len(relevant_fits_pixels):
        # merge intersecting fits 
        for i in range (0, len(relevant_fits_pixels)):
            dat_disc = read_FitsCat(
                os.path.join(
                    gdir, 
                    str(relevant_fits_pixels[i])+'_footprint'+extension
                )
            )
            ra, dec = hpix2radec(dat_disc[footprint['key_pixel']],\
                                 footprint['Nside'], footprint['nest'])
            dcen = np.degrees(dist_ang(ra, dec, racen, deccen))
            if i == 0:
                data_fp_disc = np.copy(dat_disc[dcen<radius_deg])
            else:
                data_fp_disc = np.append(
                    data_fp_disc, 
                    dat_disc[dcen<radius_deg]
                )
    else:
        data_fp_disc = None

    return data_fp_disc


def read_mosaicFitsCat_in_hpix (galcat, hpix_tile, Nside_tile, nest_tile):
    """_summary_

    Args:
        footprint (_type_): _description_
        hpix_tile (_type_): _description_
        Nside_tile (_type_): _description_
        nest_tile (_type_): _description_

    Returns:
        _type_: _description_
    """
    """
    From a list of galcat files, selects objects in a cone 
    centered on racen, deccen
    Output is a structured array
    """
    # list of available galcats => healpix pixels 
    gdir = galcat['mosaic']['dir']
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array(
        [os.path.splitext(x)[0] for x in raw_list]
    ).astype(int)
    extension = os.path.splitext(raw_list[0])[1]
    Nside_fits, nest_fits = galcat['mosaic']['Nside'], galcat['mosaic']['nest']

    # warning we assume Nside_tile > Nside_fits !!
    ra_fits, dec_fits = hpix2radec(hpix_fits, Nside_fits, nest_fits )
    hpix_fits_tile = radec2hpix(ra_fits, dec_fits, Nside_tile, nest_tile)
    relevant_fits_pixels = np.unique(
        hpix_fits[np.isin(hpix_fits_tile, hpix_tile)]
    )
    if len(relevant_fits_pixels):
        # merge intersecting fits 
        for i in range (0, len(relevant_fits_pixels)):
            dat = read_FitsCat(
                os.path.join(gdir, str(relevant_fits_pixels[i])+extension)
            )
            if i == 0:
                data_gal_hpix = np.copy(dat)
            else:
                data_gal_hpix = np.append(data_gal_hpix, dat)
    else:
        data_gal_hpix = None
    return data_gal_hpix


def read_mosaicFootprint_in_hpix (footprint, hpix_tile, Nside_tile, nest_tile):
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
    gdir = footprint['mosaic']['dir']
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array(
        [os.path.splitext(x)[0].replace('_footprint','') for x in raw_list]
    ).astype(int)
    extension = os.path.splitext(raw_list[0])[1]
    Nside_fits, nest_fits = footprint['mosaic']['Nside'],\
                            footprint['mosaic']['nest']

    # warning we assume Nside_tile > Nside_fits !!
    ra_fits, dec_fits = hpix2radec(hpix_fits, Nside_fits, nest_fits )
    hpix_fits_tile = radec2hpix(ra_fits, dec_fits, Nside_tile, nest_tile)

    relevant_fits_pixels = np.unique(
        hpix_fits[np.isin(hpix_fits_tile, hpix_tile)]
    )

    if len(relevant_fits_pixels):
        # merge intersecting fits 
        for i in range (0, len(relevant_fits_pixels)):
            dat = read_FitsCat(
                os.path.join(
                    gdir, 
                    str(relevant_fits_pixels[i])+'_footprint'+extension
                )
            )
            if i == 0:
                data_fp_hpix = np.copy(dat)
            else:
                data_fp_hpix = np.append(data_fp_hpix, dat)
    else:
        data_fp_hpix = None
    return data_fp_hpix


def create_survey_footprint_from_mosaic(footprint, survey_footprint):
    """_summary_

    Args:
        footprint (_type_): _description_
        fpath (_type_): _description_
    """
    all_files = np.array(os.listdir(footprint['mosaic']['dir']))
    flist = [os.path.join(footprint['mosaic']['dir'], f) for f in all_files]
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
    hpix0, frac0 = read_FitsFootprint(
        footprint['survey_footprint'], footprint
    )
    ra0, dec0 = hpix2radec(hpix0, footprint['Nside'], footprint['nest'])
    hpix = radec2hpix(
        ra0, dec0, 
        footprint['mosaic']['Nside'], 
        footprint['mosaic']['nest']
    )
    for hp in np.unique(hpix):
        all_cols = fits.ColDefs([
            fits.Column(
                name = footprint['key_pixel'],  
                format = 'K',
                array = hpix0[np.isin(hpix, hp)]
            ),
            fits.Column(
                name='ra',       
                format='E',
                array= ra0[np.isin(hpix, hp)]
            ),
            fits.Column(
                name='dec',      
                format='E',
                array= dec0[np.isin(hpix, hp)]
            ),
            fits.Column(
                name=footprint['key_frac'],   
                format='K',
                array= frac0[np.isin(hpix, hp)]
            )
        ])
        hdu = fits.BinTableHDU.from_columns(all_cols)
        hdu.writeto(
            os.path.join(fpath, str(hp)+'_footprint.fits'),
            overwrite=True
        )
    return


def concatenate_fits(flist, output):
    """_summary_

    Args:
        flist (_type_): _description_
        output (_type_): _description_

    Returns:
        _type_: _description_
    """

    for i in range (0, len(flist)):
        dat = read_FitsCat(flist[i])
        if i == 0:
            cdat = np.copy(dat)
        else:
            cdat = np.append(cdat, dat)

    t = Table(cdat)#, names=names)
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
    for i in range (0, len(flist)):
        dat = read_FitsCat(flist[i])
        datL = Table(dat)
        datL[label_name] = int(label[i])*np.ones(len(dat)).astype(int)
        if i == 0:
            cdat = np.copy(datL)
        else:
            cdat = np.append(cdat, datL)
    t = Table(cdat)#, names=names)
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
    dat=hdulist[1].data
    hdulist.close()

    orig_cols = dat.columns

    if key_type == 'float':
        new_col = fits.ColDefs([
            fits.Column(name=key_name, format='E',array=key_val)])
    if key_type == 'int':
        new_col = fits.ColDefs([
            fits.Column(name=key_name, format='J',array=key_val)])


    hdu = fits.BinTableHDU.from_columns(orig_cols + new_col)    
    hdu.writeto(fitsfile, overwrite = True)
    return

        
def filter_hpx_tile(data, cat, tile_specs):
    """_summary_

    Args:
        data (_type_): _description_
        cat (_type_): _description_
        tile_specs (_type_): _description_

    Returns:
        _type_: _description_
    """
    ra, dec = data[cat['keys']['key_ra']],\
              data[cat['keys']['key_dec']]
    Nside, nest = tile_specs['Nside'], tile_specs['nest']
    pixel_tile = tile_specs['hpix']
    hpx = radec2hpix(ra, dec, Nside, nest)
    return data[np.argwhere(hpx == pixel_tile).T[0]]


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
    ghpx = radec2hpix (ra, dec, Nside_tmp, nest_tmp)
    t = Table (data_gal)
    t[keyname] = ghpx
    return t


def mad(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1.4826*np.median(abs(x))


def gaussian(x, mu, sig):
    """_summary_

    Args:
        x (_type_): _description_
        mu (_type_): _description_
        sig (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.exp(-(x - mu)**2 / (2.*sig**2) ) / (sig * np.sqrt(2.*np.pi))


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
    """
    angular distance between (ra1, dec1) and (ra_ref, dec_ref)
    ra-dec in degrees
    ra1-dec1 can be arrays 
    ra_ref-dec_ref are scalars
    output is in radian
    """
    costheta = np.sin(np.radians(dec_ref)) * np.sin(np.radians(dec1)) +\
               np.cos(np.radians(dec_ref)) * np.cos(np.radians(dec1)) *\
               np.cos(np.radians(ra1-ra_ref))
    dist_ang = np.arccos(costheta)
    return dist_ang 


def area_ann_deg2(theta_1, theta_2):
    """_summary_

    Args:
        theta_1 (_type_): _description_
        theta_2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    area = 2. * np.pi * (np.cos(np.radians(theta_1)) -\
                         np.cos(np.radians(theta_2))) *\
        (180./np.pi)**2
    return area


def _mstar_ (mstar_filename, zin):
    """
    from a given (z, mstar) ascii file
    interpolate to provide the mstar at a given z_in
    """
    zst, mst = np.loadtxt(mstar_filename, usecols=(0, 1), unpack=True)
    return np.interp (zin,zst,mst)


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
        joint[:,offset:offset+size] = a.view(np.uint8).reshape(n,size)
        #print ('desc ', a.dtype.descr)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)

def radec_window_area (ramin, ramax, decmin, decmax):
    """_summary_

    Args:
        ramin (_type_): _description_
        ramax (_type_): _description_
        decmin (_type_): _description_
        decmax (_type_): _description_

    Returns:
        _type_: _description_
    """
    nstep = int((decmax-decmin)/0.1)+1
    step = (decmax-decmin)/float(nstep)
    decmini = np.arange(decmin, decmax, step)
    decmaxi = decmini+step
    decceni = (decmini + decmaxi)/2.
    darea = (ramax-ramin)*np.cos(np.pi*decceni/180.)*(decmaxi-decmini)
    return np.sum(darea)


# healpix functions
def radec2phitheta(ra, dec):
    """_summary_

    Args:
        phi (_type_): _description_
        theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    """
    Turn (ra,dec) [deg] in (phi, theta) [rad] used by healpix
    """
    phi, theta = np.radians(ra), np.radians(90.-dec)
    return phi, theta


def phitheta2radec(phi,theta):
    """_summary_

    Args:
        phi (_type_): _description_
        theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.degrees(phi), 90.-np.degrees(theta)


def radec2hpix(ra, dec, Nside, nest):
    """_summary_

    Args:
        hpix (_type_): _description_
        nside (_type_): _description_
        nest (_type_): _description_

    Returns:
        _type_: _description_
    """
    """
    From a list of ra-dec's (deg) compute the list of associated healpix index
    """
    phi, theta = radec2phitheta(ra, dec) # np.radians(ra), np.radians(90.-dec)
    return  hp.ang2pix(Nside, theta, phi, nest)


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
    return phitheta2radec(phi,theta)


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
    rac, decc = np.zeros(4*len(hpix)), np.zeros(4*len(hpix))
    i=0
    for p in hpix:
        theta, phi = hp.vec2ang(hp.boundaries(Nside, p, 1, nest).T)
        ra, dec = phitheta2radec(phi, theta)
        racen, deccen = hpix2radec(p, Nside, nest)
        for j in range(0,4):
            rac[i], decc[i] = (ra[j]+racen)/2., (dec[j]+deccen)/2. 
            i+=1

    return radec2hpix(rac, decc, Nside*2, nest)


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
    ipix = hp.ang2pix(nside, (90-dec)/180*np.pi, ra/180*np.pi, nest=nest)
    return np.bincount(ipix, weights = weights, minlength=hp.nside2npix(nside))


def all_hpx_in_annulus (ra, dec, radius_in_deg, radius_out_deg, 
                        hpx_meta, inclusive):
    """
    Get the list of all healpix pixels falling in an annulus around 
    ra-dec (deg) 
    the radii that define the annulus are in degrees    
    pixels are inclusive on radius_out but not radius_in
    """
    Nside, nest = hpx_meta['Nside'], hpx_meta['nest']
    pixels_in_disc = hp.query_disc(
        nside=Nside, nest=nest, 
        vec=hp.pixelfunc.ang2vec(
            np.radians(90.-dec), 
            np.radians(ra)
        ),
        radius = np.radians(radius_out_deg), 
        inclusive=inclusive
    )
    if radius_in_deg>0.:
        pixels_in_disc_in = hp.query_disc(
            nside=Nside, nest=nest, 
            vec=hp.pixelfunc.ang2vec(
                np.radians(90.-dec), 
                np.radians(ra)
            ),
            radius = np.radians(radius_in_deg), 
            inclusive=inclusive
        )
        id_annulus = np.isin(
            pixels_in_disc, 
            pixels_in_disc_in, 
            assume_unique=True, 
            invert=True
        )
        pixels_in_ann = pixels_in_disc[id_annulus]
    else:
        pixels_in_ann = np.copy(pixels_in_disc)

    return pixels_in_ann

def hpx_in_annulus (ra, dec, radius_in_deg, radius_out_deg, 
                    data_fp, hpx_meta, inclusive):
    """
    Given an array of healpix pixels (hpix, frac) where frac is the 
    covered fraction of each hpix pixel,
    computes the sub list of these pixels falling in an annulus around position 
    ra-dec (deg)
    the radii that define the annulus are in degrees
    hpx pixels are inclusive on radius_out but not radius_in
    """
    Nside, nest = hpx_meta['Nside'], hpx_meta['nest']
    hpix, frac = data_fp[hpx_meta['key_pixel']], data_fp[hpx_meta['key_frac']]

    area_pix = hp.pixelfunc.nside2pixarea(Nside, degrees=True)
    pixels_in_ann = all_hpx_in_annulus (
        ra, dec, radius_in_deg, radius_out_deg, hpx_meta, inclusive
    )
    npix_all = len(pixels_in_ann)
    area_deg2 = 0.
    coverfrac = 0.
    hpx_in_ann, frac_in_ann = [], []

    if npix_all:
        idx = np.isin(hpix, pixels_in_ann)
        hpx_in_ann = hpix[idx]  # visible pixels
        frac_in_ann = frac[idx] 
        npix = len(hpx_in_ann)
        if npix > 0:
            area_deg2 = np.sum(frac_in_ann) * area_pix
            coverfrac = np.sum(frac_in_ann)/float(npix_all)
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
    if ramin<0.5 and ramax>359.5:
        nbins = 360
        hist, bin_edges = np.histogram(ra, bins=nbins, range=(0., 360))
        ramin_empty = bin_edges[np.amin ( np.argwhere(hist==0 ))]
        ramax_empty = bin_edges[np.amax ( np.argwhere(hist==0 ))]
        
        ra1 = ra[(ra<ramin_empty+1.)]
        ra2 = ra[(ra>ramax_empty-1.)]-360.
        ra_new = np.hstack((ra1, ra2))
        ramin, ramax = np.amin(ra_new), np.amax(ra_new)
    return ramin, ramax


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
    nsamp = (float(nside_in)/float(nside_out))**2
    return pix_out, counts.astype(float)/nsamp

def hpx_split_survey (footprint_file, footprint, admin, output):
    """_summary_

    Args:
        footprint_file (_type_): _description_
        footprint (_type_): _description_
        admin (_type_): _description_
        output (_type_): _description_

    Returns:
        _type_: _description_
    """
    Nside_fp  , nest_fp   = footprint['Nside'], footprint['nest']
    Nside_tile, nest_tile = admin['Nside'], admin['nest']

    dat = read_FitsCat(footprint_file)
    hpix_map = dat[footprint['key_pixel']]
    frac_map = dat[footprint['key_frac']]

    '''
    # inner tile
    hmap = np.arange(hp.nside2npix(Nside_fp))
    pixel0 = np.zeros(len(hmap))
    pixel0[hpix_map]=1
    frac0 = hp.pixelfunc.ud_grade(pixel0, Nside_tile)
    hpix_tile = np.argwhere(frac0>0).T[0]
    frac_tile = frac0[(frac0>0)]
    '''
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
            nside=Nside_fp, nest=nest_fp, 
            vec=hp.pixelfunc.ang2vec(
                np.radians(90.-deccen[i]), np.radians(racen[i])
            ),
            radius = np.radians(radius_deg), 
            inclusive=False
        )
        framed_eff_area_deg2[i] = np.sum(
            frac_map[np.isin(hpix_map, pixels_in_disc, assume_unique=True)]
        ) * hp.pixelfunc.nside2pixarea(Nside_fp, degrees=True)

    data_tiles = np.zeros( (len(hpix_tile)), 
                           dtype={
                               'names':(
                                   'id', 'hpix', 
                                   'ra', 'dec', 
                                   'area_deg2', 'eff_area_deg2',
                                   'framed_eff_area_deg2',
                                   'radius_tile_deg', 
                                   'Nside', 'nest'
                               ), 
                                  'formats':(
                                      'i8', 'i8', 
                                      'f8', 'f8', 
                                      'f8', 'f8', 
                                      'f8', 
                                      'f8', 
                                      'i8', 'b'
                                  ) 
                           }
    )
    data_tiles['id'] = np.arange(len(hpix_tile))
    data_tiles['hpix'] = hpix_tile
    data_tiles['ra'], data_tiles['dec'] = np.around(racen,4),\
                                          np.around(deccen,4)
    data_tiles['area_deg2'] = np.around(area_tile,4)
    data_tiles['eff_area_deg2'] = np.around(frac_tile*area_tile,4)
    data_tiles['framed_eff_area_deg2'] = np.around(framed_eff_area_deg2,4)
    data_tiles['radius_tile_deg'] = np.ones(len(hpix_tile))*tile_radius(admin)
    data_tiles['Nside'] = Nside_tile *np.ones(len(hpix_tile)).astype(int)
    data_tiles['nest'] = len(hpix_tile)*[nest_tile]

    t = Table (data_tiles)
    t.write(output, overwrite=True)
    print ('.....tile area (deg2) = ', 
           np.round(hp.pixelfunc.nside2pixarea(Nside_tile, degrees=True), 2)
    )
    print ('.....effective survey area (deg2) = ', 
           np.around(np.sum(frac_tile)*area_tile,4)
    )
    return len(data_tiles)


def tile_radius(tiling):
    """_summary_

    Args:
        tiling (_type_): _description_

    Returns:
        _type_: _description_
    """

    Nside_tile = tiling['Nside']
    frame_deg = tiling['overlap_deg']
    tile_radius = (2.*\
                   hp.pixelfunc.nside2pixarea(
                       Nside_tile, degrees=True
                   ))**0.5 / 2.
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

    if admin['target_mode']:
        hpix = -1
        Nside, nest = -1, None
        area_deg2 = -1.
        eff_area_deg2 = -1. 
        framed_eff_area_deg2 = -1.
    else:
        hpix = tile['hpix']
        Nside, nest = admin['tiling']['Nside'], admin['tiling']['nest']
        area_deg2 = tile['area_deg2']
        eff_area_deg2 = tile['eff_area_deg2']
        framed_eff_area_deg2 = tile['framed_eff_area_deg2']

    tile_specs = {'id':tile['id'],
                  'ra': tile['ra'], 'dec': tile['dec'],
                  'hpix': hpix,
                  'Nside': Nside,
                  'nest': nest,
                  'area_deg2': area_deg2,
                  'eff_area_deg2': eff_area_deg2,
                  'framed_eff_area_deg2': framed_eff_area_deg2,
                  'radius_tile_deg': tile_radius_deg, 
                  'radius_filter_deg': -1.,  # active only if target_mode=True 
                  'target_mode': admin['target_mode'] }
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
        nside = Nside, nest=nest, 
        vec = hp.pixelfunc.ang2vec(
            np.radians(90.-deccen), np.radians(racen)
        ),
        radius = np.radians(rad_deg), 
        inclusive = False
    )
    pixels_in_disc = hp.query_disc(
        nside = Nside, nest=nest, 
        vec = hp.pixelfunc.ang2vec(
            np.radians(90.-deccen), np.radians(racen)
        ),
        radius = np.radians(rad_deg), 
        inclusive=True
    )
    pixels_edge = pixels_in_disc[np.isin(
        pixels_in_disc, pixels_in_disc_strict, assume_unique=True
    )]
    cond_strict = np.isin(hpxg, pixels_in_disc_strict)
    cond_edge  =  np.isin(hpxg, pixels_edge)

    dist2cl = np.ones(len(rag))*2.*rad_deg
    dist2cl[cond_strict] = 0.
    dist2cl[cond_edge] = np.degrees(
        dist_ang(
            rag[cond_edge], decg[cond_edge], racen, deccen
        )
    )
    return (dist2cl<rad_deg)


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
        nside=Nside, nest=nest, 
        vec=hp.pixelfunc.ang2vec(
            np.radians(90.-deccen), np.radians(racen)
        ),
        radius = np.radians(rad_deg), 
        inclusive=False
    )

    cond_strict = np.isin(hpxg, pixels_in_disc_strict)
    return cond_strict


def normal_distribution_function(x):
    value = scipy.stats.norm.pdf(x,mean,std)
    return value

def compute_gaussian_kernel_1d(kernel):
# kernel is an integer >0
    mean = 0.0 
    kk = []
    for n in range (0,3*kernel+1):
        x1 = mean - 1./2. + float(n)
        x2 = mean + 1./2. + float(n)
        res, err = quad(normal_distribution_function, x1, x2)
        kk = np.append(kk, 100.*res)
    return np.array(np.concatenate((np.sort(kk)[0:len(kk)-1], kk)))


def get_gaussian_kernel_1d(kernel):

    if kernel == 1:
        gkernel = 0.01*np.array([0.60, 6.06, 24.17, 38.29, 24.17, 6.06, 0.60])
    if kernel == 2:
        gkernel = 0.01*np.array([ 0.24, 0.924, 2.783, 6.559, 12.098, \
                                  17.467, 19.741, 17.467, 12.098, 6.559, \
                                  2.783, 0.924, 0.24])
    if kernel == 3:
        gkernel = 0.01*np.array([ 0.153, 0.391, 0.892, 1.825, 3.343, 5.487, \
                                  8.066, 10.621, 12.528, 13.237, 12.528, \
                                  10.621, 8.066, 5.487, 3.343, 1.825, 0.892,\
                                  0.391, 0.153])
    if kernel > 3:
        gkernel = compute_gaussian_kernel_1d(kernel)
    return gkernel


def concatenate_clusters(tiles_dir, infilename, clusters_outfile): 
    """_summary_

    Args:
        tiles_dir (_type_): _description_
        clusters_outfile (_type_): _description_
    """
    # assumes that clusters are called 'clusters.fits'
    # and the existence of 'tile_info.fits'
    clist = []
    for tile_dir in tiles_dir: 
        clist.append(os.path.join(tile_dir, infilename))
    clcat = concatenate_fits(clist, clusters_outfile)
    return clcat


def concatenate_members(all_tiles, list_path_members, 
                        infilename, data_clusters, members_outfile):
    # data_clusters = clusters over the whole survey
    for it in range(0, len(all_tiles)):
        tile_id = int(all_tiles['id'][it])
        clusters_tile = data_clusters[data_clusters['tile'] == tile_id]
        clusters_id_in_tile = clusters_tile['index_cl_tile']
        members = read_FitsCat(
            os.path.join(list_path_members[it], infilename)
        )
        members_kept = members[np.isin(
            members['index_cl_tile'], clusters_id_in_tile
        )]
        # sort clusters by id_in_tile 
        ids = clusters_tile[np.argsort(clusters_id_in_tile)]['id']
        nmems = clusters_tile[np.argsort(clusters_id_in_tile)]['nmem']
        idd = clusters_tile[np.argsort(clusters_id_in_tile)]['index_cl_tile']
        # sort members by id_in_tile 
        members_kept_sorted = members_kept[np.argsort(
            members_kept['index_cl_tile']
        )]
        if it == 0:
            final_members = np.copy(members_kept_sorted)
        else:
            final_members = np.hstack((final_members, members_kept_sorted))
        for i in range(0, len(clusters_id_in_tile)):
            if it == 0 and i == 0:
                ids_for_members = ids[i]*np.ones(nmems[i]).astype(int)
                tile_for_members = tile_id*np.ones(nmems[i]).astype(int)
            else:
                ids_for_members = np.hstack(
                    (ids_for_members, ids[i]*np.ones(nmems[i]).astype(int))
                )
                tile_for_members = np.hstack(
                    (tile_for_members, tile_id*np.ones(nmems[i]).astype(int))
                )
    t = Table (final_members)
    t['id_cl'] = ids_for_members
    t['tile'] = tile_for_members
    t.write(members_outfile, overwrite=True)
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
    snr = data_clusters[clkeys['key_snr']]
    data_clusters = data_clusters[np.argsort(-snr)]
    t = Table (data_clusters)#, names=names)
    t['id'] = id_in_survey.astype('str')
    return t
