import numpy as np 
import dask.array as da
import astropy.io.fits as fits
#from lib.timeit import timeit

#@timeit
def read_FitsCat(cat):
    hdulist=fits.open(cat)
    dat=hdulist[1].data
    hdulist.close()
    return dat

#@timeit
def split_equal_nr_of_tiles_in_threads(n_threads, ntiles):
    if ntiles <= n_threads:
        n_threads = ntiles
        thread_ids = da.arange(0, ntiles)

    if n_threads == 0:
        n_threads = ntiles
        thread_ids = da.arange(0, ntiles)

    if ntiles > n_threads:
        thread_ids = da.zeros(ntiles).compute()

        p = float(ntiles)/float(n_threads)

        if ntiles % n_threads == 0:
            ng = int(p)
            k=0
            for i in range(0, n_threads):
                for j in range(0, ng):
                    thread_ids[k] = i
                    k+=1    
        else:
            ng = int(p)
            nr = ntiles - ng * n_threads

            k=0
            for i in range(0, n_threads):
                if i < nr:
                    for j in range(0, ng+1):
                        thread_ids[k] = i
                        k+=1    
                else:
                    for j in range(0, ng):
                        thread_ids[k] = i
                        k+=1    

    thread_idsf = thread_ids + 1

    return n_threads, thread_idsf.astype(int)


#@timeit
def split_equal_area_in_threads(n_threads, tiles_filename):

    tiles = read_FitsCat(tiles_filename)
    ntiles = len(tiles)

    if n_threads == 0:
        n_threads = ntiles
        thread_ids = da.arange(0, ntiles)

    elif ntiles <= n_threads:
        n_threads = ntiles
        thread_ids = da.arange(0, ntiles)

    else:
        eff_area = tiles['eff_area_deg2']
        area_thread = da.sum(eff_area).compute()/float(n_threads)        
        thread_ids = np.zeros(ntiles)
        # thread_ids.flags.writeable = True
        area_per_thread = np.zeros(n_threads)
        # area_per_thread.flags.writeable = True
        for j in np.argsort(-eff_area):
            i = np.argmin(area_per_thread)
            thread_ids[j]=i
            area_per_thread[i] += eff_area[j]

        print ('.....mean, min, max area / thread (deg2) = ', np.round(area_thread, 2), 
               np.round(np.amin(area_per_thread), 2), np.round(np.amax(area_per_thread), 2)   )

    thread_idsf = thread_ids + 1

    return n_threads, thread_idsf.astype(int)

