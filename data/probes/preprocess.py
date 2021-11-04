import numpy as np
from tqdm import tqdm
from astropy.io import fits
from glob import glob
import re
from os.path import basename, getsize

def check_for_corruption(ar):
    '''
    Check if the galaxies are there and then remove the image if its
    not there.
    
    If more than 1/3 of the pixels are duds we don't want that gal.
    '''
    nanned = np.any(~np.isfinite(ar))
    zeroed = np.sum(ar == 0) > (ar.size*0.3)
    return np.any((nanned, zeroed))

for fi in tqdm(sorted(glob('./raws/*_g.fits'))):
    if any((
           getsize(fi) < 5000,
           getsize(fi[:-7] + '_r.fits') < 5000,
           getsize(fi[:-7] + '_z.fits') < 5000
          )):
        print(f'{fi} is empty, not dealing with that...')
        continue
    try:
        g = fits.open(fi)[0].data
        r = fits.open(re.sub('_g.fits', '_r.fits', fi))[0].data
        z = fits.open(re.sub('_g.fits', '_z.fits', fi))[0].data

        if any(map(check_for_corruption, (g, r, z))):
            print(f'Removing corrupted/missing pixels image: {fi}')
            continue

    except Exception as e:
        print(f'Problem with loading, skipping that one...\n{e}')
        continue

    gal = np.stack([g, r, z], axis=0)
    gal = gal[:, gal.shape[1]//2 - 128:gal.shape[1]//2 + 128, gal.shape[2]//2 - 128:gal.shape[2]//2 + 128]
    np.save(f'./gals/{basename(fi)[:-7]}.npy', gal)
