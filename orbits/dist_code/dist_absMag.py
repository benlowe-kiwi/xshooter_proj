import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from bcutil import bcstar
from numpy.random import normal,rand

def calc_absM(teff, logg, bc):
    """Calculates the absolute Gaia G magnitude and its error.
    
    Parameters
    ----------
    teff : float
        Effective temperature in K.
    logg : float
        Log surface gravity in dex.
    bc : float
        Bolometric correction.
    
    Returns
    -------
    M_G : float
        Absolute Gaia G magnitude.
    M_G_err : float
        Absolute Gaia G magnitude error.
    """
    
    # Solar values
    logg_sol = 4.438
    mass_sol = 1
    teff_sol = 5771
    M_sol = 4.75
    
    # Assumptions made on the star
    mass = 0.8
    
    # Calculating the absolute bolometric magnitude
    M = 5/2 * (logg-logg_sol-np.log10((mass*teff**4)/(mass_sol*teff_sol**4))) + M_sol
    
    # Using the bolometric correction to get the absolute gaia magnitude
    M_G = M - bc
    
    return M_G


def dist_absMag(star_id, teff, logg, feh, ebv, m_g):
    """Calculates the distance using the distance modulus. This requires the absolute
    magnitude to be calculated first, done using the stefan-boltzman law mixed with 
    various canonical equations. Bolometric corrections are calculated using the
    bcutil package.
    
    Parameters
    ----------
    star_id (str):
        Star identifier.
    teff (float): 
        Effective temperature in K.
    logg (float): 
        Log surface gravity in dex.
    feh (float): 
        Metallicity in dex.
    ebv (float): 
        E(B-V) value for extinction correction.
    m_g (float): 
        Apparent Gaia G magnitude.
        
    Returns
    -------
    dist (float):
        Distance in kpc.
    """    
    
    # Calculating the bolometric correction
    bc = bcstar(sid=star_id, 
                teff=teff, 
                logg=1.0, 
                feh=feh,
                ebv=ebv, 
                filters='G3', retarr=True, Rv=3.1)
    bc_val = bc[0][0]
    
    # If bolometric correction is NaN, then parameters are hitting grid limits
    # Will adjust according to the line of code in becutil.py that sets NaNs:
    #  rme = np.where((ebv>=0.72) |                 # Removes stars with E(B-V) >= 0.72
    #                 (ebv<0.) |                    # Removes stars with E(B-V) < 0.0
    #                 (np.isfinite(ebv)==False) |   # Removes stars with non-finite E(B-V)
    #                 ((logg<2.5) & (teff<3500)))   # Removes stars with logg<2.5 and Teff<3500 K

    if np.isnan(bc_val):
        if ebv >= 0.72 and logg < 1.0:
            bc = bcstar(sid=star_id, 
                        teff=teff, 
                        logg=1.0, 
                        feh=feh,
                        ebv=0.71, 
                        filters='G3', retarr=True, Rv=3.1)
        elif ebv >= 0.72:
            bc = bcstar(sid=star_id, 
                        teff=teff, 
                        logg=logg, 
                        feh=feh,
                        ebv=0.71, 
                        filters='G3', retarr=True, Rv=3.1)
        elif logg < 1.0:
            bc = bcstar(sid=star_id, 
                        teff=teff, 
                        logg=1.0, 
                        feh=feh,
                        ebv=ebv, 
                        filters='G3', retarr=True, Rv=3.1)
        bc_val = bc[0][0]
    

    # Calculating the absolute magnitude and its error
    M_G = calc_absM(teff=teff,
                    logg=logg,
                    bc=bc_val)
    
    # Calculating the distance from the apparent and absolute magnitudes
    dist = 10**((m_g-M_G+5)/5)
    dist = dist/1000  # Converting to kpc
    
    return dist, M_G, bc_val