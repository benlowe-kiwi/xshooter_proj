import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from bcutil import bcstar
from numpy.random import normal,rand

def dist_plx(plx):
    """Calculates distance and distance error from parallax and parallax error.
    
    Parameters
    ----------
    plx (float): 
        Parallax in mas.
    plx_err (float): 
        Parallax error in mas.
        
    Returns
    -------
    dist : float
        Distance in kpc.
    dist_err : float
        Distance error in kpc.
    """

    # Calculating the distance in kpc
    dist = 1/plx  
    
    return dist
