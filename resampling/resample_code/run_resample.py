import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import os
import matplotlib.patches as patches
import matplotlib as mpl
from scipy.special import gammaincc
from scipy import stats
import seaborn as sns
from scipy.stats import gaussian_kde
import ndtest
from scipy.stats import ks_2samp, norm
from resample_code import delta_chisqr
from resample_code import resampling_data

def run_resample(n_iterations, data_dir, chisqr_dir, element):
    
    
    # ---LOADING IN THE DATA--- #
    # Loading in the abundance data
    comb_gse_df = pd.read_csv(data_dir + "comb_gse_df.csv")
    comb_prodisk_df = pd.read_csv(data_dir + "comb_prodisk_df.csv")
    comb_halo_df = pd.read_csv(data_dir + "comb_halo_df.csv")
    comb_retdisk_df = pd.read_csv(data_dir + "comb_retdisk_df.csv")
    
    # Loading in the chisqaure data
    el_chisqr_df = pd.read_csv(chisqr_dir + f"{element}_abunds_chisqr.csv")
    el_prodisk_deltachisqr = el_chisqr_df[el_chisqr_df['Region'] == 'pro_disk']['delta_chisqr'].values[0]
    el_gse_deltachisqr = el_chisqr_df[el_chisqr_df['Region'] == 'gse']['delta_chisqr'].values[0]
    el_retdisk_deltachisqr = el_chisqr_df[el_chisqr_df['Region'] == 'ret_disk']['delta_chisqr'].values[0]
    
    
    # ---CALCULATING THE DELTA CHI-SQUARE VALUES--- #
    all_prodisk_deltachisqr_dfs = []
    all_gse_deltachisqr_dfs = []
    for n in n_iterations:
        print(f"Running resampling for {element} with {n} iterations...")
        
        # Only taking stars with [Fe/H] < -1.8 (to help with binning)
        comb_halo_feh180_df = comb_halo_df[comb_halo_df['feh'] < -1.8].copy()
        comb_prodisk_feh180_df = comb_prodisk_df[comb_prodisk_df['feh'] < -1.8].copy()
        comb_gse_feh180_df = comb_gse_df[comb_gse_df['feh'] < -1.8].copy()
        comb_retdisk_feh180_df = comb_retdisk_df[comb_retdisk_df['feh'] < -1.8].copy()
        
        
        # # ---BEGIN TEST 1--- #
        # # Add observational errors to the [Fe/H] values (±0.1 dex)
        # feh_error = np.random.uniform(-0.1, 0.1)
        # comb_halo_feh180_df['feh'] += feh_error
        # comb_prodisk_feh180_df['feh'] += feh_error
        # comb_gse_feh180_df['feh'] += feh_error
        # comb_retdisk_feh180_df['feh'] += feh_error        
        # # ---END TEST 1--- #
        
        
        # # ---BEGIN TEST 2--- #
        # # Stop the analysis at [Fe/H] = -3.5, i.e. only taking stars with -3.5 < [Fe/H] < -1.8
        # comb_halo_feh180_df = comb_halo_feh180_df[comb_halo_feh180_df['feh'] > -3.5].copy()
        # comb_prodisk_feh180_df = comb_prodisk_feh180_df[comb_prodisk_feh180_df['feh'] > -3.5].copy()
        # comb_gse_feh180_df = comb_gse_feh180_df[comb_gse_feh180_df['feh'] > -3.5].copy()
        # comb_retdisk_feh180_df = comb_retdisk_feh180_df[comb_retdisk_feh180_df['feh'] > -3.5].copy()
        # # ---END TEST 2--- #
        
        
        # # ---BEGIN TEST 3--- #
        # # Shift bins by 0.25 dex. Instead of having bins of -4.5 to -4.0, -4.0 to -3.5, etc., have bins 
        # # of -4.25 to -3.75, -3.75 to -3.25, etc.
        # bins = np.arange(-4.25, -1.75 + 0.5, 0.5)
        # # ---END TEST 3--- #
        
        
        # Binning the data by metallicity (0.5 dex bins), then adding jitter
        bin_size = 0.5 
        bins = np.arange(-4.5, -1.5 + (bin_size / 2), bin_size) # Doing (bin_size / 2) to make sure the last bin includes -1.5
        
        jitter = 0.1                
        shift = np.random.uniform(-jitter, jitter)
        jittered_bins = bins + shift
                                    
        comb_halo_feh180_df['feh_bin'] = pd.cut(comb_halo_feh180_df['feh'], bins=jittered_bins, right=False)
        comb_prodisk_feh180_df['feh_bin'] = pd.cut(comb_prodisk_feh180_df['feh'], bins=jittered_bins, right=False)
        comb_gse_feh180_df['feh_bin'] = pd.cut(comb_gse_feh180_df['feh'], bins=jittered_bins, right=False)
        comb_retdisk_feh180_df['feh_bin'] = pd.cut(comb_retdisk_feh180_df['feh'], bins=jittered_bins, right=False)
                
        # Extracting the randomly resampled data for the halo based on the number of prograde and GSE stars in each bin
        resampled_halo_prodisk_df = resampling_data.resample_group(halo_df=comb_halo_feh180_df, 
                                                                other_df=comb_prodisk_feh180_df)
        resampled_halo_gse_df = resampling_data.resample_group(halo_df=comb_halo_feh180_df, 
                                                            other_df=comb_gse_feh180_df)
                
        # print(resampled_halo_prodisk_df[['star_id', 'feh', 'feh_bin']])
        
        # Compute the delta chi-square values for the prograde disk and GSE using the resampled halo data
        prodisk_deltachisqr = delta_chisqr.calc_delta_chisqr(comb_halo_df=comb_halo_feh180_df,
                                                            comb_other_df=resampled_halo_prodisk_df,
                                                            other_label='prodisk',
                                                            other_marker='s',
                                                            other_color='lawngreen',
                                                            n_iter=n,
                                                            save_path=None, # "/data/misfit/benlowe/xshooter/p115/chisqrs_resample/scmg/tests/bin_shift_025/testplots_prograde/",
                                                            plot=False)
        gse_deltachisqr = delta_chisqr.calc_delta_chisqr(comb_halo_df=comb_halo_feh180_df,
                                                        comb_other_df=resampled_halo_gse_df,
                                                        other_label='gse',
                                                        other_marker='p',
                                                        other_color='cornflowerblue',
                                                        n_iter=n,    
                                                        save_path=None, # "/data/misfit/benlowe/xshooter/p115/chisqrs_resample/scmg/tests/bin_shift_025/testplots_gse/",
                                                        plot=False)
        
        # Storing the results into their respective lists
        all_prodisk_deltachisqr_dfs.append(prodisk_deltachisqr)
        all_gse_deltachisqr_dfs.append(gse_deltachisqr)
        
    # Combining the results into a single DataFrame for each region
    prodisk_deltachisqr_df = pd.concat(all_prodisk_deltachisqr_dfs, ignore_index=True)
    gse_deltachisqr_df = pd.concat(all_gse_deltachisqr_dfs, ignore_index=True)
    
    return prodisk_deltachisqr_df, gse_deltachisqr_df
    
    

