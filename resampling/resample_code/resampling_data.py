import numpy as np
import pandas as pd

def resample_group(halo_df, other_df):
    """"For a given other_df (e.g. prograde disk or GSE), randomly resample stars in halo_df based on the number of stars
    in each bin in other_df. This returns a new dataframe of randomly resampled halo stars that has the same number of stars 
    as the other_df, but with the same distribution as the original halo_df.
    
    Parameters:
    -----------
    halo_df: pd.DataFrame
        The dataframe containing the halo stars, which should have a column 'feh_bin' for the metallicity bins.
    other_df: pd.DataFrame
        The dataframe containing the other group of stars (e.g. prograde disk or GSE), which should also have 
        a column 'feh_bin' for the metallicity bins.
        
    Returns:
    --------
    resampled_halo_df: pd.DataFrame
        A new dataframe containing the randomly resampled halo stars based on the distribution of the other_df
    """""
    
    # Extracting the randomly resampled data for the halo based on the number of stars in each bin in other_df
    resampled_halo_df = pd.DataFrame()  
    for bin_interval, count in other_df['feh_bin'].value_counts().items():
        
        # Getting the halo stars in the same bin
        halo_bin_stars = halo_df[halo_df['feh_bin'] == bin_interval]
        
        # Resample the halo stars based on the count in the other_df
        if len(halo_bin_stars) > 0 and count > 0:
            resampled_halo_bin_stars = halo_bin_stars.sample(n=count, replace=True)
            resampled_halo_df = pd.concat([resampled_halo_df, resampled_halo_bin_stars], ignore_index=True)
    
    return resampled_halo_df