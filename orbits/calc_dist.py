import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from bcutil import bcstar
from numpy.random import normal,rand
from dist_code.dist_plx import dist_plx
from dist_code.dist_absMag import dist_absMag
from astropy.table import Table
from multiprocessing import Pool
from functools import partial


def run_mcmc_dist(df, savepath):
    """Runs an MCMC to estimate distances (mean) and their errors (standard deviation) for a set of stars.
    Will use two different methods depending on the quality of the parallax measurement:
    - For stars with 'good' parallaxes (parallax > 3 * parallax_error), distances are derived from the parallax.
    - For stars with 'bad' parallaxes (parallax <= 3 * parallax_error), distances are derived from the absolute magnitude.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the stellar parameters and Gaia data. Must contain the following columns:
        'star_id', 'teff', 'logg', 'feh', 'ebv_corr', 'phot_g_mean_mag', 'parallax', 'parallax_error'.
    savepath : str
        Path to save the output distance results.
        
    Returns
    -------
    None
    """
    all_dist_res = [] 
    for i, entry in df.iterrows():
        
        star_id = str(entry['star_id'])
        catalog = str(entry['catalog'])
        
        # Creating save directory: savepath/catalog/star_id/
        star_savepath = os.path.join(savepath, catalog)
        os.makedirs(star_savepath, exist_ok=True)
        
        # # Check if star has already been processed
        # if os.path.exists(os.path.join(star_savepath, f"{star_id}_dists.csv")):
        #     print(f"Star {star_id} already processed, skipping...")
        #     continue
        
        
        # Extracting the stellar parameters
        teff = entry['teff']
        teff_err = 100 # Assuming a constant error of 100 K
        
        logg = entry['logg']
        logg = 0.3  # Assuming a constant error of 0.3 dex
        
        feh = entry['feh']
        feh_err = 0.15 # Assuming a constant error of 0.15 dex
        
        ebv = abs(entry['ebv_corr'])
        env_err = 0.1 * float(ebv) # Taking error of reddening as 10%
        
        m_g = entry['phot_g_mean_mag']
        m_g_err = (2.5/np.log(10)) * 1/(float(entry['phot_g_mean_flux_over_error']))
        
        plx = entry['parallax']
        plx_err = entry['parallax_error']
        
        
        # Beginning MCMC calculation
        nrand = 100
        n_success = 0
        res_list = []
        print(f"Running star {star_id} ({i+1}/{len(df)})")
        
        while n_success < nrand:
            print(f"  Iteration {n_success+1}/{nrand}")
            # Randomising the input parameters based on their errors
            teff_rand = normal(loc=teff, scale=teff_err)
            logg_rand = normal(loc=logg, scale=logg)
            feh_rand = normal(loc=feh, scale=feh_err)
            m_g_rand = normal(loc=m_g, scale=m_g_err)
            plx_rand = normal(loc=plx, scale=plx_err)
            ebv_rand = normal(loc=ebv, scale=env_err)
                        
            # Calculating distances based on their parallax quality
            # For stars with 'good' parallaxes, derive distance from parallax
            if plx_rand > 3 * plx_err:
                plx_sigma = plx_rand/plx_err
                plx_quality = 'good'
                
                dist_rand = dist_plx(plx=plx_rand)
                        
            # For stars with 'bad' parallaxes, derive distance from absolute magnitude
            elif plx_rand <= 3 * plx_err:
                plx_sigma = plx_rand/plx_err
                plx_quality = 'bad'      
                
                dist_rand, M_G_rand, bc_rand = dist_absMag(star_id=star_id, 
                                                           teff=teff_rand, 
                                                           logg=logg_rand, 
                                                           feh=feh_rand, 
                                                           ebv=ebv_rand, 
                                                           m_g=m_g_rand)
            
            # Only accepting physical distances, otherwise re-sample
            if dist_rand > 0:
                res_dict = {
                    'it': n_success+1,
                    'dist': dist_rand,
                    'teff_rand': teff_rand,
                    'logg_rand': logg_rand,
                    'feh_rand': feh_rand,
                    'ebv_rand': ebv_rand,
                    'm_g_rand': m_g_rand,
                    'plx_rand': plx_rand,
                    'M_G_rand': M_G_rand if plx_quality == 'bad' else np.nan,
                    'bc_rand': bc_rand if plx_quality == 'bad' else np.nan,
                    'plx_sigma': plx_sigma,
                    'plx_quality': plx_quality
                }
                n_success += 1
                res_list.append(res_dict)
            else:
                print("Unphysical distance, re-sampling...")
                continue
        
        # End of MCMC calculation
        res_df = pd.DataFrame(res_list)
        
        res_df.to_csv(os.path.join(star_savepath, f"{star_id}_dists.csv"), index=False)


        # Calculating the mean distance and its error, then storing to list
        dist_mean = res_df['dist'].mean()
        dist_std = res_df['dist'].std()
        dist_dict = {
            'star_id': star_id,
            'catalog': catalog,
            'dist_mean': dist_mean,
            'dist_std': dist_std
        }
        all_dist_res.append(dist_dict)
        print(f"Star {star_id}: Distance = {dist_mean:.3f} +/- {dist_std:.3f} kpc")
        
        
        # Plotting the distance distribution
        plt.figure(figsize=(8,6))
        plt.hist(res_df['dist'], bins=20, color='chocolate', alpha=1, edgecolor='black')
        ylims = plt.gca().get_ylim()

        plt.axvline(dist_mean, color='black', linestyle='dashed', linewidth=2)
        plt.fill_betweenx(y=ylims, x1=dist_mean - dist_std, x2=dist_mean + dist_std, color='gray', alpha=0.5)

        plt.xlabel('Distance (kpc)')
        plt.ylabel('Number of Stars')
        plt.ylim(ylims)
        plt.title(f"{star_id}\nDist = {dist_mean:.2f} +/- {dist_std:.2f} kpc")
        plt.tight_layout()
        plt.savefig(f"{star_savepath}/{star_id}_dist_hist.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        i += 1
    
    # Combining all distance results into a single dataframe
    all_dist_df = pd.DataFrame(all_dist_res)    
    return all_dist_df
        


if __name__ == "__main__":
    
    # Loading in the data
    lit_df = pd.read_csv('/data/misfit/benlowe/xshooter/lit_abund_data/all_lit_combined_data.csv')
    # lit_df = lit_df[lit_df['star_id'] == 'SMSS J062609.83-590503.2']
    # lit_df = lit_df.iloc[[0,1,2,3,4,5,6,7,8,9]]  # Testing on a smaller subset
    
    # Savepath for distance results
    savepath = '/data/misfit/benlowe/xshooter/lit_abund_data/distances/'
    
    # Running code in parallel
    num_cores = 20
    lit_df_split = np.array_split(lit_df, num_cores)
    run_mcmc_dist_partial = partial(run_mcmc_dist, savepath=savepath)
    
    with Pool(num_cores) as pool:
        res_df = pool.map(run_mcmc_dist_partial, lit_df_split)
        
    # Combining the results from all processes
    final_res_df = pd.concat(res_df, ignore_index=True)
    final_res_df.to_csv(os.path.join(savepath, 'all_dist_res.csv'), index=False)
    
