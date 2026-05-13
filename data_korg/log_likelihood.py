import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy
import scipy.integrate
import scipy.stats
import pandas as pd
from multiprocessing import Pool
from functools import partial
import os
import ast

def lnL_obs(npix, flux_err, chisqrs):
    """Computes the log likelihood of a given observed dataset.

    Parameters
    ----------
    npix : int
        Number of pixels within the given wavelength region.
    flux_err : ndarray
        Array of observed flux errors.
    chisqrs : ndarray
        Array of chi-squared values.

    Returns
    -------
    ndarray
        Array of the observed log likelihoods for the dataset.
    """
    
    return -1/2 * (chisqrs * npix) + 1/2 * np.sum(np.log(np.sqrt(2 * np.pi) * flux_err))

def lnL_model(npix, flux_err, chisqrs):
    """Computes the log likelihood of a given model dataset.

    Parameters
    ----------
    npix : int
        Number of pixels within the given wavelength region.
    flux_err : ndarray
        Array of observed flux errors.
    chisqrs : ndarray
        Array of model chi-squared values.

    Returns
    -------
    ndarray
        Array of the observed log likelihoods for the dataset.
    """
    return -1/2 * (chisqrs * npix) + 1/2 * np.sum(np.log(np.sqrt(2 * np.pi) * flux_err))

def closest_val(array, x):
    """Within an array, finds the value closest to x.

    Parameters
    ----------
    array : ndarray
        Array of values.
    x : float
        Desired value to find closest entry to in array.

    Returns
    -------
    min_index : int
        Index of closest value to x in array.
    """
    arr_diff = np.abs(array - x) # Take x, then subtract that from every entry in the array
    min_index = np.argmin(arr_diff) # Finding the index of the smallest difference
    return min_index


def p_calc(lnL, abund_arr):
    """Performs the cumlative integral on a given log likelihood and [C/Fe] values.

    Parameters
    ----------
    lnL : ndarray
        Array of log likelihoods of dataset
    abund_arr : ndarray
        Array of [C/Fe] values.
        
    Returns
    -------
    p_norm : ndarray
        Array of normalized P values
    """
    
    # Performing the cumulative integral
    p = scipy.integrate.cumulative_trapezoid(y=np.exp(lnL - np.max(lnL)), x=10**abund_arr) # '- np.max(thing1)' to make numbers sensible.
    
    # Normalizing
    p_norm = p/np.max(p)
    
    return p_norm



def log_likelihood(star_paths, chisqr_path, save_path, bestfit_path, sigma):
    """Calculates the log likelihoods for each star and element, and saves the results.
    
    Parameters
    ----------
    star_paths : list
        List of paths to the stars.
    chisqr_path : str
        Path to the chisquare data.
    save_path : str
        Path to save the results.
    bestfit_path : str
        Path to the best fit abundances.
    sigma : float
        Sigma level for detection limit (e.g., 2 for 2-sigma).
        
    Returns
    -------
    None
        The function saves the results to the specified save_path.
        
    """
    
    for i, star in enumerate(star_paths):
        
        # Getting the star id
        star_id = star.split('/')[-2]
        print(f"Running {star_id}")
        
        # Making save directory for the star
        star_save_path = f"{save_path}{star_id}/"
        os.makedirs(star_save_path, exist_ok=True)
        
        # Identifying elements that had their chisquares measured (i.e. those that were identifed as uncertain for detection)
        # Do this by checking the chisquare path to see if the element folder exists
        els_path = chisqr_path + star_id 
        els = [el for el in os.listdir(els_path) if os.path.isdir(os.path.join(els_path, el))]
        els = ["C_I", "N_I"]
        print(f"Running log likelihoods for elements: {els}")

        # Looping through each element
        for el in els:
            print(f"Element: {el}")
            
            # Making a save directory for the element
            el_save_path = f"{star_save_path}{el}/"
            os.makedirs(el_save_path, exist_ok=True)
            
            # Loading in the observed and empty chisquared data 
            el_chisqrs_path = f"{els_path}/{el}/chisqrs.csv"
            el_chisqrs_df = pd.read_csv(el_chisqrs_path)
            el_xfe_abunds = el_chisqrs_df["xfe"].to_numpy()
            # el_xh_abunds = el_chisqrs_df["xh"].to_numpy()
            el_xh_abunds = el_xfe_abunds 
            el_obs_chisqrs = el_chisqrs_df["chisq_obs"].to_numpy()
            el_empty_chisqrs = el_chisqrs_df["chisq_empty"].to_numpy()

             
            # Extracting the observed flux errors 
            # They're all the same for each abundance, so we can just take the first one
            flux_err_path = f"{els_path}/{el}/synth_specs/"
            flux_err_file = [flux_err_path + f for f in os.listdir(flux_err_path) if f.endswith('.csv')][0]
            flux_err = pd.read_csv(flux_err_file)["obs_flux_err"].to_numpy()
            # flux_err = ast.literal_eval(pd.read_csv(flux_err_file)["obs_flux_err"].to_numpy()[0])
            npixels = len(flux_err)
            
            # Extracting the synthetic flux errors
            empty_flux_err_path = f"{els_path}/{el}/empty_specs/"
            empty_flux_err_file = [empty_flux_err_path + f for f in os.listdir(empty_flux_err_path) if f.endswith('.csv')][0]
            empty_flux_err = pd.read_csv(empty_flux_err_file)["empty_flux_err"].to_numpy()
            # empty_flux_err = flux_err
            
            # Getting the best-fit abundance for the element
            el_bestfit_path = f"{bestfit_path}{star_id}/{el}_abund.csv"
            el_bestfit_df = pd.read_csv(el_bestfit_path)

            el_bestfit_xfe = el_bestfit_df[f"{el}_xfe_it1"].to_numpy()[0]
            el_bestfit_xh = el_bestfit_df[f"{el}_xh_it1"].to_numpy()[0]
            el_bestfit_feh = el_bestfit_df[f"mh"].to_numpy()[0]
            

            # ---COMPUTING THE LOG LIKELIHOOD--- #
            # Calculating the log likelihood for the observed data
            star_lnL = lnL_obs(npix=npixels, flux_err=np.array(flux_err), chisqrs=el_obs_chisqrs)
            
            # Calculating the cumulative integral for the observed log likelihood
            star_pnorm = p_calc(lnL=star_lnL, abund_arr=el_xfe_abunds)
            
            # Determining P_norm between P(x=1) and P(x=-1)
            pnorm_m1 = scipy.stats.norm.cdf(x=-1)
            pnorm_0 = scipy.stats.norm.cdf(x=0)
            pnorm_p1 = scipy.stats.norm.cdf(x=1)
            
            # Using these P_norm values to find corresponding [C/Fe] values
            index_pnorm_m1 = closest_val(array=star_pnorm, x=pnorm_m1)
            star_abund_m1 = el_xfe_abunds[index_pnorm_m1]

            index_pnorm_0 = closest_val(array=star_pnorm, x=pnorm_0)
            star_abund_p0 = el_xfe_abunds[index_pnorm_0]
            
            index_pnorm_p1 = closest_val(array=star_pnorm, x=pnorm_p1)
            star_abund_p1 = el_xfe_abunds[index_pnorm_p1]
            
            # Finding correspondiong [C/Fe] 2sigma and 1 sigma errors
            star_abund_2sig_err = star_abund_p1 - star_abund_m1
            star_abund_1sig_err = star_abund_2sig_err/2
            

            # ---COMPUTING THE NULL HYPOTHESIS--- #
            # Calculating the log likelihood for the synthetic data
            synth_lnL = lnL_model(npix=npixels, flux_err=np.array(empty_flux_err), chisqrs=el_empty_chisqrs)
            
            # Calculating the cumulative integral
            synth_pnorm_xfemin = p_calc(lnL=synth_lnL, abund_arr=el_xfe_abunds)
            synth_pnorm_xhmin = p_calc(lnL=synth_lnL, abund_arr=el_xh_abunds)

            # Determining detection limit at P(x=2)
            pnorm_p2 = scipy.stats.norm.cdf(x=sigma)

            index_synth_pnorm_p2 = closest_val(array=synth_pnorm_xfemin, x=pnorm_p2)
            star_abund_p2 = el_xfe_abunds[index_synth_pnorm_p2] # 2sigma detection limit
            index_synth_pnorm_p2_xh = closest_val(array=synth_pnorm_xhmin, x=pnorm_p2)
            star_abund_p2_xh = el_xh_abunds[index_synth_pnorm_p2_xh]
            
            
            # ---DETERMINING IF LINE IS DETECTED--- #
            # If 2 sigma detection limit is greater than observed, then no detection. Otherwise detected.
            if star_abund_p2 >= star_abund_p0:
                detect_flag = 'N'
                
            if star_abund_p2 < star_abund_p0:
                detect_flag = 'Y'
                
            # # For stars with stupid values, we set the detection flag to 'N'
            # if el_bestfit_xfe <= -1.5:
            #     detect_flag = 'N'
                
            # Star ra_1656-1433_s143 is 100% a detection (through visible inspection), idk why it failed 
            if star_id == 'ra_1656-1433_s143':
                detect_flag = 'Y'
                el_bestfit_xfe = el_bestfit_df[f"{el}_xfe_it1"].to_numpy()[0]
                el_bestfit_xh = el_bestfit_df[f"{el}_xh_it1"].to_numpy()[0]

            
            # ---PLOTTING AND SAVING THE RESULTS--- #
            # Plotting the following:
            # 1. [X/Fe] vs Chisquares
            # 2. [X/Fe] vs Log likelihood 
            # 3. [X/Fe] vs P_norm
            fig, axs = plt.subplots(1, 4, figsize=(16, 6), tight_layout=True)
            axs[0].plot(el_xfe_abunds, el_obs_chisqrs, label='Observed Chisquares', color='darkblue')
            axs[0].plot(el_xfe_abunds, el_empty_chisqrs, label='Empty Chisquares', color='darkorange')
            axs[0].legend(fancybox=False)
            axs[0].set_ylabel(r'χ$^2$')
            
            axs[1].plot(el_xfe_abunds, el_obs_chisqrs, label='Observed Chisquares', color='darkblue')
            axs[1].plot(el_xfe_abunds, el_empty_chisqrs, label='Empty Chisquares', color='darkorange')
            axs[1].set_yscale('log')
            axs[1].legend(fancybox=False)
            
            axs[2].plot(el_xfe_abunds, star_lnL, label='Observed Log Likelihood', color='darkblue')
            axs[2].plot(el_xfe_abunds, synth_lnL, label='Empty Log Likelihood', color='darkorange')
            axs[2].set_ylabel(r'Log Likelihood')
            
            axs[3].plot(el_xfe_abunds[1:], star_pnorm, color='darkblue')
            axs[3].plot(el_xfe_abunds[1:], synth_pnorm_xfemin, color='darkorange')
            # axs[3].scatter(star_abund_m1, pnorm_m1, label='P(x=-1)', marker='x', color='black', s=100)
            axs[3].scatter(star_abund_p0, pnorm_0, label='P(x=0)', marker='x', color='black', s=100)
            # axs[3].scatter(star_abund_p1, pnorm_p1, label='P(x=1)', marker='x', color='black', s=100)
            # axs[23].scatter(star_abund_p2, pnorm_p2, label='P(x=2)', marker='x', color='black', s=00)
            axs[3].axvline(x=star_abund_p2, color='brown', linestyle='--', label='2 Sigma Detection Limit')
            axs[3].set_ylabel(r'P_norm')
            # axs[3].legend(fancybox=False)
            
            
            fig.suptitle(t = f"Detection: {detect_flag}\n{sigma}σ Detection limit = {star_abund_p2:.2f}")
            fig.supxlabel(r'[X/Fe]')
            
            plot_savepath = f"{el_save_path}log_likelihood.png"
            plt.savefig(plot_savepath, dpi=300, transparent=False, facecolor='white')
            plt.close()
            
            
            # ---SAVING THE RESULTS--- #
            # Saving the log likelihood results
            el_loglikelihood_df = pd.DataFrame({
                'star_id': star_id,
                'el': el,
                'detect_flag': detect_flag,
                'star_abund_m1': star_abund_m1,
                'star_abund_p0': star_abund_p0,
                'star_abund_p1': star_abund_p1,
                'star_abund_2sig_err': star_abund_2sig_err,
                'star_abund_1sig_err': star_abund_1sig_err,
                'star_abund_p2': star_abund_p2
            }, index=[0])
            el_loglikelihood_df.to_csv(f"{el_save_path}{el}_log_likelihoods.csv", index=False)

            # Saving the detection results
            # bestfit_abund_xfe = star_abund_p0
            # bestfit_abund_xh = star_abund_p0 + el_bestfit_feh
            
            if el == 'N_I':
                line_peak = 3360.22
                
            if el == 'C_I':
                line_peak = 4300.313
                
                # Adding evolutionary corrections
                carb_df = pd.read_csv("/data/misfit/benlowe/xshooter/p113/C_evolve_corrections/cfe_corrections.csv")
                carb_corrections = carb_df[carb_df['star_id'] == star_id]['cfe_corr'].to_numpy()[0]
                # bestfit_abund_xfe += carb_corrections
                # bestfit_abund_xh = bestfit_abund_xfe + el_bestfit_feh
                
            
            el_detection_df = pd.DataFrame({
                'el': el,
                'line_peak': line_peak,
                'detection': detect_flag,
                'min_abund_xh': star_abund_p2_xh,
                'min_abund_xfe': star_abund_p2,
                'bestfit_abund_xh': el_bestfit_xh,
                'bestfit_abund_xfe': el_bestfit_xfe}, index=[0]) # el_bestfit_feh
            el_detection_df.to_csv(f"{star_save_path}{el}_abunds.csv", index=False)
    
    return None
    
    
    
    
    
if __name__ == '__main__':
    
    # ---INPUTS--- #
    # Getting the list of star paths
    path = '/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter/'
    star_paths = [path + star + '/' for star in os.listdir(path) if os.path.isdir(os.path.join(path, star))] # and star == 'ra_1633-2814_s284'

    # Path to the chisquare data
    # chisqr_path = '/data/misfit/benlowe/xshooter/korg_data/upper_limits_iter/tests/newsynth_test_mask/'
    chisqr_path = '/data/misfit/benlowe/xshooter/p113/korg_data/upper_limits_v6/'
    
    # Path to the best fit abundances
    bestfit_path = '/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter9/'
    
    # Setting up the savepath (will store with the chisquare data)
    savepath = chisqr_path
    
    # Sigma level for detection limit
    sigma = 3
    
    
    # ---RUNNING THE CODE--- #
    # Running parallel processing    
    num_cores = 8
    star_paths_split = np.array_split(star_paths, num_cores)  # Split DataFrame into equal parts
    run_partial = partial(log_likelihood, save_path=savepath, chisqr_path=chisqr_path, bestfit_path=bestfit_path, sigma=sigma)

    with Pool(processes=num_cores) as pool:  
        pool.map(run_partial, star_paths_split)
    
    # # Combining
    # final_info_df = pd.concat(info_df, ignore_index=True)
    
    # # Saving
    # final_info_df.to_csv(save_path + 'star_info.csv', index=False) 
    # print('Done!')    