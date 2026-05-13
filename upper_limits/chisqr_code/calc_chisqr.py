import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import bokeh.plotting as bok
import os
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from PyAstronomy import pyasl
from scipy.signal import correlate
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
from astroscrappy import detect_cosmics
import ast
from scipy.stats import chisquare
import scipy
import scipy.integrate
import scipy.stats


def get_chisqr(masks_df, el, synth_path, savepath, plots=True):
    """Calculates the chi-square between observed and synthetic spectra for a given element and set of stars. Will apply masks 
    to only include certain wavelength regions.

    Parameters
    ----------
    masks_df : pd.DataFrame
        DataFrame containing the masks for each star. Must contain columns 'star_id', '{el}_maskmin', and '{el}_maskmax'.
    el : str
        Element to calculate chi-square for. Options are 'N_I' or 'C_I'.
    synth_path : str
        Path to the synthetic spectra directory. Should contain subdirectories for each star and element.
    savepath : str
        Path to save the chi-square results and plots.
    plots : bool, optional
        Whether to make and save fit plots, by default True

    Returns
    -------
    None
    """

    stars_list = masks_df['star_id'].values
    
    for star in stars_list:
    
        print(f"Star {star}")
        
        # Extracting the mask for the star
        if el == 'N_I':
            el_ = 'nh'
        elif el == 'C_I':
            el_ = 'ch'
        
        star_maskmin = masks_df[masks_df['star_id'] == star][f'{el_}_maskmin'].values[0]
        star_maskmax = masks_df[masks_df['star_id'] == star][f'{el_}_maskmax'].values[0]
        masks = [(star_maskmin, star_maskmax)]
        
        # Getting the list of synthetic spectra
        path = f'{synth_path}{star}/{el}/synth_specs/'
        specs = [path + f for f in os.listdir(path) if f.endswith('.csv') and not f.startswith(el)]
        specs.sort()
        
        # Loading in the observed spectra (all the same, so will take the first entry)
        obs_df = pd.read_csv(specs[0])
        obs_flux = obs_df['obs_flux']
        obs_flux_err = obs_df['obs_flux_err']
        obs_wave = obs_df['obs_wave']    
            
        # Loading in the empty spectra
        empty_path = f'{synth_path}{star}/{el}/empty_specs/'
        empty_specs = [empty_path + f for f in os.listdir(empty_path) if f.endswith('.csv') and not f.startswith(el)]
        empty_df = pd.read_csv(empty_specs[0])
        empty_flux = empty_df['empty_flux']
        empty_wave = empty_df['empty_wave']
        
        # Setting up save directories
        star_savepath = f"{savepath}{star}/"
        os.makedirs(star_savepath, exist_ok=True)
        
        el_savepath = f"{star_savepath}{el}/"
        os.makedirs(el_savepath, exist_ok=True)
        
        el_empty_savepath = f"{el_savepath}empty_specs/"
        os.makedirs(el_empty_savepath, exist_ok=True)
        
        el_obs_savepath = f"{el_savepath}synth_specs/"
        os.makedirs(el_obs_savepath, exist_ok=True)
        
        # Looping through each synthetic spectra
        el_ = el.split('_')[0]
        chisq_all = []
        for f in specs:

            # Loading in the synthetic spectra
            synth_abund = f.split('/')[-1].split('_')[0]
            print(f"Running [{el_}/Fe] = {synth_abund}")
            
            synth_df = pd.read_csv(f)
            synth_flux = synth_df['synth_flux'].values
            synth_wave = synth_df['synth_wave'].values
            
            # Masking out regions from masks
            mask_obs = np.ones_like(obs_wave, dtype=bool)
            mask_synth = np.ones_like(synth_wave, dtype=bool)
            mask_empty = np.ones_like(synth_wave, dtype=bool)
            for start, end in masks:
                mask_obs &= ((obs_wave >= start) & (obs_wave <= end))
                mask_synth &= ((synth_wave >= start) & (synth_wave <= end))
                mask_empty &= ((empty_wave >= start) & (empty_wave <= end))
                        
            # Calculating chi-square only in unmasked regions
            chisq_obs = np.sum((obs_flux[mask_obs] - synth_flux[mask_obs])**2 / obs_flux_err[mask_obs]**2)
            chisq_red_obs = chisq_obs / len(obs_flux[mask_obs])
                    
            chisq_empty = np.sum((empty_flux[mask_empty] - synth_flux[mask_empty])**2 / obs_flux_err[mask_empty]**2)        

            # Storing into dictionary
            chisq_dict = {'star_id': star, 'element': el, 'xfe': synth_abund, 
                        'chisq_obs': chisq_obs, 'chisq_red_obs': chisq_red_obs, 'chisq_empty': chisq_empty}
            chisq_all.append(chisq_dict)
            
            if plots:
                # Plotting the observed and synthetic spectra with masked-out regions
                plt.figure(figsize=(10,6))
                plt.plot(obs_wave, obs_flux, label='Observed', color='black')
                plt.errorbar(obs_wave, obs_flux, yerr=obs_flux_err, fmt='o', markersize=2, color='grey', alpha=0.5)
                plt.plot(synth_wave, synth_flux, label=f'[{el_}/Fe] = {synth_abund}]\nChisqr = {chisq_obs:.2f}', color='chocolate')

                ymin, ymax = plt.gca().get_ylim()        
                for start, end in masks:
                    fill_range = (obs_wave >= start) & (obs_wave <= end)
                    plt.fill_between(obs_wave[fill_range], y1=ymin, y2=ymax, color="grey", alpha=0.5)
                
                plt.legend(fancybox=False)
                plt.xlabel('Wavelength (Å)')
                plt.ylabel('Normalised Flux')
                plt.tight_layout()
                plt.savefig(f"{el_obs_savepath}{synth_abund}_spec.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                
                # Plotting the empty and synthetic spectra with masked-out regions
                plt.figure(figsize=(10,6))
                plt.plot(empty_wave, empty_flux, label='Empty', color='black')
                plt.errorbar(empty_wave, empty_flux, yerr=obs_flux_err, fmt='o', markersize=2, color='grey', alpha=0.5)
                plt.plot(synth_wave, synth_flux, label=f'[{el_}/Fe] = {synth_abund}]\nChisqr = {chisq_empty:.2f}', color='chocolate')

                ymin, ymax = plt.gca().get_ylim()        
                for start, end in masks:
                    fill_range = (empty_wave >= start) & (empty_wave <= end)
                    plt.fill_between(empty_wave[fill_range], y1=ymin, y2=ymax, color="grey", alpha=0.5)
                
                plt.legend(fancybox=False)
                plt.xlabel('Wavelength (Å)')
                plt.ylabel('Normalised Flux')
                plt.tight_layout()
                plt.savefig(f"{el_empty_savepath}{synth_abund}_spec.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                pass

        # Saving the chi-square results to a CSV file
        chisq_df = pd.DataFrame(chisq_all)
        chisq_df.to_csv(f"{el_savepath}chisqrs.csv", index=False)
    
    return None


def delta_chisqr(path_list, el):
    
    for path in path_list:
        # Loading in the chi-square results
        star = path.split('/')[-1]
        chisq_df = pd.read_csv(f"{path}/{el}/chisqrs.csv")
        chisq_df.sort_values(by='xfe', inplace=True)
        
        print(f"Running star {star}")
        
        # Extracting the chisquare, finding the minimum, then shifting all by that
        # For observed
        chisq_obs = chisq_df['chisq_obs'].values
        min_chisq_obs = np.min(chisq_obs)
        min_abund_obs = chisq_df['xfe'][np.argmin(chisq_obs)]
        delta_chisq_obs = chisq_obs - min_chisq_obs
        
        # For empty
        chisq_empty = chisq_df['chisq_empty'].values
        min_chisq_empty = np.min(chisq_empty)
        min_abund_empty = chisq_df['xfe'][np.argmin(chisq_empty)]
        delta_chisq_empty = chisq_empty - min_chisq_empty
        
        # Getting the 1-, 2- and 3-sigma confidence levels, then identifying the abundance values at those levels
        conf_levels = [1.0, 4.0, 9.0]
        conf_abunds_obs = []
        conf_abunds_empty = []
        for level in conf_levels:
            
            # For observed
            indices_obs = np.where(delta_chisq_obs <= level)[0]
            if len(indices_obs) > 0:
                conf_abund = chisq_df['xfe'].values[indices_obs[-1]]
                conf_abunds_obs.append(conf_abund)
            else:
                conf_abunds_obs.append(None)
                
            # For empty
            indices_empty = np.where(delta_chisq_empty <= level)[0]
            if len(indices_empty) > 0:
                conf_abund = chisq_df['xfe'].values[indices_empty[-1]]
                conf_abunds_empty.append(conf_abund)
            else:
                conf_abunds_empty.append(None)
                
                    
        # Plotting
        # For observed
        xfe_val = chisq_df['xfe'].values.astype(float)
        plt.figure(figsize=(8,5))
        plt.plot(xfe_val, delta_chisq_obs, color='black', linewidth=0.7, label='Observed')
        plt.plot(xfe_val, delta_chisq_empty, color='black', linewidth=1, linestyle='dotted', label='Empty')
            
        colors = ['darkgreen', 'darkorange', 'darkred']
        for i, level in enumerate(conf_levels):
            abund = conf_abunds_empty[i]
            sigma = i + 1
            
            plt.hlines(level, xmin=-4, xmax=conf_abunds_obs[-1]+1, linestyles='dashed', label=f"{sigma}σ = {abund:.2f}", colors=colors[i])
        plt.ylim(-0.5, 10)
        plt.xlim(left=-4, right=conf_abunds_obs[-1]+1)
        plt.legend(fancybox=False, loc='upper left')
        plt.title(f"{star} - {el}")
        plt.xlabel(f"[{el.split('_')[0]}/Fe]")
        plt.ylabel(r"$\Delta \chi^2$")
        plt.tight_layout()
        plt.savefig(f"{path}/{el}/delta_chisq.png", dpi=300, bbox_inches='tight')

        # # For empty
        # plt.figure(figsize=(8,5))
        # # plt.scatter(min_abund, 0, color='red')
        
        # colors = ['darkgreen', 'darkorange', 'darkred']
        # for i, level in enumerate(conf_levels):
        #     abund = conf_abunds_empty[i]
        #     sigma = i + 1
            
        #     plt.hlines(level, xmin=-4, xmax=conf_abunds_empty[-1]+1, linestyles='dashed', label=f"{sigma}σ = {abund:.2f}", colors=colors[i])
        # plt.ylim(-0.5, 10)
        # plt.xlim(left=-4, right=conf_abunds_empty[-1]+1)
        # plt.legend(fancybox=False)
        # plt.title(f"{star} - {el} (Empty)")
        # plt.xlabel(f"[{el.split('_')[0]}/Fe]")
        # plt.ylabel(r"$\Delta \chi^2$")
        # plt.tight_layout()
        # plt.savefig(f"{path}/{el}/delta_chisq_empty.png", dpi=300, bbox_inches='tight')
    
    return None
