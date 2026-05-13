import numpy as np
from astropy.io import fits
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from astropy.io import fits
from astropy.table import Table
import bokeh.plotting as bok
from bokeh.models import Span
from bokeh.palettes import Category10
from lmfit.models import VoigtModel
from PyAstronomy.pyasl import dopplerShift
from multiprocessing import Pool
from scipy.signal import medfilt
from functools import partial
from tqdm import tqdm
import warnings
import sys
from scipy.signal import correlate

warnings.simplefilter("ignore", RuntimeWarning)


def mkdir(dir_path):
    """Makes a directory at a given path. If it already exists, then it will skip.

    Parameters
    ----------
    dir_name : str
        Pathway to directory.
    """
    
    os.makedirs(dir_path, exist_ok=True)
    
    return None

def grab_fits(star_path):
    """
    Grabs the UVB, VIS and NIR fits files for a given star.
    
    Parameters
    ----------
    star_path : str
        Path to the directory containing the UVB, VIS and NIR data.
    
    Returns
    -------
    uvb_fits : list
        List of UVB fits files.
    vis_fits : list
        List of VIS fits files.
    nir_fits : list
        List of NIR fits files.
    """
    
    # Getting UVB data
    uvb_path = star_path + 'uvb/'
    uvb_fits = [uvb_path + uvb for uvb in os.listdir(uvb_path) if uvb[-5:] == '.fits']

    # Getting VIS data
    vis_path = star_path + 'vis/'
    vis_fits = [vis_path + vis for vis in os.listdir(vis_path) if vis[-5:] == '.fits']
    
    # Getting NIR data
    nir_path = star_path + 'nir/'
    nir_fits = [nir_path + nir for nir in os.listdir(nir_path) if nir[-5:] == '.fits']
    
    return uvb_fits, vis_fits, nir_fits


def grab_norm(star_norm_path):
    """
    Grabs the UVB, VIS and NIR .csv norm files for a given star.
    
    Parameters
    ----------
    star_path : str
        Path to the directory containing the UVB, VIS and NIR data.
    
    Returns
    -------
    uvb_fits : list
        List of UVB fits files.
    vis_fits : list
        List of VIS fits files.
    nir_fits : list
        List of NIR fits files.
    """
    
    # Getting UVB data
    uvb_path = star_norm_path + 'uvb/'
    uvb_csv = [uvb_path + uvb for uvb in os.listdir(uvb_path) if uvb[-4:] == '.csv']

    # Getting VIS data
    vis_path = star_norm_path + 'vis/'
    vis_csv = [vis_path + vis for vis in os.listdir(vis_path) if vis[-4:] == '.csv']
    
    # Getting NIR data
    nir_path = star_norm_path + 'nir/'
    nir_csv = [nir_path + nir for nir in os.listdir(nir_path) if nir[-4:] == '.csv']
    
    return uvb_csv, vis_csv, nir_csv


def get_data(fits_path, norm_path):
    """Getting data from .fits files for a given star. If normalised data is provided, will use that instead.

    Parameters
    ----------
    fits_path : list
        List of paths to each .fits file for the star.
    norm_path : list, optional
        List of paths to each .csv file for the normalised star, by default None

    Returns
    -------
    df : pandas.DataFrame
        Pandas dataframe containing the data from each .fits file.
    fits_headers : dict
        Dictionary containing the header information of the star.
    """
    obs = []
    for i, (star_fits, star_norm) in enumerate(zip(fits_path, norm_path)):
        
        # Opening the .fits file
        star_fit = fits.open(star_fits)
        
        # Extracting S/N and starname information
        star_id = star_fit[0].header['OBJECT']
        snr = star_fit[0].header['SNR']
        ra = star_fit[0].header['RA']
        dec = star_fit[0].header['DEC']
        rv_barycor = star_fit[0].header['HIERARCH ESO QC VRAD BARYCOR']
        rv_helicor = star_fit[0].header['HIERARCH ESO QC VRAD HELICOR']
        fits_headers = {'star_id': star_id, 'snr': snr, 'ra': ra, 'dec': dec, 'rv_barycor': rv_barycor, 'rv_heliocor': rv_helicor}
        
        # Extracting wavelength, flux and flux error data
        ob = 'OB' + str(i + 1)
        
        wave = star_fit[1].data['WAVE'][0] * 10
        flux = star_fit[1].data['FLUX_REDUCED'][0]
        flux_err = star_fit[1].data['ERR_REDUCED'][0]
        flux_weight = 1 / (flux_err**2)
        
        # Saving to dictionary, then appending to list
        norm_df = pd.read_csv(star_norm)
        
        wave_norm = norm_df['wave'].to_numpy()
        flux_norm = norm_df['normed_flux'].to_numpy()
        
        
        wave_index = np.where((wave >= wave_norm[0]) & (wave <= wave_norm[-1]))[0]
        
        smooth_cont = flux[wave_index]/flux_norm
        flux_err_cut = flux_err[wave_index]
        flux_err_norm = flux_err_cut/smooth_cont
        flux_weight_norm = 1 / (flux_err_norm**2)
        
        ith_ob = dict(ob=ob, star_id=star_id, snr=snr, wave=wave, flux=flux, flux_err=flux_err, flux_weight=flux_weight, wave_norm=wave_norm, flux_norm=flux_norm, flux_err_norm=flux_err_norm, flux_weight_norm=flux_weight_norm)
        obs.append(ith_ob)
        
    # Turning into pandas dataframe
    df = pd.DataFrame(obs)
    
    return df, fits_headers

def html_plotter(x_data, y_data, x_label, y_label, title, filename, path):
    """Creates and saves interactive .html plots using the bokeh package.

    Parameters
    ----------
    x_data : ndarray
        Data on the x-axis
    y_data : ndarray
        Data on the y-axis
    x_label : str
        Label for the x-axis
    y_label : str
        Label for the y-axis
    title : str
        Title to be put at the top of the plot
    filename : str
        Name of the saved .html file.
    path : str
        Location for where to save the plot

    Returns
    -------
    None
    """
    # Defining the bokeh figure
    p = bok.figure(title=title, x_axis_label=x_label, y_axis_label=y_label, width = 1000)
    p.sizing_mode = 'scale_height'
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.title.text_font_size = "16pt"

    # Plotting the spectra
    p.line(x_data, y_data, line_width=1.3, color='black')

    # Saving the spectra to the given directory
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bok.save(obj=p, filename=path + filename, title=title)
    
    return None


def cross_corr(star_df, minwave, maxwave, plot=False):
    """Using exposure with highest S/N, cross-correlating with all other exposures in star_df for a given wavelength region. Recommend strong absorption features.

    Parameters
    ----------
    star_df : pandas.DataFrame
        Dataframe containing all spectra for a given star.
    minwave : float
        Minimum wavelength for cross-correlation
    maxwave : float
        Maximum wavelength for cross-correlation
    plot : bool, optional
        To make test plots showing cross-correlation over sliced region for all exposures, by default False.

    Returns
    -------
    star_df : pandas.DataFrame
        Pandas dataframe containing spectral data alongside shifted spectra for all exposures.
    """
    
    # Selecting exposure with highest S/N to be the reference spectra
    star_id = star_df['star_id'].to_numpy()[0]
    
    ref_exp = star_df[star_df['snr'] == star_df['snr'].max()]
    ref_ob = ref_exp['ob'].to_numpy()[0]
    ref_wave = ref_exp['wave_norm'].to_numpy()[0]
    ref_flux = ref_exp['flux_norm'].to_numpy()[0]
        
    waveindex = np.where((ref_wave >= minwave) & (ref_wave <= maxwave))[0]
    ref_wavecut = ref_wave[waveindex]
    ref_fluxcut = ref_flux[waveindex]

    # Cross-correlating over each exposure in star_df
    dict_list = []
    for i, exp in star_df.iterrows():
        
        # Getting individual exposure
        exp_ob = exp['ob']
        exp_wave = exp['wave_norm']
        exp_flux = exp['flux_norm']
        exp_err = exp['flux_err_norm']
        
        # Cutting wavelengths to match reference for shifting
        exp_wavecut = exp_wave[waveindex]
        exp_fluxcut = exp_flux[waveindex]
        exp_errcut = exp_err[waveindex]

        # Cross-correlating by taking peak absorption feature, then shifting to template
        peak_ref = np.argmin(ref_fluxcut)
        peak_exp = np.argmin(exp_fluxcut)
        manual_shift = peak_ref - peak_exp
        # ccf = correlate(exp_fluxcut, ref_fluxcut, mode='full')
        # manual_shift = np.argmax(ccf) - (len(ref_flux) - 1)

        # Applying shift to whole spectra. If shift is 0, then will use original spectrum. Otherwise will shift using np.roll
        if manual_shift == 0:
            spec_shift = exp_flux
            spec_err_shift = exp_err
        else:
            spec_shift = np.roll(exp_flux, manual_shift)
            spec_err_shift = np.roll(exp_err, manual_shift)
           
        # Storing into dictionary
        ob_dict = dict(ob=exp_ob, shift=manual_shift, spec_shift=spec_shift, spec_err_shift=spec_err_shift)
        dict_list.append(ob_dict)
        
    # Storing data into pandas.DataFrame, then merging with star_df
    df_shift = pd.DataFrame(dict_list)
    star_df = pd.merge(star_df, df_shift, how='inner', on='ob')
    
    # Plotting results
    if plot:
        plt.figure(figsize=[8, 6])
        
        for i, exp in star_df.iterrows():
            exp_ob = exp['ob']
            exp_wavecut = exp['wave'][waveindex]
            exp_spec_shift = exp['spec_shift'][waveindex]
            
            if ref_ob == exp_ob:
                plt.plot(exp_wavecut, exp_spec_shift, label=exp['ob'] + ' (ref)')
            else:
                plt.plot(exp_wavecut, exp_spec_shift, label=exp['ob'])
        
        plt.legend(fancybox=False)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Normalised Flux')
        plt.title(star_id)
        plt.tight_layout()

        plt.savefig(plot, dpi=300, transparent=False, facecolor='white')
        plt.close()
        
    return star_df, ref_ob


def stack_spec(star_df, plot=None):
    """Stacks spectra for all exposures of a star using weighted averages.

    Parameters
    ----------
    star_df : pandas.DataFrame
        DataFrame containing spectral data for all exposures of a star, including wavelength, flux, flux error, 
        and signal-to-noise ratio (SNR).
        
    Returns
    -------
    stack_df : pandas.DataFrame
        DataFrame containing the stacked spectral data
    """
    
    # Loading in the data
    all_waves = star_df['wave_norm'].to_numpy()[0]
    all_flux = star_df['flux_no_cosmic'].to_list() # cosmic rays removed
    all_flux_err = star_df['flux_err_no_cosmic'].to_list() # cosmic rays removed
    all_weights = star_df['weights'].to_list()
    all_snr = star_df['snr'].to_numpy()
    star_id = star_df['star_id'].to_numpy()[0]
    
    # Using pandas to compute weighted average for those with multiple entries
    # Pandas automatically deals with nan entries, so taking average of [1, 2, np.nan] and [3, 2, 1] will 
    # yield [1.5, 2, 1]. np.average does not do this, hence using this approach
    flux_df = pd.DataFrame(all_flux)
    flux_err_df = pd.DataFrame(all_flux_err)
    weights_df = pd.DataFrame(all_weights)

    stack_spec = (flux_df * weights_df).sum(axis=0) / weights_df.sum(axis=0)
    stack_spec_err = np.sqrt((flux_err_df**2).sum(axis=0))/len(flux_err_df) # Computing errors of errors by addature in quadature
    
    # For stars with single exposures (or those badly destroyed by cosmics in all exposures), will replace nans with average of nearby points
    if stack_spec.isnull().any():
        
        # Fluxes        
        spec_forward = stack_spec.ffill()  # Propagate last valid value forward
        spec_backward = stack_spec.bfill()  # Propagate next valid value backward
        stack_spec = stack_spec.fillna((spec_forward + spec_backward) / 2).to_numpy()
        
        # Flux errors
        flux_err_series = pd.Series(all_flux_err[0])
        flux_err_forward = flux_err_series.ffill()  # Propagate last valid value forward
        flux_err_backward = flux_err_series.bfill()  # Propagate next valid value backward
        stack_spec_err = flux_err_series.fillna(np.sqrt((flux_err_forward**2 + flux_err_backward**2))).to_numpy()
    
    # Creating new stacked dataframe
    stack_snr = np.sqrt(np.sum(all_snr**2))
    stack_df = pd.DataFrame({'star_id': star_id, 'snr': stack_snr, 'wave': [all_waves], 'flux': [np.array(stack_spec)], 'flux_err': [np.array(stack_spec_err)]})
    
    # Plotting results if requested
    if plot is not None:
        p = bok.figure(x_axis_label='Wavelength (Å)', y_axis_label='Normalised Flux', width = 1000, y_range=(0,1.5))
        p.sizing_mode = 'scale_height'
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"
        p.title.text_font_size = "16pt"
        
        colors = Category10[10]  
        for i, exp in star_df.iterrows():
            
            # Extracting out the data
            wave = exp['wave_norm']
            flux = exp['spec_shift']
            snr = exp['snr']

            p.line(wave, flux, line_width=1.3, legend_label='OB {0} (SNR = {1:.2f})'.format(i+1, snr), line_color=colors[i], line_alpha=0.35)
        
        p.line(stack_df['wave'].to_numpy()[0], stack_df['flux'].to_numpy()[0], line_width=1.3, 
               legend_label='Stacked (SNR = {:.2f})'.format(stack_snr), line_color='black')
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bok.save(obj=p, filename=plot)
        
        # plt.figure(figsize=[8,6])
    
        # for k, uvb_exp in uvb_df.iterrows():
        #     wave = uvb_exp['wave_norm']
        #     flux = uvb_exp['spec_shift']
        #     flux_no_cosmic = uvb_exp['flux_no_cosmic']
            
        #     # reg = np.where((wave >= 4961) & (wave <= 4964))[0]
        #     # reg = np.where((wave >= 4950) & (wave <= 4965))[0]
        #     # reg = np.where((wave >= 5268) & (wave <= 5271))[0] 
        #     reg = np.where((wave >= 4924) & (wave <= 4928))[0]  
        #     wave_reg = wave[reg]
        #     flux_reg = flux[reg]
        #     flux_no_reg = flux_no_cosmic[reg]
            
        #     plt.plot(wave_reg, flux_no_reg, label='OB {}'.format(k+1), alpha=0.5)
            
        # plt.plot(uvb_stack_df['wave'].to_numpy()[0][reg], uvb_stack_df['flux'].to_numpy()[0][reg], color='black', label='Stacked')
        # plt.legend(fancybox=False)
        # plt.savefig('/data/misfit/benlowe/xshooter/reduced_data/cosmic_ray_tests/noise_cut/4924-4928_v3.png', dpi=300, transparent=False, facecolor='white')


    return stack_df


def spec_save(uvb_df, vis_df, rv_df, fits_headers, save_path):
    """Saves the stacked spectral data for each star into a .fits file and 
    generates a .html plot for each arm.

    Parameters
    ----------
    uvb_df : pandas dataframe
        Contains the stacked spectral data for the UVB arm
    vis_df : pandas dataframe
        Contains the stacked spectral data for the VIS arm
    rv_df : pandas dataframe
        Contains the radial velocities for the UVB and VIS arms
    fits_headers : dict
        Contains the header information for the .fits files
    save_path : str
        The path to save the .fits and .html files

    Returns
    -------
    None
    """
    
    # ---SAVING INTO .FITS FILES--- #    
    # Making up the PrimaryHDU headers
    hdr = fits.Header() 
    hdr['star_id'] = (fits_headers['star_id'], 'Unique star identifier')
    hdr['snr_uvb'] = (float(uvb_df['snr'].iloc[0]), 'Signal-to-noise ratio in UVB')
    hdr['snr_vis'] = (float(vis_df['snr'].iloc[0]), 'Signal-to-noise ratio in VIS')
    hdr['ra'] = (fits_headers['ra'], 'Right Ascension')
    hdr['dec'] = (fits_headers['dec'], 'Declination')
    hdr['barycor'] = (fits_headers['rv_barycor'], 'Barycentric radial velocity')
    hdr['heliocor'] = (fits_headers['rv_heliocor'], 'Heliocentric radial velocity')
    hdr['rv_avg'] = (float(rv_df['rv_avg'].iloc[0]), 'Average radial velocity')
    hdr['erv_avg'] = (float(rv_df['rv_avg_err'].iloc[0]), 'Average radial velocity err')
    hdr['rv_uvb'] = (float(rv_df['rv_uvb'].iloc[0]), 'UVB radial velocity')
    hdr['erv_uvb'] = (float(rv_df['rv_uvb_err'].iloc[0]), 'UVB radial velocity err')
    hdr['rv_vis'] = (float(rv_df['rv_vis'].iloc[0]), 'VIS radial velocity')
    hdr['erv_vis'] = (float(rv_df['rv_vis_err'].iloc[0]), 'VIS radial velocity err')
    hdr['rv_hd'] = (float(rv_df['rv_hdelta'].iloc[0]), 'Hdelta radial velocity')
    hdr['erv_hd'] = (float(rv_df['rv_hdelta_err'].iloc[0]), 'Hdelta radial velocity err')
    hdr['rv_hg'] = (float(rv_df['rv_hgamma'].iloc[0]), 'Hgamma radial velocity')
    hdr['erv_hg'] = (float(rv_df['rv_hgamma_err'].iloc[0]), 'Hgamma radial velocity err')
    hdr['rv_hb'] = (float(rv_df['rv_hbeta'].iloc[0]), 'Hbeta radial velocity')
    hdr['erv_hb'] = (float(rv_df['rv_hbeta_err'].iloc[0]), 'Hbeta radial velocity err')
    hdr['rv_ha'] = (float(rv_df['rv_halpha'].iloc[0]), 'Halpha radial velocity')
    hdr['erv_ha'] = (float(rv_df['rv_halpha_err'].iloc[0]), 'Halpha radial velocity err')
    hdr['rv_c849'] = (float(rv_df['rv_caii849'].iloc[0]), 'CaII849 radial velocity')
    hdr['erv_c849'] = (float(rv_df['rv_caii849_err'].iloc[0]), 'CaII849 radial velocity err')
    hdr['rv_c854'] = (float(rv_df['rv_caii854'].iloc[0]), 'CaII854 radial velocity')
    hdr['erv_c854'] = (float(rv_df['rv_caii854_err'].iloc[0]), 'CaII854 radial velocity err')
    hdr['rv_c866'] = (float(rv_df['rv_caii866'].iloc[0]), 'CaII866 radial velocity')
    hdr['erv_c866'] = (float(rv_df['rv_caii866_err'].iloc[0]), 'CaII866 radial velocity err')
    
    
    # Saving spectral data for UVB arm
    uvb_wave = uvb_df['wave_shift'].to_numpy()[0]
    uvb_flux = uvb_df['flux'].to_numpy()[0]
    uvb_flux_err = uvb_df['flux_err'].to_numpy()[0]

    uvb_tab = Table([uvb_wave, uvb_flux, uvb_flux_err], names=['wave', 'flux', 'flux_err'])
    uvb_hdu = fits.BinTableHDU(data=uvb_tab, name='UVB data')
    
    # Saving spectral data for VIS arm
    vis_wave = vis_df['wave_shift'].to_numpy()[0]
    vis_flux = vis_df['flux'].to_numpy()[0]
    vis_flux_err = vis_df['flux_err'].to_numpy()[0]
    
    vis_wave_index = np.where(vis_wave >= 5600)[0]
    vis_wave = vis_wave[vis_wave_index]
    vis_flux = vis_flux[vis_wave_index]
    vis_flux_err = vis_flux_err[vis_wave_index]
    
    vis_tab = Table([vis_wave, vis_flux, vis_flux_err], names=['wave', 'flux', 'flux_err'])
    vis_hdu = fits.BinTableHDU(data=vis_tab, name='VIS data')
    
    # Combining tables, then saving
    hdu_primary = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([hdu_primary, uvb_hdu, vis_hdu])
    
    spec_savepath = save_path + fits_headers['star_id'] + '_data.fits'
    hdul.writeto(spec_savepath, overwrite=True)
        
    
    # Making .html plots 
    html_plotter(x_data=uvb_wave, 
                 y_data=uvb_flux, 
                 x_label='Wavelength (Å)', 
                 y_label='Normalised Flux', 
                 title='{0} ({1}, SNR = {2:.2f})'.format(fits_headers['star_id'], 'UVB', float(uvb_df['snr'].iloc[0])), 
                 filename=fits_headers['star_id'] + '_uvb.html', 
                 path=save_path)
    
    html_plotter(x_data=vis_wave, 
                 y_data=vis_flux, 
                 x_label='Wavelength (Å)', 
                 y_label='Normalised Flux', 
                 title='{0} ({1}, SNR = {2:.2f})'.format(fits_headers['star_id'], 'VIS', float(vis_df['snr'].iloc[0])), 
                 filename=fits_headers['star_id'] + '_vis.html', 
                 path=save_path)    
    
    return None


def spec_norm(wave, flux, wave_normmin, wave_normmax, cont_region, plot_path=False):
    """Normalize a spectrum within a given wavelength range, using a 2nd degree polynomial fit to the continuum.

    Parameters
    ----------
    wave : numpy.ndarray
        Wavelengths of the spectrum.
    flux : numpy.ndarray
        Fluxes of the spectrum.
    wave_normmin : float
        Minimum wavelength for normalization.
    wave_normmax : float
        Maximum wavelength for normalization.
    cont_region : list of tuples
        List of continuum regions, given as [(start1, end1), (start2, end2), ...].
    plot_path : str, optional
        Directory path to save plots of the continuum fit and normalised spectrum. If not given, no plots will be generated.

    Returns
    -------
    wavecut : numpy.ndarray
        Wavelengths of the spectrum, sliced to only include the desired region.
    fluxnorm : numpy.ndarray
        Fluxes of the spectrum, normalised to the continuum fit.
    """
    
    # Slicing spectra to only include desired region
    region_index = np.where((wave >= wave_normmin) & (wave <= wave_normmax))[0]
    wavecut = wave[region_index]
    fluxcut = flux[region_index]
    
    # Defining continuum regions
    cont_mask = np.logical_or.reduce([(wavecut >= start) & (wavecut <= end) for start, end in cont_region])
    
    # Normalizing via 2nd degree polynomials
    coeffs = np.polyfit(wavecut[cont_mask], fluxcut[cont_mask], 2)
    cont_fit = np.polyval(coeffs, wavecut)
    fluxnorm = fluxcut / cont_fit
    
    # Plotting results
    if plot_path:
        plt.figure()
        plt.plot(wavecut, fluxcut)
        plt.plot(wavecut, cont_fit)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Normalised Flux')
        plt.savefig(plot_path + '_contfit.png', dpi=300, transparent=False, facecolor='white')
        plt.close()
        
        plt.figure()
        plt.plot(wavecut, fluxnorm)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Normalised Flux')
        plt.savefig(plot_path + '_norm.png', dpi=300, transparent=False, facecolor='white')
        plt.close()
        
    return wavecut, fluxnorm


def line_fit(wave, fluxnorm, wave_fitmin, wave_fitmax, plot_path=False):
    """Fits a Voigt profile to the observed spectrum in the given region.

    Parameters
    ----------
    wave : ndarray
        Wavelengths of the spectrum
    fluxnorm : ndarray
        Fluxes of the spectrum, normalised to the continuum fit
    wave_fitmin : float
        Minimum wavelength for fitting
    wave_fitmax : float
        Maximum wavelength for fitting
    plot_path : str, optional
        If provided, will save a plot of the observed spectrum and the fit to the given directory.

    Returns
    -------
    fit_cent : float
        Centroid of the fit
    fit_cent_stdrr : float
        Standard error of the centroid
    """
    # Slicing spectra to only include desired region
    region_index = np.where((wave >= wave_fitmin) & (wave <= wave_fitmax))[0]
    wavecut = wave[region_index]
    fluxcut = fluxnorm[region_index]
    fluxcut_copy = np.copy(fluxcut) # Making copy to avoid altering original 
    y_offset = 0.99
    
    # If nans in continuum, then set to 1
    flux_nans = np.isnan(fluxcut_copy)
    fluxcut_copy[flux_nans] = y_offset
    
    # Defining the model
    mod = VoigtModel()
    
    # Fitting the model to the data
    pars = mod.guess(y_offset - fluxcut_copy, x=wavecut)
    # pars['gamma'].set(value=0.7, vary=True, expr='')
    fit = mod.fit(y_offset - fluxcut_copy, pars, x=wavecut)
     
    # Extracting centroid from fit
    fit_cent = fit.params['center'].value
    fit_cent_stdrr = fit.params['center'].stderr
    fit_chisq = fit.chisqr
     
    # Plotting
    if plot_path:
        plt.figure()
        plt.plot(wave, fluxnorm, label='Observed')
        plt.plot(wavecut, y_offset - fit.best_fit, '--', label='Fit')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('normalised Flux')
        plt.legend()
        
        plt.xlim(wave_fitmin, wave_fitmax)
        plt.savefig(plot_path + '_fit.png', dpi=300, transparent=False, facecolor='white')
        plt.close()

    return fit_cent, fit_cent_stdrr


def doppler_shift(rest_cent, obs_cent, obs_cent_err):
    """Calculates the radial velocity of a star given the rest and observed centroid wavelengths using the Doppler shift equation.

    Parameters
    ----------
    rest_cent : float
        Rest centroid wavelength of the spectral line in Å.
    obs_cent : float
        Observed centroid wavelength of the spectral line in Å.
    obs_cent_err : float
        Uncertainty in the observed centroid wavelength of the spectral line in Å.

    Returns
    -------
    rad_vel : float
        Radial velocity of the star in km/s.
    """
    # Using doppler shift equation to calculate radial velocity in km/s
    c = 299792.458 # km/s
    rad_vel = c * ((obs_cent - rest_cent)/ rest_cent)
    rad_vel_uncrt = c * (obs_cent_err / rest_cent)
    
    return rad_vel, rad_vel_uncrt

def rv_calc(uvb_df, vis_df, plot_path, lines):
    """Calculates the radial velocity of a star given the observed UVB and VIS spectra by fitting notable lines with Voigt functions,
    then takes the centroid of the fits in the Doppler shift equation.

    Parameters
    ----------
    uvb_df : pd.DataFrame
        DataFrame containing the observed UVB spectrum.
    vis_df : pd.DataFrame
        DataFrame containing the observed VIS spectrum.
    plot_path : str
        Path to save the plots of the fits and shifted spectra.

    Returns
    -------
    rv_df : pd.DataFrame
        DataFrame containing the calculated radial velocities for the Ca-II 8498Å, 8542Å, and 8662Å lines.
    """
    
    # Getting the wavelength and flux data
    star_id = uvb_df['star_id']
    uvb_wave = uvb_df['wave'].to_numpy()[0]
    uvb_flux = uvb_df['flux'].to_numpy()[0]
    vis_wave = vis_df['wave'].to_numpy()[0]
    vis_flux = vis_df['flux'].to_numpy()[0]
    
    # print(uvb_df['wave_norm'])
    # print(vis_df['wave_norm'])
    
    # We will be applying RV corrections to both UVB and VIS by adopting the corrections from 
    # Sana H., et al., 2024, Astronomy & Astrophysics, 688, A104:
    # - UVB: 4.477 km/s
    # - VIS: 1.001 km/s
    
    rv_vals = []
    rv_err_vals = []
    
    if 'hdelta' in lines:
        # ----------#    
        # Calculating RV for Hbeta
        hdelta_fitmin = 4090
        hdelta_fitmax = 4112
        
        # Slicing spectra to only include desired region
        hdelta_index = np.where((uvb_wave >= hdelta_fitmin) & (uvb_wave <= hdelta_fitmax))[0]
        hdelta_wave = uvb_wave[hdelta_index]
        hdelta_flux = uvb_flux[hdelta_index]
        
        # Fitting Voigt model
        hdelta_cent, hdelta_cent_stdrr = line_fit(wave=hdelta_wave, fluxnorm=hdelta_flux, wave_fitmin=hdelta_fitmin, 
                                                wave_fitmax=hdelta_fitmax, plot_path=plot_path + 'hdelta')
        
        # Calculating radial velocity, then applying wavelength calibration corrections
        hdelta_rv, hdelta_rv_err = doppler_shift(rest_cent=4101.73, obs_cent=hdelta_cent, obs_cent_err=hdelta_cent_stdrr)
        hdelta_rv += 4.477
        
        hdelta_dict = {'line': 'hdelta', 'cent': hdelta_cent, 'cent_err': hdelta_cent_stdrr, 'rv': hdelta_rv, 'rv_err': hdelta_rv_err}
        
        rv_vals.append(hdelta_dict)
        
        # print('hdelta: centroid = %.2f (rest = 4101.73), rv = %.2f' % (hdelta_cent, hdelta_rv))
        # ----------# 
    
    if 'hgamma' in lines:
        # ----------#    
        # Calculating RV for Hgamma
        hgamma_fitmin = 4326
        hgamma_fitmax = 4365
        
        # Slicing spectra to only include desired region
        hgamma_index = np.where((uvb_wave >= hgamma_fitmin) & (uvb_wave <= hgamma_fitmax))[0]
        hgamma_wave = uvb_wave[hgamma_index]
        hgamma_flux = uvb_flux[hgamma_index]
        
        # Fitting Voigt model
        hgamma_cent, hgamma_cent_stdrr = line_fit(wave=hgamma_wave, fluxnorm=hgamma_flux, wave_fitmin=hgamma_fitmin, 
                                                wave_fitmax=hgamma_fitmax, plot_path=plot_path + 'hgamma')
        
        # Calculating radial velocity, then applying wavelength calibration corrections
        hgamma_rv, hgamma_rv_err = doppler_shift(rest_cent=4340.47, obs_cent=hgamma_cent, obs_cent_err=hgamma_cent_stdrr)
        hgamma_rv += 4.477
        
        hgamma_dict = {'line': 'hgamma', 'cent': hgamma_cent, 'cent_err': hgamma_cent_stdrr, 'rv': hgamma_rv, 'rv_err': hgamma_rv_err}
        rv_vals.append(hgamma_dict)
        
        # print('hgamma: centroid = %.2f (rest = 4340.47), rv = %.2f' % (hgamma_cent, hgamma_rv))
        # ----------# 
    
    if 'hbeta' in lines:
        # ----------#    
        # Calculating RV for Hbeta
        hbeta_fitmin = 4750
        hbeta_fitmax = 4950
        
        # Slicing spectra to only include desired region
        hbeta_index = np.where((uvb_wave >= hbeta_fitmin) & (uvb_wave <= hbeta_fitmax))[0]
        hbeta_wave = uvb_wave[hbeta_index]
        hbeta_flux = uvb_flux[hbeta_index]
        
        # Fitting Voigt model
        hbeta_cent, hbeta_cent_stdrr = line_fit(wave=hbeta_wave, fluxnorm=hbeta_flux, wave_fitmin=hbeta_fitmin, 
                                                wave_fitmax=hbeta_fitmax, plot_path=plot_path + 'hbeta')
        
        # Calculating radial velocity, then applying wavelength calibration corrections
        hbeta_rv, hbeta_rv_err = doppler_shift(rest_cent=4861.35, obs_cent=hbeta_cent, obs_cent_err=hbeta_cent_stdrr)
        hbeta_rv += 4.477
        
        hbeta_dict = {'line': 'hbeta', 'cent': hbeta_cent, 'cent_err': hbeta_cent_stdrr, 'rv': hbeta_rv, 'rv_err': hbeta_rv_err}
        rv_vals.append(hbeta_dict)
        
        # print('hbeta: centroid = %.2f (rest = 4861.35), rv = %.2f' % (hbeta_cent, hbeta_rv))
        # ----------#  
    
    if 'halpha' in lines:
        # ----------#
        # Calculating RV for Halpha
        halpha_fitmin = 6450
        halpha_fitmax = 6650
        
        # Slicing spectra to only include desired region
        halpha_index = np.where((vis_wave >= halpha_fitmin) & (vis_wave <= halpha_fitmax))[0]
        halpha_wave = vis_wave[halpha_index]
        halpha_flux = vis_flux[halpha_index]
                
        # Fitting Voigt model
        halpha_cent, halpha_cent_stdrr = line_fit(wave=halpha_wave, fluxnorm=halpha_flux, wave_fitmin=halpha_fitmin, 
                                                wave_fitmax=halpha_fitmax, plot_path=plot_path + 'halpha')
        
        # Calculating radial velocity, then applying wavelength calibration corrections
        halpha_rv, halpha_rv_err = doppler_shift(rest_cent=6562.79, obs_cent=halpha_cent, obs_cent_err=halpha_cent_stdrr)
        halpha_rv += 1.001
        
        halpha_dict = {'line': 'halpha', 'cent': halpha_cent, 'cent_err': halpha_cent_stdrr, 'rv': halpha_rv, 'rv_err': halpha_rv_err}
        rv_vals.append(halpha_dict)
        
        # print('halpha: centroid = %.2f (rest = 6562.79), rv = %.2f' % (halpha_cent, halpha_rv))
        # ----------#  
    
    if 'caii849' in lines:
        # ----------#  
        # Calculating RV for Ca-II 8498Å and 8542Å
        caii849_fitmin = 8480
        caii849_fitmax = 8520
        
        # Slicing spectra to only include desired region
        caii_index = np.where((vis_wave >= caii849_fitmin) & (vis_wave <= caii849_fitmax))[0]
        caii_wave = vis_wave[caii_index]
        caii_flux = vis_flux[caii_index]
                
        # Fitting Voigt model
        caii849_cent, caii849_cent_stdrr = line_fit(wave=caii_wave, fluxnorm=caii_flux, wave_fitmin=caii849_fitmin, 
                                                wave_fitmax=caii849_fitmax, plot_path=plot_path + 'caii849')

        # Calculating radial velocity, then applying wavelength calibration corrections
        caii849_rv, caii849_rv_err = doppler_shift(rest_cent=8498.02, obs_cent=caii849_cent, obs_cent_err=caii849_cent_stdrr)
        caii849_rv += 1.001
        
        caii849_dict = {'line': 'caii849', 'cent': caii849_cent, 'cent_err': caii849_cent_stdrr, 'rv': caii849_rv, 'rv_err': caii849_rv_err}
        rv_vals.append(caii849_dict)
        
        # print('caii849: centroid = %.2f (rest = 8498.02), rv = %.2f' % (caii849_cent, caii849_rv))
        # ----------#  
    
    if 'caii854' in lines:
        # ----------#  
        # Calculating RV for Ca-II 8498Å and 8542Å
        if star_id.to_numpy()[0] == 'ra_1604-2712_s24': # This star has missing data around line, so need to manually set region 
            caii854_fitmin = 8541
            caii854_fitmax = 8554
        else:    
            caii854_fitmin = 8520
            caii854_fitmax = 8560

        # Slicing spectra to only include desired region
        caii_index = np.where((vis_wave >= caii854_fitmin) & (vis_wave <= caii854_fitmax))[0]
        caii_wave = vis_wave[caii_index]
        caii_flux = vis_flux[caii_index]
                
        # Fitting Voigt model
        caii854_cent, caii854_cent_stdrr = line_fit(wave=caii_wave, fluxnorm=caii_flux, wave_fitmin=caii854_fitmin, 
                                                wave_fitmax=caii854_fitmax, plot_path=plot_path + 'caii854')
        
        # Calculating radial velocity, then applying wavelength calibration corrections
        caii854_rv, caii854_rv_err = doppler_shift(rest_cent=8542.09, obs_cent=caii854_cent, obs_cent_err=caii854_cent_stdrr)
        caii854_rv += 1.001
        
        caii854_dict = {'line': 'caii854', 'cent': caii854_cent, 'cent_err': caii854_cent_stdrr, 'rv': caii854_rv, 'rv_err': caii854_rv_err}
        rv_vals.append(caii854_dict)
        
        # print('caii854: centroid = %.2f (rest = 8542.09), rv = %.2f' % (caii854_cent, caii854_rv))
        # ----------#  
        
    if 'caii866' in lines:    
        # ----------#  
        # Calculating RV for Ca-II 8662Å
        if star_id.to_numpy()[0] == 'ra_1604-2712_s24': # This star has missing data around line, so need to manually set region 
            caii866_fitmin = 8660
            caii866_fitmax = 8673
        else:
            caii866_fitmin = 8640
            caii866_fitmax = 8680

        # Slicing spectra to only include desired region
        caii866_index = np.where((vis_wave >= caii866_fitmin) & (vis_wave <= caii866_fitmax))[0]
        caii866_wave = vis_wave[caii866_index]
        caii866_flux = vis_flux[caii866_index]
        
        # Fitting Voigt model
        caii866_cent, caii866_cent_stdrr = line_fit(wave=caii866_wave, fluxnorm=caii866_flux, wave_fitmin=caii866_fitmin, 
                                                wave_fitmax=caii866_fitmax, plot_path=plot_path + 'caii866')
            
        # Calculating radial velocity, then applying wavelength calibration corrections
        caii866_rv, caii866_rv_err = doppler_shift(rest_cent=8662.14, obs_cent=caii866_cent, obs_cent_err=caii866_cent_stdrr)
        caii866_rv += 1.001
        
        caii866_dict = {'line': 'caii866', 'cent': caii866_cent, 'cent_err': caii866_cent_stdrr, 'rv': caii866_rv, 'rv_err': caii866_rv_err}
        rv_vals.append(caii866_dict)
        
        # print('caii866: centroid = %.2f (rest = 8662.14), rv = %.2f' % (caii866_cent, caii866_rv))
        # ----------#  
    

    # Calculate the average radial velocities for each arm
    rv_df = pd.DataFrame(rv_vals)
    
    # rv_uvb_df = rv_df[rv_df['line'].isin(['hdelta', 'hgamma', 'hbeta'])]
    # rv_uvb_vals = rv_uvb_df['rv'].to_numpy()
    # rv_uvb_errs = rv_uvb_df['rv_err'].to_numpy()
    # rv_uvb_avg = np.average(rv_uvb_vals)
    # rv_uvb_err_avg = np.sqrt(sum(x**2 for x in rv_uvb_errs) / len(rv_uvb_errs))
    
    # rv_vis_df = rv_df[rv_df['line'].isin(['halpha', 'caii849', 'caii854', 'caii866'])]
    # rv_vis_vals = rv_vis_df['rv'].to_numpy()
    # rv_vis_errs = rv_vis_df['rv_err'].to_numpy()
    # rv_vis_avg = np.average(rv_vis_vals)
    # rv_vis_err_avg = np.sqrt(sum(x**2 for x in rv_vis_errs) / len(rv_vis_errs))
    
    rv_uvb_df = rv_df[rv_df['line'].isin(['hdelta', 'hgamma', 'hbeta'])]
    rv_uvb_vals = rv_uvb_df['rv'].to_numpy()
    rv_uvb_errs = rv_uvb_df['rv_err'].to_numpy()
    rv_uvb_weights = 1 / (rv_uvb_errs**2)
    rv_uvb_avg = np.sum(rv_uvb_weights * rv_uvb_vals) / np.sum(rv_uvb_weights)
    rv_uvb_err_avg = np.sqrt(1 / sum(rv_uvb_weights))
    
    rv_vis_df = rv_df[rv_df['line'].isin(['halpha', 'caii849', 'caii854', 'caii866'])]
    rv_vis_vals = rv_vis_df['rv'].to_numpy()
    rv_vis_errs = rv_vis_df['rv_err'].to_numpy()
    rv_vis_weights = 1 / (rv_vis_errs**2)
    rv_vis_avg = np.sum(rv_vis_weights * rv_vis_vals) / np.sum(rv_vis_weights)
    rv_vis_err_avg = np.sqrt(1 / sum(rv_vis_weights))
    
    # Calculating the weighted average of the radial velocities from both arms
    rv_weight_avg = (rv_uvb_avg * (1 / rv_uvb_err_avg**2) + rv_vis_avg * (1 / rv_vis_err_avg**2)) / \
                     ((1 / rv_uvb_err_avg**2) + (1 / rv_vis_err_avg**2))
    rv_weight_err_avg = np.sqrt(1 / ((1 / rv_uvb_err_avg**2) + (1 / rv_vis_err_avg**2)))
    
        
    # Storing these radial velocities into a pandas DataFrame    
    rv_df = pd.DataFrame({'star_id': star_id, 'rv_avg': rv_weight_avg, 'rv_avg_err': rv_weight_err_avg, 'rv_uvb': rv_uvb_avg, 'rv_uvb_err': rv_uvb_err_avg, 'rv_vis': rv_vis_avg, 'rv_vis_err': rv_vis_err_avg,
                          'rv_hdelta': rv_df[rv_df['line'] == 'hdelta']['rv'].to_numpy()[0], 'rv_hdelta_err': rv_df[rv_df['line'] == 'hdelta']['rv_err'].to_numpy()[0],
                          'rv_hgamma': rv_df[rv_df['line'] == 'hgamma']['rv'].to_numpy()[0], 'rv_hgamma_err': rv_df[rv_df['line'] == 'hgamma']['rv_err'].to_numpy()[0],
                          'rv_hbeta': rv_df[rv_df['line'] == 'hbeta']['rv'].to_numpy()[0], 'rv_hbeta_err': rv_df[rv_df['line'] == 'hbeta']['rv_err'].to_numpy()[0],
                          'rv_halpha': rv_df[rv_df['line'] == 'halpha']['rv'].to_numpy()[0], 'rv_halpha_err': rv_df[rv_df['line'] == 'halpha']['rv_err'].to_numpy()[0],
                          'rv_caii849': rv_df[rv_df['line'] == 'caii849']['rv'].to_numpy()[0], 'rv_caii849_err': rv_df[rv_df['line'] == 'caii849']['rv_err'].to_numpy()[0],
                          'rv_caii854': rv_df[rv_df['line'] == 'caii854']['rv'].to_numpy()[0], 'rv_caii854_err': rv_df[rv_df['line'] == 'caii854']['rv_err'].to_numpy()[0],
                          'rv_caii866': rv_df[rv_df['line'] == 'caii866']['rv'].to_numpy()[0], 'rv_caii866_err': rv_df[rv_df['line'] == 'caii866']['rv_err'].to_numpy()[0]})
    
    return rv_df


def spec_shift(waves, vel):
    """
    Performs radial shift on spectra.

    Parameters
    ----------
    waves : ndarray
        Wavelength array being shifted.

    vel : float
        Radial velocity value being used to perform the correction.

    Returns
    -------
    waves_shifted : ndarray
        Velocity-corrected wavelength array.
    """
    c = 299792.458

    # COMPUTING SHIFT IN LOG SPACE
    shift = np.log(1 + vel / c)

    # TRANSFORMING SHIFTED AXIS INTO WAVELENGTH SPACE
    waves_shifted = np.exp(np.log(waves) - shift)

    return waves_shifted


def rv_shift(uvb_df, vis_df, rv_df, plot=False):
    """Shifts UVB and VIS data by given radial velocity.

    Parameters
    ----------
    uvb_df : pandas.DataFrame
        Dataframe containing UVB data
    vis_df : pandas.DataFrame
        Dataframe containing VIS data
    rv_val : float
        Radial velocity to shift by (in km/s)
    plot_path : str
        Path to save plots

    Returns
    -------
    uvb_df : pandas.DataFrame
        UVB data with shifted columns
    vis_df : pandas.DataFrame
        VIS data with shifted columns
    """

    # Shifting UVB data by given radial velocity 
    rv_uvb_val = rv_df['rv_uvb'].to_numpy()[0]
    
    uvb_wave = uvb_df['wave'].to_numpy()[0]
    uvb_flux = uvb_df['flux'].to_numpy()[0]
    uvb_wave_shift = spec_shift(waves=uvb_wave, vel=rv_uvb_val)
    uvb_df['wave_shift'] = [uvb_wave_shift]
    
    # Plotting
    if plot:
        # linelist = pd.read_csv('/data/misfit/benlowe/xshooter/linelist/lines.csv')
        # linelist_uvb = linelist[linelist['wave'] <= 5560]
        # lines_uvb = linelist_uvb['wave'].to_numpy()
        colors = Category10[10]
        
        # Making figure
        p = bok.figure(x_axis_label='Wavelength (Å)', y_axis_label='Normalised Flux', width = 1000)
        p.sizing_mode = 'scale_height'
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"
        p.title.text_font_size = "16pt"

        # Plotting original and shifted spectra
        p.line(uvb_wave, uvb_flux, line_width=1.3, legend_label='Original', color=colors[0])
        p.line(uvb_wave_shift, uvb_flux, line_width=1.3, legend_label='Shifted', color=colors[1])
        
        # Plotting lines used for rv calculations
        vline = Span(location=4101.73, dimension='height', line_color='black', line_dash='dashed', line_width=1.3)
        p.add_layout(vline)
        vline = Span(location=4340.47, dimension='height', line_color='black', line_dash='dashed', line_width=1.3)
        p.add_layout(vline)
        vline = Span(location=4861.35, dimension='height', line_color='black', line_dash='dashed', line_width=1.3)
        p.add_layout(vline)
        
        # # Plotting lines from listlist
        # for line in lines_uvb:
        #     vline = Span(location=line, dimension='height', line_color='black', line_dash='dashed', line_width=1.3, line_alpha=0.1)
        #     p.add_layout(vline)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bok.save(obj=p, filename=plot + 'uvb_shift.html')

    
    
    # Shifting VIS data by given radial velocity
    rv_vis_val = rv_df['rv_vis'].to_numpy()[0]
    
    vis_wave = vis_df['wave'].to_numpy()[0]
    vis_flux = vis_df['flux'].to_numpy()[0]
    vis_wave_shift = spec_shift(waves=vis_wave, vel=rv_vis_val)
    vis_df['wave_shift'] = [vis_wave_shift]
    
    # Plotting
    if plot:
        # linelist_vis = linelist[linelist['wave'] > 5560]
        # lines_vis = linelist_vis['wave'].to_numpy()
        
        # Making figure
        p = bok.figure(x_axis_label='Wavelength (Å)', y_axis_label='Normalised Flux', width = 1000)
        p.sizing_mode = 'scale_height'
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"
        p.title.text_font_size = "16pt"

        # Plotting original and shifted spectra
        p.line(vis_wave, vis_flux, line_width=1.3, legend_label='Original', color=colors[0])
        p.line(vis_wave_shift, vis_flux, line_width=1.3, legend_label='Shifted', color=colors[1])
        
        # Plotting lines used for rv calculations
        vline = Span(location=6562.79, dimension='height', line_color='black', line_dash='dashed', line_width=1.3)
        p.add_layout(vline)
        vline = Span(location=8498.02, dimension='height', line_color='black', line_dash='dashed', line_width=1.3)
        p.add_layout(vline)
        vline = Span(location=8542.09, dimension='height', line_color='black', line_dash='dashed', line_width=1.3)
        p.add_layout(vline)
        vline = Span(location=8662.14, dimension='height', line_color='black', line_dash='dashed', line_width=1.3)
        p.add_layout(vline)
        
        # # Plotting lines from listlist
        # for line in lines_vis:
        #     vline = Span(location=line, dimension='height', line_color='black', line_dash='dashed', line_width=1.3, line_alpha=0.1)
        #     p.add_layout(vline)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bok.save(obj=p, filename=plot + 'vis_shift.html')


        
    return uvb_df, vis_df      
    

def remove_cosmic_snr(uvb_df, vis_df, sigma=3): 
    """ Removes cosmic rays from the spectrum data by identifying and masking outliers.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing spectral data with columns 'ob', 'wave_norm', 'spec_shift', and 'snr'.
    sigma : int, optional
        The sigma value to determine the noise threshold for outlier detection. Default is 3.

    Returns
    -------
    pandas.DataFrame
        Dataframe with an additional column 'flux_no_cosmic' containing the flux with cosmic rays removed.
    """
    
    def remove_cosmic(df, sigma=3):
        """Removes cosmic rays from the spectrum data by identifying and masking outliers.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing spectral data with columns 'ob', 'wave_norm', 'spec_shift', and 'snr'.
        sigma : int, optional
            The sigma value to determine the noise threshold for outlier detection. Default is 3.

        Returns
        -------
        pandas.DataFrame
            Dataframe with an additional column 'flux_no_cosmic' containing the flux with cosmic rays removed.
        """
        # Using SNR 
        dict_list = []
        for i, exp in df.iterrows():
            # Extracting out the data
            ob = exp['ob']
            wave = exp['wave_norm']
            flux = exp['spec_shift']
            flux_copy = np.copy(flux) # Doing this to not alter the original flux array
            flux_err = exp['flux_err_norm']
            flux_err_copy = np.copy(flux_err) # Doing this to not alter the original flux error array
            snr = exp['snr']
            
            # Using noise = 1/SNR to determine noise, then take the sigma
            noise = 1 / snr
            noise_std = sigma * noise
            flux_upper = 1 + noise_std
            
            # Removing anything that is above this threshold, and removing ± 1 pixels either side of the flux
            mask = flux_copy > flux_upper
            mask = np.logical_or(mask, np.roll(mask, 1))  # Include ± 1 pixel on either side
            mask = np.logical_or(mask, np.roll(mask, -1)) # Include ± 1 pixel on either side
            
            cosmics = flux_copy[mask]
            flux_mask = flux_copy[mask] # Masking out fluxes that surpass this threshold
            wave_mask = wave[mask]

            # Replacing these fluxes with nans
            flux_copy[mask] = np.nan
            flux_err_copy[mask] = np.nan
            
            # Getting the weights for the exposures using the SNR
            weight = snr**2
            weight_all = np.array([weight] * len(wave))
            weight_all[mask] = np.nan
                        
            # Storing
            dict_ = {'ob': ob, 'flux_no_cosmic': flux_copy, 'flux_err_no_cosmic': flux_err_copy, 'weights': weight_all}
            dict_list.append(dict_)
            
            
        # # Using median-filtering
        # dict_list = []
        # plt.figure(figsize=[8,6])
        # for i, exp in df.iterrows():
            
        #     # Extracting out the data
        #     ob = exp['ob']
        #     wave = exp['wave_norm']
        #     flux = exp['spec_shift']
        #     flux_copy = np.copy(flux) # Doing this to not alter the original flux array
        #     snr = exp['snr']
            
        #     # test_reg = np.where((wave >= 4961) & (wave <= 4964))[0] 
        #     # test_reg = np.where((wave >= 4950) & (wave <= 4965))[0]
        #     # test_reg = np.where((wave >= 5268) & (wave <= 5271))[0] 
        #     # test_reg = np.where((wave >= 4924) & (wave <= 4928))[0] 
            
        #     wave = wave[test_reg]
        #     flux_copy = flux_copy[test_reg]            
            
        #     # Taking the median of each entry in each array, then subtracting it from the original stack
        #     test_med = np.median(flux_copy, axis=0)
        #     test_diff = flux_copy - test_med
        #     test_abs = np.abs(test_diff)
            
        #     # Taking outliers as being the 96th percentile
        #     threshold = np.percentile(test_abs, 95)

        #     # Find the max outlier above the threshold
        #     mask = test_abs > threshold  # Mask of values that are above the threshold
        #     outliers = test_diff[mask]  # Extract actual outlier values

        #     # Replacing these fluxes with nans
        #     flux_mask = flux_copy[mask] 
            
        #     for j in flux_mask:
        #         if j >= 1:
        #             flux_copy[flux_copy == j] = np.nan
               
                    
        #     loc = np.isnan(flux_copy)
        #     cosmics = wave[loc]
        #     print(cosmics)
                    
            
        #     # Getting the weights for the exposures using the SNR
        #     weight = snr**2
        #     weight_all = np.array([weight] * len(wave))
        #     weight_all[mask] = np.nan
            
        #     plt.plot(wave, flux_copy, label='OB {0} (SNR = {1:.2f})'.format(i+1, snr))
        #     # if len(cosmics) != 0:
        #     #     for k in cosmics:
        #     #         plt.axvline(k, color='black', linestyle='--')
            
        #     # Storing
        #     dict_ = {'ob': ob, 'flux_no_cosmic': flux_copy, 'weights': weight_all}
        #     dict_list.append(dict_)
            
            
        df_no_cosmic = pd.DataFrame(dict_list)
        df = pd.merge(df, df_no_cosmic, on='ob')    
        
        return df
    
    # Removing cosmic rays
    uvb_df = remove_cosmic(df=uvb_df, sigma=3)
    vis_df = remove_cosmic(df=vis_df, sigma=3)
    
    return uvb_df, vis_df

    
def run_stack(star_paths, save_path):
    """Processes spectral data for multiple stars by extracting, cross-correlating, stacking, calculating radial
    velocities, and saving the processed data.

    Parameters
    ----------
    star_paths : pd.DataFrame
        Dataframe containing the paths to both the .fits files and the normalised .csv files.
    save_path : str
        Path to the directory where processed data and plots will be saved.
    norm_paths : str
        Path to the directory containing the normalization data.

    Returns
    -------
    None
    """
    star_info_all = []
    j = 1
    process_id = os.getpid()
    # Running this for each star
    for i, star in star_paths.iterrows():
        
        # Getting paths to both the .fits (sci spec) and the .csv (norm spec) files
        sci_path = star['sci_paths']
        norm_path = star['norm_paths']
        
        # Getting UVB, VIS and NIR .fits data, then extracting into pd.DataFrames
        uvb_fits, vis_fits, nir_fits = grab_fits(star_path=sci_path)
        uvb_norm, vis_norm, nir_norm = grab_norm(star_norm_path=norm_path)
        
        uvb_df, fits_headers = get_data(fits_path=uvb_fits, norm_path=uvb_norm)
        vis_df, fits_headers = get_data(fits_path=vis_fits, norm_path=vis_norm)
        star_id = uvb_df['star_id'].to_numpy()[0]
        
        print('Running {0}\n- Process {1} \n- Star {2}/{3}\n'.format(star_id, process_id, j, len(star_paths)))
        j += 1

        # Making directories to store data
        star_save_path = save_path + star_id + '/'
        mkdir(star_save_path)
        
        test_plots = star_save_path + 'test_plots/'
        mkdir(test_plots)
        
                
        # Cross-correlating by manually shifting spectra to given strong absorption feature
        # UVB - Cross-correlating over Hbeta region (476-496Å)
        # VIS - Cross-correlating over Halpha region (647-667Å)
        # print('Cross-correlating ...')
        # sys.stdout.write(f"\rCross-correlating... ")
        # sys.stdout.flush()
        uvb_corr_plot = test_plots + 'corr_uvb.png'
        vis_corr_plot = test_plots + 'corr_vis.png'
        uvb_df, uvb_ref_ob = cross_corr(star_df=uvb_df, minwave=4760, maxwave=4960, plot=uvb_corr_plot)
        vis_df, vis_ref_ob = cross_corr(star_df=vis_df, minwave=6470, maxwave=6670, plot=vis_corr_plot)        
        
        
        # Removing cosmic rays
        # print('Removing cosmic rays ...')  
        # sys.stdout.write(f"\rRemoving cosmic rays... ")
        # sys.stdout.flush()
        uvb_df, vis_df = remove_cosmic_snr(uvb_df=uvb_df, vis_df=vis_df, sigma=2.5)
        

        # Stacking spectra
        # print('Stacking the spectra ...')
        # sys.stdout.write(f"\rStacking spectra... ")
        # sys.stdout.flush()
        uvb_stack_df = stack_spec(star_df=uvb_df, plot=test_plots + 'uvb_stack.html')
        vis_stack_df = stack_spec(star_df=vis_df, plot=test_plots + 'vis_stack.html')
    
        
        # Calculating and correcting for radial velocities (for stacked data)
        # print('Calculating radial velocities ...')
        # sys.stdout.write(f"\rCalculating RVs... ")
        # sys.stdout.flush()
        
        # star ra_1604-2712_s24 has missing data for Ca-II triplet, so will use hydrogen lines to determine rvs
        # star ra_ 1624-2150_s278 has jank Ca-II 8490 line, so will only take the other two
        # if star_id == 'ra_1604-2712_s24':
        #     lines = ['halpha', 'hbeta']
        # elif star_id == 'ra_1624-2150_s278':
        #     lines = ['caii854', 'caii866']    
        # else:
        #     lines = ['caii849', 'caii854', 'caii866', 'halpha', 'hbeta', 'hgamma', 'hdelta']
        
        # if star_id == 'ra_1659-2154_s347':
        #     lines = ['caii854', 'caii866', 'halpha', 'hbeta', 'hgamma', 'hdelta']    
        # else:
        lines = ['caii849', 'caii854', 'caii866', 'halpha', 'hbeta', 'hgamma', 'hdelta']
        
        rv_df = rv_calc(uvb_df=uvb_stack_df, vis_df=vis_stack_df, plot_path=test_plots, lines=lines)        
        uvb_stack_df, vis_stack_df = rv_shift(uvb_df=uvb_stack_df, vis_df=vis_stack_df, rv_df=rv_df, plot=test_plots)
                    

        # Storing data into .fits table
        # Getting the .fits file for template spectra - used for storing header information
        # print('Saving ...\n')
        # sys.stdout.write(f"\rSaving results... ")
        # sys.stdout.flush()
        spec_save(uvb_df=uvb_stack_df, vis_df=vis_stack_df, rv_df=rv_df, fits_headers=fits_headers, save_path=star_save_path)

        # Storing header data into dict
        star_info = {'star_id': star_id, 'snr_uvb': float(uvb_stack_df['snr'].iloc[0]), 'snr_vis': float(vis_stack_df['snr'].iloc[0]), 'ra': fits_headers['ra'], 'dec': fits_headers['dec'],
                     'barycor': fits_headers['rv_barycor'], 'heliocor': fits_headers['rv_heliocor'], 'rv_avg': rv_df['rv_avg'].to_numpy()[0], 'rv_avg_err': rv_df['rv_avg_err'].to_numpy()[0], 'rv_uvb': rv_df['rv_uvb'].to_numpy()[0], 
                     'rv_uvb_err': rv_df['rv_uvb_err'].to_numpy()[0], 'rv_vis': rv_df['rv_vis'].to_numpy()[0], 'rv_vis_err': rv_df['rv_vis_err'].to_numpy()[0],
                     'rv_hdelta': rv_df['rv_hdelta'].to_numpy()[0], 'rv_hdelta_err': rv_df['rv_hdelta_err'].to_numpy()[0], 
                     'rv_hgamma': rv_df['rv_hgamma'].to_numpy()[0], 'rv_hgamma_err': rv_df['rv_hgamma_err'].to_numpy()[0], 
                     'rv_hbeta': rv_df['rv_hbeta'].to_numpy()[0], 'rv_hbeta_err': rv_df['rv_hbeta_err'].to_numpy()[0], 
                     'rv_halpha': rv_df['rv_halpha'].to_numpy()[0], 'rv_halpha_err': rv_df['rv_halpha_err'].to_numpy()[0],
                     'rv_caii849': rv_df['rv_caii849'].to_numpy()[0], 'rv_caii849_err': rv_df['rv_caii849_err'].to_numpy()[0],
                     'rv_caii854': rv_df['rv_caii854'].to_numpy()[0], 'rv_caii854_err': rv_df['rv_caii854_err'].to_numpy()[0],
                     'rv_caii866': rv_df['rv_caii866'].to_numpy()[0], 'rv_caii866_err': rv_df['rv_caii866_err'].to_numpy()[0]}
        star_info_all.append(star_info)
        
    # Storing star info into dataframe file
    star_info_df = pd.DataFrame(star_info_all)
    return star_info_df              

    
    
if __name__ == '__main__':
    
    # List of folders containing the science spec .fits files
    sci_path = '/data/misfit/benlowe/xshooter/p113/reduced_data/science_spec/'
    sci_specs = [sci_path + star + '/' for star in os.listdir(sci_path)] # if star == 'ra_1604-2712_s292'
    sci_specs.sort()
    
    # List of folders containing the normalised spec .csv files
    norm_path = '/data/misfit/benlowe/xshooter/p113/reduced_data/norm_spec/all/'
    # norm_path = '/data/misfit/benlowe/xshooter/p115/suppnet_data/final/'
    norm_specs = [norm_path + star + '/' for star in os.listdir(norm_path)] # if star == 'ra_1604-2712_s292'
    norm_specs.sort()
    
    # Storing into pandas dataframe for easy access
    star_paths = pd.DataFrame({'sci_paths': sci_specs, 'norm_paths': norm_specs})
    
    # Location to store saved stacked data
    # save_path = '/data/misfit/benlowe/xshooter/reduced_data/comb_spec_2/'
    save_path = '/data/misfit/benlowe/xshooter/p113/reduced_data/comb_spec_3/'
    
    
    # Running parallel processing    
    num_cores = 16
    star_paths_split = np.array_split(star_paths, num_cores)  # Split DataFrame into equal parts
    run_partial = partial(run_stack, save_path=save_path)

    with Pool(processes=num_cores) as pool:  
        info_df = pool.map(run_partial, star_paths_split)
    
    # Combining
    final_info_df = pd.concat(info_df, ignore_index=True)
    
    # Saving
    final_info_df.to_csv(save_path + 'star_info.csv', index=False) 
    print('Done!')    
    