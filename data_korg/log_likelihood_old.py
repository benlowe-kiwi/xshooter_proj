import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from specutils.spectra import Spectrum1D
from specutils.fitting.continuum import fit_continuum
import scipy
import scipy.integrate
import scipy.stats
from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils
import yaml
import code
import pandas as pd
import os
from multiprocessing import Process, Pool

def spec_normal(spectra, cont_band):
    """
    Normalizers spectral region according to the continuum bandpass chosen using Specutils functions Spectrum1D and
    fit_continuum.

    Parameters
    ----------
    spectra : Spectrum1D
        Spectrum container involving spectral flux (target_spec.flux) and spectral wavelength (target_spec.spectral_axis).

    cont_band : list
        A [min1, max1, min2, max2] list containing the minimum and maximum wavelengths corresponding to two continuum
        regions used to normalize the spectra for a particular absorption line.

    Returns
    -------
    spec_normalized : Spectrum1D
        Normalized spectral wave and flux values contained within the Specutils Spectrum1D object. Use spec_normalized.flux
        to access the normalized flux, and spec_normalized.spectral_axis to access the wavelength.
    """

    # DEFINING THE CONTINUUM BANDPASS REGIONS IN ASTROPY UNITS
    if len(cont_band) == 2:
        region = [cont_band[0] * u.AA, cont_band[1] * u.AA]

    if len(cont_band) == 4:
        region = [(cont_band[0] * u.AA, cont_band[1] * u.AA), (cont_band[2] * u.AA, cont_band[3] * u.AA)]

    if len(cont_band) == 6:
        region = [(cont_band[0] * u.AA, cont_band[1] * u.AA), (cont_band[2] * u.AA, cont_band[3] * u.AA),
                  (cont_band[4] * u.AA, cont_band[5] * u.AA)]

    if len(cont_band) == 8:
        region = [(cont_band[0] * u.AA, cont_band[1] * u.AA), (cont_band[2] * u.AA, cont_band[3] * u.AA),
                  (cont_band[4] * u.AA, cont_band[5] * u.AA), (cont_band[6] * u.AA, cont_band[7] * u.AA)]

    # FITTING THE CONTINUUM
    fitted_continuum = fit_continuum(spectra, window=region)
    y_fit = fitted_continuum(spectra.spectral_axis)

    # NORMALIZING THE SPECTRA
    spec_normalized = spectra / y_fit

    return spec_normalized

def mkdir(dir_path):
    """Makes a directory at a given path. If it already exists, then it will skip.

    Parameters
    ----------
    dir_name : str
        Pathway to directory.
    """
    
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass
    
    return None

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

def lnL_obs(npix, flux_err, chisq_red):
    """Computes the log likelihood of a given observed dataset.

    Parameters
    ----------
    npix : int
        Number of pixels within the given wavelength region.
    flux_err : ndarray
        Array of observed flux errors.
    chisq_red : ndarray
        Array of chi-squared values.

    Returns
    -------
    ndarray
        Array of the observed log likelihoods for the dataset.
    """
    return -1/2 * (chisq_red * npix) + 1/2 * np.sum(np.log(np.sqrt(2 * np.pi) * flux_err))

def lnL_model(npix, flux_err, chisq_red):
    """Computes the log likelihood of a given model dataset.

    Parameters
    ----------
    npix : int
        Number of pixels within the given wavelength region.
    flux_err : ndarray
        Array of observed flux errors.
    chisq_red : ndarray
        Array of model chi-squared values.

    Returns
    -------
    ndarray
        Array of the observed log likelihoods for the dataset.
    """
    return -1/2 * (chisq_red * npix) + 1/2 * np.sum(np.log(np.sqrt(2 * np.pi) * flux_err))

def p_calc(lnL, cfe_arr):
    """Performs the cumlative integral on a given log likelihood and [C/Fe] values.

    Parameters
    ----------
    lnL : ndarray
        Array of log likelihoods of dataset
    cfe_arr : ndarray
        Array of [C/Fe] values.
        
    Returns
    -------
    p_norm : ndarray
        Array of normalized P values
    """
    
    # Performing the cumulative integral
    p = scipy.integrate.cumulative_trapezoid(y=np.exp(lnL - np.max(lnL)), x=10**cfe_arr) # '- np.max(thing1)' to make numbers sensible.
    
    # Normalizing
    p_norm = p/np.max(p)
    
    return p_norm

def chisq_run_cfemin(wave, flux, eflux, conf_name, atm_params, config, carb_vals, min_cfe, cont_npoly=2):
        """Uses rvspecfit's chi-square minimisation code to manually fit the G-band region in the blue spectra.

        Parameters
        ----------
        wave : ndarray
            Array of wavelengths for the star
        flux : ndarray
            Array of fluxes for the star
        eflux : ndarray
            Array of flux errors for the star
        conf_name : str
            Name of blue interpolation configuration.
        atm_params : dict
           Dictonary of stellar paramaters. Format is:
            {'star_id': val,
            'teff': val,
            'logg': val,
            'feh': val,
            'rv': val}
        carb_vals : ndarray
            Array of carbon values used for fitting
        min_cfe : float
            Lowest [C/Fe] value used for null-hypothesis.
        cont_npoly : int, optional
            Number of polynomial points to be used in continuum fitting. For the G-band, 2 polynomial points is ideal. 

        Returns
        -------
        chisq_list : list
            List of chisquared values used over the range of given [C/Fe] values
        synth_spec_cfemin : Spectrum1D
            A Spectrum1D object containing the synthetic spectrum spectra with the lowest [C/Fe] value.

        """
        
        # Making Spectrum1D pnkect
        b_star_spec = Spectrum1D(spectral_axis=wave * u.AA, flux=flux * u.Jy)      
        
        
        # ---DEFINING THE SPECDATA OBJECTS--- #
        specdata_b = spec_fit.SpecData(name = conf_name,
                                    lam = wave,
                                    spec = flux,
                                    espec= eflux)
                                    

        # ---PERFORMING THE FITTING--- #
        options = {'npoly': cont_npoly,
                'rbf_continuum': False}
        
        # Running this for lowest [C/Fe] value
        stellar_params = [atm_params['teff'], atm_params['logg'], atm_params['feh'], min_cfe]
        rv = atm_params['rv']
        
        spec_chisq_cfemin = spec_fit.get_chisq(specdata=[specdata_b],
                                                vel=rv,
                                                atm_params=stellar_params,
                                                rot_params=None,
                                                config=config,
                                                options=options,
                                                full_output=True)
        
        
        # Taking the model of this to be our 'empty' spectrum
        synth_model_cfemin = spec_chisq_cfemin['models'][0]
        synth_spec_cfemin = Spectrum1D(flux=synth_model_cfemin * u.Jy, spectral_axis=wave * u.AA)
        
        synthdata_cfemin = spec_fit.SpecData(name = conf_name,
                                lam = wave,
                                spec = synth_model_cfemin,
                                espec= eflux)
        
        # Now using this to get range of chisquare values for range of [C/Fe] values
        chisq_list_cfemin = []
        for carb in carb_vals:
            stellar_params = [atm_params['teff'], atm_params['logg'], atm_params['feh'], carb]
            rv = atm_params['rv']
            
            synth_chisq_cfemin = spec_fit.get_chisq(specdata=[synthdata_cfemin],
                                vel=rv,
                                atm_params=stellar_params,
                                rot_params=None,
                                config=config,
                                options=options,
                                full_output=True)
            
            chisq_list_cfemin.append(synth_chisq_cfemin)  

        return chisq_list_cfemin, synth_spec_cfemin



def synth_extract(wave, flux, eflux, conf_name, atm_params, config, one_sigma, cfe_val, cont_npoly=2):
        """Uses rvspecfit's chi-square minimisation code to manually fit the G-band region in the blue spectra.

        Parameters
        ----------
        fits_file : fits HDU
            The opened .fits file containing all the spectral data.
        b_wave_min : float
            Minimum wavelength range for blue data.
        b_wave_max : float
            Maximum wavelength range for blue data.
        conf_name : str
            Name of blue interpolation configuration.
        atm_params : dict
           Dictonary of stellar paramaters. Format is:
            {'star_id': val,
            'teff': val,
            'logg': val,
            'feh': val,
            'rv': val}
        carb_vals : ndarray
            Array of carbon values used for fitting
        cont_npoly : int, optional
            Number of polynomial points to be used in continuum fitting. For the G-band, 2 polynomial points is ideal. 

        Returns
        -------
        chisq_list : dict
            Dictionary contianing the fitting results. The keys are: 'chisq', 'chisq_array', 'red_chisq_array', 'models' and 'raw_models'.
        b_star_spec : Spectrum1D
            A Spectrum1D object containing the observed spectra. To get flux use star_spec.flux, and to get wavelength use star_spec.spectral_axis.

        """
        
        # Making Spectrum1D pnkect
        b_star_spec = Spectrum1D(spectral_axis=wave * u.AA, flux=flux * u.Jy)      
            
        
        # ---DEFINING THE SPECDATA OBJECTS--- #
        specdata_b = spec_fit.SpecData(name = conf_name,
                                    lam = wave,
                                    spec = flux,
                                    espec= eflux)
                                    

        # ---PERFORMING THE FITTING--- #
        options = {'npoly': cont_npoly,
                'rbf_continuum': False}
        
        # Running this for the star's [C/Fe] value
        rv = atm_params['rv']
        
        stellar_params = [atm_params['teff'], atm_params['logg'], atm_params['feh'], cfe_val]
        synth_chisq = spec_fit.get_chisq(specdata=[specdata_b],
                                            vel=rv,
                                            atm_params=stellar_params,
                                            rot_params=None,
                                            config=config,
                                            options=options,
                                            full_output=True)
        synth_spec = synth_chisq['models'][0]
        
        # Running this for 1sigma synthetic spectra around the star's [C/Fe] value
        rv = atm_params['rv']
        
        synth_cfe_1 = cfe_val + one_sigma
        stellar_params_1 = [atm_params['teff'], atm_params['logg'], atm_params['feh'], synth_cfe_1]
        synth_chisq_1 = spec_fit.get_chisq(specdata=[specdata_b],
                                            vel=rv,
                                            atm_params=stellar_params_1,
                                            rot_params=None,
                                            config=config,
                                            options=options,
                                            full_output=True)
        synth_spec_1 = synth_chisq_1['models'][0]
        
        synth_cfe_2 = cfe_val - one_sigma
        stellar_params_2 = [atm_params['teff'], atm_params['logg'], atm_params['feh'], synth_cfe_2]
        synth_chisq_2 = spec_fit.get_chisq(specdata=[specdata_b],
                                            vel=rv,
                                            atm_params=stellar_params_2,
                                            rot_params=None,
                                            config=config,
                                            options=options,
                                            full_output=True)
        synth_spec_2 = synth_chisq_2['models'][0]
        
        return synth_spec, synth_spec_1, synth_spec_2



def run_carb_err(data_df, info_df, reduce_path, chisq_path, wave_min, wave_max, conf_name, config, min_cfe, save_path, plot=True, gc=False):
    """Computes the log likelihood of a given model dataset to get [C/Fe] values and uncertainties. Will also flag if stars are
    above the detection limit. 

    Parameters
    ----------
    data_df : pandas.dataframe
        Dataframe housing the stellar data
    info_df : pandas.dataframe
        Dataframe housing information about the star like S/N ratios and field names
    reduce_path : str
        Pathway to the reduction files 
    chisq_path : str
        Pathway to the chisquare values
    wave_min : float
        Minimum wavelength for synthesis
    wave_max : float
        Maximum wavelength for synthesis
    conf_name : str
        Name of blue interpolation configuration.
    config : .ymal 
        The .ymal file to configure rvspecfit
    min_cfe : float
        Lowest detectable [C/Fe] abundances avaliable from the grid
    save_path : str
        Save path for plots. Only used when plot == True
    plot : bool, optional
        Whether to plot statistical infographics, by default =True
    gc : bool, optional
        Whether the data is globular cluster data or not. True for when it is, False otherwise.
    """
    # # Loading in the EMP dataframe (which contains the standard deviations for [C/Fe] from randomizations)
    # emp_df = pd.read_csv('/data/mustard/benlowe/data/rvspecfit_data/aat_stars/results/emp_all_data_reduced.csv')
    emp_new_carb = pd.read_csv('/data/mustard/benlowe/data/rvspecfit_data/aat_stars/cfe_stat_tests/cfe_min_-5/aat_cfe_data.csv')
    data_df = pd.merge(data_df, emp_new_carb[['star_id', 'detect_flag', '2sig_detect']], how='inner', on='star_id')
    
    dict_list = []
    star_ids = info_df['Star ID'].to_numpy()
    
    # Looping over every star in data_df
    for i, entry in data_df.iterrows():
        
        # ---OBSERVED--- #
        # Getting star name and its stellar parameters
        star = str(entry['star_id'])
        teff = float(entry['teff'])
        logg = float(entry['logg'])
        feh = float(entry['z'])
        rv = float(entry['rv'])
        cfe_std = float(entry['randcarb_std'])
        star_params = {'star_id': star, 'teff': teff, 'logg': logg, 'feh': feh, 'rv': rv}
        # star_params = {'star_id': 'ra_1639-2632_s419', 'teff': 5015.57, 'logg': 3.77, 'feh': -3.02, 'rv': 157.62}
        
        # Checking if star_id is from field ra_1648-0653, as it should be ra_1648-0652
        if star[:12] == 'ra_1648-0653':
            first_id = star[:8]
            second_id = '0652'
            third_id = star[12:]
            
            star = first_id + second_id + third_id
            
        # Checking if star_id is from field ra_1650-1651, as it should be ra_1650-1650
        if star[:12] == 'ra_1650-1651':
            first_id = star[:8]
            second_id = '1650'
            third_id = star[12:]
            
            star = first_id + second_id + third_id
            
        # Checking if star_id is from field ra_1656-1433, as it should be ra_1656-1432
        if star[:12] == 'ra_1656-1433':
            first_id = star[:8]
            second_id = '1432'
            third_id = star[12:]
            
            star = first_id + second_id + third_id
        
        # Locating .fits file containing the star's reduced spectral data
        # Fixing up GC naming
        if gc:
            if 'N288' in star:
                star_gc = 'NGC 288 RGB (and AGB) likely members_s' + star.split('N288-')[1]
                star_gc_index = np.where(star_ids == star_gc)[0][0]

                star_sn = info_df.loc[star_gc_index]['SN Blue']
                field = info_df.loc[star_gc_index]['Field']
                
                fits_path = reduce_path + field + '/combined/spectral_data/' + star_gc + '_data.fits'
                star_fits = fits.open(fits_path)
                
            if 'N7099' in star:
                star_gc = 'M30 (NGC7099) Red Giants_s' + star.split('N7099-')[1]
                star_gc_index = np.where(star_ids == star_gc)[0][0]

                star_sn = info_df.loc[star_gc_index]['SN Blue']
                field = info_df.loc[star_gc_index]['Field']
                
                fits_path = reduce_path + field + '/combined/spectral_data/' + star_gc + '_data.fits'
                star_fits = fits.open(fits_path)
                
            if 'N2298' in star:
                star_gc = 'NGC2298 Red Giants_sN2298-' + star.split('N2298-')[1]
                star_gc_index = np.where(star_ids == star_gc)[0][0]

                star_sn = info_df.loc[star_gc_index]['SN Blue']
                field = info_df.loc[star_gc_index]['Field']
                
                fits_path = reduce_path + field + '/combined/spectral_data/' + star_gc + '_data.fits'
                star_fits = fits.open(fits_path)
                
            if 'N1904' in star:
                star_gc = 'NGC1904 RGB (and AGB) stars_sN1904-' + star.split('N1904-')[1]
                star_gc_index = np.where(star_ids == star_gc)[0][0]

                star_sn = info_df.loc[star_gc_index]['SN Blue']
                field = info_df.loc[star_gc_index]['Field']
                
                fits_path = reduce_path + field + '/combined/spectral_data/' + star_gc + '_data.fits'
                star_fits = fits.open(fits_path)
                
            if 'N104' in star:
                star_gc = 'NGC 104 RGB (and AGB) likely members_sN104-' + star.split('N104-')[1]
                star_gc_index = np.where(star_ids == star_gc)[0][0]

                star_sn = info_df.loc[star_gc_index]['SN Blue']
                field = info_df.loc[star_gc_index]['Field']
                
                fits_path = reduce_path + field + '/combined/spectral_data/' + star_gc + '_data.fits'
                star_fits = fits.open(fits_path)
                
            if 'ra_0103-7050' in star:
                star_info = info_df[info_df['Star ID'] == star]
                field = star_info['Field'].astype(str).str.cat(sep=', ') # Doing magic to convert from panda.Series to string
                star_sn = float(star_info['SN Blue']) 
                
                fits_path = reduce_path + field + '/combined/spectral_data/' + star + '_data.fits'
                star_fits = fits.open(fits_path)
        
        # For everything else
        if gc == False:
            star_info = info_df[info_df['Star ID'] == star]
            field = star_info['Field'].astype(str).str.cat(sep=', ') # Doing magic to convert from panda.Series to string
            star_sn = float(star_info['SN Blue']) 
            
            # fits_path = reduce_path + field + '/combined/spectral_data/' + star + '_data.fits'
            fits_path = reduce_path + '/' + star + '_data.fits'
            
            star_fits = fits.open(fits_path)

        
        # Extracting spectral data
        print('Running {0} ({1}/{2})'.format(star, i+1, len(data_df)))
        star_wave = star_fits[1].data['Wave']
        star_flux = star_fits[1].data['Flux']
        star_eflux = star_fits[1].data['Variance']
        
        # Cutting 
        cut = np.where((star_wave >= wave_min) & (star_wave <= wave_max))
        star_wave_cut = star_wave[cut]
        star_flux_cut = star_flux[cut]
        star_eflux_cut = star_eflux[cut]
        
        # Locating .csv file containing the star's chisquare values
        star_chisq_path = chisq_path + field + '/' + star + '_chisq.csv'
        star_chisq_df = pd.read_csv(star_chisq_path)
        star_cfes = star_chisq_df['cfe'].to_numpy()
        star_chisqs = star_chisq_df['chisq'].to_numpy()

        star_best_cfe = star_cfes[np.argmin(star_chisqs)]
        
        # Computing log likelihood for the observed dataset
        star_lnL = lnL_obs(npix=len(star_flux_cut), flux_err=star_eflux_cut, chisq_red=star_chisqs)
        
        # Calculating the cumulative integral for the observed log likelihood
        star_pnorm = p_calc(lnL=star_lnL, cfe_arr=star_cfes)
        
        # Determining P_norm between P(x=1) and P(x=-1)
        pnorm_m1 = scipy.stats.norm.cdf(x=-1)
        pnorm_0 = scipy.stats.norm.cdf(x=0)
        pnorm_p1 = scipy.stats.norm.cdf(x=1)
        
        # Using these P_norm values to find corresponding [C/Fe] values
        index_pnorm_m1 = closest_val(array=star_pnorm, x=pnorm_m1)
        star_cfe_m1 = star_cfes[index_pnorm_m1]

        index_pnorm_0 = closest_val(array=star_pnorm, x=pnorm_0)
        star_cfe_p0 = star_cfes[index_pnorm_0]
        
        index_pnorm_p1 = closest_val(array=star_pnorm, x=pnorm_p1)
        star_cfe_p1 = star_cfes[index_pnorm_p1]
        
        # Finding correspondiong [C/Fe] 2sigma and 1 sigma errors
        star_cfe_2sig_err = star_cfe_p1 - star_cfe_m1
        star_cfe_1sig_err = star_cfe_2sig_err/2
        
        
        # ---SYNTHETIC (NULL HYPOTHESIS)--- #
        # Getting the synthetic spectra at lowest [C/Fe]
        try:
            synth_chisqs_cfemin, synth_spec_cfemin = chisq_run_cfemin(wave=star_wave_cut, flux=star_flux_cut, eflux=star_eflux_cut, 
                                                                    conf_name=conf_name, atm_params=star_params, config=config,
                                                                    carb_vals=star_cfes, min_cfe=min_cfe, cont_npoly=2)
        except: 
            print('Synthetic spectrum not found - skipping null hypothesis')
            cfe_dict = {'star_id': star, 'detect_flag': 'broke','cfe': star_cfe_p0, 'cfe_err': star_cfe_1sig_err, '2sig_detect': np.nan}
            dict_list.append(cfe_dict)
            continue

            
        synth_flux_cfemin = synth_spec_cfemin.flux
        synth_wave_cfemin = synth_spec_cfemin.spectral_axis

        # Extracting list of reduced chisquares
        synth_redchisqs_cfemin = []
        for entry in synth_chisqs_cfemin:
            chisq = entry['chisq_array'][0]
            synth_redchisqs_cfemin.append(chisq)
        synth_redchisq_cfemin_arr = np.array(synth_redchisqs_cfemin)
        
        # Computing log likelihood of model dataset
        synth_lnL = lnL_model(npix=len(synth_flux_cfemin), flux_err=star_eflux_cut, chisq_red=synth_redchisq_cfemin_arr)
        
        # Calculating the cumulative integral
        synth_pnorm_cfemin = p_calc(lnL=synth_lnL, cfe_arr=star_cfes)

        # Determining detection limit at P(x=2)
        pnorm_p2 = scipy.stats.norm.cdf(x=2)

        index_synth_pnorm_p2 = closest_val(array=synth_pnorm_cfemin, x=pnorm_p2)
        star_cfe_p2 = star_cfes[index_synth_pnorm_p2] # 2sigma detection limit
        
        # If 2 sigma detection limit greater than observed, then flag as 'no'. Otherwise detections are 'yes'
        if star_cfe_p2 >= star_cfe_p0:
            detect_flag = 'no'
            
        if star_cfe_p2 < star_cfe_p0:
            detect_flag = 'yes'
        
        if plot is True:            
            star_cfe_1sig_err = float(data_df[data_df['star_id'] == star]['randcarb_std'])
            star_cfe_2sig_err = float(data_df[data_df['star_id'] == star]['2sig_detect'])
            star_best_cfe = float(data_df[data_df['star_id'] == star]['cfe'])
            # # Getting [C/Fe] errors
            # star_cfe_std = float(emp_df[emp_df['star_id'] == star]['randcarb_std'])
            # star_best_cfe = float(emp_df[emp_df['star_id'] == star]['cfe'])
            
            # # Plotting the spectra for stars with no detectable [C/Fe] using upper limits
            # no_detect_stars = ['ra_1604-2712_s188', 'ra_1624-2150_s36', 'ra_1633-2814_s284', 'ra_1659-2154_s114', 'ra_1832-3457_s438']
            # no_detect_2sig = [1.39, 0.69, 1.33, 0.93, 1.14]
            
            if detect_flag == 'no':
                # # Getting corresponding [C/Fe] 2sig upper limit for star
                # star_index = no_detect_stars.index(star)
                # star_cfe_2sig_err = no_detect_2sig[star_index]
                
                # Extracting the synthetic spectrum
                synth_spec_final, synth_spec_psig, synth_spec_msig = synth_extract(wave=star_wave_cut, flux=star_flux_cut, eflux=star_eflux_cut, 
                                                                                    conf_name=conf_name, atm_params=star_params, config=config, one_sigma=star_cfe_1sig_err, 
                                                                                    cfe_val=star_cfe_2sig_err, cont_npoly=2)
                
                # Normalizing
                bandpass = [4150, 4190, 440, 4440]
                star_flux_norm = spec_normal(spectra=Spectrum1D(flux=star_flux_cut * u.Jy, spectral_axis=star_wave_cut*u.AA), cont_band=bandpass)
                synth_final_flux_norm = spec_normal(spectra=Spectrum1D(flux=synth_spec_final * u.Jy, spectral_axis=star_wave_cut*u.AA), cont_band=bandpass)
                synth_empty_flux_norm = spec_normal(spectra=Spectrum1D(flux=synth_flux_cfemin * u.Jy, spectral_axis=star_wave_cut*u.AA), cont_band=bandpass)
                
                # Cutting to desired wavelength region
                plot_cut = np.where((star_wave_cut >= 4260) & (star_wave_cut <= 4370))
                star_wave_plt = star_wave_cut[plot_cut]
                star_flux_plt = star_flux_norm.flux[plot_cut]
                synth_final_flux_plt = synth_final_flux_norm.flux[plot_cut]
                synth_empty_flux_plt = synth_empty_flux_norm.flux[plot_cut]
                
                # Plotting the results 
                plt.figure(figsize=[8,6])
                plt.plot(star_wave_plt, star_flux_plt, linewidth=0.65, color='black', 
                        label='{0}\n[{1:.0f}, {2:.2f}, {3:.2f}]\nS/N = {4:.2f}'.format(star, teff, logg, feh, star_sn))
                plt.plot(star_wave_plt, synth_final_flux_plt, linewidth=2, color='darkblue', label='[C/Fe] < {0:.2f}'.format(star_cfe_2sig_err))
                # plt.plot(star_wave_plt, synth_empty_flux_plt, linewidth=0.8, linestyle='--', color='black')

                # plt.fill_between(star_wave_plt, synth_final_flux_plt, synth_empty_flux_plt, color='blue', alpha=0.3)

                plt.xlim(4260, 4370)
                plt.xlabel('Wavelength (Å)')
                plt.ylabel('Normalised Flux')
                plt.legend(fancybox=False)
                plt.tight_layout()
                
                plt.savefig(save_path + '{0}_fits.pdf'.format(star), dpi=300, transparent=False, facecolor='white',
                            bbox_inches='tight')
                plt.close()
            
                
                
            else:
                # Extracting the synthetic spectrum
                synth_spec_final, synth_spec_psig, synth_spec_msig = synth_extract(wave=star_wave_cut, flux=star_flux_cut, eflux=star_eflux_cut, 
                                                                                    conf_name=conf_name, atm_params=star_params, config=config, one_sigma=star_cfe_1sig_err, 
                                                                                    cfe_val=star_best_cfe, cont_npoly=2)
                
                # Normalizing
                bandpass = [4150, 4190, 440, 4440]
                star_flux_norm = spec_normal(spectra=Spectrum1D(flux=star_flux_cut * u.Jy, spectral_axis=star_wave_cut*u.AA), cont_band=bandpass)
                synth_psig_flux_norm = spec_normal(spectra=Spectrum1D(flux=synth_spec_psig * u.Jy, spectral_axis=star_wave_cut*u.AA), cont_band=bandpass)
                synth_msig_flux_norm = spec_normal(spectra=Spectrum1D(flux=synth_spec_msig * u.Jy, spectral_axis=star_wave_cut*u.AA), cont_band=bandpass)
                synth_final_flux_norm = spec_normal(spectra=Spectrum1D(flux=synth_spec_final * u.Jy, spectral_axis=star_wave_cut*u.AA), cont_band=bandpass)
                synth_empty_flux_norm = spec_normal(spectra=Spectrum1D(flux=synth_flux_cfemin * u.Jy, spectral_axis=star_wave_cut*u.AA), cont_band=bandpass)
                
                # Cutting to desired wavelength region
                plot_cut = np.where((star_wave_cut >= 4260) & (star_wave_cut <= 4370))
                star_wave_plt = star_wave_cut[plot_cut]
                star_flux_plt = star_flux_norm.flux[plot_cut]
                synth_final_flux_plt = synth_final_flux_norm.flux[plot_cut]
                synth_psig_flux_plt = synth_psig_flux_norm.flux[plot_cut]
                synth_msig_flux_plt = synth_msig_flux_norm.flux[plot_cut]
                synth_empty_flux_plt = synth_empty_flux_norm.flux[plot_cut]

                # Plotting the results 
                plt.figure(figsize=[8,6])
                plt.plot(star_wave_plt, star_flux_plt, linewidth=0.65, color='black', 
                        label='{0}\n[{1:.0f}, {2:.2f}, {3:.2f}]\nS/N = {4:.2f}'.format(star, teff, logg, feh, star_sn))
                plt.plot(star_wave_plt, synth_final_flux_plt, linewidth=2, color='darkred', label='[C/Fe] = {0:.2f} ± {1:.2f}'.format(star_best_cfe, cfe_std))
                plt.plot(star_wave_plt, synth_psig_flux_plt, linewidth=0.8, color='darkred')
                plt.plot(star_wave_plt, synth_msig_flux_plt, linewidth=0.8, color='darkred')
                # plt.plot(star_wave_plt, synth_empty_flux_plt, linewidth=0.8, linestyle='--', color='black')

                plt.fill_between(star_wave_plt, synth_psig_flux_plt, synth_msig_flux_plt, color='red', alpha=0.3)
                
                plt.xlim(4260, 4370)
                plt.xlabel('Wavelength (Å)')
                plt.ylabel('Normalised Flux')
                plt.legend(fancybox=False)
                plt.tight_layout()
                
                plt.savefig(save_path + '{0}_fits.pdf'.format(star), dpi=300, transparent=False, facecolor='white',
                            bbox_inches='tight')
                plt.close()
        
        # ---SAVING DATA--- #
        cfe_dict = {'star_id': star, 'detect_flag': detect_flag,'cfe': star_cfe_p0, 'cfe_err': star_cfe_1sig_err, '2sig_detect': star_cfe_p2, 'cfe_randstd': cfe_std, 'chisq': np.argmin(star_chisqs)}
        dict_list.append(cfe_dict)
        
    # cfe_df = pd.DataFrame(dict_list)
    # cfe_df.to_csv(save_path + 'gc_cfe_data.csv')


if __name__ == '__main__':
    # ---RVSPECFIT PARAMETERS--- #
    # Configuration for rvspecfit parameters
    config_save_path = '/data/mustard/benlowe/data_analysis/rvspecfit/interpolation/'
    config = {
        'template_lib' : '/data/mustard/benlowe/data_analysis/rvspecfit/interpolation/580V_Turbo_cfe_v2/',
        'min_vel': -600,
        'max_vel': 600,
        'min_vel_step': 0.2,
        'vel_step0': 5,
        'min_vsini': 0.1,
        'max_vsini': 11,
        'second_minimizer': 1
        }
    with open(config_save_path + 'config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    config_ = utils.read_config('/data/mustard/benlowe/data_analysis/rvspecfit/interpolation/config.yaml')

    # Names of the configuration files
    b_conf = 'blueconf_turbo_fixed_cfe'

    # Wavelength ranges for blue spectra
    min_b_wave = 4150
    max_b_wave = 4450

    # Number of polynomial continuum points
    cont_npol = 2
    
    # Minimum [C/Fe]
    cfe_min = -5.0
    
    
    # ---DATA DIRECTORIES + DATAFRAMES--- #    
    # # AAT data
    aat_data_df = pd.read_csv('/data/mustard/benlowe/data/rvspecfit_data/aat_stars/results/aat_all_data_reduced.csv')
    emp_stars = ['ra_0103-7050_s236',
                 'ra_0752-5047_s47',
                 'ra_1604-2712_s188',
                 'ra_1604-2712_s292',
                 'ra_1633-2814_s130',
                 'ra_1633-2814_s284',
                 'ra_1639-2632_s419',
                 'ra_1648-2642_s91',
                 'ra_1659-2154_s114',
                 'ra_1659-2154_s261',
                 'ra_1659-2154_s347',
                 'ra_1752-4300_s155',
                 'ra_1752-4300_s269',
                 'ra_1832-3457_s438',
                 'ra_1853-3255_s51']
    aat_data_df = aat_data_df[aat_data_df['star_id'].isin(emp_stars)]
    
    aat_info_df = pd.read_csv('/data/mustard/benlowe/aat_data_reduction/extracted_data_v2/aat_comb_info.csv')
    aat_reduce_path = '/data/mustard/benlowe/aat_data_reduction/extracted_data_v2/EMP/'
    aat_chisq_path = '/data/mustard/benlowe/data/rvspecfit_data/aat_stars/new_carb_res_wider_v2/'
    aat_save_path = '/data/mustard/benlowe/results/paper_plots/paper1/emp_cfe_fits/FINAL/'
    # cfe_save_path = '/data/mustard/benlowe/data/rvspecfit_data/aat_stars/results/'
    # emp_cfe_save_path = '/data/mustard/benlowe/results/paper_plots/paper1/emp_cfe_fits/version3/'

    # # GC data
    # gc_data_df = pd.read_csv('/data/mustard/benlowe/data/rvspecfit_data/gc_stars/results/gc_all_data_reduced.csv')
    # gc_info_df = pd.read_csv('/data/mustard/benlowe/aat_data_reduction/extracted_data_v2/gc_final/gc_comb_info_v2.csv')
    # gc_reduce_path = '/data/mustard/benlowe/aat_data_reduction/extracted_data_v2/gc_final/'
    # gc_chisq_path = '/data/mustard/benlowe/data/rvspecfit_data/gc_stars/results/cfe_results/'
    # gc_save_path = '/data/mustard/benlowe/data/rvspecfit_data/gc_stars/results/cfe_upper_limits/'
        
    
    # --RUNNING [C/FE] DETERMINATION CODE--- #
    num_processes = 60  
    with Pool(processes=num_processes) as pool:
        try:
            result = pool.map(run_carb_err(data_df=aat_data_df, info_df=aat_info_df, reduce_path=aat_reduce_path, chisq_path=aat_chisq_path,
                                           wave_min=min_b_wave, wave_max=max_b_wave, conf_name=b_conf, config=config_, min_cfe=cfe_min, save_path=aat_save_path,
                                           plot=True, gc=False),
                              range(num_processes))
        except TypeError:
            pass
    
    
    