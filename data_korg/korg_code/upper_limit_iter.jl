using Korg
using CSV
using DataFrames
using PyPlot
using PyCall
using Distributed
using Statistics
using GLM
import Logging
gammaincc = pyimport("scipy.special").gammaincc
fits = pyimport("astropy.io.fits")
plt = pyimport("matplotlib.pyplot")
ast = pyimport("ast")

function get_abunds(path, elements)
    """
    Loading in the stellar abundances from the given path.

    Parameters
    ----------
    path : Str
        Path to the stellar abundances.
    elements : Vector{Str}
        List of elements to load abundances for.

    Returns
    -------
    star_abunds_xh_df : DataFrame
        DataFrame containing the stellar [X/H] abundances for the specified elements.
    star_abunds_xfe_df : DataFrame
        DataFrame containing the stellar [X/Fe] abundances for the specified elements.
    """
    
    # Loading in the stellar abundances from the given path
    all_abunds_xh = []
    all_abunds_xfe = []

    for el in elements
        # Ignoring the Ca-I abundances
        if el == "Ca_I"
            continue
        end

        star_abund_xh = CSV.read(path * el * "_abund.csv", DataFrame)[!, ["$(el)_xh_it1"]]
        star_abund_xfe = CSV.read(path * el * "_abund.csv", DataFrame)[!, ["$(el)_xfe_it1"]]
        push!(all_abunds_xh, star_abund_xh)
        push!(all_abunds_xfe, star_abund_xfe)
    end

    # Combining the abundances into a single DataFrame
    star_abunds_xh = reduce(hcat, all_abunds_xh)
    star_abunds_xfe = reduce(hcat, all_abunds_xfe)

    # Convert to dictionaries with element names as keys
    df_xh_cols = names(star_abunds_xh)
    df_xh_cols = [split(col, "_")[1] for col in df_xh_cols if split(col, "_")[3] == "xh"] # Getting the element names from the column names
    star_abunds_xh_df = DataFrame(Dict(df_xh_cols .=> collect(star_abunds_xh[1,:])))

    df_xfe_cols = names(star_abunds_xfe)
    df_xfe_cols = [split(col, "_")[1] for col in df_xfe_cols if split(col, "_")[3] == "xfe"] # Getting the element names from the column names
    star_abunds_xfe_df = DataFrame(Dict(df_xfe_cols .=> collect(star_abunds_xfe[1,:])))

    return star_abunds_xh_df, star_abunds_xfe_df
end


function get_window(path)
    """
    Loading in the window from the given path.

    Parameters
    ----------
    path : Str
        Path to the window.

    Returns
    -------
    windows : Dict
        Dictionary with the windows for each element.
    """

    # Loading in the window from the given path
    windowtable = CSV.read(path, DataFrame)

    # Creating a dictionary with the windows
    windows = Dict()
    for row in eachrow(windowtable)
        # atomic number
        Z = Korg.get_atoms(Korg.Species(row.species))[1]
        # get the windows for this element so far 
        wins = get(windows, Z, [])
        # add the new window
        push!(wins, (row.wavepeak, Korg.air_to_vacuum(row.wavebase), Korg.air_to_vacuum(row.wavetop)))
        # update the dictionary
        windows[Z] = wins
    end

    return windows
end

function reduced_chisqr(obs_flux, obs_flux_err, fit_flux, n_params)
    """ Calculate the chi-squared and reduced chi-squared for the given observed and the fitted flux.

    Parameters
    ----------
    obs_flux : Vector{Float64}
        The observed flux values.
    obs_flux_err : Vector{Float64}
        The errors in the observed flux values.
    fit_flux : Vector{Float64}
        The fitted flux values.
    n_params : Int
        The number of parameters in the fit.

    Returns
    -------
    chisq : Float64
        The chi-squared value.
    reduced_chisq : Float64
        The reduced chi-squared value.
    """
    # Calculate chi-squared
    chisq = sum(((obs_flux .- fit_flux) .^ 2) ./ (obs_flux_err .^ 2))

    # Degrees of freedom
    dof = length(obs_flux) - n_params

    # Reduced chi-squared
    reduced_chisq = chisq / dof

    return chisq, reduced_chisq
end

function p_value(chisq, dof)
    """ Calculate the p-value for the given chi-squared and degrees of freedom.

    Parameters
    ----------
    chisq : Float64
        The chi-squared value.
    dof : Int
        The degrees of freedom.

    Returns
    -------
    p_val : Float64
        The p-value.
    """
    # Using the incomplete gamma function to calculate the p-value
    p_val = gammaincc(dof / 2, chisq / 2)

    return p_val
end


function ew_min(wave_cent, sn, pixel_step, res)
    """ Calculate the minimum equivalent width (EW) for a given signal-to-noise ratio (sn), pixel step, and resolution (res).

    Parameters:
    ----------
    sn : float
        The signal-to-noise ratio of the spectra.
    pixel_step : float
        The pixel step size in the spectra.
    res : float
        The resolution of the spectra.
    
    Returns:
    -------
    min_ew : float
        The minimum observed equivalent width of the given spectra.
    """

    # Defining the parameters
    epsilon = 1/sn
    delta_x = pixel_step
    fwhm = wave_cent / res
    
    # Calculate the minimum equivalent width (in Angstroms)
    min_ew = 3 * (1.5 * sqrt(fwhm * delta_x) * epsilon)
    return min_ew
end

function gauss(waves, wave_cent, res, ew, empty_spec)
    """ Creates a Gaussian function rescaled to the minimum EW, with a FWHM determined by the spectral resolution.
    
    Parameters:
    ----------
    waves : Vector{Float64}
        The wavelength array for which the Gaussian is calculated.
    wave_cent : Float64
        The central wavelength of the line.
    res : Float64
        The spectral resolution of the instrument.
    ew : Float64
        The equivalent width of the line.
    empty_spec : Vector{Float64}
        The empty spectrum to be used to inject the Gaussian into.

    Returns:
    -------
    gaussian : Vector{Float64}
        The Gaussian function values at the specified wavelengths.
    """

    # Calculate the FWHM given the spectral resolution
    fwhm = wave_cent / res

    # Calculate the standard deviation of the Gaussian using the FWHM
    sigma = fwhm / (2 * sqrt(2 * log(2) * 2))

    # Calculate the amplitude
    A = ew / (sigma * sqrt(2 * pi))

    # Create the Gaussian function
    gaussian = empty_spec .- A * exp.(-1 * (waves .- wave_cent).^2 ./ (2 * sigma^2))

    return gaussian
end

function run_upperlimit(path, vald_linelist, el_linelist_path, params_stars, abunds_path, nondetects, pvalue_regions, savepath)

    # ---LOADING IN THE SPECTRAL DATA--- #
    # Finding and extracting the .fits data
    star_id = split(path, "/")[end]
    println("Running $star_id")

    fits_file = string(path, "/", star_id, "_data.fits")
    star_fits = fits.open(fits_file)
    fits_header = convert(Array, star_fits[1].header)
    
    # Getting the UVB data
    uvb_data = star_fits[2].data
    uvb_min = Korg.air_to_vacuum(3000)
    uvb_max = Korg.air_to_vacuum(5560)

    # uvb_wave = Korg.air_to_vacuum.(uvb_data["wave"]) # Converting from air to vacuum
    uvb_wave = convert(Array, uvb_data["wave"])
    uvb_flux = convert(Array, uvb_data["flux"])
    uvb_flux_err = convert(Array, uvb_data["flux_err"])
    uvb_res = 5600
    uvb_snr = fits_header[6] # SNR from the header    

    # Getting the VIS data
    vis_data = star_fits[3].data
    vis_min = Korg.air_to_vacuum(5600)
    vis_max = Korg.air_to_vacuum(10200)

    # vis_wave = Korg.air_to_vacuum.(vis_data["wave"]) # Converting from air to vacuum
    vis_wave = convert(Array, vis_data["wave"])
    vis_flux = convert(Array, vis_data["flux"])
    vis_flux_err = convert(Array, vis_data["flux_err"])
    vis_res = 8900
    vis_snr = fits_header[7] # SNR from the header    


    # ---EXTRACTING THE STELLAR PARAMETERS--- #
    # Getting the stellar parameters
    star_params = params_stars[params_stars.star_id .== star_id, :]
    teff = star_params.teff[1]
    logg = star_params.logg[1]
    feh = star_params.Fe_it1[1]
    afe = 0.4 # [Alpha/Fe]
    ah = afe + feh # [Alpha/H]
    vmic = star_params.vmic[1]
    vsini = star_params.vsini[1]
    epsilon = 0


    # ---SOLAR ABUNDANCES--- #
    # Loading them in directly from Korg
    solar_abunds_ax = Korg.asplund_2020_solar_abundances


    # ---EXTRACTING ELEMENTS WITH NO DETECTIONS--- #
    # Extracting data from the non-detections list
    element_cols = names(nondetects)[2:end] # Getting elements, which are column names. This excludes the star_id column.
    detects_df = nondetects[nondetects.star_id .== star_id, :] # Getting non-detections for the star
    detect_flags = [col[1] for col in eachcol(detects_df)][2:end] # Getting the detection flags for each element, excluding the star_id column
    no_detects = [element_cols[i] for (i, flag) in pairs(detect_flags) if flag == "N"] # Elements with no detections
    yes_detects = [element_cols[i] for (i, flag) in pairs(detect_flags) if flag == "Y"] # Elements with yes detections

    # Gettting the stellar abundances for both yes and no detections
    star_abunds_xh_yes, star_abunds_xfe_yes = get_abunds(abunds_path * star_id * "/", yes_detects)
    star_abunds_xh_no, star_abunds_xfe_no = get_abunds(abunds_path * star_id * "/", no_detects)

    star_abunds_xfe_all = hcat(star_abunds_xfe_yes, star_abunds_xfe_no) # Combining the [X/Fe] abundances for all elements
    star_abunds_xh_all = hcat(star_abunds_xh_yes, star_abunds_xh_no) # Combining the [X/H] abundances for all elements
    println("Checking for non-detections in $star_id: $(no_detects)\n")


    # ---DETERMINING THE UPPER-LIMITS--- #
    star_savepath = savepath * star_id * "/" 
    mkpath(star_savepath)
     

    # Looping through the elements with no detections
    # no_detects = ["Sr_II"]
    

    for el in no_detects
        if el == "C_I" || el == "N_I"
            println("Skipping $el - run log likelihood code instead\n")
            continue
        end
        el_savepath = star_savepath * el * "/"
        mkpath(el_savepath) # Making the directory for the element
        el_spec_savepath = el_savepath * "specs/"
        mkpath(el_spec_savepath) # Making the directory for the element's spectra

 
        # ---EXTRACTING THE ELEMENTS' WINDOWS AND LINELISTS--- #
        # Getting the path to the linelist and window files for the element
        el_path = el_linelist_path * el * "/"
        el_dir = readdir(el_path)

        el_lines_path = [i for i in el_dir if occursin("lines_$(el)_ALL.csv", i)][1] # Getting the linelist file
        el_window_path = [j for j in el_dir if occursin("windows_$(el)_ALL.csv", j)][1] # Getting the window file

        # Loading in the linelist window for the element
        el_ = split(el, "_")[1] # Getting the element name without the ionization state
        
        el_linelist = CSV.read(el_path * el_lines_path, DataFrame) # In CSV format, will put into Korg format later
        el_lines = el_linelist.wave # Getting the wavelengths of the lines in the linelist
        el_ews = el_linelist.ew # Getting the equivalent widths of the lines in the linelist
        el_windows_korg = get_window(el_path * el_window_path)[Korg.atomic_numbers[el_]] # In Korg format

         # Getting the p-value regions for the element
        pval_region = pvalue_regions[pvalue_regions.element .== el, :] 

        # Looping through the windows
        abunds_all = []
        for (k, wind) in enumerate(el_windows_korg)

            # Getting the window properties
            line_peak = wind[1]
            windmin = wind[2]
            windmin_air = Korg.vacuum_to_air(windmin) # Converting from vacuum to air
            windmax = wind[3]
            windmax_air = Korg.vacuum_to_air(windmax) # Converting from vacuum to air

            println("Running $el for lines in the $(line_peak) Å window: $(round(windmin_air, digits=3)) - $(round(windmax_air, digits=3)) Å")

            # Getting the individual line abundances for this star
            el_line_abunds_path = abunds_path * star_id * "/" * el * "/iter_1/$(el_)_results.csv"
            el_line_abunds_df = CSV.read(el_line_abunds_path, DataFrame)

            line_entry = "$(el_)_$(line_peak)"
            el_bestfit_abund = el_line_abunds_df[el_line_abunds_df.line .== line_entry, :][!, ["abund_xfe"]][1,1]

            # Making sure to use the correct spectral arm
            if windmin < uvb_max
                wave = uvb_wave
                wavemin = uvb_min
                wavemax = uvb_max
                flux = uvb_flux
                flux_err = uvb_flux_err
                res = uvb_res
                snr = uvb_snr
            end

            if windmin > vis_min && windmin < vis_max
                wave = vis_wave
                wavemin = vis_min
                wavemax = vis_max
                flux = vis_flux
                flux_err = vis_flux_err
                res = vis_res
                snr = vis_snr
            end


            # ---EXTRACTING THE ELEMENT'S BEST FIT SPECTRA--- #
            # Using linepeak to get the best fit spectra
            bestfit_path = abunds_path * star_id * "/" * el * "/iter_1/fits_data/$(el_)_$(line_peak)_fit.csv"
            bestfit_data = CSV.read(bestfit_path, DataFrame)

            bestfit_wave = bestfit_data.wave
            bestfit_flux = bestfit_data.best_fit


            # ---GETTING THE OBSERVED SPECTRA OVER SAME REGION--- #
            # Slicing the observed spectra to the window
            obs_wave = wave[(wave .>= windmin_air) .& (wave .<= windmax_air)]
            obs_flux = flux[(wave .>= windmin_air) .& (wave .<= windmax_air)]
            obs_flux_err = flux_err[(wave .>= windmin_air) .& (wave .<= windmax_air)]


            # ---GETTING EMPTY SPECTRA--- #
            # Fitting a straight line to the observed flux. Will only fit the top 95% of the flux values to
            # ensure we only fit the continuum and not the absorption lines.
            threshold = quantile(obs_flux, 0.05)   # 5th percentile (top 95% of the flux values)
            mask = obs_flux .> threshold  
            empty_wave = obs_wave[mask]
            empty_fit = lm(@formula(flux ~ wave), DataFrame(wave=empty_wave, flux=obs_flux[mask]))
            empty_flux = predict(empty_fit)


            # ---PERFORMING THE P-TEST--- #
            println("Performing p-test")

            # Getting the p-value regions for the specific line of the element
            line_pval_reg = pval_region[pval_region.line .== line_peak, :]
            pval_min = line_pval_reg.wavemin[1]
            pval_max = line_pval_reg.wavemax[1]

            pval_mask_obs = (obs_wave .>= pval_min) .& (obs_wave .<= pval_max)
            pval_mask_bestfit = (bestfit_wave .>= pval_min) .& (bestfit_wave .<= pval_max)
            pval_mask_empty = (empty_wave .>= pval_min) .& (empty_wave .<= pval_max)

            # Calculating the chi-squared and reduced chi-squared between the following:
            # 1. The best fit spectra and the observed spectra
            # 2. The empty spectra and the observed spectra
            obs_fit_nparams = 3 # Number of parameters in the best fit (slope, intercept, abundance)
            obs_fit_chisqr, obs_fit_redchisqr = reduced_chisqr(obs_flux[pval_mask_obs], obs_flux_err[pval_mask_obs], bestfit_flux[pval_mask_bestfit], obs_fit_nparams)

            obs_empty_nparams = 2 # Number of parameters in the empty fit (slope, intercept)
            obs_empty_chisqr, obs_empty_redchisqr = reduced_chisqr(obs_flux[mask][pval_mask_empty], obs_flux_err[mask][pval_mask_empty], empty_flux[pval_mask_empty], obs_empty_nparams)


            # Calculating the p-value for the following:
            # 1. The best fit spectra and the observed spectra
            # 2. The empty spectra and the observed spectra
            # 3. The difference between the two
            obs_fit_dof = length(obs_flux[pval_mask_obs]) - obs_fit_nparams
            obs_fit_pval = p_value(obs_fit_chisqr, obs_fit_dof)

            obs_empty_dof = length(obs_flux[mask][pval_mask_empty]) - obs_empty_nparams
            obs_empty_pval = p_value(obs_empty_chisqr, obs_empty_dof)

            diff_chisqr = abs(obs_fit_chisqr - obs_empty_chisqr)
            diff_dof = obs_fit_dof - obs_empty_dof
            diff_pval = p_value(diff_chisqr, diff_dof)


            # Plotting the results
            # el_bestfit_abund = star_abunds_xfe_all[!, el_][1] # Getting the best fit abundance for the element

            figure(figsize=(10, 6))
            title("Star $(star_id)\nBest fit reduced chisqr: $(round(obs_fit_redchisqr, digits=2)) (p = $(round(obs_fit_pval * 100, digits=2))%)\nEmpty reduced chisqr: $(round(obs_empty_redchisqr, digits=2)) (p = $(round(obs_empty_pval * 100, digits=2))%)", fontsize=12)
            plot(bestfit_wave, bestfit_flux, label="Best fit\n[$(el_)/Fe] = $(round(el_bestfit_abund, digits=2))", color="chocolate")
            plot(obs_wave, obs_flux, label="Observed", color="black", linewidth=0.9)
            plot(empty_wave, empty_flux, label="Empty fit", color="green", linestyle="--")
            axvspan(pval_min, pval_max, color="grey", alpha=0.3)
            xlabel("Wavelength (Å)")
            ylabel("Normalised Flux")
            legend(fancybox=false)
            savefig(el_savepath * "$(el_)_$(line_peak)_ptest.png", dpi=300, bbox_inches="tight")
            close(gcf())
            # display(gcf())

            
            # ---CALCULATING MINIMUM EW--- #
            println("Calculating minimum EW")

            # Get the wavelengths of lines in the window
            lines_wind = el_lines[(el_lines .>= windmin) .& (el_lines .<= windmax)] 

            # Will use Eqn. 6' from Cayrel (1988) to calculate the minimum equivalent width (EW) for the given SNR, 
            # pixel step, and resolution. 
            pixel_step = mean(diff(obs_wave)) # Calculating the pixel step size

            min_ew_list = [] 
            for line in lines_wind
                # Calculate the minimum EW for the line
                min_ew = ew_min(line, snr, pixel_step, res) # In Å
                push!(min_ew_list, min_ew)
            end
        

            # ---CALCULATING GAUSSIANS--- #
            println("Calculating Gaussians")

            # For each line in the window, calculate the Gaussian function
            gaussians = []
            for (i, line) in enumerate(lines_wind)
                # Calculate the Gaussian function for the line
                min_ew = min_ew_list[i]
                gaussian = gauss(empty_wave, line, res, min_ew, empty_flux)
                push!(gaussians, gaussian)
            end
            # Combine the gaussians into a single array
            gaussians_combined = reduce(hcat, gaussians)

            # Plot the Gaussians
            figure(figsize=(10, 6))
            plot(obs_wave, obs_flux, label="Observed", color="black", linewidth=0.9)
            linestyles = [":", "--", "-.", "-"]
            for (l, gauss) in enumerate(gaussians)
                plot(empty_wave, gauss, color="orchid", label="Line $(lines_wind[l]) Å\nEW_min = $(round(min_ew_list[l], digits=4)) Å", linestyle=linestyles[l % length(linestyles)])
            end
            axvspan(pval_min, pval_max, color="grey", alpha=0.3)
            xlabel("Wavelength (Å)")
            ylabel("Normalised Flux")
            legend(fancybox=false)
            savefig(el_savepath * "$(el_)_$(line_peak)_gaussians.png", dpi=300, bbox_inches="tight")
            close(gcf())


            # ---CONVERTING MINIMUM EQUIVALENT WIDTH TO ABUNDANCE--- #
            println("Converting minimum EW to [$el_/Fe]")

            # Taking the strongest line in the window from the linelist
            ews_wind = el_ews[(el_lines .>= windmin) .& (el_lines .<= windmax)] # Equivalent widths of the lines in the window
            strongest_line = lines_wind[findmax(ews_wind)[2]] # Getting the wavelength of the strongest line
            ew_strongest = min_ew_list[findmax(ews_wind)[2]] # Getting the equivalent width of the strongest line

            # Creating linelist with just the strongest line, then converting to Korg format
            min_ew_linelist = el_linelist[el_linelist.wave .== strongest_line, :]
            min_ew_linelist_korg = Korg.Line.(Korg.air_to_vacuum.(min_ew_linelist.wave),
                                              min_ew_linelist.loggf,
                                              Korg.Species.(min_ew_linelist.species),
                                              min_ew_linelist.elow) 


            # For elements with upper-limits:
            # - If [X/H] >= 0, then set [X/H] = 0
            # - If 0 > [X/H] > -1.5, then use current [X/H] value
            # - If [X/H] <= -1.5, then set [X/H] = 0
            abunds_yes_xh = Dict(pairs(eachcol(star_abunds_xh_yes)))
            abunds_yes_xfe = Dict(pairs(eachcol(star_abunds_xfe_yes)))
            abunds_no_xh = Dict(pairs(eachcol(star_abunds_xh_no)))
            abunds_no_xfe = Dict(pairs(eachcol(star_abunds_xfe_no)))

            for (el, abund) in pairs(abunds_no_xfe)
                # If [X/H] >= 0, then set [X/H] = 0
                if abund[1] > 0
                    abunds_no_xfe[el] = [0.0]
                    abunds_no_xh[el] = [0.0 + feh]
                end
                # If [X/H] < 1.5, then use current [X/H] value as it is likely stupid
                if abund[1] < -1.5
                    abunds_no_xfe[el] = [0.0]
                    abunds_no_xh[el] = [0.0 + feh] 
                end
            end

            # Combining the abundances into a single dictionary
            # Note: all abundances are stored as a list in each key, so will will need to 
            # make them into a single value
            abunds_all_xh = merge(abunds_yes_xh, abunds_no_xh)
            abunds_all_xh = Dict(el => abund[1] for (el, abund) in abunds_all_xh)
            abunds_all_xfe = merge(abunds_yes_xfe, abunds_no_xfe)
            abunds_all_xfe = Dict(el => abund[1] for (el, abund) in abunds_all_xfe)

            # Converting the abundances to Korg format
            A_x = Korg.format_A_X(feh, ah, abunds_all_xh)

            # Interpolating the MARCS model atmosphere
            atm = Korg.interpolate_marcs(teff, logg, A_x, clamp_abundances=true)

            # Converting from EW to A(X), then A(X) to [X/H], then [X/H] to [X/Fe]
            # Korg requires the EW to be in mÅ, so we multiply min_ew_list by 1000
            # println(ew_strongest)
            @time min_abund = Korg.Fit.ews_to_abundances(atm, min_ew_linelist_korg, A_x, [ew_strongest] * 1000)
            min_abund_xh = min_abund .- solar_abunds_ax[Korg.atomic_numbers[el_]] 
            min_abund_xh = min_abund_xh[1]

            min_abund_xfe = min_abund_xh .- feh # [X/Fe] = [X/H] - [Fe/H]
            min_abund_xfe = min_abund_xfe[1]


            # ---GENERATE SYNTHETIC SPECTRA AT ABUNDANCE--- #
            println("Generating synthetic spectra at abundance")

            # Generating the synthetic spectra at the minimum abundance
            # Including given element abundance in the dictionary
            guesses = Dict("cntm_offset" => 0.0, "cntm_slope" => 0.0)
            params = Dict("Teff" => teff, "logg" => logg, "m_H" => feh, "vmic" => vmic, "vsini" => vsini, "epsilon" => epsilon, "$(split(el, "_")[1])" => min_abund_xh)

            # If the minimum abundance is NaN, then its a detection. Line is too strong, so Korg blows up. Thus,
            # will save the results and continue to the next element.
            if isnan(min_abund_xh)
                println("Minimum abundance is NaN, so automatic detection. Skipping element $el")
                min_abund_df = DataFrame(el=el, line_peak=line_peak, detection='Y', min_abund_xh=min_abund_xh, min_abund_xfe=min_abund_xfe, bestfit_abund_xh=el_bestfit_abund + feh, bestfit_abund_xfe=el_bestfit_abund)
                push!(abunds_all, min_abund_df)
                CSV.write(el_savepath * "$(el_)_$(line_peak)_abund.csv", min_abund_df)

                # Saving the p-values
                pval_df = DataFrame(obs_fit_chisqr=obs_fit_chisqr, obs_fit_redchisqr=obs_fit_redchisqr, obs_fit_pval=obs_fit_pval,
                                    obs_empty_chisqr=obs_empty_chisqr, obs_empty_redchisqr=obs_empty_redchisqr, obs_empty_pval=obs_empty_pval,
                                    diff_chisqr=diff_chisqr, diff_dof=diff_dof, diff_pval=diff_pval)
                CSV.write(el_savepath * "$(el_)_$(line_peak)_pval.csv", pval_df)
                println()

                continue
            end

            spec_fit = nothing
            Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do 
                @time spec_fit = Korg.Fit.fit_spectrum(Korg.air_to_vacuum.(wave), flux, flux_err, vald_linelist, guesses, params, R=res, windows=[(windmin, windmax)], 
                                    adjust_continuum=false, precision=1e-6)
            end

            # Extracting the results
            min_fit = spec_fit.best_fit_flux
            fit_mask = spec_fit.obs_wl_mask
            trace = spec_fit.trace

            figure(figsize=(10, 6))
            if min_abund_xfe >= el_bestfit_abund
                detection = "N"
                title("No detection: [$(el_)/Fe] < $(round(min_abund_xfe, digits=2))")
                printstyled("No detection: [$(el_)/Fe] < $(round(min_abund_xfe, digits=2))\n"; bold=true, color=:blue)
            else
                detection = "Y"
                title("Detection: [$(el_)/Fe] = $(round(el_bestfit_abund, digits=2))")
                printstyled("Detection: [$(el_)/Fe] = $(round(el_bestfit_abund, digits=2))\n"; bold=true, color=:blue)
                
            end
            plot(bestfit_wave, bestfit_flux, label="Best fit\n[$(el_)/Fe] = $(round(el_bestfit_abund, digits=2))", color="chocolate")
            plot(obs_wave, obs_flux, label="Observed", color="black", linewidth=0.9)
            plot(obs_wave, min_fit, label="Minimum\n[$(el_)/Fe] = $(round(min_abund_xfe, digits=2))", color="dodgerblue")
            xlabel("Wavelength (Å)")
            ylabel("Normalized Flux")
            legend(fancybox=false)
            savefig(el_savepath * "$(el_)_$(line_peak)_minspec.png", dpi=300, bbox_inches="tight")
            close(gcf())


            # ---SAVING THE RESULTS--- #
            println("Saving results")
            # Saving the fitted spectra (observed, best fit, minimum fit, empty fit)
            spec_fit_df = DataFrame(obs_wave=[obs_wave], obs_flux=[obs_flux], obs_flux_err=[obs_flux_err], minfit_wave=[obs_wave], 
                                    minfit_flux=[min_fit], bestfit_wave=[bestfit_wave], bestfit_flux=[bestfit_flux], empty_wave=[empty_wave], 
                                    empty_flux=[empty_flux])
            CSV.write(el_spec_savepath * "$(el_)_$(line_peak)_spec_fit.csv", spec_fit_df)

            # Saving the gaussians
            gaussians_df = DataFrame(gauss_wave=[empty_wave], gauss_flux=[gaussians_combined])
            CSV.write(el_spec_savepath * "$(el_)_$(line_peak)_gaussians.csv", gaussians_df)

            # Saving the minimum abundance
            min_abund_df = DataFrame(el=el, line_peak=line_peak, detection=detection, min_abund_xh=min_abund_xh, min_abund_xfe=min_abund_xfe, bestfit_abund_xh=el_bestfit_abund + feh, bestfit_abund_xfe=el_bestfit_abund)
            push!(abunds_all, min_abund_df)
            CSV.write(el_savepath * "$(el_)_$(line_peak)_abund.csv", min_abund_df)

            # Saving the p-values
            pval_df = DataFrame(obs_fit_chisqr=obs_fit_chisqr, obs_fit_redchisqr=obs_fit_redchisqr, obs_fit_pval=obs_fit_pval,
                                obs_empty_chisqr=obs_empty_chisqr, obs_empty_redchisqr=obs_empty_redchisqr, obs_empty_pval=obs_empty_pval,
                                diff_chisqr=diff_chisqr, diff_dof=diff_dof, diff_pval=diff_pval)
            CSV.write(el_savepath * "$(el_)_$(line_peak)_pval.csv", pval_df)
            println()

        end

        # ---SAVING ALL THE ABUNDANCES FOR THE ELEMENT--- #
        # Combining all the abundances into a single DataFrame
        all_abunds_df = reduce(vcat, abunds_all)
        CSV.write(star_savepath * "$(el)_abunds.csv", all_abunds_df)
    end
end


