using Korg
using CSV
using DataFrames
using PyPlot
matplotlib.use("Agg")
using PyCall
using Distributed
fits = pyimport("astropy.io.fits")
plt = pyimport("matplotlib.pyplot")
ast = pyimport("ast")

function create_mask(wave, ranges)
    mask = falses(length(wave))  # Initialize all false
    for r in ranges
        mask .|= (wave .>= Korg.air_to_vacuum(first(r))) .& (wave .<= Korg.air_to_vacuum(last(r)))
    end
    return .!mask  # Invert to exclude specified ranges
end

function get_abunds(star, abunds_path, cfe_corr)
    """
    Loading in the stellar abundances from the given path.

    Parameters
    ----------
    star : Str
        Name of star to cross-match abundances for.
    abunds_path : Str
        Path to the directory containing the stellar abundances.
    cfe_corr : DataFrame
        DataFrame containing the C/Fe evolutionary corrections for each star.

    Returns
    -------
    star_abunds_xh_df : DataFrame
        DataFrame containing the stellar [X/H] abundances for the specified elements.
    star_abunds_xfe_df : DataFrame
        DataFrame containing the stellar [X/Fe] abundances for the specified elements.
    """
    
    # Getting the list of elemental abundance files from the path
    elements = filter(f -> endswith(f, "I_abunds.csv"), readdir(abunds_path, join=true))

    # Extracting the abundances 
    all_abunds_xh = []
    all_abunds_xfe = []
    all_abunds_xh_err = []
    all_abunds_xfe_err = []
    els_list = []

    for el in elements

        # Loading in the elemental abundance file, then getting the entry for the star
        abund_df = CSV.read(el, DataFrame)
        star_abund_df = abund_df[abund_df.star_id .== star, :]

        # For Na-I, some stars are impacted by interstellar lines, and thus don't have
        # abundances. In this case, we skip the star.
        if isempty(star_abund_df)
            continue
        end

        feh = star_abund_df[!, "feh"][1]
        teff = star_abund_df[!, "teff"][1]
        logg = star_abund_df[!, "logg"][1]
        el = star_abund_df[!, "el"][1]
        el_detect = star_abund_df[!, "detection"][1]

        # Extracting the abundances based on the detection flag
        # Those with detections
        if el_detect == "Y"
            star_abund_xh = star_abund_df[!, "bestfit_abund_xh"][1]
            star_abund_xfe = star_abund_df[!,"bestfit_abund_xfe"][1]

            star_abund_xh_err = star_abund_df[!, "bestfit_abund_comberr_xh"][1]
            star_abund_xfe_err = star_abund_df[!, "bestfit_abund_comberr_xfe"][1]

            # Un-correcting the [C/Fe] abundance for C_evolve
            if el == "C_I"
                star_cfe = cfe_corr[cfe_corr.star_id .== star, :][!, "cfe_corr"][1]

                star_abund_xfe = star_abund_xfe - star_cfe
                star_abund_xh = star_abund_xfe + feh

            # # Uncorrecting the [Ca/Fe] abundance for NLTE
            # # Need to get the NLTE corrections for all three Ca II lines, then average them
            # elseif el == "Ca_II"

            #     # Getting the NLTE corrections for each Ca II line
            #     nlte_corr_8498 = nlte_correct(teff, logg, feh, 8498.02)
            #     nlte_corr_8535 = nlte_correct(teff, logg, feh, 8542.198)
            #     nlte_corr_8662 = nlte_correct(teff, logg, feh, 8662.138)

            #     # Averaging the NLTE corrections
            #     nlte_corr = mean([nlte_corr_8498, nlte_corr_8535, nlte_corr_8662])

            #     # Uncorrecting the [Ca/Fe] abundance
            #     star_abund_xh = star_abund_xh - nlte_corr
            #     star_abund_xfe = star_abund_xh - feh
            end

        # Those without detections
        elseif el_detect == "N"
            star_abund_xh = star_abund_df[!, "upper_limit_xh"][1]
            star_abund_xfe = star_abund_df[!, "upper_limit_xfe"][1]

            star_abund_xh_err = star_abund_df[!, "bestfit_abund_comberr_xh"][1]
            star_abund_xfe_err = star_abund_df[!, "bestfit_abund_comberr_xfe"][1]


            # For elements with upper-limits:
            # - If [X/Fe] >= 0, then set [X/Fe] = 0
            # - If [X/Fe] < 0, then use current [X/Fe] value
            if star_abund_xfe >= 0
                star_abund_xfe = 0.0
                star_abund_xh = 0.0 + feh
            end

            if star_abund_xfe < 0
                star_abund_xfe = star_abund_xfe
                star_abund_xh = star_abund_xh           
            end

        end

        # Removing ionisation from the element name
        el = split(el, "_")[1] 

        push!(all_abunds_xh, star_abund_xh)
        push!(all_abunds_xfe, star_abund_xfe)
        push!(all_abunds_xh_err, star_abund_xh_err)
        push!(all_abunds_xfe_err, star_abund_xfe_err)
        push!(els_list, el)
    end

    # Combining the abundances into a single DataFrame
    star_abunds_xh = reduce(hcat, all_abunds_xh)
    star_abunds_xfe = reduce(hcat, all_abunds_xfe)
    star_abunds_xh_err = reduce(hcat, all_abunds_xh_err)
    star_abunds_xfe_err = reduce(hcat, all_abunds_xfe_err)

    # Convert to dictionaries with element names as keys
    # df_xh_cols = names(star_abunds_xh)
    # df_xh_cols = [split(col, "_")[1] for col in df_xh_cols if split(col, "_")[3] == "xh"] # Getting the element names from the column names
    star_abunds_xh_df = DataFrame(Dict(els_list .=> collect(star_abunds_xh[1,:])))
    star_abunds_xh_err_df = DataFrame(Dict(els_list .=> collect(star_abunds_xh_err[1,:])))

    # df_xfe_cols = names(star_abunds_xfe)
    # df_xfe_cols = [split(col, "_")[1] for col in df_xfe_cols if split(col, "_")[3] == "xfe"] # Getting the element names from the column names
    star_abunds_xfe_df = DataFrame(Dict(els_list .=> collect(star_abunds_xfe[1,:])))
    star_abunds_xfe_err_df = DataFrame(Dict(els_list .=> collect(star_abunds_xfe_err[1,:])))

    return star_abunds_xh_df, star_abunds_xfe_df, star_abunds_xh_err_df, star_abunds_xfe_err_df
end


function korg_run(uvb_wave, uvb_flux, uvb_flux_err, vis_wave, vis_flux, vis_flux_err, linelist, path_lines,
    params, el_windows, el; el_guess, star_el_path, star_el_fit_path, mask_regions=false)
    """
    Running Korg for a given star and element.

    Parameters
    ----------
    uvb_wave : Vector
        Wavelength data for the UVB arm.
    uvb_flux : Vector
        Flux data for the UVB arm.
    uvb_flux_err : Vector
        Flux error data for the UVB arm.
    vis_wave : Vector
        Wavelength data for the VIS arm.
    vis_flux : Vector
        Flux data for the VIS arm.
    vis_flux_err : Vector
        Flux error data for the VIS arm.
    linelist : Vector
        List with the desired linelist.
    path_lines : Str
        Path to the directories containing the linelists.
    params : Dict
        Dictionary with the stellar parameters.
    el_windows : Dict
        Dictionary with the windows for each element.
    el : Str
        Element to fit.
    el_guess : Float
        Initial guess for the element abundance.
    star_el_path : Str
        Path to save the results.
    star_el_fit_path : Str
        Path to save the fits.
    mask_regions : Bool
        If true, will mask the regions around the lines in the linelist when given as a list. Example input
        is [4399:0.1:4401, 4407:0.1:4409, 4401.5:0.1:4402.5, 4405.7:0.1:4406.6]. Default is false.

    Returns
    -------
    el_res_df : DataFrame
    DataFrame containing the fitting results for each line. Also contains the observed and fitted fluxes.
    el_vals_df : DataFrame
    DataFrame containing the fit abundances and weights for each line.
    el_abund : Float
    Weighted average abundance for the given element.
    """

    # Resolutions of UVB and VIS arms
    R_ubv = 5600
    R_vis = 8900
    # R_vis = 12000

    # Ranges for the UVB and VIS arms
    uvb_min = Korg.air_to_vacuum(3000)
    uvb_max = Korg.air_to_vacuum(5560)
    vis_min = Korg.air_to_vacuum(5600)
    vis_max = Korg.air_to_vacuum(10200)

    # Looping over each window for the given element
    el_res = []
    el_vals_all = []
    for wind in el_windows

        # Getting the window properties
        line_peak = wind[1]
        windmin = wind[2]
        windmax = wind[3]

        # Making sure to use the correct spectral arm
        if windmin < uvb_max
            wave = uvb_wave
            wavemin = uvb_min
            wavemax = uvb_max
            flux = uvb_flux
            flux_err = uvb_flux_err
            res = R_ubv
        end

        if windmin > vis_min && windmin < vis_max
            wave = vis_wave
            wavemin = vis_min
            wavemax = vis_max
            flux = vis_flux
            flux_err = vis_flux_err
            res = R_vis
        end

        if windmin > uvb_max && windmin < vis_min
            println("Line in-between UVB and VIS arms with no data - skipping")
            continue
        end

        if wind[1] > vis_max 
            println("Line in the NIR arm - skipping")
            continue
        end

        # If the mask_regions is set to true, will mask the regions around the line by the given mask list.
        # Regions masked will have their flux errors inflated to 1e10, to ensure Korg 'ignores' them.
        # This is useful for when you have a lot of lines in the linelist, and you want to mask them out.

        if mask_regions
            # Getting the mask regions for the element
            el_mask_path = path_lines * "caffauFeI_masksALL.csv"
            el_mask_df = CSV.read(el_mask_path, DataFrame)

            println(el_mask_df)
            
            # Exracting mask specific to the derired line
            line_masks = nothing
            try
                line_masks = el_mask_df[el_mask_df.line .== line_peak, :].mask[1]
                line_masks = ast.literal_eval(line_masks)
            catch e
                error("No mask found for line $(line_peak) - please check mask exists in file at $(el_mask_path)")
            end

            # Converting the mask into a range (i.e. taking the min and max of the mask, then creating a 
            # range at 0.1A intervals)
            line_mask_regs = []
            step = 0.1
            for mask in line_masks
                reg = mask[1]:step:mask[2]
                push!(line_mask_regs, reg)
            end

            # Identifying the wavelength indices to mask, then inflating their flux errors
            wave_indices = create_mask(wave, line_mask_regs)
            flux_err[.!(wave_indices)] .= 1e10
        end

        # Running Korg
        # Running korg once to get fit ranges
        guesses = Dict(el => el_guess)

        println("Adjusting window ranges")
        spec_fit = Korg.Fit.fit_spectrum(wave, flux, flux_err, linelist, guesses, params, R=res,
        windows=[(windmin, windmax)], adjust_continuum=true, precision=1e-6)

        # Adjusting window sizes based on the fit range
        windmin_fix = windmin - (wave[spec_fit.obs_wl_mask][1] - windmin)
        windmax_fix = windmax - (wave[spec_fit.obs_wl_mask][end] - windmax)

        println("Fitting line $(round(line_peak, digits=3))")
        @time spec_fit = Korg.Fit.fit_spectrum(wave, flux, flux_err, linelist, guesses, params, R=res,
        windows=[(windmin_fix, windmax_fix)], adjust_continuum=true, precision=1e-6)


        # Extracting the results
        best_fit = spec_fit.best_fit_flux
        fit_mask = spec_fit.obs_wl_mask
        trace = spec_fit.trace

        # Converting to air, then getting the central wavelength
        wave_air = Korg.vacuum_to_air.(wave)
        windmin = Korg.vacuum_to_air(windmin)
        windmax = Korg.vacuum_to_air(windmax)

        obs_spec_wind = 5 # A
        spec_mask = (wave_air .> windmin-obs_spec_wind) .& (wave_air .< windmax+obs_spec_wind)

        # Getting the abundances
        el_xh_abund = spec_fit.best_fit_params["$el"] # in [X/H]
        el_xfe_abund = el_xh_abund - params["m_H"][1] # in [X/Fe]

        el_chisqr = spec_fit.trace[end]["chi2"]
        el_cov = spec_fit.covariance[end][1]
        el_err = sqrt(el_cov)
        el_vals = Dict("line" => "$(el)_$(round(line_peak, digits=3))", "abund_xh" => el_xh_abund, "abund_xfe" => el_xfe_abund, "chisqr" => el_chisqr, 
                "err" => el_err, "weight" => 1/el_err^2)

        # Saving the results
        # data_dict = Dict("$(el)_$(round(line_peak, digits=3))_abund" => el_abund, "$(el)_$(round(line_peak, digits=3))_chisqr" => el_chisqr)
        data_dict = Dict("line" => "$(el)_$(round(line_peak, digits=3))", "abund_xh" => el_xh_abund, "abund_xfe" => el_xfe_abund, 
                "chisqr" => el_chisqr, "err" => el_err, "weight" => 1/el_err^2, "obs_wave" => [wave_air[spec_mask]], "obs_flux" => [flux[spec_mask]],
                "fit_wave" => [wave_air[fit_mask]], "fit_flux" => [best_fit])

        fits_dict = Dict("wave" => wave_air[fit_mask], "flux" => flux[fit_mask], "best_fit" => best_fit)
        fits_df = DataFrame(fits_dict)
        CSV.write(star_el_fit_path * "$(el)_$(round(line_peak, digits=3))_fit.csv", fits_df)

        trace_dict = Dict("abund_trace" => [t["$el"] for t in trace], "chi2_trace" => [t["chi2"] for t in trace])
        trace_df = DataFrame(trace_dict)
        CSV.write(star_el_fit_path * "$(el)_$(round(line_peak, digits=3))_trace.csv", trace_df)

        push!(el_vals_all, el_vals)
        push!(el_res, data_dict)


        # Plotting the fit (with regions masked if mask_regions is set to true)
        figure(figsize=(8, 6))
        plot(wave_air[spec_mask], flux[spec_mask], label="Observed", color="black", linewidth=0.8)
        plot(wave_air[fit_mask], best_fit, label="[$(el)/H] = $(round(el_xh_abund, digits=2))\n(chi2 = $(round(el_chisqr, digits=2)))", color="chocolate", linewidth=1.5)
        
        # Shading masked regions if mask_regions is set to true
        if mask_regions
            for mask in line_mask_regs
                fill_between(mask, maximum(flux[spec_mask])+0.01, minimum(flux[spec_mask])-0.01, color="grey", alpha=0.5)
            end
        end

        axvline(wave_air[fit_mask][1], color="grey", linestyle="--")
        axvline(wave_air[fit_mask][end], color="grey", linestyle="--")
        xlabel("Wavelength (Å)")
        ylabel("Normalised Flux")
        legend(fancybox=true)
        tight_layout()
        ylim(minimum(flux[spec_mask])-0.01, maximum(flux[spec_mask])+0.01)
        savefig(star_el_path * "$(el)_$(round(line_peak, digits=3))_fit.png", dpi=300, transparent=false, facecolor="white")
        close()


        # Plotting the fit - overplotting errors (with regions masked if mask_regions is set to true)
        figure(figsize=(8, 6))
        # scatter(wave_air[spec_mask], flux[spec_mask], label="Unaltered errors", color="green", zorder=10)
        errorbar(wave_air[spec_mask], flux[spec_mask], yerr=flux_err[spec_mask], fmt=".", color="red", zorder=10, alpha=0.3)
        plot(wave_air[spec_mask], flux[spec_mask], color="black")
        plot(wave_air[fit_mask], best_fit, label="[$(el)/H] = $(round(el_xh_abund, digits=2))\n(chi2 = $(round(el_chisqr, digits=2)))", color="chocolate", linewidth=1.5)

        # Shading masked regions if mask_regions is set to true
        if mask_regions
            for mask in line_mask_regs
                fill_between(mask, maximum(flux[spec_mask])+0.01, minimum(flux[spec_mask])-0.01, color="grey", alpha=0.5)
            end
        end

        axvline(wave_air[fit_mask][1], color="grey", linestyle="--")
        axvline(wave_air[fit_mask][end], color="grey", linestyle="--")
        xlabel("Wavelength (Å)")
        ylabel("Normalised Flux")
        legend(fancybox=true)
        tight_layout()
        ylim(minimum(flux[spec_mask])-0.01, maximum(flux[spec_mask])+0.01)
        savefig(star_el_path * "$(el)_$(round(line_peak, digits=3))_errors.png", dpi=300, transparent=false, facecolor="white")
        close()


        # Plotting the trace
        fig, axs = subplots(1, 2, figsize=[15,5])
        axs[1].plot([t[el] for t in trace], color="darkred")
        axs[1].set_ylabel("[$el/H]")

        axs[2].plot([t["chi2"] for t in trace], color="darkblue")
        axs[2].set_ylabel("χ^2")

        fig.supxlabel("Optimizer step", fontsize=14)
        tight_layout()
        savefig(star_el_path * "$(el)_$(round(line_peak, digits=3))_trace.png", dpi=300, transparent=false, facecolor="white")
        close()
    end

    # Saving the results to DataFrame
    el_res_df = DataFrame(el_res)
    CSV.write(star_el_path * "$(el)_results.csv", el_res_df)

    # Calculating the weighted average abundance for the given element
    el_vals_df = DataFrame(el_vals_all)
    mean_el_xh_abund = sum(el_vals_df[!, "abund_xh"] .* el_vals_df[!, "weight"]) / sum(el_vals_df[!, "weight"])
    mean_el_xfe_abund = sum(el_vals_df[!, "abund_xfe"] .* el_vals_df[!, "weight"]) / sum(el_vals_df[!, "weight"])

    return el_res_df, el_vals_df, mean_el_xh_abund, mean_el_xfe_abund

end


function run_feh(path, path_lines, linelist, params_stars, windows, elements, savepath, abunds_path, abunds_detects, cfe_corr; mask_regions=false)
    """
    Using Korg to derive new [Fe/H] abundances for each star in the sample. Will use the Caffau+2013 line list for this.

    Parameters
    ----------
    path : Vector
        List of paths to the stacked spectral data for each star.
    path_lines : Str
        Path to the directories containing the linelists for the specific element.
    linelist : Vector
        List with the desired linelist.
    params_stars : DataFrame
        Stellar parameters for each star.
    windows : Dict
        Dictionary with the windows for each element.
    elements : Vector
        List with the desired elements.
    savepath : Str
        Path to save the results to.
    abunds_path : Str
        Path to the pre-derived chemical abundances for each star.
    abunds_detects : DataFrame
        DataFrame containing the detection flags for each element in the pre-derived chemical abundances.
    cfe_corr : DataFrame
        DataFrame containing the C_evolve corrections for each star. This will be used to un-correct the [C/Fe] abundances before running Korg.
    mask_regions : Bool
        If true, will mask the regions around the lines in the linelist. Need separate file to specify what regions are being
        masked, which will be read inside the code. Default is false.
            
    Returns
    -------
    None
    """
    params_list = []  

    # Finding and extracting the .fits data
    star_id = split(path, "/")[end]
    println("Running $star_id")

    fits_file = string(path, "/", star_id, "_data.fits")
    star_fits = fits.open(fits_file)
    
    # Getting the UVB data
    uvb_data = star_fits[2].data
    uvb_wave = Korg.air_to_vacuum.(uvb_data["wave"]) # Converting from air to vacuum
    uvb_flux = convert(Array,uvb_data["flux"])
    uvb_flux_err = convert(Array, uvb_data["flux_err"])

    # Getting the VIS data
    vis_data = star_fits[3].data
    vis_wave = Korg.air_to_vacuum.(vis_data["wave"]) # Converting from air to vacuum
    vis_flux = vis_data["flux"]
    vis_flux_err = vis_data["flux_err"]

    # Getting the stellar parameters
    star_params = params_stars[params_stars.star_id .== star_id, :]
    teff = star_params.teff[1] 
    teff_err = star_params.teff_err[1]
    logg = star_params.logg[1]
    logg_err = star_params.logg_err[1]
    feh = star_params.Fe_it1[1]
    feh_err = star_params.Fe_comberr[1]
    afe = 0.4 # [Alpha/Fe]
    ah = afe + feh # [Alpha/H]

    # From Buder+2024
    teff_ = teff - 5500
    logg_ = logg - 4.0
    vmic = 1.198 + 3.16 * 10^-4 * teff_ - 0.253 * logg_ - 2.86 * 10^-4 * teff_ * logg_ + 0.165 * logg_^2
    vmic_err = sqrt((3.16E-4 - 2.86E-4 * logg_)^2 * teff_err^2 + (-0.253 - 2.86E-4 * teff_ + 0.330 * logg_)^2 * logg_err^2)
    vsini = 0
    epsilon = 0.6 # limb darkening

    # Getting the stellar abundances
    star_abunds_xh_all, star_abunds_xfe_all, star_abunds_xh_err_all, star_abunds_xfe_err_all = get_abunds(star_id, abunds_path, cfe_corr)

    # Converting the abundances to dictionaries
    star_abund_dict_xh = Dict(col => star_abunds_xh_all[!, col][1] for col in names(star_abunds_xh_all))
    star_abund_dict_xfe = Dict(col => star_abunds_xfe_all[!, col][1] for col in names(star_abunds_xfe_all))
    star_abund_err_dict_xh = Dict(col => star_abunds_xh_err_all[!, col][1] for col in names(star_abunds_xh_err_all))
    star_abund_err_dict_xfe = Dict(col => star_abunds_xfe_err_all[!, col][1] for col in names(star_abunds_xfe_err_all))

    # Using the abundances to determine the [alpha/H] value by getting weighted mean of Mg, Si, Ca, Ti
    alpha_arr = [star_abund_dict_xh["Mg"], star_abund_dict_xh["Si"], star_abund_dict_xh["Ca"], star_abund_dict_xh["Ti"]]
    alpha_weights = 1 ./ [star_abund_err_dict_xh["Mg"]^2, star_abund_err_dict_xh["Si"]^2, star_abund_err_dict_xh["Ca"]^2, star_abund_err_dict_xh["Ti"]^2]
    alpha_vals = sum(alpha_arr .* alpha_weights) / sum(alpha_weights)

    ah = alpha_vals
    afe = ah - feh

    for el in elements
        println("Fitting $(el)")

        # Selecting windows of element
        el_windows = windows[Korg.atomic_numbers[el]]
        

        # ITERATION 1
        # Making directories for each element
        iter = 1
        println("Iteration $iter")
        star_el_path_it1 = savepath * "$(star_id)/$(el)/iter_$(iter)/"
        mkpath(star_el_path_it1)
        star_el_fit_path_it1 = star_el_path_it1 * "fits_data/"
        mkpath(star_el_fit_path_it1)

        # Running Korg
        params = Dict("Teff" => teff, "logg" => logg, "m_H" => feh, "vmic" => vmic, "vsini" => vsini, "epsilon" => epsilon, "alpha_H" => ah)
        params = merge(params, star_abund_dict_xh) # Adding the stellar abundances to the parameters

        el_res_df_it1, el_vals_df_it1, el_abund_xh_it1, el_abund_xfe_it1 = korg_run(uvb_wave, uvb_flux, uvb_flux_err, vis_wave, vis_flux, vis_flux_err, linelist, 
                                                                                    path_lines, params, el_windows, el, el_guess=feh, star_el_path=star_el_path_it1, 
                                                                                    star_el_fit_path=star_el_fit_path_it1, mask_regions=mask_regions)
    
            
        # ITERATION 2 - Perturbing Teff by +100K
        # Making directories for each element
        iter = 2
        println("Iteration $iter - perturbing Teff by +100K")
        star_el_path_it2 = savepath * "$(star_id)/$(el)/teff_perturb/"
        mkpath(star_el_path_it2)
        star_el_fit_path_it2 = star_el_path_it2 * "fits_data/"
        mkpath(star_el_fit_path_it2)

        # Running Korg
        params = Dict("Teff" => teff + 100, "logg" => logg, "m_H" => feh, "vmic" => vmic, "vsini" => vsini, "epsilon" => epsilon, "alpha_H" => ah)        
        params = merge(params, star_abund_dict_xh) # Adding the stellar abundances to the parameters

        el_res_df_it2, el_vals_df_it2, el_abund_xh_it2, el_abund_xfe_it2 = korg_run(uvb_wave, uvb_flux, uvb_flux_err, vis_wave, vis_flux, vis_flux_err, linelist, 
                                                                                    path_lines, params, el_windows, el, el_guess=el_abund_xh_it1, star_el_path=star_el_path_it2,
                                                                                    star_el_fit_path=star_el_fit_path_it2, mask_regions=mask_regions)


        # ITERATION 3 - Perturbing Logg by +0.3dex
        # Making directories for each element
        iter = 3
        println("Iteration $iter - perturbing Logg by +0.3dex")
        star_el_path_it3 = savepath * "$(star_id)/$(el)/logg_perturb/"
        mkpath(star_el_path_it3)
        star_el_fit_path_it3 = star_el_path_it3 * "fits_data/"
        mkpath(star_el_fit_path_it3)

        # Running Korg
        params = Dict("Teff" => teff, "logg" => logg + 0.3, "m_H" => feh, "vmic" => vmic, "vsini" => vsini, "epsilon" => epsilon, "alpha_H" => ah)
        params = merge(params, star_abund_dict_xh) # Adding the stellar abundances to the parameters

        el_res_df_it3, el_vals_df_it3, el_abund_xh_it3, el_abund_xfe_it3 = korg_run(uvb_wave, uvb_flux, uvb_flux_err, vis_wave, vis_flux, vis_flux_err, linelist, 
                                                                                    path_lines, params, el_windows, el, el_guess=el_abund_xh_it1, star_el_path=star_el_path_it3,
                                                                                    star_el_fit_path=star_el_fit_path_it3, mask_regions=mask_regions)


        # ITERATION 4 - Perturbing vmic by +0.3km/s
        # Making directories for each element
        iter = 4
        println("Iteration $iter - perturbing vmic by +0.3km/s")
        star_el_path_it4 = savepath * "$(star_id)/$(el)/vmic_perturb/"
        mkpath(star_el_path_it4)
        star_el_fit_path_it4 = star_el_path_it4 * "fits_data/"
        mkpath(star_el_fit_path_it4)

        # Running Korg
        params = Dict("Teff" => teff, "logg" => logg, "m_H" => feh, "vmic" => vmic + 0.3, "vsini" => vsini, "epsilon" => epsilon, "alpha_H" => ah)
        params = merge(params, star_abund_dict_xh) # Adding the stellar abundances to the parameters

        el_res_df_it4, el_vals_df_it4, el_abund_xh_it4, el_abund_xfe_it4 = korg_run(uvb_wave, uvb_flux, uvb_flux_err, vis_wave, vis_flux, vis_flux_err, linelist, 
                                                                                    path_lines, params, el_windows, el, el_guess=el_abund_xh_it1, star_el_path=star_el_path_it4,
                                                                                    star_el_fit_path=star_el_fit_path_it4, mask_regions=mask_regions)


        # Calculating uncertainties
        el_abund_xh_teff_err = abs(el_abund_xh_it2-el_abund_xh_it1)
        el_abund_xh_logg_err = abs(el_abund_xh_it3-el_abund_xh_it1)
        el_abund_xh_vmic_err = abs(el_abund_xh_it4-el_abund_xh_it1)
        el_abund_xh_comb_err = sqrt(el_abund_xh_teff_err^2 + el_abund_xh_logg_err^2 +el_abund_xh_vmic_err^2 )/3

        # Saving the stellar parameters
        params_dict = Dict("star_id" => star_id, "teff" => teff, "teff_err" => teff_err, "logg" => logg, "logg_err" => logg_err, "vmic" => vmic, 
                            "vmic_err" => vmic_err, "vsini" => vsini, "epsilon" => epsilon, "$(el)_init" => feh, "$(el)_it1" => el_abund_xh_it1, 
                            "$(el)_teff_perturb" => el_abund_xh_it2, "$(el)_logg_perturb" => el_abund_xh_it3, "$(el)_vmic_perturb" => el_abund_xh_it4, 
                            "$(el)_tefferr" => el_abund_xh_teff_err, "$(el)_loggerr" => el_abund_xh_logg_err, "$(el)_vmicerr" => el_abund_xh_vmic_err, 
                            "$(el)_comberr" => el_abund_xh_comb_err)

        params_df = DataFrame(params_dict)
        CSV.write(savepath * "$(star_id)/" * "stellar_params.csv", params_df)

        return params_df
    end
end

