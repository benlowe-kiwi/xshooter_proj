using Korg
using CSV
using DataFrames
using PyPlot
using PyCall
using Statistics
using StatsBase
using Distributed
fits = pyimport("astropy.io.fits")
plt = pyimport("matplotlib.pyplot")
ast = pyimport("ast")
scipy_interpolate = pyimport("scipy.interpolate")
LinearNDInterpolator = scipy_interpolate."LinearNDInterpolator"

function create_mask(wave, ranges)
    mask = falses(length(wave))  # Initialize all false
    for r in ranges
        mask .|= (wave .>= Korg.air_to_vacuum(first(r))) .& (wave .<= Korg.air_to_vacuum(last(r)))
    end
    return .!mask  # Invert to exclude specified ranges
end


function nlte_correct(star_teff, star_logg, star_feh, caII_line)
    """
    Function to apply NLTE corrections for the Ca II lines based on the Osorio et al. (2023) grid. The grid will be
    linearly interpolated to the stellar parameters provided. The function will also check if the parameters are within
    the grid limits, and if not, will set them to the nearest grid limit. The grid limits are:
        4500 ≤ Teff ≤ 7000 K
        1.0 ≤ logg ≤ 4.0 dex
        -2.5 ≥ [Fe/H] ≥ -6.0 dex

    Parameters
    ----------
    star_teff : Float
        Effective temperature of the star in Kelvin.
    star_logg : Float
        Surface gravity of the star in dex.
    star_feh : Float
        Metallicity of the star in [Fe/H] dex.
    caII_line : Float
        Wavelength of the Ca II line to apply the correction for. Should be one of:
        8498.02, 8542.198, or 8662.138 A.

    Returns
    -------
    nlte_corr : Float
        The NLTE correction to be applied to the abundance of the Ca II line.
    """

    # Loading in the NLTE grid
    nlte_grid = CSV.read("/data/misfit/benlowe/xshooter/p115/CaII_nlte_corrections/osorio_corrs.csv", DataFrame)
    
    # Performing linear interpolation on the grid
    grid_teff = nlte_grid[:, "Teff"]
    grid_logg = nlte_grid[:, "log_g"]
    grid_feh = nlte_grid[:, "Fe_H"]
    points = hcat(grid_teff, grid_logg, grid_feh)

    if caII_line == 8498.02
        nlte_grid = LinearNDInterpolator(points, nlte_grid[:, "CaII_8498"])
    elseif caII_line == 8542.198
        nlte_grid = LinearNDInterpolator(points, nlte_grid[:, "CaII_8542"])
    elseif caII_line == 8662.138
        nlte_grid = LinearNDInterpolator(points, nlte_grid[:, "CaII_8662"])
    else
        error("Invalid Ca II line: $(caII_line)")
    end
    

    # Checking if the parameters are within the grid limits
    # Grid limits are:
    #   4500 ≤ Teff ≤ 7000 K
    #   1.0 ≤ logg ≤ 4.0 dex
    #   -2.5 ≥ [Fe/H] ≥ -6.0 dex
    if star_teff < 4500
        println("NLTE correction: Teff below 4500K grid limit - setting to 4500K")
        star_teff = 4500
    end
    if star_teff > 7000
        println("NLTE correction: Teff above 7000K grid limit - setting to 7000K")
        star_teff = 7000
    end
    if star_logg < 1.0
        println("NLTE correction: Logg below 1.0 dex grid limit - setting to 1.0 dex")
        star_logg = 1.0
    end
    if star_logg > 4.0
        println("NLTE correction: Logg above 4.0 dex grid limit - setting to 4.0 dex")
        star_logg = 4.0
    end
    if star_feh < -6.0
        println("NLTE correction: [Fe/H] below -6.0 dex grid limit - setting to -6.0 dex")
        star_feh = -6.0
    end
    if star_feh > -2.5
        println("NLTE correction: [Fe/H] above -2.5 dex grid limit - setting to -2.5 dex")
        star_feh = -2.5
    end

    # Determining the NLTE correction from the grid
    nlte_corr = nlte_grid(star_teff, star_logg, star_feh)[1]

    return nlte_corr
end



function korg_run(uvb_wave, uvb_flux, uvb_flux_err, vis_wave, vis_flux, vis_flux_err, linelist, path_lines,
    params, el_windows, el_, star_id, cfe_corr_path; el_guess, star_el_path, star_el_fit_path, mask_regions=false)
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
    el_ : Str
        Element to fit. Should be in format e.g. "Fe I" or "Fe II"
    el_guess : Float
        Initial guess for the element abundance.
    star_el_path : Str
        Path to save the results.
    star_el_fit_path : Str
        Path to save the fits.
    cfe_corr_path : Str
        Path to the C/Fe evolutionary corrections.
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
            el_mask_path = path_lines * split(el_, " ")[1] * "_" * split(el_, " ")[2] * "_masksALL.csv"
            el_mask_df = CSV.read(el_mask_path, DataFrame)

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


        # # If the line is the Ca-II triplet, mask out the bottom 10% of the line
        # if el_ == "Ca II"
        #     if line_peak == 8498.02
        #         bottom_percent = 30
        #         caII_windmin = 8496.02
        #         caII_windmax = 8500.02
        #     elseif line_peak == 8542.198
        #         bottom_percent = 30
        #         caII_windmin = 8540.198
        #         caII_windmax = 8544.198
        #     elseif line_peak == 8662.138
        #         bottom_percent = 30
        #         caII_windmin = 8660.138
        #         caII_windmax = 8664.138
        #     end

        #     wave_caII = wave[(wave .> Korg.air_to_vacuum(caII_windmin)) .&& (wave .< Korg.air_to_vacuum(caII_windmax))]
        #     flux_caII = flux[(wave .> Korg.air_to_vacuum(caII_windmin)) .&& (wave .< Korg.air_to_vacuum(caII_windmax))]
        #     flux_err_caII = flux_err[(wave .> Korg.air_to_vacuum(caII_windmin)) .&& (wave .< Korg.air_to_vacuum(caII_windmax))]

        #     # Getting the bottom 10% of the line
        #     flux_threshold = quantile(flux_caII, bottom_percent/100)
        #     mask_caII = findall(flux_caII .< flux_threshold)

        #     flux_err[flux .< flux_threshold] .= 1e10
        #     wave_caII = Korg.vacuum_to_air.(wave_caII)
        #     wave_errrange_caII = wave_caII[flux_caII .<= flux_threshold]
        #     # println(wave_caII[flux_caII .< flux_threshold])
        #     # println(flux_caII[flux_caII .< flux_threshold])
        #     # println(flux_threshold)

        # end

        # HOW TO FIT MULTIPLE LINES AT ONCE:
        # As long as your linelist contains all the elements in your window, simply give their guesses below,
        # and Korg will fit them all at once. It takes significantly longer, but should be more accurate for 
        # your desired element.
        # if el == "Sr"
        #     feh = params.m_H[1]
        #     guesses = Dict(el => el_guess, "Fe" => feh)
        # else
        #     guesses = Dict(el => el_guess)
        # end

        # Running Korg
        # Running korg once to get fit ranges
        el = split(el_, " ")[1] # Removing the number at the end of the element
        guesses = Dict(el => el_guess)

        println("Line $(round(line_peak, digits=3)): Adjusting window ranges")
        spec_fit = Korg.Fit.fit_spectrum(wave, flux, flux_err, linelist, guesses, params, R=res,
        windows=[(windmin, windmax)], adjust_continuum=false, precision=1e-6)

        # Adjusting window sizes based on the fit range
        windmin_fix = windmin - (wave[spec_fit.obs_wl_mask][1] - windmin)
        windmax_fix = windmax - (wave[spec_fit.obs_wl_mask][end] - windmax)


        # For Ca-II, applying continuum offset 
        if el_ == "Ca II"

            # Loading in the continuum offset data
            cntm_offset_df = CSV.read(path_lines * split(el_, " ")[1] * "_" * split(el_, " ")[2] * "_cntm_offsets.csv", DataFrame)
            star_cntm_offset = cntm_offset_df[cntm_offset_df.star_id .== star_id, :]

            # Extracting the continuum offset for the given line
            if line_peak == 8498.02
                cntm_offset = -star_cntm_offset[!, "8498.02"][1]
            elseif line_peak == 8542.198
                cntm_offset = -star_cntm_offset[!, "8542.198"][1]
            elseif line_peak == 8662.138
                cntm_offset = -star_cntm_offset[!, "8662.138"][1]
            end

            # Step 2: Fit for the element abundance, keeping the continuum offset fixed
            println("Line $(round(line_peak, digits=3)): Fitting")
            guesses = Dict(el => el_guess)
            params2 = merge(params, Dict("cntm_offset" => cntm_offset))

            @time spec_fit = Korg.Fit.fit_spectrum(wave, flux, flux_err, linelist, guesses, params2, R=res,
            windows=[(windmin_fix, windmax_fix)], adjust_continuum=false, precision=1e-6)
        else
            println("Fitting line $(round(line_peak, digits=3))")
            @time spec_fit = Korg.Fit.fit_spectrum(wave, flux, flux_err, linelist, guesses, params, R=res,
            windows=[(windmin_fix, windmax_fix)], adjust_continuum=true, precision=1e-6)
        end


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
        el_xfe_abund = el_xh_abund - params["m_H"] # in [X/Fe]

        el_chisqr = spec_fit.trace[end]["chi2"]
        el_cov = spec_fit.covariance[end][1]
        el_err = sqrt(el_cov)
        el_vals = Dict("line" => "$(el)_$(round(line_peak, digits=3))", "abund_xh" => el_xh_abund, "abund_xfe" => el_xfe_abund, "chisqr" => el_chisqr, 
                "err" => el_err, "weight" => 1/el_err^2)

        # Saving the results
        # data_dict = Dict("$(el)_$(round(line_peak, digits=2))_abund" => el_abund, "$(el)_$(round(line_peak, digits=2))_chisqr" => el_chisqr)
        data_dict = Dict("line" => "$(el)_$(round(line_peak, digits=3))", "abund_xh" => el_xh_abund, "abund_xfe" => el_xfe_abund, 
                "chisqr" => el_chisqr, "err" => el_err, "weight" => 1/el_err^2, "obs_wave" => [wave_air[spec_mask]], "obs_flux" => [flux[spec_mask]],
                "fit_wave" => [wave_air[fit_mask]], "fit_flux" => [best_fit])

        # For Ca-II: applying NLTE correction and saving the continuum offset
        if el_ == "Ca II"
            # Saving the continuum offset
            el_vals["cntm_offset"] = cntm_offset
            data_dict["cntm_offset"] = cntm_offset
            
            # Computing the NLTE correction
            println("Line $(round(line_peak, digits=3)): Calculating NLTE correction and applying")
            nlte_corr = nlte_correct(params["Teff"], params["logg"], params["m_H"], line_peak)

            # Applying the NLTE correction to the abundance
            el_xh_abund = el_xh_abund + nlte_corr
            el_xfe_abund = el_xh_abund - params["m_H"]

            # Updating values in the dictionaries
            el_vals["abund_xh"] = el_xh_abund
            data_dict["abund_xh"] = el_xh_abund
            el_vals["abund_xfe"] = el_xfe_abund
            data_dict["abund_xfe"] = el_xfe_abund
            el_vals["nlte_corr"] = nlte_corr
            data_dict["nlte_corr"] = nlte_corr

        # For C-I: applying evolutionary corrections
        elseif el_ == "C I"
            # Loading in the evolutionnary corrections
            carb_evol_corrs = CSV.read(cfe_corr_path, DataFrame)

            # Extracting the evolutionary correction for the given star
            star_carb_corr = carb_evol_corrs[carb_evol_corrs.star_id .== star_id, :][!, "cfe_corr"]

            # Applying the correction to the abundance
            el_xfe_abund_evol = el_xfe_abund + star_carb_corr[1]
            el_xh_abund_evol = el_xfe_abund + params["m_H"]

            # Updating values in the dictionaries
            el_vals["abund_xh_corr"] = el_xh_abund_evol
            data_dict["abund_xh_corr"] = el_xh_abund_evol
            el_vals["abund_xfe_corr"] = el_xfe_abund_evol
            data_dict["abund_xfe_corr"] = el_xfe_abund_evol
            el_vals["carb_corr"] = star_carb_corr[1]
            data_dict["carb_corr"] = star_carb_corr[1]
        end

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

        if el_ == "Ca II"
            label = "[$(el)/Fe]_NLTE = $(round(el_xfe_abund, digits=2))]\n(corr = $(round(nlte_corr, digits=2)))\n(chi2 = $(round(el_chisqr, digits=2)))"
        elseif el_ == "C I"
            label = "[$(el)/Fe]_corr = $(round(el_xfe_abund_evol, digits=2))\n(corr = $(round(star_carb_corr[1], digits=3)))\n(chi2 = $(round(el_chisqr, digits=2)))"
        else
            label = "[$(el)/Fe] = $(round(el_xfe_abund, digits=2))\n(chi2 = $(round(el_chisqr, digits=2)))"
        end
        plot(wave_air[fit_mask], best_fit, label=label, color="chocolate", linewidth=1.5)
        
        # Shading masked regions if mask_regions is set to true
        if mask_regions
            for mask in line_mask_regs
                fill_between(mask, maximum(flux[spec_mask])+0.01, minimum(flux[spec_mask])-0.01, color="grey", alpha=0.5)
            end
        end

        # # Shading bottom 10% of the line if Ca II
        # if el_ == "Ca II"
        #     fill_between(wave_errrange_caII, maximum(flux[spec_mask])+0.01, minimum(flux[spec_mask])-0.01, color="grey", alpha=0.3)
        # end

        axvline(line_peak, color="darkred", linestyle=":")
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
        errorbar(wave_air[spec_mask], flux[spec_mask], yerr=flux_err[spec_mask], fmt=".", color="red", zorder=10, alpha=0.3)
        plot(wave_air[spec_mask], flux[spec_mask], color="black")
        label = "[$(el)/Fe] = $(round(el_xfe_abund, digits=2))\n(chi2 = $(round(el_chisqr, digits=2)))"
        if el_ == "Ca II"
            label = "[$(el)/Fe]_NLTE = $(round(el_xfe_abund, digits=2))\n(corr = $(round(nlte_corr, digits=2)))\n(chi2 = $(round(el_chisqr, digits=2)))"
        elseif el_ == "C I"
            label = "[$(el)/Fe]_corr = $(round(el_xfe_abund_evol, digits=2))\n(corr = $(round(star_carb_corr[1], digits=3)))\n(chi2 = $(round(el_chisqr, digits=2)))"
        else
            label = "[$(el)/Fe] = $(round(el_xfe_abund, digits=2))\n(chi2 = $(round(el_chisqr, digits=2)))"
        end
        plot(wave_air[fit_mask], best_fit, label=label, color="chocolate", linewidth=1.5)

        # Shading masked regions if mask_regions is set to true
        if mask_regions
            for mask in line_mask_regs
                fill_between(mask, maximum(flux[spec_mask])+0.01, minimum(flux[spec_mask])-0.01, color="grey", alpha=0.5)
            end
        end

        # # Shading bottom 10% of the line if Ca II
        # if el_ == "Ca II"
        #     fill_between(wave_errrange_caII, maximum(flux[spec_mask])+0.01, minimum(flux[spec_mask])-0.01, color="grey", alpha=0.3)        
        # end

        axvline(line_peak, color="darkred", linestyle=":")
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
    el = split(el_, " ")[1] # Removing the number at the end of the element

    el_res_df = DataFrame(el_res)
    CSV.write(star_el_path * "$(el)_results.csv", el_res_df)

    # Calculating the weighted average abundance for the given element
    el_vals_df = DataFrame(el_vals_all)
    mean_el_xh_abund = sum(el_vals_df[!, "abund_xh"] .* el_vals_df[!, "weight"]) / sum(el_vals_df[!, "weight"])
    mean_el_xfe_abund = sum(el_vals_df[!, "abund_xfe"] .* el_vals_df[!, "weight"]) / sum(el_vals_df[!, "weight"])

    return el_res_df, el_vals_df, mean_el_xh_abund, mean_el_xfe_abund

end



function run_abund(path, path_lines, linelist, params_stars, windows, elements, savepath, cfe_corr_path; mask_regions=false)
    """
    Using Korg to derive the given abundances for each star in the sample.

    Parameters
    ----------
    path : Vector
        List of paths to the stacked spectral data for each star.
    path_lines : Str
        Path to the directories containing the linelists.
    linelist : Vector
        List with the desired linelist.
    params_stars : DataFrame
        Stellar parameters for each star.
    windows : Dict
        Dictionary with the windows for each element.
    elements : Vector
        List with the desired elements.
    cfe_corr_path : Str
        Path to the C/Fe evolutionary corrections.
    savepath : Str
        Path to save the results to.
            
    Returns
    -------
    None
    """

    # Finding and extracting the .fits data
    star_id = split(path, "/")[end]
    println("Running $star_id")

    fits_file = string(path, "/", star_id, "_data.fits")
    star_fits = fits.open(fits_file)
    
    # Getting the UVB data
    uvb_data = star_fits[2].data
    uvb_wave = Korg.air_to_vacuum.(uvb_data["wave"]) # Converting from air to vacuum
    uvb_flux = convert(Array, uvb_data["flux"])
    uvb_flux_err = convert(Array, uvb_data["flux_err"])

    # Getting the VIS data
    vis_data = star_fits[3].data
    vis_wave = Korg.air_to_vacuum.(vis_data["wave"]) # Converting from air to vacuum
    vis_flux = convert(Array, vis_data["flux"])
    vis_flux_err = convert(Array, vis_data["flux_err"])

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

    vsini = 0
    epsilon = 0.6 # limb darkening

    for el_ in elements
        println("Fitting $(el_)")
        # Selecting windows of element
        el = split(el_, " ")[1] # Removing the number at the end of the element

        if el_ == "CH I" || el_ == "NH I"
            el_windows = windows[0x01]
        else
            el_windows = windows[Korg.atomic_numbers[el]]
        end
        

        # ITERATION 1
        # Making directories for each element
        iter = 1
        el_guess_it1 = 0 + feh

        println("Iteration $iter - initial fit")
        star_el_path_it1 = savepath * "$(star_id)/$(split(el_, " ")[1])_$(split(el_, " ")[2])/iter_$(iter)/"
        mkpath(star_el_path_it1)
        star_el_fit_path_it1 = star_el_path_it1 * "fits_data/"
        mkpath(star_el_fit_path_it1)

        # Running Korg
        params = Dict("Teff" => teff, "logg" => logg, "m_H" => feh, "vmic" => vmic, "vsini" => vsini, "epsilon" => epsilon, "alpha_H" => ah)

        el_res_df_it1, el_vals_df_it1, el_abund_xh_it1, el_abund_xfe_it1 = korg_run(uvb_wave, uvb_flux, uvb_flux_err, vis_wave, vis_flux, vis_flux_err, linelist, 
                                                                                    path_lines, params, el_windows, el_, star_id, cfe_corr_path, el_guess=el_guess_it1, star_el_path=star_el_path_it1, 
                                                                                    star_el_fit_path=star_el_fit_path_it1, mask_regions=mask_regions)
        

        # ITERATION 2
        # Making directories for each element
        iter = 2
        el_guess_it2 = el_abund_xh_it1

        println("Iteration $iter - final fit")
        star_el_path_it2 = savepath * "$(star_id)/$(split(el_, " ")[1])_$(split(el_, " ")[2])/iter_$(iter)/"
        mkpath(star_el_path_it2)
        star_el_fit_path_it2 = star_el_path_it2 * "fits_data/"
        mkpath(star_el_fit_path_it2)

        # Running Korg
        params = Dict("Teff" => teff, "logg" => logg, "m_H" => feh, "vmic" => vmic, "vsini" => vsini, "epsilon" => epsilon, "alpha_H" => ah)

        el_res_df_it2, el_vals_df_it2, el_abund_xh_it2, el_abund_xfe_it2 = korg_run(uvb_wave, uvb_flux, uvb_flux_err, vis_wave, vis_flux, vis_flux_err, linelist, 
                                                                                    path_lines, params, el_windows, el_, star_id, cfe_corr_path, el_guess=el_guess_it2, star_el_path=star_el_path_it2,
                                                                                    star_el_fit_path=star_el_fit_path_it2, mask_regions=mask_regions)

    
        # ITERATION 3 - Perturbing Teff by +100K
        # Making directories for each element
        iter = 3
        el_guess_it3 = el_abund_xh_it2

        println("Iteration $iter - perturbing Teff by +100K")
        star_el_path_it3 = savepath * "$(star_id)/$(split(el_, " ")[1])_$(split(el_, " ")[2])/iter_$(iter)/"
        mkpath(star_el_path_it3)
        star_el_fit_path_it3 = star_el_path_it3 * "fits_data/"
        mkpath(star_el_fit_path_it3)

        # Running Korg
        params = Dict("Teff" => teff + 100, "logg" => logg, "m_H" => feh, "vmic" => vmic, "vsini" => vsini, "epsilon" => epsilon, "alpha_H" => ah)
        
        el_res_df_it3, el_vals_df_it3, el_abund_xh_it3, el_abund_xfe_it3 = korg_run(uvb_wave, uvb_flux, uvb_flux_err, vis_wave, vis_flux, vis_flux_err, linelist, 
                                                                                    path_lines, params, el_windows, el_, star_id, cfe_corr_path, el_guess=el_guess_it3, star_el_path=star_el_path_it3,
                                                                                    star_el_fit_path=star_el_fit_path_it3, mask_regions=mask_regions)


        # ITERATION 4 - Perturbing Logg by +0.3dex
        # Making directories for each element
        iter = 4
        el_guess_it4 = el_abund_xh_it2

        println("Iteration $iter - perturbing Logg by +0.3dex")
        star_el_path_it4 = savepath * "$(star_id)/$(split(el_, " ")[1])_$(split(el_, " ")[2])/iter_$(iter)/"
        mkpath(star_el_path_it4)
        star_el_fit_path_it4 = star_el_path_it4 * "fits_data/"
        mkpath(star_el_fit_path_it4)

        # Running Korg
        params = Dict("Teff" => teff, "logg" => logg + 0.3, "m_H" => feh, "vmic" => vmic, "vsini" => vsini, "epsilon" => epsilon, "alpha_H" => ah)
        
        el_res_df_it4, el_vals_df_it4, el_abund_xh_it4, el_abund_xfe_it4 = korg_run(uvb_wave, uvb_flux, uvb_flux_err, vis_wave, vis_flux, vis_flux_err, linelist, 
                                                                                    path_lines, params, el_windows, el_, star_id, cfe_corr_path, el_guess=el_guess_it4, star_el_path=star_el_path_it4,
                                                                                    star_el_fit_path=star_el_fit_path_it4, mask_regions=mask_regions)


        # ITERATION 5 - Perturbing vmic by +0.3km/s
        # Making directories for each element
        iter = 5
        el_guess_it5 = el_abund_xh_it2

        println("Iteration $iter - perturbing vmic by +0.3km/s")
        star_el_path_it5 = savepath * "$(star_id)/$(split(el_, " ")[1])_$(split(el_, " ")[2])/vmic_perturb/"
        mkpath(star_el_path_it5)
        star_el_fit_path_it5 = star_el_path_it5 * "fits_data/"
        mkpath(star_el_fit_path_it5)

        # Running Korg
        params = Dict("Teff" => teff, "logg" => logg, "m_H" => feh, "vmic" => vmic + 0.3, "vsini" => vsini, "epsilon" => epsilon, "alpha_H" => ah)
        
        el_res_df_it5, el_vals_df_it5, el_abund_xh_it5, el_abund_xfe_it5 = korg_run(uvb_wave, uvb_flux, uvb_flux_err, vis_wave, vis_flux, vis_flux_err, linelist, 
                                                                                    path_lines, params, el_windows, el_, star_id, cfe_corr_path, el_guess=el_guess_it5, star_el_path=star_el_path_it5,
                                                                                    star_el_fit_path=star_el_fit_path_it5, mask_regions=mask_regions)

        
        # Calculating uncertainties
        el_abund_xh_teff_err = abs(el_abund_xh_it3-el_abund_xh_it2)
        el_abund_xh_logg_err = abs(el_abund_xh_it4-el_abund_xh_it2)
        el_abund_xh_vmic_err = abs(el_abund_xh_it5-el_abund_xh_it2)
        el_abund_xh_comb_err = sqrt(el_abund_xh_teff_err^2 + el_abund_xh_logg_err^2 + el_abund_xh_vmic_err^2)/3

        el_abund_xfe_teff_err = abs(el_abund_xfe_it3-el_abund_xfe_it2)
        el_abund_xfe_logg_err = abs(el_abund_xfe_it4-el_abund_xfe_it2)
        el_abund_xfe_vmic_err = abs(el_abund_xfe_it5-el_abund_xfe_it2)
        el_abund_xfe_comb_err = sqrt(el_abund_xfe_teff_err^2 + el_abund_xfe_logg_err^2 + el_abund_xfe_vmic_err^2)/3
        

        # Saving the stellar parameters
        params_dict = Dict("star_id" => star_id, "teff" => teff, "teff_err" => teff_err, "logg" => logg, "logg_err" => logg_err, "feh" => feh, "feh_err" => feh_err, 
        "vmic" => vmic, "vmic_err" => vmic_err, "vsini" => vsini, "epsilon" => epsilon,
        "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xfe_init" => el_guess_it1 - feh, "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xfe_it1" => el_abund_xfe_it1,
        "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xfe_it2" => el_abund_xfe_it2, "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xfe_teff_perturb" => el_abund_xfe_it3, 
        "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xfe_logg_perturb" => el_abund_xfe_it4, "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xfe_vmic_perturb" => el_abund_xfe_it5, 
        "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xfe_tefferr" => el_abund_xfe_teff_err, "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xfe_loggerr" => el_abund_xfe_logg_err, 
        "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xfe_vmicerr" => el_abund_xfe_vmic_err, "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xfe_comberr" => el_abund_xfe_comb_err,
        "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xh_init" => el_guess_it1, "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xh_it1" => el_abund_xh_it1, 
        "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xh_it2" => el_abund_xh_it2, "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xh_teff_perturb" => el_abund_xh_it3, 
        "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xh_logg_perturb" => el_abund_xh_it4, "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xh_vmic_perturb" => el_abund_xh_it5, 
        "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xh_tefferr" => el_abund_xh_teff_err, "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xh_loggerr" => el_abund_xh_logg_err, 
        "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xh_vmicerr" => el_abund_xh_it5, "$(split(el_, " ")[1])_$(split(el_, " ")[2])_xh_comberr" => el_abund_xh_comb_err)

        params_df = DataFrame(params_dict)
        el_save = split(el_, " ")[1] * "_" * split(el_, " ")[2] * "_abund.csv"
        CSV.write(savepath * "$(star_id)/" * el_save, params_df)
    end
end
