using Korg
using CSV
using DataFrames
using PyPlot
matplotlib.use("Agg")
using PyCall
using Distributed
import Logging
fits = pyimport("astropy.io.fits")
ast = pyimport("ast")

function chisqr(obs_flux, obs_flux_err, synth_flux)
    """ Calculates the chi-squared value between the observed and synthetic fluxes.

    Parameters
    ----------
    obs_flux : Array{Float64,1}
        Observed flux values.
    obs_flux_err : Array{Float64,1}
        Observed flux errors.
    synth_flux : Array{Float64,1}
        Synthetic flux values.

    Returns
    -------
    chi_squared : Float64
        Chi-squared value.
    """
    
    return sum((obs_flux .- synth_flux).^2 ./ obs_flux_err.^2)
end

function create_mask(wave, ranges)
    mask = falses(length(wave))  # Initialize all false
    for r in ranges
        mask .|= (wave .>= first(r)) .& (wave .<= last(r))
    end
    return .!mask  # Invert to exclude specified ranges
end

function create_synth(teff, logg, feh, ah, linelist, R, vsini, vmic, abund_xh, allabunds_xh, wavemin, wavemax, el, wavelengths, cntm_offset)
    """ Creates a synthetic spectrum for the given stellar parameters and abundance of the element of interest.

    Parameters
    ----------
    teff : Float64
        Effective temperature of the star.
    logg : Float64
        Surface gravity of the star.
    feh : Float64
        [Fe/H] of the star.
    ah : Float64
        [Alpha/H] of the star.
    linelist : Linelist
        Linelist to use for the synthetic spectrum.
    R : Float64
        Spectral resolution to convolve the synthetic spectrum to.
    vsini : Float64
        v sin i of the star.
    vmic : Float64
        Microturbulence of the star.
    abund_xh : Float64
        [X/H] abundance of the element of interest.
    allabunds_xh : Dict
        Dictionary containing the [X/H] abundances of all elements for the star.
    wavemin : Float64
        Minimum wavelength of the spectral window.
    wavemax : Float64
        Maximum wavelength of the spectral window.
    el : Str
        Element of interest.
    wavelengths : Wavelengths
        Wavelengths to synthesise the spectrum over.
    cntm_offset : Float64
        Continuum offset to apply to the synthetic spectrum.

    Returns
    -------
    synth_wave : Array{Float64,1}
        Wavelengths of the synthetic spectrum.
    synth_flux : Array{Float64,1}
        Fluxes of the synthetic spectrum.
    """

    # Updating the abundance of the element of interest
    allabunds_xh[el] = abund_xh

    # Creating synthetic spectra
    A_x = Korg.format_A_X(feh, ah, allabunds_xh)
    atm = Korg.interpolate_marcs(teff, logg, A_x, clamp_abundances=true)
    
    @time synth_spec = Korg.synthesize(atm, linelist, A_x, wavelengths; vmic=vmic)
    
    # Convolving to the instrumental resolution
    synth_wave = Korg.vacuum_to_air.(synth_spec.wavelengths)
    synth_flux = Korg.apply_LSF(synth_spec.flux, synth_wave, R)
    synth_flux = synth_flux ./ synth_spec.cntm

    # Adjusting continuum
    synth_flux = synth_flux .* (1 - cntm_offset)

    return synth_wave, synth_flux
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
        el = star_abund_df[!, "el"][1]
        el_detect = star_abund_df[!, "detection"][1]

        # Extracting the abundances based on the detection flag
        # Those with detections
        if el_detect == "Y"
            star_abund_xh = star_abund_df[!, "bestfit_abund_xh"][1]
            star_abund_xfe = star_abund_df[!,"bestfit_abund_xfe"][1]

            if el == "C_I"
                # Un-correcting the [C/Fe] abundance for C_evolve
                star_cfe = cfe_corr[cfe_corr.star_id .== star, :][!, "cfe_corr"][1]

                star_abund_xfe = star_abund_xfe - star_cfe
                star_abund_xh = star_abund_xfe + feh
            end

        # Those without detections
        elseif el_detect == "N"
            star_abund_xh = star_abund_df[!, "upper_limit_xh"][1]
            star_abund_xfe = star_abund_df[!, "upper_limit_xfe"][1]

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
        push!(els_list, el)
    end

    # Combining the abundances into a single DataFrame
    star_abunds_xh = reduce(hcat, all_abunds_xh)
    star_abunds_xfe = reduce(hcat, all_abunds_xfe)

    # Convert to dictionaries with element names as keys
    # df_xh_cols = names(star_abunds_xh)
    # df_xh_cols = [split(col, "_")[1] for col in df_xh_cols if split(col, "_")[3] == "xh"] # Getting the element names from the column names
    star_abunds_xh_df = DataFrame(Dict(els_list .=> collect(star_abunds_xh[1,:])))

    # df_xfe_cols = names(star_abunds_xfe)
    # df_xfe_cols = [split(col, "_")[1] for col in df_xfe_cols if split(col, "_")[3] == "xfe"] # Getting the element names from the column names
    star_abunds_xfe_df = DataFrame(Dict(els_list .=> collect(star_abunds_xfe[1,:])))

    return star_abunds_xh_df, star_abunds_xfe_df
end


function run_synth(path, windows_path, linelist, params_stars, abunds_path, nondetects, savepath, els, el_xfe_min, el_xfe_max, synth_step, cfe_corr; mask_regions=false)
    """ Uses Korg to generate synthetic spectra for a given star and element (from -2 to +2 dex at 0.01 stepsize), 
    then calculates the chisqaures for all. Does the same for the lowest synthetic spectra at -2 dex.

    Parameters
    ----------
    path : Str
        Path to the star data.
    windows_path : Str
        Path to the windows for the elements.
    linelist : Linelist
        Linelist to use for the synthetic spectra.
    params_stars : DataFrame
        DataFrame containing the stellar parameters for the stars.
    abunds_path : Str
        Path to the stellar abundances.
    nondetects : DataFrame
        DataFrame containing the non-detections for the stars.
    savepath : Str
        Path to save the synthetic spectra and chisquares.
    els : Vector{Str}
        List of elements to synthesise spectra for.
    el_xfe_min : Float64
        Minimum [X/Fe] abundance to synthesise spectra for.
    el_xfe_max : Float64
        Maximum [X/Fe] abundance to synthesise spectra for.
    synth_step : Float64
        Step size in [X/Fe] to synthesise spectra for.
    cfe_corr : DataFrame
        DataFrame containing the C/Fe evolutionary corrections for each star.
    mask_regions : Bool, optional
        Whether to mask regions in the synthetic spectra. Default is false.
    """
    # ---LOADING IN THE SPECTRAL DATA--- #
    # Finding and extracting the .fits data
    star_id = split(path, "/")[end]
    println("Running $star_id")

    fits_file = string(path, "/", star_id, "_data.fits")
    star_fits = fits.open(fits_file)
    fits_header = convert(Array, star_fits[1].header)
    
    # Getting the UVB data
    uvb_data = star_fits[2].data
    uvb_wave = convert(Array, uvb_data["wave"])
    uvb_flux = convert(Array, uvb_data["flux"])
    uvb_flux_err = convert(Array, uvb_data["flux_err"])
    uvb_min = 3000
    uvb_max = 5560
    uvb_res = 5600
    uvb_snr = fits_header[6]

    # Getting the VIS data
    vis_data = star_fits[3].data
    vis_wave = convert(Array, vis_data["wave"])
    vis_flux = convert(Array, vis_data["flux"])
    vis_flux_err = convert(Array, vis_data["flux_err"])
    vis_min = 5600
    vis_max = 10200
    vis_res = 8900
    vis_snr = fits_header[7] # SNR from the header    


    # ---EXTRACTING THE STELLAR PARAMETERS AND ABUNDANCES--- #
    # Getting the stellar parameters
    star_params = params_stars[params_stars.star_id .== star_id, :]
    teff = star_params.teff[1]
    feh = star_params.Fe_it1[1]
    afe = 0.4 # [Alpha/Fe]
    ah = afe + feh # [Alpha/H]
    vsini = star_params.vsini[1]
    epsilon = 0

    # Looping over logg values
    logg_range = 1.0:0.1:4.5
    for logg in logg_range
        println("Generating spectra for logg = $(round(logg, digits=2))")
        teff_ = teff - 5500
        logg_ = logg - 4.0
        vmic = 1.198 + 3.16 * 10^-4 * teff_ - 0.253 * logg_ - 2.86 * 10^-4 * teff_ * logg_ + 0.165 * logg_^2

        # Getting the stellar abundances
        star_abunds_xh_all, star_abunds_xfe_all = get_abunds(star_id, abunds_path, cfe_corr)
        star_abunds_xh_dict = Dict(col => star_abunds_xh_all[!, col][1] for col in names(star_abunds_xh_all))


        # ---BEGINNING SYNTHETIC CODE--- #
        # Looping over the elements
        for el in els
            el_ = split(el, "_")[1] # Removing ionisation state

            # Making the save directories
            star_logg_path = savepath * "/logg_" * string(round(logg, digits=2)) * "/"
            mkpath(star_logg_path)
            star_el_path = star_logg_path * star_id * "/" * el * "/"
            mkpath(star_el_path) 
            star_synth_spec_path = star_el_path * "synth_specs/"
            mkpath(star_synth_spec_path) 
            star_empty_spec_path = star_el_path * "empty_specs/"
            mkpath(star_empty_spec_path)

            # Loading in the continuum offsets for the element
            el_cont_path = windows_path * "$(el)/$(el)_cntm_offsets.csv"
            el_cont_df = CSV.read(el_cont_path, DataFrame)
            star_cont_df = el_cont_df[el_cont_df.star_id .== star_id, :] # Getting the continuum offsets for the star
            el_cntm_offset = star_cont_df.cntm_offset[1]

            # Loading in the window ranges
            el_windows_path = windows_path * "$(el)/windows_synth_$(el).csv"
            el_windows = CSV.read(el_windows_path, DataFrame)
            windmin = el_windows.wavebase[1]
            windmax = el_windows.wavetop[1]
            wavepeak = el_windows.wavepeak[1]

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

            # Getting wavelengths into suitable Korg format
            # Also ensuring correct wavelength step
            wave_indices = (wave .>= windmin) .& (wave .<= windmax)
            wave_step = (Korg.air_to_vacuum(windmax) - Korg.air_to_vacuum(windmin)) / (length(wave[wave_indices]) - 1)
            wavelengths = Korg.Wavelengths(Korg.air_to_vacuum(windmin), Korg.air_to_vacuum(windmax), wave_step)

            # Masking out regions if required
            if mask_regions

                # Getting the mask regions for the element
                el_mask_path = windows_path * "$(el)/$(el)_masksALL.csv"
                el_mask_df = CSV.read(el_mask_path, DataFrame)

                if el == "N_I"
                    line_peak = 3360.220
                end

                # Exracting mask specific to the derired line
                line_masks = nothing
                try
                    line_masks = el_mask_df[float(el_mask_df.line) .== float(line_peak), :].mask[1]
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
                println(line_mask_regs)

                # Identifying the wavelength indices to mask, then inflating their flux errors
                wave_indices = create_mask(wave, line_mask_regs)

                empty_flux_err = copy(flux_err)
                empty_flux_err[.!(wave_indices)] .= 1e10
            end



            # ---GENERATING EMPTY SPECTRUM--- #
            # Generating an "empty" spectrum with a very low abundance of the element
            min_abund_xfe = -10
            min_abund_xh = min_abund_xfe + feh
            println("Generating empty spectrum with [$el/Fe] = $(round(min_abund_xfe, digits=2))")
            empty_wave, empty_flux = create_synth(teff, logg, feh, ah, linelist, res, vsini, vmic, min_abund_xh, star_abunds_xh_dict, windmin, windmax, el_, wavelengths, el_cntm_offset)


            # ---GENERATING SYNTHETIC SPECTRUM ACROSS ABUNDANCE RANGE--- #
            # Getting range of [X/H] abundances to synthesise
            el_xfe_range = el_xfe_min:synth_step:el_xfe_max # Range in [X/Fe]
            el_xh_range = el_xfe_range .+ feh # Range in [X/H]

            # Looping over the abundance range
            chisqrs_all = []
            for (i, abund_xh) in enumerate(el_xh_range)

                println("Generating [$el_/Fe] = $(round(abund_xh - feh, digits=2))")
                synth_wave, synth_flux = create_synth(teff, logg, feh, ah, linelist, res, vsini, vmic, abund_xh, star_abunds_xh_dict, windmin, windmax, el_, wavelengths, el_cntm_offset)            


                # ---CALCULATE CHISQUARE BETWEEN EMPTY + SYNTH, ALONGSIDE OBS + SYNTH--- #
                spec_mask = (wave .> windmin) .& (wave .< windmax)

                # For empty + synth
                if mask_regions
                    empty_chisqr = chisqr(empty_flux, empty_flux_err[spec_mask], synth_flux)
                else
                    empty_chisqr = chisqr(empty_flux, flux_err[spec_mask], synth_flux)
                end
                
                # For obs + synth
                obs_chisqr = chisqr(flux[spec_mask], flux_err[spec_mask], synth_flux)

                # Storing the chisquares
                chisqr_dict = Dict(
                    "star_id" => star_id,
                    "element" => el,
                    "logg" => logg,
                    "xfe" => round(abund_xh - feh, digits=2),
                    "xh" => abund_xh,
                    "chisq_obs" => obs_chisqr,
                    "chisq_empty" => empty_chisqr
                )
                push!(chisqrs_all, chisqr_dict)


                # ---PLOTTING AND SAVING THE SYNTHETIC SPECTRUM--- #
                # Plotting obs + synth
                figure(figsize=(10,6))
                plot(wave[spec_mask], flux[spec_mask], color="black", linewidth=0.8, label="Observed Spectrum")
                plot(synth_wave, synth_flux, color="chocolate", linewidth=1.5, label="[$(el_)/Fe] = $(round(abund_xh - feh, digits=2))\n(chi2 = $(round(empty_chisqr, digits=2)))")
                errorbar(wave[spec_mask], flux[spec_mask], yerr=flux_err[spec_mask], fmt="o", color="grey", alpha=0.5, markersize=2)
                legend(fancybox=false)
                xlabel("Wavelength (Å)")
                ylabel("Normalised Flux")
                title("Logg = $(round(logg, digits=2))")
                ylim(minimum(flux[spec_mask])-0.01, maximum(flux[spec_mask])+0.01)
                tight_layout()
                savefig(star_synth_spec_path * "$(round(abund_xh - feh, digits=2))_spec.png", dpi=300)
                close()
                
                # Plotting empty + synth
                figure(figsize=(10,6))
                plot(empty_wave, empty_flux, color="black", linewidth=0.8, label="Empty Spectrum")
                if mask_regions
                    errorbar(empty_wave, empty_flux, yerr=empty_flux_err[spec_mask], fmt="o", color="grey", alpha=0.5, markersize=2) 
                else
                    errorbar(empty_wave, empty_flux, yerr=flux_err[spec_mask], fmt="o", color="grey", alpha=0.5, markersize=2)
                end           
                plot(synth_wave, synth_flux, color="chocolate", linewidth=1.5, label="[$(el_)/Fe] = $(round(abund_xh - feh, digits=2))\n(chi2 = $(round(empty_chisqr, digits=2)))")
                legend(fancybox=false)
                xlabel("Wavelength (Å)")
                ylabel("Normalised Flux")
                title("Logg = $(round(logg, digits=2))")
                ylim(minimum(flux[spec_mask])-0.01, maximum(flux[spec_mask])+0.01)
                tight_layout()
                savefig(star_empty_spec_path * "$(round(abund_xh - feh, digits=2))_spec.png", dpi=300)
                close()


                # ---SAVING THE SYNTHETIC SPECTRUM--- #
                # For obs + synth
                synth_spec_df = DataFrame(synth_wave=synth_wave, synth_flux=synth_flux, obs_wave=wave[spec_mask], obs_flux=flux[spec_mask], obs_flux_err=flux_err[spec_mask])
                CSV.write(star_synth_spec_path * "$(round(abund_xh - feh, digits=2))_spec.csv", synth_spec_df)

                # For empty + synth
                if mask_regions
                    empty_spec_df = DataFrame(synth_wave=synth_wave, synth_flux=synth_flux, empty_wave=empty_wave, empty_flux=empty_flux, empty_flux_err=empty_flux_err[spec_mask])
                else
                    empty_spec_df = DataFrame(synth_wave=synth_wave, synth_flux=synth_flux, empty_wave=empty_wave, empty_flux=empty_flux, empty_flux_err=flux_err[spec_mask])
                end

                CSV.write(star_empty_spec_path * "$(round(abund_xh - feh, digits=2))_spec.csv", empty_spec_df)
            end

            # ---SAVING THE CHISQUARES--- #
            chisqrs_df = DataFrame(chisqrs_all)
            CSV.write(savepath * "/chisqrs_$(round(logg, digits=2)).csv", chisqrs_df)
        end
    end

end