using Korg
using CSV
using DataFrames
using PyPlot
using PyCall
using Distributed
import Logging
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

# function get_abunds(path, elements)
#     """
#     Loading in the stellar abundances from the given path.

#     Parameters
#     ----------
#     path : Str
#         Path to the stellar abundances.
#     elements : Vector{Str}
#         List of elements to load abundances for.

#     Returns
#     -------
#     star_abunds_xh_df : DataFrame
#         DataFrame containing the stellar [X/H] abundances for the specified elements.
#     star_abunds_xfe_df : DataFrame
#         DataFrame containing the stellar [X/Fe] abundances for the specified elements.
#     """
    
#     # Loading in the stellar abundances from the given path
#     all_abunds_xh = []
#     all_abunds_xfe = []

#     for el in elements
#         # Ignoring the Ca-I abundances
#         if el == "Ca_I"
#             nothing
#         end

#         star_abund_xh = CSV.read(path * el * "_abund.csv", DataFrame)[!, ["$(el)_xh_it2"]]
#         star_abund_xfe = CSV.read(path * el * "_abund.csv", DataFrame)[!, ["$(el)_xfe_it2"]]
#         push!(all_abunds_xh, star_abund_xh)
#         push!(all_abunds_xfe, star_abund_xfe)
#     end

#     # Combining the abundances into a single DataFrame
#     star_abunds_xh = reduce(hcat, all_abunds_xh)
#     star_abunds_xfe = reduce(hcat, all_abunds_xfe)

#     # Convert to dictionaries with element names as keys
#     df_xh_cols = names(star_abunds_xh)
#     df_xh_cols = [split(col, "_")[1] for col in df_xh_cols if split(col, "_")[3] == "xh"] # Getting the element names from the column names
#     star_abunds_xh_df = DataFrame(Dict(df_xh_cols .=> collect(star_abunds_xh[1,:])))

#     df_xfe_cols = names(star_abunds_xfe)
#     df_xfe_cols = [split(col, "_")[1] for col in df_xfe_cols if split(col, "_")[3] == "xfe"] # Getting the element names from the column names
#     star_abunds_xfe_df = DataFrame(Dict(df_xfe_cols .=> collect(star_abunds_xfe[1,:])))

#     return star_abunds_xh_df, star_abunds_xfe_df
# end


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
        el = star_abund_df[!, "el"][1]
        el_detect = star_abund_df[!, "detection"][1]

        # Extracting the abundances based on the detection flag
        # Those with detections
        if el_detect == "Y"
            star_abund_xh = star_abund_df[!, "bestfit_abund_xh"][1]
            star_abund_xfe = star_abund_df[!,"bestfit_abund_xfe"][1]

            star_abund_xh_err = star_abund_df[!, "bestfit_abund_comberr_xh"][1]
            star_abund_xfe_err = star_abund_df[!, "bestfit_abund_comberr_xfe"][1]

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


# function run_synth(path, windows_path, linelist, params_stars, abunds_path, nondetects, savepath; mask_regions=false)
#     """ Uses Korg to generate synthetic spectra for a given star and element (from -2 to +2 dex at 0.01 stepsize), 
#     then calculates the chisqaures for all. Does the same for the lowest synthetic spectra at -2 dex.

#     Parameters
#     ----------
#     path : Str
#         Path to the star data.
#     windows_path : Str
#         Path to the windows for the elements.
#     linelist : Linelist
#         Linelist to use for the synthetic spectra.
#     params_stars : DataFrame
#         DataFrame containing the stellar parameters for the stars.
#     abunds_path : Str
#         Path to the stellar abundances.
#     nondetects : DataFrame
#         DataFrame containing the non-detections for the stars.
#     savepath : Str
#         Path to save the synthetic spectra and chisquares.
#     mask_regions : Bool, optional
#         Whether to mask regions in the synthetic spectra. Default is false.
#     """
#     # ---LOADING IN THE SPECTRAL DATA--- #
#     # Finding and extracting the .fits data
#     star_id = split(path, "/")[end]
#     println("Running $star_id")

#     fits_file = string(path, "/", star_id, "_data.fits")
#     star_fits = fits.open(fits_file)
#     fits_header = convert(Array, star_fits[1].header)
    
#     # Getting the UVB data
#     uvb_data = star_fits[2].data
#     uvb_wave = convert(Array, uvb_data["wave"])
#     uvb_flux = convert(Array, uvb_data["flux"])
#     uvb_flux_err = convert(Array, uvb_data["flux_err"])
#     uvb_min = 3000
#     uvb_max = 5560
#     uvb_res = 5600
#     uvb_snr = fits_header[6]

#     # Getting the VIS data
#     vis_data = star_fits[3].data
#     vis_wave = convert(Array, vis_data["wave"])
#     vis_flux = convert(Array, vis_data["flux"])
#     vis_flux_err = convert(Array, vis_data["flux_err"])
#     vis_min = 5600
#     vis_max = 10200
#     vis_res = 8900
#     vis_snr = fits_header[7] # SNR from the header    



#     # ---EXTRACTING THE STELLAR PARAMETERS--- #
#     # Getting the stellar parameters
#     star_params = params_stars[params_stars.star_id .== star_id, :]
#     teff = star_params.teff[1]
#     logg = star_params.logg[1]
#     feh = star_params.Fe_it2[1]
#     afe = 0.4 # [Alpha/Fe]
#     ah = afe + feh # [Alpha/H]
#     vmic = star_params.vmic[1]
#     vsini = star_params.vsini[1]
#     epsilon = 0


#     # ---EXTRACTING ELEMENTS WITH NO DETECTIONS--- #
#     # Extracting data from the non-detections list
#     element_cols = names(nondetects)[2:end] # Getting elements, which are column names. This excludes the star_id column.
#     detects_df = nondetects[nondetects.star_id .== star_id, :] # Getting non-detections for the star
#     detect_flags = [col[1] for col in eachcol(detects_df)][2:end] # Getting the detection flags for each element, excluding the star_id column
#     no_detects = [element_cols[i] for (i, flag) in pairs(detect_flags) if flag == "N"] # Elements with no detections
#     yes_detects = [element_cols[i] for (i, flag) in pairs(detect_flags) if flag == "Y"] # Elements with yes detections
    
#     # Gettting the stellar abundances for both yes and no detections
#     # If there are no non-detections, then just use the yes detections
#     if length(no_detects) == 0
#         star_abunds_xh_yes, star_abunds_xfe_yes = get_abunds(abunds_path * star_id * "/", yes_detects)
#         star_abunds_xfe_all = star_abunds_xfe_yes # Combining the [X/Fe] abundances for all elements
#         star_abunds_xh_all = star_abunds_xh_yes # Combining the [X/H] abundances for all elements
#     else
#         star_abunds_xh_yes, star_abunds_xfe_yes = get_abunds(abunds_path * star_id * "/", yes_detects)
#         star_abunds_xh_no, star_abunds_xfe_no = get_abunds(abunds_path * star_id * "/", no_detects)

#         star_abunds_xfe_all = hcat(star_abunds_xfe_yes, star_abunds_xfe_no) # Combining the [X/Fe] abundances for all elements
#         star_abunds_xh_all = hcat(star_abunds_xh_yes, star_abunds_xh_no) # Combining the [X/H] abundances for all elements
#     end

#     # Creating synthetic spectra grids for elements with no detections
#     # no_detects = ["N_I", "C_I"]
#     no_detects = ["N_I", "C_I"]
#     println("Checking for non-detections in $star_id: $(no_detects)")
#     for el in no_detects

#         # ---LOADING IN CONTINUUM OFFSET--- #
#         # Loading in the continuum offsets for the element
#         el_cont_path = windows_path * "$(el)/$(el)_cntm_offsets.csv"
#         el_cont_df = CSV.read(el_cont_path, DataFrame)
#         star_cont_df = el_cont_df[el_cont_df.star_id .== star_id, :] # Getting the continuum offsets for the star
#         el_cntm_offset = star_cont_df.cntm_offset[1]

#         if el_cntm_offset == 0.0
#             println("$el - skipping")
#             continue
#         end

#         println("Running $el with continuum offset of $(round(el_cntm_offset, digits=2))")

#         # ---RANGE OF ABUNDANCES TO FIT--- #
#         # Range of abundances to test detection for given element
#         el_xfe_max = 4.0 
#         el_xfe_min = -4.0
#         el_xfe_range = el_xfe_min:0.01:el_xfe_max # Range in [X/Fe]
#         el_xh_range = el_xfe_range .+ feh # Range in [X/H]

#         # el_xh_max = el_xfe_max .+ feh
#         # el_xh_min = el_xfe_min .+ feh 

#         # el_xh_range = el_xh_min:0.01:el_xh_max # Range in [X/H]
#         # el_xh_range = [el_xh_max]

#         # el_xfe_range = [-3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, +0.0, +0.3, +0.6, +0.9, +1.2, +1.5, +1.8, +2.1, +2.4, +2.7, +3.0] # Range in [X/Fe]
#         # el_xfe_max = maximum(el_xfe_range)
#         # el_xfe_min = minimum(el_xfe_range)
#         # el_xh_range = el_xfe_range .+ feh # Range in [X/H]

#         # Getting the window ranges for the element
#         # If multiple lines, then will loop over each one
#         el_windows_path = windows_path * "$(el)/windows_$(el).csv"
#         el_windows = CSV.read(el_windows_path, DataFrame)
#         windmin = el_windows.wavebase[1]
#         windmax = el_windows.wavetop[1]
#         wavepeak = el_windows.wavepeak[1]

#         # el_windows = get_window(el_windows_path)[Korg.atomic_numbers[split(el, "_")[1]]]


#         # ---GATHERING ABUNDANCES--- #
#         # For elements with upper-limits:
#         # - If [X/H] >= 0, then set [X/H] = 0
#         # - If 0 > [X/H] > -1.5, then use current [X/H] value
#         # - If [X/H] <= -1.5, then set [X/H] = 0
#         no_detects_ = [element_cols[i] for (i, flag) in pairs(detect_flags) if flag == "N"] 
#         if length(no_detects_) == 0
#             abunds_yes_xh = Dict(pairs(eachcol(star_abunds_xh_yes)))
#             abunds_yes_xfe = Dict(pairs(eachcol(star_abunds_xfe_yes)))

#             abunds_all_xh = abunds_yes_xh
#             abunds_all_xh = Dict(el => abund[1] for (el, abund) in abunds_all_xh)
#             abunds_all_xfe = abunds_yes_xfe
#             abunds_all_xfe = Dict(el => abund[1] for (el, abund) in abunds_all_xfe)
#         else
#             abunds_yes_xh = Dict(pairs(eachcol(star_abunds_xh_yes)))
#             abunds_yes_xfe = Dict(pairs(eachcol(star_abunds_xfe_yes)))
#             abunds_no_xh = Dict(pairs(eachcol(star_abunds_xh_no)))
#             abunds_no_xfe = Dict(pairs(eachcol(star_abunds_xfe_no)))
        
#             for (el, abund) in pairs(abunds_no_xfe)
#                 # If [X/H] >= 0, then set [X/H] = 0
#                 if abund[1] > 0
#                     abunds_no_xfe[el] = [0.0]
#                     abunds_no_xh[el] = [0.0 + feh]
#                 end
#                 # If [X/H] < 1.5, then use current [X/H] value as it is likely stupid
#                 if abund[1] < -1.5
#                     abunds_no_xfe[el] = [0.0]
#                     abunds_no_xh[el] = [0.0 + feh] 
#                 end
#             end

#             # Combining the abundances into a single dictionary
#             # Note: all abundances are stored as a list in each key, so will will need to 
#             # make them into a single value
#             abunds_all_xh = merge(abunds_yes_xh, abunds_no_xh)
#             abunds_all_xh = Dict(el => abund[1] for (el, abund) in abunds_all_xh)
#             abunds_all_xfe = merge(abunds_yes_xfe, abunds_no_xfe)
#             abunds_all_xfe = Dict(el => abund[1] for (el, abund) in abunds_all_xfe)
#         end



#         # ---SETTING UP THE OBSERVED SPECTRA--- #
#         # Making sure to use the correct spectral arm
#         if windmin < uvb_max
#             wave = uvb_wave
#             wavemin = uvb_min
#             wavemax = uvb_max
#             flux = uvb_flux
#             flux_err = uvb_flux_err
#             res = uvb_res
#             snr = uvb_snr
#         end

#         if windmin > vis_min && windmin < vis_max
#             wave = vis_wave
#             wavemin = vis_min
#             wavemax = vis_max
#             flux = vis_flux
#             flux_err = vis_flux_err
#             res = vis_res
#             snr = vis_snr
#         end

#         # Slicing the observed spectra to the window
#         wave_indices = (wave .>= windmin) .& (wave .<= windmax)
#         wave_obs = wave[wave_indices]
#         flux_obs = flux[wave_indices]
#         flux_err_obs = flux_err[wave_indices]


#         # ---GENERATE EMPTY SPECTRA--- #
#         # Setting the abundance for the element to the lowest value
#         if el == "N_I"
#             abunds_all_xh[:N] = el_xfe_min .+ feh
#         end
#         if el == "C_I"
#             abunds_all_xh[:C] = el_xfe_min .+ feh
#         end

#         # Converting the abundances to Korg format
#         A_x_empty = Korg.format_A_X(feh, ah, abunds_all_xh)

#         # Interpolating the MARCS model atmosphere
#         atm_empty = Korg.interpolate_marcs(teff, logg, A_x_empty, clamp_abundances=true)

#         # Generating the synthetic spectra for the lowest abundance
#         println("Generating empty spectra for $el at [$(split(el, "_")[1])/Fe] = $(round(el_xfe_min, digits=2))")
#         wave_step = (Korg.air_to_vacuum(windmax) - Korg.air_to_vacuum(windmin)) / (length(wave_obs) - 1)
#         @time empty_spec = Korg.synthesize(atm_empty, linelist, A_x_empty, Korg.Wavelengths(Korg.air_to_vacuum(windmin), Korg.air_to_vacuum(windmax), wave_step))
#         empty_wave = Korg.vacuum_to_air.(empty_spec.wavelengths)
#         empty_flux = Korg.apply_LSF(empty_spec.flux, empty_wave, res)
#         empty_flux = empty_flux ./ empty_spec.cntm


#         # ---GENERATING SYNTHETIC SPECTRA--- #
#         println("Generating synthetic spectra for $el")

#         # Making the save directories
#         star_el_path = savepath * star_id * "/" * el * "/"
#         mkpath(star_el_path) 
#         star_synth_spec_path = star_el_path * "synth_specs/"
#         mkpath(star_synth_spec_path) 
#         star_empty_spec_path = star_el_path * "empty_specs/"
#         mkpath(star_empty_spec_path)

#         # Empty list to store chisquares
#         chisqrs_all = []

#         for (i, abund) in enumerate(el_xh_range)
            
#             ith_xfe = el_xfe_range[i]
#             println("[$(split(el, "_")[1])/Fe] = $(round(ith_xfe, digits=2))")


#             # ---CALCULATING CHISQUARES FOR OBSERVED SPECTRA--- #
#             # Updating the abundance for the element for the model atmosphere
#             if el == "N_I"
#                 abunds_all_xh[:N] = abund # Updating the abundance for the element
#             end
#             if el == "C_I"
#                 abunds_all_xh[:C] = abund # Updating the abundance for the element
#             end
            
#             # Converting the abundances to Korg format
#             A_x = Korg.format_A_X(feh, ah, abunds_all_xh)

#             # Interpolating the MARCS model atmosphere
#             atm = Korg.interpolate_marcs(teff, logg, A_x, clamp_abundances=true)

#             # Generating the synthetic spectra for the element, then convolving with the LSF
#             wave_step = (Korg.air_to_vacuum(windmax) - Korg.air_to_vacuum(windmin)) / (length(wave_obs) - 1)
#             @time synth_spec = Korg.synthesize(atm, linelist, A_x, Korg.Wavelengths(Korg.air_to_vacuum(windmin), Korg.air_to_vacuum(windmax), wave_step))
#             synth_wave = Korg.vacuum_to_air.(synth_spec.wavelengths)
#             convolved_flux = Korg.apply_LSF(synth_spec.flux, synth_wave, res)
#             convolved_flux = convolved_flux ./ synth_spec.cntm
            
#             # Adjusting the continuum
#             synth_flux = convolved_flux .* (1 - el_cntm_offset)

#             # Calculating the chi-squared for the synthetic spectra
#             chisq_obs = sum((flux_obs .- synth_flux).^2 ./ flux_err_obs.^2)

#             # Plotting the synthetic spectra
#             figure(figsize=(10,6))
#             plot(synth_wave, synth_flux, label="Synthetic\n[$(split(el, "_")[1])/Fe] = $(round(ith_xfe, digits=2))\nChisqr = $(round(chisq_obs, digits=2))", color="chocolate")
#             plot(wave_obs, flux_obs, label="Observed", color="black", linewidth=0.9)
#             xlabel("Wavelength (Å)")
#             ylabel("Normalised Flux")
#             legend(fancybox=false)
#             savefig(star_synth_spec_path * "$(el)_$(round(ith_xfe, digits=2))_spec.png", dpi=300, transparent=false, facecolor="white")
#             close(gcf())

#             # Saving the synthetic spectra to a file
#             synth_spec_path = star_synth_spec_path * "$(el)_$(round(ith_xfe, digits=2))_spec.csv"
#             synth_spec_df = DataFrame(synth_wave=[synth_wave], synth_flux=[synth_flux], obs_wave=[wave_obs], obs_flux=[flux_obs], obs_flux_err=[flux_err_obs])
#             CSV.write(synth_spec_path, synth_spec_df)


#             # ---CALCULATING CHISQUARES FOR EMPTY SPECTRA--- #
#             # Calculating the chi-squared for the synthetic spectra
#             chisq_empty = sum((empty_flux .- convolved_flux).^2 ./ flux_err_obs.^2)

#             # Plotting the synthetic spectra
#             figure(figsize=(10,6))
#             plot(synth_wave, convolved_flux, label="Synthetic\n[$(split(el, "_")[1])/Fe] = $(round(ith_xfe, digits=2))\nChisqr = $(round(chisq_obs, digits=2))", color="chocolate")
#             plot(empty_wave, empty_flux, label="Empty\n[$(split(el, "_")[1])/Fe] = $(round(el_xfe_min, digits=2))", color="black", linewidth=0.9)
#             xlabel("Wavelength (Å)")
#             ylabel("Normalised Flux")
#             legend(fancybox=false)
#             savefig(star_empty_spec_path * "$(el)_$(round(ith_xfe, digits=2))_spec.png", dpi=300, transparent=false, facecolor="white")
#             close(gcf())

#             # Saving the synthetic spectra to a file
#             empty_spec_path = star_empty_spec_path * "$(el)_$(round(ith_xfe, digits=2))_spec.csv"
#             empty_spec_df = DataFrame(empty_wave=[empty_wave], empty_flux=[empty_flux], synth_wave=[synth_wave], synth_flux=[synth_flux], obs_flux_err=[flux_err_obs])
#             CSV.write(empty_spec_path, empty_spec_df)


#             # ---SAVING CHISQUARES TO FILE--- #
#             # Storing the chisquare values in dictionary, then appending to list
#             chisqr_dict = Dict(
#                 "star_id" => star_id,
#                 "element" => el,
#                 "xfe" => ith_xfe,
#                 "xh" => abund,
#                 "chisq_obs" => chisq_obs,
#                 "chisq_empty" => chisq_empty
#             )
#             push!(chisqrs_all, chisqr_dict)
#         end

#         # Writing list of dictionaries to DataFrame
#         chisqrs_df = DataFrame(chisqrs_all)

#         # Saving the chisquares to a file
#         chisqrs_path = star_el_path * "chisquares.csv"
#         CSV.write(chisqrs_path, chisqrs_df)
#         println()

#     end
# end



function run_synth_fitspec(path, windows_path, linelist, params_stars, abunds_path, nondetects, savepath, els, el_xfe_min, el_xfe_max, cfe_corr; mask_regions=false)
    """ Uses Korg to generate synthetic spectra for a given star and element, then calculates the chisqaures for all. Does the same for the lowest synthetic spectra at -2 dex.

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
    els : List
        List of elements to generate synthetic spectra for.
    mask_regions : Bool, optional
        Whether to mask regions in the synthetic spectra. Default is false.
    """
    # ---LOADING IN THE SPECTRAL DATA--- #
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


    # ---EXTRACTING THE STELLAR PARAMETERS--- #
    # Getting the stellar parameters
    star_params = params_stars[params_stars.star_id .== star_id, :]
    teff = star_params.teff[1]
    logg = star_params.logg[1]
    feh = star_params.Fe_it1[1]
    # afe = 0.4 # [Alpha/Fe]
    # ah = afe + feh # [Alpha/H]
    vmic = star_params.vmic[1]
    vsini = star_params.vsini[1]
    epsilon = 0


    # ---EXTRACTING ELEMENTS WITH NO DETECTIONS--- #
    # Extracting data from the non-detections list
    element_cols = names(nondetects)[2:end] # Getting elements, which are column names. This excludes the star_id column.
    detects_df = nondetects[nondetects.star_id .== star_id, :] # Getting non-detections for the star
    detect_flags = [col[1] for col in eachcol(detects_df)][2:end] # Getting the detection flags for each element, excluding the star_id column
    no_detects = [element_cols[i] for (i, flag) in pairs(detect_flags) if flag == "N"] # Elements with no detections
    yes_detects = [element_cols[i] for (i, flag) in pairs(detect_flags) if flag == "Y"] # Elements with yes detections
    
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
    
    # Creating synthetic spectra grids for elements with no detections
    # no_detects = ["N_I", "C_I"]
    no_detects = els

    for el in no_detects

        # Range of abundances to test detection for given element
        el_xfe_max = el_xfe_max
        el_xfe_min = el_xfe_min

        el_xh_max = el_xfe_max .+ feh
        el_xh_min = el_xfe_min .+ feh 

        el_xh_range = el_xh_min:0.01:el_xh_max # Range in [X/H]
``
        # el_xfe_range = [-3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, +0.0, +0.3, +0.6, +0.9, +1.2, +1.5, +1.8, +2.1, +2.4, +2.7, +3.0] # Range in [X/Fe]
        # el_xfe_max = maximum(el_xfe_range)
        # el_xfe_min = minimum(el_xfe_range)
        # el_xh_range = el_xfe_range .+ feh # Range in [X/H]

        # Getting the window ranges for the element
        # If multiple lines, then will loop over each one
        el_windows_path = windows_path * "$(el)/windows_$(el).csv"
        el_windows = get_window(el_windows_path)[Korg.atomic_numbers[split(el, "_")[1]]]
        

        # ---CALCULATING CHISQUARES FOR OBSERVED SPECTRA--- #
        println("Calculating chisquares for $el with the obseerved spectra")
        obs_el_savepath = savepath * star_id * "/" * el * "/obs/"
        mkpath(obs_el_savepath) # Making the directory for the star and element
        for wind in el_windows
            # Getting the line properties
            line_peak = wind[1]
            windmin = wind[2] - 10
            windmax = wind[3] + 10
            # line_peak = wind[1]
            # windmin = line_peak - 5
            # windmax = line_peak + 5

            line_savepath = obs_el_savepath * "line_$(line_peak)/"
            mkpath(line_savepath) # Making the directory for the line

            # Resolutions of UVB and VIS arms
            R_ubv = 5600
            R_vis = 8900
            # R_vis = 12000

            # Ranges for the UVB and VIS arms
            uvb_min = Korg.air_to_vacuum(3000)
            uvb_max = Korg.air_to_vacuum(5560)
            vis_min = Korg.air_to_vacuum(5600)
            vis_max = Korg.air_to_vacuum(10200)

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

            # Getting mask regions if required
            if mask_regions
                if el == "N_I"
                    # N_I has no mask regions, so skipping
                    continue
                end

                # Getting the mask regions for the element
                el_mask_path = windows_path * "$(el)/$(el)_masksALL.csv"
                el_mask_df = CSV.read(el_mask_path, DataFrame)

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

                # Identifying the wavelength indices to mask, then inflating their flux errors
                wave_indices = create_mask(wave, line_mask_regs)
                flux_err[.!(wave_indices)] .= 1e10
            end


            # Generating the synthetic spectra for each abundance in the range
            chisqrs_all = []
            for el_xh in el_xh_range

                # Including given element abundance in the dictionary
                guesses = Dict("cntm_offset" => 0.0, "cntm_slope" => 0.0)
                params = Dict("Teff" => teff, "logg" => logg, "m_H" => feh, "vmic" => vmic, "vsini" => vsini, "epsilon" => epsilon, "$(split(el, "_")[1])" => el_xh, "alpha_H" => ah)

                # Removing the desired elemenet from the fix params dictionary
                star_abund_dict_noel_xh = copy(star_abund_dict_xh)
                delete!(star_abund_dict_noel_xh, "$(split(el, "_")[1])")
                star_abund_dict_noel_xfe = copy(star_abund_dict_xfe)
                delete!(star_abund_dict_noel_xfe, "$(split(el, "_")[1])")
                params = merge(params, star_abund_dict_noel_xh) # Adding the stellar abundances to the parameters

                println("$(el) $(line_peak): generating [$(split(el, "_")[1])/Fe] = $(round(el_xh - feh, digits=3))")

                # I'm doing something very naughty here, but inorder for me to have everything fixed, I need to use Korg's old way
                # of fitting the continuum, which is to have "cntm_offset" and "cntm_slope" as guesses and adjust_continuum=false
                # in  Korg.Fit.fit_spectrum. I do this because you can't have guesses being empty, so I need to rig a dumb solution to
                # overcome this. Korg does not appreciate the old way of fitting the continuum, and so will spit out a shit ton of warnings, which 
                # I don't care about in the slightest. Thus, I use Logging around the function to splat them away into oblivion. Don't tell anyone...
                spec_fit = nothing
                Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do 
                    @time spec_fit = Korg.Fit.fit_spectrum(wave, flux, flux_err, linelist, guesses, params, R=res,windows=[(windmin, windmax)], 
                                                    adjust_continuum=false, precision=1e-6)
                end

                # Extracting the results
                best_fit = spec_fit.best_fit_flux
                fit_mask = spec_fit.obs_wl_mask
                trace = spec_fit.trace

                # Converting to air, then getting the central wavelength
                wave_air = Korg.vacuum_to_air.(wave)
                windmin_air = Korg.vacuum_to_air(windmin)
                windmax_air = Korg.vacuum_to_air(windmax)

                obs_spec_wind = 5 # A
                spec_mask = (wave_air .> windmin_air-obs_spec_wind) .& (wave_air .< windmax_air+obs_spec_wind)

                # Getting the abundances
                el_xh_abund = el_xh
                el_xfe_abund = el_xh_abund - feh # in [X/Fe]
                el_chisqr = spec_fit.trace[end]["chi2"]
                

                # Storing the chisquares values in a dictionary
                chisqr_dict = Dict("el" => [el], "el_xh" => [el_xh_abund], "el_xfe" => [round(el_xfe_abund, digits=2)], "chisqr" => [el_chisqr])
                push!(chisqrs_all, chisqr_dict)

                # Saving the synthetic spectra
                filename_spec = line_savepath * "spectra/"
                mkpath(filename_spec)

                spec_dict = Dict("obs_wave" => [wave_air[fit_mask]], "obs_flux" => [flux[fit_mask]],
                                 "obs_flux_err" => [flux_err[fit_mask]], "fit_wave" => [wave_air[fit_mask]], "fit_flux" => [best_fit])
      
                spec_df = DataFrame(spec_dict)
                CSV.write(filename_spec * "$(round(el_xh - feh, digits=3))_spec.csv", spec_df)
                
                # Plotting the fit (with regions masked if mask_regions is set to true)
                filename_plot = line_savepath * "plots/"
                mkpath(filename_plot) 

                label = "[$(split(el, "_")[1])/Fe] = $(round(el_xfe_abund, digits=2))\n(chi2 = $(round(el_chisqr, digits=2)))"

                plt.figure(figsize=(8, 6))
                plt.plot(wave_air[spec_mask], flux[spec_mask], label="Observed", color="black", linewidth=0.8)
                plt.plot(wave_air[fit_mask], best_fit, label=label, color="chocolate", linewidth=1.5)
                
                # Shading masked regions if mask_regions is set to true
                if mask_regions
                    for mask in line_mask_regs
                        plt.fill_between(mask, maximum(flux[spec_mask])+0.01, minimum(flux[spec_mask])-0.01, color="grey", alpha=0.5)
                    end
                end

                plt.axvline(line_peak, color="darkred", linestyle=":")
                plt.axvline(wave_air[fit_mask][1], color="grey", linestyle="--")
                plt.axvline(wave_air[fit_mask][end], color="grey", linestyle="--")
                plt.xlabel("Wavelength (Å)")
                plt.ylabel("Normalised Flux")
                plt.legend(fancybox=true)
                plt.tight_layout()
                plt.ylim(minimum(flux[spec_mask])-0.01, maximum(flux[spec_mask])+0.01)
                plt.savefig(filename_plot * "$(round(el_xh - feh, digits=3))_spec.png", dpi=300, transparent=false, facecolor="white")
                plt.close()
            end

            # Saving the chisquares for the element
            chisqr_df = vcat(DataFrame.(chisqrs_all)...)
            CSV.write(line_savepath * "$(line_peak)_chisqrs.csv", chisqr_df)

        end
    end
end


#         # ---CALCULATING CHISQUARES FOR LOWEST SYNTHETIC SPECTRA--- #
#         println("Calculating chisquares for $el with [$(split(el, "_")[1])/Fe] = $el_xfe_min synthetic spectra")
#         lowsynth_el_savepath = savepath * star_id * "/" * el * "/lowest_synth/"
#         mkpath(lowsynth_el_savepath) # Making the directory for the star and element

#         for wind in el_windows
#             # Getting the line properties
#             line_peak = wind[1]
#             windmin = wind[2] - 10
#             windmax = wind[3] + 10
            
#             # Loading in the lowest synthetic spectra for the element (calculated from above)
#             lowsynth_el_path = obs_el_savepath * "line_$(line_peak)/spectra/$(el_xfe_min)_spec.csv" 
#             lowsynth_df = CSV.read(lowsynth_el_path, DataFrame)
#             lowsynth_wave = Korg.air_to_vacuum.(ast.literal_eval(lowsynth_df.fit_wave[1]))
#             lowsynth_flux = repeat([1.0], length(lowsynth_wave))
#             lowsynth_flux_err = ast.literal_eval(lowsynth_df.obs_flux_err[1]) 

#             line_savepath = lowsynth_el_savepath * "line_$(line_peak)/"
#             mkpath(line_savepath) # Making the directory for the line
 
#             # Resolutions of UVB and VIS arms
#             R_ubv = 5600
#             R_vis = 8900
#             # R_vis = 12000

#             # Ranges for the UVB and VIS arms
#             uvb_min = Korg.air_to_vacuum(3000)
#             uvb_max = Korg.air_to_vacuum(5560)
#             vis_min = Korg.air_to_vacuum(5600)
#             vis_max = Korg.air_to_vacuum(10200)

#             # Making sure to use the correct spectral arm
#             if windmin < uvb_max
#                 wave = uvb_wave
#                 wavemin = uvb_min
#                 wavemax = uvb_max
#                 res = R_ubv
#             end

#             if windmin > vis_min && windmin < vis_max
#                 wave = vis_wave
#                 wavemin = vis_min
#                 wavemax = vis_max
#                 res = R_vis
#             end

#             # # Getting mask regions if required
#             # if mask_regions
#             #     if el == "N_I"
#             #         # N_I has no mask regions, so skipping
#             #         continue
#             #     end

#             #     # Getting the mask regions for the element
#             #     el_mask_path = windows_path * "$(el)/$(el)_masksALL.csv"
#             #     el_mask_df = CSV.read(el_mask_path, DataFrame)

#             #     # Exracting mask specific to the derired line
#             #     line_masks = nothing
#             #     try
#             #         line_masks = el_mask_df[float(el_mask_df.line) .== float(line_peak), :].mask[1]
#             #         line_masks = ast.literal_eval(line_masks)
#             #     catch e
#             #         error("No mask found for line $(line_peak) - please check mask exists in file at $(el_mask_path)")
#             #     end

#             #     # Converting the mask into a range (i.e. taking the min and max of the mask, then creating a 
#             #     # range at 0.1A intervals)
#             #     line_mask_regs = []
#             #     step = 0.1
#             #     for mask in line_masks
#             #         reg = mask[1]:step:mask[2]
#             #         push!(line_mask_regs, reg)
#             #     end

#             #     # Identifying the wavelength indices to mask, then inflating their flux errors
#             #     wave_indices = create_mask(lowsynth_wave, line_mask_regs)
#             #     lowsynth_flux_err[.!(wave_indices)] .= 1e10
#             # end


#             # Generating the synthetic spectra for each abundance in the range
#             chisqrs_all_synth = []
#             for el_xh in el_xh_range

#                 # Including given element abundance in the dictionary
#                 guesses = Dict("cntm_offset" => 0.0, "cntm_slope" => 0.0)
#                 params = Dict("Teff" => teff, "logg" => logg, "m_H" => feh, "vmic" => vmic, "vsini" => vsini, "epsilon" => epsilon, "$(split(el, "_")[1])" => el_xh)

#                 println("$(el) $(line_peak): generating [$(split(el, "_")[1])/Fe] = $(round(el_xh - feh, digits=3))")

#                 # I'm doing something very naughty here, but inorder for me to have everything fixed, I need to use Korg's old way
#                 # of fitting the continuum, which is to have "cntm_offset" and "cntm_slope" as guesses and adjust_continuum=false
#                 # in  Korg.Fit.fit_spectrum. Korg does not appreciate the old way, and so will spit out a shit ton of warnings, which 
#                 # I don't care about in the slightest. Thus, I use Logging to splat them away into oblivion. Don't tell anyone...
#                 spec_fit = nothing
#                 Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do 
#                     @time spec_fit = Korg.Fit.fit_spectrum(lowsynth_wave, lowsynth_flux, lowsynth_flux_err, linelist, guesses, params, R=res,windows=[(windmin, windmax)], 
#                                                     adjust_continuum=false, precision=1e-6)
#                 end

#                 # Extracting the results
#                 best_fit = spec_fit.best_fit_flux
#                 fit_mask = spec_fit.obs_wl_mask
#                 trace_synth = spec_fit.trace

#                 # Converting to air, then getting the central wavelength
#                 wave_air = Korg.vacuum_to_air.(lowsynth_wave)
#                 windmin_air = Korg.vacuum_to_air(windmin)
#                 windmax_air = Korg.vacuum_to_air(windmax)

#                 spec_mask = (wave_air .> windmin_air) .& (wave_air .< windmax_air)

#                 # Getting the abundances
#                 el_xh_abund = el_xh
#                 el_xfe_abund = el_xh_abund - feh # in [X/Fe]
#                 el_chisqr = spec_fit.trace[end]["chi2"]
                

#                 # Storing the chisquares values in a dictionary
#                 chisqr_dict = Dict("el" => [el], "el_xh" => [el_xh_abund], "el_xfe" => [round(el_xfe_abund, digits=2)], "chisqr" => [el_chisqr])
#                 push!(chisqrs_all_synth, chisqr_dict)

#                 # Saving the synthetic spectra
#                 filename_spec = line_savepath * "spectra/"
#                 mkpath(filename_spec)

#                 spec_dict = Dict("obs_wave" => [wave_air[fit_mask]], "obs_flux" => [lowsynth_flux[fit_mask]], 
#                                  "obs_flux_err" => [lowsynth_flux_err[fit_mask]], "fit_wave" => [wave_air[fit_mask]], "fit_flux" => [best_fit])
      
#                 spec_df = DataFrame(spec_dict)
#                 CSV.write(filename_spec * "$(round(el_xh - feh, digits=3))_spec.csv", spec_df)
                
#                 # Plotting the fit (with regions masked if mask_regions is set to true)
#                 filename_plot = line_savepath * "plots/"
#                 mkpath(filename_plot) 

#                 label = "[$(split(el, "_")[1])/Fe] = $(round(el_xfe_abund, digits=2))\n(chi2 = $(round(el_chisqr, digits=2)))"

#                 plt.figure(figsize=(8, 6))
#                 plt.plot(wave_air[spec_mask], lowsynth_flux[spec_mask], label="Empty\n[$(split(el, "_")[1])/Fe] = $(el_xfe_min)", color="black", linewidth=0.8)
#                 plt.plot(wave_air[fit_mask], best_fit, label=label, color="chocolate", linewidth=1.5)
                
#                 # # Shading masked regions if mask_regions is set to true
#                 # if mask_regions
#                 #     for mask in line_mask_regs
#                 #         plt.fill_between(mask, maximum(lowsynth_flux[spec_mask])+0.01, minimum(lowsynth_flux[spec_mask])-0.01, color="grey", alpha=0.5)
#                 #     end
#                 # end

#                 plt.axvline(line_peak, color="darkred", linestyle=":")
#                 plt.axvline(wave_air[fit_mask][1], color="grey", linestyle="--")
#                 plt.axvline(wave_air[fit_mask][end], color="grey", linestyle="--")
#                 plt.xlabel("Wavelength (Å)")
#                 plt.ylabel("Normalised Flux")
#                 plt.legend(fancybox=true)
#                 plt.tight_layout()
#                 plt.ylim(minimum(lowsynth_flux[spec_mask])-0.01, maximum(lowsynth_flux[spec_mask])+0.01)
#                 plt.savefig(filename_plot * "$(round(el_xh - feh, digits=3))_spec.png", dpi=300, transparent=false, facecolor="white")
#                 plt.close()
#             end

#             # Saving the chisquares for the element
#             chisqr_synth_df = vcat(DataFrame.(chisqrs_all_synth)...)
#             CSV.write(line_savepath * "$(line_peak)_chisqrs.csv", chisqr_synth_df)

#         end



#     end
# end


