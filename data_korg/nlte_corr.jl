using CSV
using DataFrames
using PyPlot
using PyCall
using Distributed
# Table = pyimport("astropy.table").Table
fits = pyimport("astropy.io.fits")
scipy_interpolate = pyimport("scipy.interpolate")
np = pyimport("numpy")
LinearNDInterpolator = scipy_interpolate."LinearNDInterpolator"

function lind2022_nlte_correction(star, el, el_lines, grid_path, star_teff, star_logg, star_feh, star_vmic, star_xfes)
    """Calculates the NLTE correction for a given star, element, and line(s) using the Lind et al. (2022) NLTE grid. 
    The grid will be linearly interpolated across the grid of stellar parameters.

    Parameters
    ----------
    star : String
        Star identifier
    el : String
        The element for which to calculate the NLTE correction (e.g., "al", "mg", "na")
    el_lines : Vector{Int}
        The line(s) for which to calculate the NLTE correction (e.g., [3962] for Al, [5167, 5172, 5183, 8806] for Mg, [5889, 5895] for Na)
    grid_path : String
        The file path to the directory containing the Lind et al. (2022) NLTE grid files.
    star_teff : Float64
        The effective temperature of the star (in K).
    star_logg : Float64
        The surface gravity of the star (in dex).
    star_feh : Float64
        The metallicity of the star (in dex).
    star_vmic : Float64
        The microturbulent velocity of the star (in km/s).
    star_xfes : Vector{Float64}
        The [X/Fe] abundance(s) of the star (in dex) for the given element and line(s). Each [X/Fe] abundance should correspond to the respective 
        line in el_lines. For example, for Na, this would be a vector of length 2 containing the [Na/Fe] abundances for the 2 Na lines.

    Returns
    -------
    star_nlte_df : DataFrame
        A DataFrame containing the NLTE correction for the given star, element, and line(s), as well as any notes about the interpolation 
            (e.g., if any stellar parameters were outside the grid limits and were forced to be within the grid for interpolation).
    """

    function interpolate_nlte_correction(el, el_line_df, star_teff, star_logg, star_feh, star_vmic, star_xfe)
        """For a given element and line, this function will perform a linear interpolation across the 5D grid 
        of stellar parameters (Teff, logg, [Fe/H], Vmic, [X/Fe]) to get the corresponding LTE EW for the 
        star's parameters. Then, it will adopt this LTE EW as the NLTE EW, and perform another linear 
        interpolation across the grid of stellar parameters + NLTE EW to get the corresponding NLTE abundance 
        for the star's parameters. Finally, it will return the difference between the NLTE abundance and the 
        input [X/Fe] as the NLTE correction for that line.

        Parameters
        ----------
        el : String
            The element for which to calculate the NLTE correction (e.g., "al", "mg", "na")
        el_line_df : DataFrame
            A DataFrame containing the grid of stellar parameters and corresponding LTE and NLTE EWs for the 
            given element and line.
        star_teff : Float64
            The effective temperature of the star (in K).
        star_logg : Float64
            The surface gravity of the star (in dex).
        star_feh : Float64
            The metallicity of the star (in dex).
        star_vmic : Float64
            The microturbulent velocity of the star (in km/s).
        star_xfe : Float64
            The [X/Fe] abundance of the star (in dex).

        Returns
        -------
        xfe_nlte_correction : Float64
            The NLTE correction for the given element and line, defined as the difference between the NLTE 
            abundance and the input LTE abundance (in dex).
        notes : Vector{String}
            A vector of notes indicating if any of the stellar parameters were outside the grid limits and 
            were forced to be within the grid for interpolation (e.g., "teff_below_grid", "logg_above_grid", etc.)
        """
        # Extracting the relevant columns for interpolation
        teff_grid = el_line_df."Teff"
        logg_grid = el_line_df."logg"
        feh_grid = el_line_df."[Fe/H]"
        vmic_grid = el_line_df."Vturb"
        ew_lte_grid = el_line_df."logWL"
        ew_nlte_grid = el_line_df."logWN"
        
        if el == "al"
            xfe_grid = el_line_df."[Al/Fe]"
        elseif el == "mg"
            xfe_grid = el_line_df."[Mg/Fe]"
        elseif el == "na"
            xfe_grid = el_line_df."[Na/Fe]"
        end

        # If the star's parameters full outside the grid, force the parameter to be within the grid limits to avoid extrapolation
        notes = String[]
        if star_teff < minimum(teff_grid)
            star_teff = minimum(teff_grid)
            note = "teff_below_grid"
            push!(notes, note)
        elseif star_teff > maximum(teff_grid)
            star_teff = maximum(teff_grid)
            note = "teff_above_grid"
            push!(notes, note)
        end

        if star_logg < minimum(logg_grid)
            star_logg = minimum(logg_grid)
            note = "logg_below_grid"
            push!(notes, note)
        elseif star_logg > maximum(logg_grid)
            star_logg = maximum(logg_grid)
            note = "logg_above_grid"
            push!(notes, note)
        end

        if star_feh < minimum(feh_grid)
            star_feh = minimum(feh_grid)
            note = "feh_below_grid"
            push!(notes, note)
        elseif star_feh > maximum(feh_grid)
            star_feh = maximum(feh_grid)
            note = "feh_above_grid"
            push!(notes, note)
        end

        if star_vmic < minimum(vmic_grid)
            star_vmic = minimum(vmic_grid)
            note = "vmic_below_grid"
            push!(notes, note)
        elseif star_vmic > maximum(vmic_grid)
            star_vmic = maximum(vmic_grid)
            note = "vmic_above_grid"
            push!(notes, note)
        end

        if star_xfe < minimum(xfe_grid)
            star_xfe = minimum(xfe_grid)
            note = "xfe_below_grid"
            push!(notes, note)
        elseif star_xfe > maximum(xfe_grid)
            star_xfe = maximum(xfe_grid)
            note = "xfe_above_grid"
            push!(notes, note)
        end
        
        # For stars that were within the grid limit, set the note to be blank
        if length(notes) == 0
            push!(notes, "")
        end 

        # Running the linear interpolation to get the corresponding EW_LTE for the given stellar parameters
        points = hcat(teff_grid, logg_grid, feh_grid, vmic_grid, xfe_grid)
        lte_grid = LinearNDInterpolator(points, el_line_df."logWL")
        ew_lte = lte_grid(star_teff, star_logg, star_feh, star_vmic, star_xfe)[1]

        # Now that we have the LTE EW, we adopt this as NLTE EW, and interpolate to see what the corresponding 
        # NLTE abundance is for this EW
        points_nlte = hcat(teff_grid, logg_grid, feh_grid, vmic_grid, ew_nlte_grid)

        if el == "al"
            nlte_grid = LinearNDInterpolator(points_nlte, el_line_df."[Al/Fe]")
        elseif el == "mg"
            nlte_grid = LinearNDInterpolator(points_nlte, el_line_df."[Mg/Fe]")
        elseif el == "na"
            nlte_grid = LinearNDInterpolator(points_nlte, el_line_df."[Na/Fe]")
        end

        xfe_nlte = nlte_grid(star_teff, star_logg, star_feh, star_vmic, ew_lte)[1]
        xfe_nlte_correction = xfe_nlte - star_xfe

        return xfe_nlte_correction, xfe_nlte, notes
    end

    # Loading in the NLTE grid
    grid_file = joinpath(grid_path, "lind2022_$(el)_nlte.csv")
    el_nlte_grid = CSV.read(grid_file, DataFrame)

    # Since the grids are MASSIVE, alongside the fact that we're performing a linear interpolation across
    # 5 dimension, we will slice the grid to only include the relevant points for interpolation. This will 
    # speed up the interpolation process significantly, while still providing accurate results.
    min_teff = star_teff - 500
    max_teff = star_teff + 500
    min_logg = star_logg - 1.0
    max_logg = star_logg + 1.0
    min_feh = star_feh - 1.0
    max_feh = star_feh + 1.0
    
    mask = (el_nlte_grid.Teff .>= min_teff) .& (el_nlte_grid.Teff .<= max_teff) .&
            (el_nlte_grid.logg .>= min_logg) .& (el_nlte_grid.logg .<= max_logg) .&
            (el_nlte_grid[!, "[Fe/H]"] .>= min_feh) .& (el_nlte_grid[!, "[Fe/H]"] .<= max_feh)
    el_nlte_grid_masked = el_nlte_grid[mask, :]

    # Extracting the relevant lines for the element
    # If multiple lines are provided, we will average the NLTE corrections across those lines
    if length(el_lines) == 1
        el_line_df = el_nlte_grid_masked[el_nlte_grid_masked."Line" .== el_lines[1], :]

        println("Running NLTE correction for $(star), element $(el), line $(el_lines[1])")
        el_line_xfe = star_xfes[1]
        el_nlte_corr, el_xfe_nlte, el_nlte_note = interpolate_nlte_correction(el, el_line_df, star_teff, star_logg, star_feh, star_vmic, el_line_xfe)
        
        star_mean_xfe_lte = np.nanmean(star_xfes)
        star_all_xfe_nlte = [el_xfe_nlte]
        star_mean_xfe_nlte = np.nanmean(star_all_xfe_nlte)
        star_all_nlte_corr = [el_nlte_corr]
        star_mean_nlte_corr = np.nanmean(star_all_nlte_corr)

    elseif length(el_lines) == 2
        el_line1_df = el_nlte_grid_masked[el_nlte_grid_masked."Line" .== el_lines[1], :]
        el_line2_df = el_nlte_grid_masked[el_nlte_grid_masked."Line" .== el_lines[2], :]

        println("Running NLTE correction for $(star), element $(el), line $(el_lines[1])")
        el_line1_xfe = star_xfes[1]
        el_line1_nlte_corr, el_line1_xfe_nlte, el_line1_nlte_note = interpolate_nlte_correction(el, el_line1_df, star_teff, star_logg, star_feh, star_vmic, el_line1_xfe)
        println("Running NLTE correction for $(star), element $(el), line $(el_lines[2])")
        el_line2_xfe = star_xfes[2]
        el_line2_nlte_corr, el_line2_xfe_nlte, el_line2_nlte_note = interpolate_nlte_correction(el, el_line2_df, star_teff, star_logg, star_feh, star_vmic, el_line2_xfe)

        star_mean_xfe_lte = np.nanmean(star_xfes)
        star_all_xfe_nlte = [el_line1_xfe_nlte, el_line2_xfe_nlte]
        star_mean_xfe_nlte = np.nanmean(star_all_xfe_nlte)
        star_all_nlte_corr = [el_line1_nlte_corr, el_line2_nlte_corr]
        star_mean_nlte_corr = np.nanmean(star_all_nlte_corr)
        el_nlte_note = vcat(el_line1_nlte_note, el_line2_nlte_note)

    elseif length(el_lines) == 4
        el_line1_df = el_nlte_grid_masked[el_nlte_grid_masked."Line" .== el_lines[1], :]
        el_line2_df = el_nlte_grid_masked[el_nlte_grid_masked."Line" .== el_lines[2], :]
        el_line3_df = el_nlte_grid_masked[el_nlte_grid_masked."Line" .== el_lines[3], :]
        el_line4_df = el_nlte_grid_masked[el_nlte_grid_masked."Line" .== el_lines[4], :]

        println("Running NLTE correction for $(star), element $(el), line $(el_lines[1])")
        el_line1_xfe = star_xfes[1]
        el_line1_nlte_corr, el_line1_xfe_nlte, el_line1_nlte_note = interpolate_nlte_correction(el, el_line1_df, star_teff, star_logg, star_feh, star_vmic, el_line1_xfe)
        println("Running NLTE correction for $(star), element $(el), line $(el_lines[2])")
        el_line2_xfe = star_xfes[2]
        el_line2_nlte_corr, el_line2_xfe_nlte, el_line2_nlte_note = interpolate_nlte_correction(el, el_line2_df, star_teff, star_logg, star_feh, star_vmic, el_line2_xfe)
        println("Running NLTE correction for $(star), element $(el), line $(el_lines[3])")
        el_line3_xfe = star_xfes[3]
        el_line3_nlte_corr, el_line3_xfe_nlte, el_line3_nlte_note = interpolate_nlte_correction(el, el_line3_df, star_teff, star_logg, star_feh, star_vmic, el_line3_xfe)
        println("Running NLTE correction for $(star), element $(el), line $(el_lines[4])")
        el_line4_xfe = star_xfes[4]
        el_line4_nlte_corr, el_line4_xfe_nlte, el_line4_nlte_note = interpolate_nlte_correction(el, el_line4_df, star_teff, star_logg, star_feh, star_vmic, el_line4_xfe)

        star_mean_xfe_lte = np.nanmean(star_xfes)
        star_all_xfe_nlte = [el_line1_xfe_nlte, el_line2_xfe_nlte, el_line3_xfe_nlte, el_line4_xfe_nlte]
        star_mean_xfe_nlte = np.nanmean(star_all_xfe_nlte)
        star_all_nlte_corr = [el_line1_nlte_corr, el_line2_nlte_corr, el_line3_nlte_corr, el_line4_nlte_corr]
        star_mean_nlte_corr = np.nanmean(star_all_nlte_corr)
        el_nlte_note = vcat(el_line1_nlte_note, el_line2_nlte_note, el_line3_nlte_note, el_line4_nlte_note)
    end

    # Storing the results in a dictionary
    star_nlte_dict = Dict("star_id" => star, "element" => el, "lines" => el_lines, "xfe_mean_lte" => star_mean_xfe_lte, 
                        "nlte_corr_mean" => star_mean_nlte_corr, "xfe_all_lte" => star_xfes, "nlte_corr_all" => star_all_nlte_corr, 
                        "notes" => el_nlte_note)


    # Converting to DataFrame 
    star_nlte_df = DataFrame([star_nlte_dict])

    return star_nlte_df

end

function run_lind2022_nlte_corrections(stars_df, el, el_lines, grid_path, savepath)
    """Runs the Lind et al. (2022) NLTE correction function for a given element and line(s) across all stars in the input DataFrame,
    and saves the results to a CSV file.

    Parameters
    ----------
    stars_df : DataFrame
        A DataFrame containing the stellar parameters and abundances for all stars. The DataFrame should have the following columns: 
        "star_id" (String), "teff" (Float64), "logg" (Float64), "feh" (Float64), "vmic" (Float64), "abund_xfe" (Vector{Float64}), 
        "line" (Vector{String}).
    el : String
        The element for which to calculate the NLTE correction (e.g., "al", "mg", "na")
    el_lines : Vector{Int}
        The line(s) for which to calculate the NLTE correction (e.g., [3962] for Al, [5167, 5172, 5183, 8806] for Mg, [5889, 5895] for Na)
    grid_path : String
        The file path to the directory containing the Lind et al. (2022) NLTE grid files.
    savepath : String
        The file path to the directory where the output CSV file containing the NLTE corrections will be saved.

    Returns
    -------
    all_nlte_df : DataFrame
        A DataFrame containing the NLTE corrections for all stars, elements, and lines, as well as any notes about the interpolation 
        (e.g., if any stellar parameters were outside the grid limits and were forced to be within the grid for interpolation). 
        The DataFrame will have the following columns: "star_id" (String), "element" (String), "lines" (Vector{Int}), "xfe_mean_lte" (Float64), 
        "nlte_corr_mean" (Float64), "xfe_all_lte" (Vector{Float64}), "nlte_corr_all" (Vector{Float64}), "notes" (Vector{String}).

    """

    # Looping through each star in the DataFrame
    all_nlte_dfs = []
    stars = unique(stars_df.star_id)
    # stars = ["ra_1659-2154_s114"]
    for star in stars
        # Extracting the stellar parameters and abundances for the star from the DataFrame
        star_entry = stars_df[stars_df.star_id .== star, :]
        star_teff = star_entry.teff[1]
        star_logg = star_entry.logg[1]
        star_feh = star_entry.feh[1]
        star_vmic = star_entry.vmic[1]
        star_xfes = star_entry.abund_xfe
        star_lines = star_entry.line

        # For Mg: the first abundance comprises of two lines: 5167 and 5172A, so will duplicate 
        # the Mg abundance for both
        if el == "mg"
            star_xfe_5167 = star_xfes[1]
            star_xfe_5172 = star_xfes[1]
            star_xfe_5183 = star_xfes[2]
            star_xfe_8806 = star_xfes[3]
            star_xfes = [star_xfe_5167, star_xfe_5172, star_xfe_5183, star_xfe_8806]
        end

        # For Na: severals stars do not have Na abundances due to interstellar contamination, so will skip them.
        # Other stars have Na abundances for only 1 of the 2 Na lines, so will only run the NLTE correction for the l
        # ine with abundance measurement.
        if el == "na"
            # Removing stars with interstellar contamination of Na lines
            na_contaminate = ["ra_1604-2712_s292", "ra_1648-0653_s38", "ra_1752-4300_s269", "ra_1752-4300_s155"]
            if star in na_contaminate
                println("Skipping star $(star) for Na NLTE correction due to interstellar contamination of Na lines")
                continue

            # For stars with only 1 Na line abundance measurement, will only run the NLTE correction for that line
            else
                if length(star_xfes) == 1
                    # Identiftying the line with abundance measurement
                    na_line = round(parse(Float64, split(star_lines[1], "Na_")[2]))
                    
                    el_lines = [na_line]
                end
            end
        end



        # Running the NLTE correction function for the star and element
        println("Running star $(star)")
        star_nlte_df = lind2022_nlte_correction(star, el, el_lines, grid_path, star_teff, star_logg, star_feh, star_vmic, star_xfes)
        push!(all_nlte_dfs, star_nlte_df)

    end

    # Concatenating all the individual star DataFrames into one DataFrame
    all_nlte_df = vcat(all_nlte_dfs...)

    # Saving the DataFrame to a CSV file
    save_file = joinpath(savepath, "$(el)_nlte_corrs.csv")
    CSV.write(save_file, all_nlte_df)

    return all_nlte_df
end


grid_path = "/data/misfit/benlowe/xshooter/nlte_corrs/grids/"
savepath = "/data/misfit/benlowe/xshooter/nlte_corrs/"

# Loading in the stellar abundances
xshooter_al_abunds = CSV.read("/data/misfit/benlowe/xshooter/p115/korg_data/all_abunds/Al_I_allabunds_line.csv", DataFrame)
xshooter_mg_abunds = CSV.read("/data/misfit/benlowe/xshooter/p115/korg_data/all_abunds/Mg_I_allabunds_line.csv", DataFrame)
xshooter_na_abunds = CSV.read("/data/misfit/benlowe/xshooter/p115/korg_data/all_abunds/Na_I_allabunds_line.csv", DataFrame)

# Running the NLTE corrections
al_lines = [3962]
al_nlte_df = run_lind2022_nlte_corrections(xshooter_al_abunds, "al", al_lines, grid_path, savepath)

mg_lines = [5167, 5173, 5184, 8807]
mg_nlte_df = run_lind2022_nlte_corrections(xshooter_mg_abunds, "mg", mg_lines, grid_path, savepath)

na_lines = [5890, 5896]
na_nlte_df = run_lind2022_nlte_corrections(xshooter_na_abunds, "na", na_lines, grid_path, savepath)

