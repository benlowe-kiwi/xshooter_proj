using Korg
using CSV
using DataFrames
using PyPlot
matplotlib.use("Agg")
using Distributed

addprocs(4)

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

function get_linelist(path)
    """
    Retrieves the line list from the given path, then converts to format accepted by Korg.

    Parameters
    ----------
    path : Str
        Path to the linelist.

    Returns
    -------
    linelist : Vector
        The linelist in a suitable format for Korg.
    """

    # Loading in the linelist from the given path
    linetable = CSV.read(path, DataFrame)
    linetable = sort(linetable, :wave)

    # Extracting
    linelist = Korg.Line.(Korg.air_to_vacuum.(linetable.wave),
                          linetable.loggf,
                          Korg.Species.(linetable.species),
                          linetable.elow) 

    return linelist 
end

function get_star_windows(star_path_all, window_path, el)
    """
    Retrieves the windows for a lsit of stars. Used if you have windows for each star.
    This is not the default, but can be used if you have a different set of windows for each star.

    Parameters
    ----------
    star_path_all : Str
        Path containing all the individual star paths.
    window_path : Dict
        Path to the windows.
=
    Returns
    -------
    star_windows : Dict
        Dictionary with the windows for the star.
    """

    star_windows_all = []
    for star in star_paths

        # Getting the individual windows for each star
        star_id = split(star, "/")[end]

        # For star ra_0752-5047_s47, the windows file is "ra_0752-5047_s47_windows_Al_I.csv" for Al
        if el == "Al I"
            if star_id == "ra_0752-5047_s47"
                star_window_path = windows_path * "$(split(el, " ")[1])_$(split(el, " ")[2])/$(star_id)_windows_$(split(el, " ")[1])_$(split(el, " ")[2]).csv"
                star_windows = get_window(star_window_path)
                push!(star_windows_all, star_windows)
            else
                star_window_path = windows_path * "$(split(el, " ")[1])_$(split(el, " ")[2])/windows_$(split(el, " ")[1])_$(split(el, " ")[2]).csv"
                star_windows = get_window(star_window_path)
                push!(star_windows_all, star_windows)
            end

        # For star ra_1639-2632_s419, the windows file is "ra_1639-2632_s419_windows_Mg_I.csv" for Mg
        elseif el == "Mg I"
            if star_id == "ra_1639-2632_s419"
                star_window_path = windows_path * "$(split(el, " ")[1])_$(split(el, " ")[2])/$(star_id)_windows_$(split(el, " ")[1])_$(split(el, " ")[2]).csv"
                star_windows = get_window(star_window_path)
                push!(star_windows_all, star_windows)
            else
                star_window_path = windows_path * "$(split(el, " ")[1])_$(split(el, " ")[2])/windows_$(split(el, " ")[1])_$(split(el, " ")[2]).csv"
                star_windows = get_window(star_window_path)
                push!(star_windows_all, star_windows)
            end

        elseif el == "Sc II"
            if star_id in ["ra_0103-7050_s236", "ra_1752-4300_s155", "ra_1752-4300_s269"]
                star_window_path = windows_path * "$(split(el, " ")[1])_$(split(el, " ")[2])/$(star_id)_windows_$(split(el, " ")[1])_$(split(el, " ")[2]).csv"
                star_windows = get_window(star_window_path)
                push!(star_windows_all, star_windows)
                println(star_window_path)
                println(star_windows)
            else
                star_window_path = windows_path * "$(split(el, " ")[1])_$(split(el, " ")[2])/windows_$(split(el, " ")[1])_$(split(el, " ")[2]).csv"
                star_windows = get_window(star_window_path)
                push!(star_windows_all, star_windows)
                println("no :(")
            end
        
        # For all stars with element Na I, extract the windows individually 
        elseif el == "Na I"
            star_window_path = windows_path * "$(split(el, " ")[1])_$(split(el, " ")[2])/$(star_id)_windows_$(split(el, " ")[1])_$(split(el, " ")[2]).csv"
            star_windows = get_window(star_window_path)
            push!(star_windows_all, star_windows)
        end
    end

    return star_windows_all
end

# List of stars
# star_paths = sort(filter(isdir, readdir("/data/misfit/benlowe/xshooter/p113/reduced_data/comb_spec_2", join=true)))
star_paths = sort(filter(isdir, readdir("/data/misfit/benlowe/xshooter/p115/reduced_data/comb_spec", join=true)))

# star_paths = [star for star in star_paths if split(star, "/")[end] == "ra_1752-4300_s269"] #  "ra_1752-4300_s155"
# star_paths = [star for star in star_paths if split(star, "/")[end] in ["ra_1639-2632_s419", "ra_1659-2154_s261"]]
# star_paths = [star for star in star_paths if split(star, "/")[end] in ["ra_0103-7050_s236", "ra_0752-5047_s47", "ra_1648-2642_s91",
#                                                                         "ra_1659-2154_s261", "ra_1659-2154_s347", "ra_1752-4300_s155"]]


# Loading in the linelist
# vald_linelist = Korg.read_linelist("/data/misfit/benlowe/xshooter/p113/linelist/vald_3300-9000_linelist_metal-poor.lin", format="vald", isotopic_abundances=Korg.isotopic_abundances)
vald_linelist = Korg.read_linelist("/data/misfit/benlowe/xshooter/p115/linelist/vald_3300-9000_linelist_metal-poor.lin", format="vald", isotopic_abundances=Korg.isotopic_abundances)


# # ---GETTING [Fe/H]--- #
# # Loading in the VALD linelist and windows
# # feh_linelist_path = "/data/misfit/benlowe/xshooter/p113/linelist/caffau_lines/Fe_I/"
# feh_linelist_path = "/data/misfit/benlowe/xshooter/p115/linelist/caffau_lines/Fe_I/"

# feh_windows = get_window(feh_linelist_path * "caffauFeI_windowsALL.csv")
# mask_lines = false # Used to mask lines in the windows. Need separate file for this.


# # # Running Korg - iteration 1 (straight run)
# # @everywhere include("korg_code/korg_feh.jl")

# # # Stellar parameters
# # # emp_params = CSV.read("/data/misfit/benlowe/xshooter/p113/stellar_params/emp_params_FINAL.csv", DataFrame)
# # emp_params = CSV.read("/data/misfit/benlowe/xshooter/p115/stellar_params/emp_params_FINAL.csv", DataFrame)

# # # savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final/"
# # savepath = "/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final/"


# # results = pmap(path -> run_feh(path, feh_linelist_path, vald_linelist, emp_params, feh_windows, ["Fe"], savepath, mask_regions=mask_lines), star_paths)
# # final_df = vcat(results...)
# # CSV.write(savepath * "stellar_params.csv", final_df)


# # Running Korg - iteration 2 (using pre-derived chemical abundances as fixed parameters)
# # Loading in the pre-derived abundances
# # savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter9/"
# # savepath = "/data/misfit/benlowe/xshooter/p113/p113/korg_data/tests/updated_rv_tests/"
# # savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/tests/alpha_test/"
# savepath = "/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final_iter9/"

# # emp_abunds_path = "/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter8/"
# emp_abunds_path = "/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final_iter8/"

# emp_params = CSV.read(emp_abunds_path * "stellar_params.csv", DataFrame)
# abund_detects = CSV.read("/data/misfit/benlowe/xshooter/p115/korg_data/abund_detections.csv", DataFrame) # "/data/misfit/benlowe/xshooter/p113/korg_data/abund_detections.csv"
# cfe_corrections = CSV.read("/data/misfit/benlowe/xshooter/p115/C_evolve_corrections/cfe_corrections.csv", DataFrame) # "/data/misfit/benlowe/xshooter/p113/C_evolve_corrections/cfe_corrections.csv"

# @everywhere include("korg_code/korg_feh_iterate.jl")
# results = pmap(path -> run_feh(path, feh_linelist_path, vald_linelist, emp_params, feh_windows, ["Fe"], savepath, emp_abunds_path, abund_detects, cfe_corrections, mask_regions=mask_lines), star_paths)
# final_df = vcat(results...)
# CSV.write(savepath * "stellar_params.csv", final_df)


# ---GETTING ABUNDANCES--- #
# Elements for abundances
# elements = ["Mg I", "Al I", "Si I", "C I", "Ca II", "Sc II", "Ti II", "N I", "Cr I", "Mn I", "Co I",
#             "Na I", "Ni I", "Sr II", "Ba II", "Eu II"] 
elements = ["Sc II"] 
# elements = ["Al I", "Si I", "C I", "Ca II", "Sc II", "Ti II", "N I", "Cr I", "Mn I", "Co I",
#             "Na I", "Ni I", "Sr II", "Ba II", "Eu II"] 
# elements = ["Co I", "Ni I", "Sr II", "Ba II", "Eu II"] 

# windows_path = "/data/misfit/benlowe/xshooter/p113/linelist/abund_lines/v3/"
windows_path = "/data/misfit/benlowe/xshooter/p115/linelist/abund_lines/v3/"
mask_lines = false # Used to mask lines in the windows. Need separate file for this.

# # Running Korg - iteration 1 (straight run)
# @everywhere include("korg_code/korg_abund.jl")

# # emp_params = CSV.read("/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final/stellar_params.csv", DataFrame)
# emp_params = CSV.read("/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final/stellar_params.csv", DataFrame)
# cfe_corr_path = "/data/misfit/benlowe/xshooter/p115/C_evolve_corrections/cfe_corrections.csv"


# # savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final/"
# savepath = "/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final/"


# # If Na-I, then because of interstellar Na, we need to get the windows for each star
# for el in elements
#     # el_linelist_path = "/data/misfit/benlowe/xshooter/p113/linelist/abund_lines/v3/$(split(el, " ")[1])_$(split(el, " ")[2])/"
#     el_linelist_path = "/data/misfit/benlowe/xshooter/p115/linelist/abund_lines/v3/$(split(el, " ")[1])_$(split(el, " ")[2])/"

#     if el == "Na I"
#         # Getting the windows for each star
#         star_windows_all = get_star_windows(star_paths, windows_path, el)

#         # li_linelist = Korg.read_linelist("/data/misfit/benlowe/xshooter/p113/linelist/Kdwarf5.lin", format="vald", isotopic_abundances=Korg.isotopic_abundances)

#         # Running Korg 
#         pmap(((path, window),) -> run_abund(path, el_linelist_path, vald_linelist, emp_params, window, [el], savepath, cfe_corr_path, mask_regions=mask_lines), zip(star_paths, star_windows_all))

#     elseif el == "Al I"
#         # Getting the windows for each star
#         star_windows_all = get_star_windows(star_paths, windows_path, el)

#         # Running Korg 
#         pmap(((path, window),) -> run_abund(path, el_linelist_path, vald_linelist, emp_params, window, [el], savepath, cfe_corr_path, mask_regions=mask_lines), zip(star_paths, star_windows_all))

#     elseif el == "Mg I"
#         # Getting the windows for each star
#         star_windows_all = get_star_windows(star_paths, windows_path, el)

#         # Running Korg 
#         pmap(((path, window),) -> run_abund(path, el_linelist_path, vald_linelist, emp_params, window, [el], savepath, cfe_corr_path, mask_regions=mask_lines), zip(star_paths, star_windows_all))


#     else
#         # Loading in the VALD linelist and windows
#         el_windows_path = windows_path * "$(split(el, " ")[1])_$(split(el, " ")[2])/windows_$(split(el, " ")[1])_$(split(el, " ")[2]).csv"
#         el_windows = get_window(el_windows_path)

#         # el_linelist_path = "/data/misfit/benlowe/xshooter/p113/linelist/abund_lines/v3/$(split(el, " ")[1])_$(split(el, " ")[2])/"
#         el_linelist_path = "/data/misfit/benlowe/xshooter/p115/linelist/abund_lines/v3/$(split(el, " ")[1])_$(split(el, " ")[2])/"

#         # li_linelist = Korg.read_linelist("/data/misfit/benlowe/xshooter/p113/linelist/Kdwarf5.lin", format="vald", isotopic_abundances=Korg.isotopic_abundances)

#         # Running Korg
#         pmap(path -> run_abund(path, el_linelist_path, vald_linelist, emp_params, el_windows, [el], savepath, cfe_corr_path, mask_regions=mask_lines), star_paths)
#     end
# end


# Running Korg - iteration 2 (using pre-derived chemical abundances as fixed parameters)
@everywhere include("korg_code/korg_abund_iterate.jl")

# savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter9/"
# savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/tests/small_sc_windows/"
# savepath = "/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final_iter9/"
savepath = "/data/misfit/benlowe/xshooter/p115/korg_data/test/small_sc_windows/"

# emp_abunds_path = "/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter9/"
emp_abunds_path = "/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final_iter9/"

# emp_params = CSV.read("/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter9/stellar_params.csv", DataFrame) 
emp_params = CSV.read("/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final_iter9/stellar_params.csv", DataFrame) # "/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter8/stellar_params.csv"


# cfe_corr_path = "/data/misfit/benlowe/xshooter/p113/C_evolve_corrections/cfe_corrections.csv"
cfe_corr_path = "/data/misfit/benlowe/xshooter/p115/C_evolve_corrections/cfe_corrections.csv"


# If Na-I, then because of interstellar Na, we need to get the windows for each star
for el in elements

    # el_linelist_path = "/data/misfit/benlowe/xshooter/p113/linelist/abund_lines/v3/$(split(el, " ")[1])_$(split(el, " ")[2])/"
    el_linelist_path = "/data/misfit/benlowe/xshooter/p115/linelist/abund_lines/v3/$(split(el, " ")[1])_$(split(el, " ")[2])/"

    if el == "Na I"
        # Getting the windows for each star
        star_windows_all = get_star_windows(star_paths, windows_path, el)

        # Running Korg 
        pmap(((path, window),) -> run_abund(path, el_linelist_path, vald_linelist, emp_params, window, [el], savepath, emp_abunds_path, cfe_corr_path, mask_regions=mask_lines), zip(star_paths, star_windows_all))

    elseif el == "Al I"
        # Getting the windows for each star
        star_windows_all = get_star_windows(star_paths, windows_path, el)

        # Running Korg 
        pmap(((path, window),) -> run_abund(path, el_linelist_path, vald_linelist, emp_params, window, [el], savepath, emp_abunds_path, cfe_corr_path, mask_regions=mask_lines), zip(star_paths, star_windows_all))

    elseif el == "Mg I"
        # Getting the windows for each star
        star_windows_all = get_star_windows(star_paths, windows_path, el)

        # Running Korg 
        pmap(((path, window),) -> run_abund(path, el_linelist_path, vald_linelist, emp_params, window, [el], savepath, emp_abunds_path, cfe_corr_path, mask_regions=mask_lines), zip(star_paths, star_windows_all))

    # elseif el == "Sc II"
    #     # Getting the windows for each star
    #     star_windows_all = get_star_windows(star_paths, windows_path, el)

    #     # Running Korg 
    #     pmap(((path, window),) -> run_abund(path, el_linelist_path, vald_linelist, emp_params, window, [el], savepath, emp_abunds_path, cfe_corr_path, mask_regions=mask_lines), zip(star_paths, star_windows_all))

    else
        # Loading in the VALD linelist and windows
        el_windows_path = windows_path * "$(split(el, " ")[1])_$(split(el, " ")[2])/windows_$(split(el, " ")[1])_$(split(el, " ")[2]).csv"
        el_windows = get_window(el_windows_path)

        # Running Korg
        pmap(path -> run_abund(path, el_linelist_path, vald_linelist, emp_params, el_windows, [el], savepath, emp_abunds_path, cfe_corr_path, mask_regions=mask_lines), star_paths)
    end
end



# pmap(path -> run_abund(path, el_linelist, emp_params, el_windows, [elements], savepath), star_paths)


# # Removing all multi-processing workers
# rmprocs(workers())