using Korg
using CSV
using DataFrames
using PyPlot
matplotlib.use("Agg")
using Distributed

# addprocs(7)

# # ---GENERATING SPECTRA USING KORG'S SYNTHETIC SPECTRA CODE--- #
# # ---INPUTS--- #
# # List of stars
# star_paths = sort(filter(isdir, readdir("/data/misfit/benlowe/xshooter/p113/reduced_data/comb_spec", join=true)))
# star_paths = [star for star in star_paths if split(star, "/")[end] == "ra_1853-3255_s45"] 
# # star_paths = [star for star in star_paths if split(star, "/")[end] in ["ra_0103-7050_s163", "ra_1633-2814_s284"]]
# # star_paths = [star for star in star_paths if split(star, "/")[end]  in ["ra_0103-7050_s163", "ra_1633-2814_s284", "ra_1658-2454_s22"]]

# # Loading in the linelist
# vald_linelist = Korg.read_linelist("/data/misfit/benlowe/xshooter/p113/linelist/vald_3300-9000_linelist_metal-poor.lin", format="vald", isotopic_abundances=Korg.isotopic_abundances)
# el_linelist_path = "/data/misfit/benlowe/xshooter/p113/linelist/abund_lines/v3"

# windows_path = "/data/misfit/benlowe/xshooter/p113/linelist/abund_lines/v3/"

# # Loading in the stellar parameters
# emp_params = CSV.read("/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final/stellar_params.csv", DataFrame)
# emp_abunds_path = "/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final/"

# # Loading in the abundances
# emp_abunds = CSV.read("/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final/stellar_abunds.csv", DataFrame)

# # Loading in the non-detections list
# non_detections = CSV.read("/data/misfit/benlowe/xshooter/p113/korg_data/abund_detections.csv", DataFrame)

# # Save path
# # savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/synths/"
# savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/upper_limits/"


# # ---RUNNING UPPER-LIMIT CODE--- #
#  include("korg_code/korg_synth.jl")
# pmap(path -> run_synth(path, windows_path, vald_linelist, emp_params, emp_abunds_path, non_detections, savepath), star_paths)



# ---GENERATING SPECTRA USING KORG'S SPECTRAL FITTING CODE--- #
# ---INPUTS--- #
# List of stars
# star_paths = sort(filter(isdir, readdir("/data/misfit/benlowe/xshooter/p113/reduced_data/comb_spec_2", join=true)))
star_paths = sort(filter(isdir, readdir("/data/misfit/benlowe/xshooter/p115/reduced_data/comb_spec", join=true)))

# star_paths = [star for star in star_paths if split(star, "/")[end] in ["ra_0103-7050_s236", "ra_0752-5047_s47", "ra_1648-2642_s91",
#                                                                         "ra_1659-2154_s261", "ra_1659-2154_s347", "ra_1752-4300_s155"]]
# star_paths = [star for star in star_paths if split(star, "/")[end] == "ra_1639-2632_s419"] 
# star_paths = [star for star in star_paths if split(star, "/")[end] in ["ra_0103-7050_s236"]]
# star_paths = [star for star in star_paths if split(star, "/")[end]  in ["ra_0103-7050_s163", "ra_1633-2814_s284", "ra_1658-2454_s22"]]


# Loading in the linelist
# vald_linelist = Korg.read_linelist("/data/misfit/benlowe/xshooter/p113/linelist/vald_3300-9000_linelist_metal-poor.lin", format="vald", isotopic_abundances=Korg.isotopic_abundances)
# el_linelist_path = "/data/misfit/benlowe/xshooter/p113/linelist/abund_lines/v3/"
vald_linelist = Korg.read_linelist("/data/misfit/benlowe/xshooter/p115/linelist/vald_3300-9000_linelist_metal-poor.lin", format="vald", isotopic_abundances=Korg.isotopic_abundances)
el_linelist_path = "/data/misfit/benlowe/xshooter/p115/linelist/abund_lines/v3/"

# windows_path = "/data/misfit/benlowe/xshooter/p113/linelist/abund_lines/v3/"
windows_path = "/data/misfit/benlowe/xshooter/p115/linelist/abund_lines/v3/"

# Loading in the stellar parameters
# emp_params = CSV.read("/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter9/stellar_params.csv", DataFrame)
emp_params = CSV.read("/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final_iter9/stellar_params.csv", DataFrame)

# Loading in the abundances
# emp_abunds_path = "/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter9/"
# emp_params = CSV.read("/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter9/stellar_params.csv", DataFrame)
# cfe_corrections = CSV.read("/data/misfit/benlowe/xshooter/p113/C_evolve_corrections/cfe_corrections.csv", DataFrame)
emp_abunds_path = "/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final_iter9/"
emp_params = CSV.read("/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final_iter9/stellar_params.csv", DataFrame)
cfe_corrections = CSV.read("/data/misfit/benlowe/xshooter/p115/C_evolve_corrections/cfe_corrections.csv", DataFrame)


# Masking regions
mask_regions = false

# Loading in the non-detections list
# non_detections = CSV.read("/data/misfit/benlowe/xshooter/p113/korg_data/abund_detections.csv", DataFrame)
non_detections = CSV.read("/data/misfit/benlowe/xshooter/p115/korg_data/abund_detections.csv", DataFrame)

# Elements to generate synthetic spectra for
els = ["C_I", "N_I"]
el_xfe_min = -5.0
el_xfe_max = 2.0
el_xfe_step = 0.01
# els = ["C_I"]
# el_xfe_min = 0
# el_xfe_max = 0
# el_xfe_step = 0.01


# ---RUNNING SYNTHETIC FITTING CODE--- #
# addprocs(0)
@everywhere include("korg_code/korg_synth.jl")
# savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/synths_v3/"
savepath = "/data/misfit/benlowe/xshooter/p115/korg_data/synths_v2/"
# pmap(path -> run_synth_fitspec(path, windows_path, vald_linelist, emp_params, emp_abunds_path, non_detections, savepath, els, el_xfe_min, el_xfe_max, cfe_corrections), star_paths)
for path in star_paths
    run_synth_fitspec(path, windows_path, vald_linelist, emp_params, emp_abunds_path, non_detections, savepath, els, el_xfe_min, el_xfe_max, cfe_corrections)
end



# # ---RUNNING SYNTHETIC GENERATION CODE--- #
# # @everywhere include("korg_code/korg_synth2_loggtest.jl")
# @everywhere include("korg_code/korg_synth2.jl")

# #  savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/upper_limits_v6/"
# savepath = "/data/misfit/benlowe/xshooter/p115/korg_data/upper_limits_v2/"
# # savepath = "/data/misfit/benlowe/xshooter/p115/korg_data/test/fix_ch_synthspec/"
# pmap(path -> run_synth(path, windows_path, vald_linelist, emp_params, emp_abunds_path, non_detections, savepath, els, el_xfe_min, el_xfe_max, el_xfe_step, cfe_corrections, mask_regions=mask_regions), star_paths)

# # for path in star_paths
# #     run_synth(path, windows_path, vald_linelist, emp_params, emp_abunds_path, non_detections, savepath, els, el_xfe_min, el_xfe_max, cfe_corrections, mask_regions=mask_regions)
# # end