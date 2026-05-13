using Korg
using CSV
using DataFrames
using PyPlot
using Distributed

addprocs(7)


# ---INPUTS--- #
# List of stars
# star_paths = sort(filter(isdir, readdir("/data/misfit/benlowe/xshooter/p113/reduced_data/comb_spec_2", join=true)))
star_paths = sort(filter(isdir, readdir("/data/misfit/benlowe/xshooter/p115/reduced_data/comb_spec", join=true)))

# star_paths = [star for star in star_paths if split(star, "/")[end] in ["ra_0103-7050_s236", "ra_0752-5047_s47", "ra_1648-2642_s91",
#                                                                         "ra_1659-2154_s261", "ra_1659-2154_s347", "ra_1752-4300_s155"]]
# star_paths = [star for star in star_paths if split(star, "/")[end] == "ra_1633-2814_s130"] 
# star_paths = [star for star in star_paths if split(star, "/")[end] in ["ra_1752-4300_s214"]]
# star_paths = [star for star in star_paths if split(star, "/")[end] in ["ra_0103-7050_s163", "ra_1633-2814_s284", "ra_1658-2454_s22"]]

# Loading in the linelist
# vald_linelist = Korg.read_linelist("/data/misfit/benlowe/xshooter/p113/linelist/vald_3300-9000_linelist_metal-poor.lin", format="vald", isotopic_abundances=Korg.isotopic_abundances)
vald_linelist = Korg.read_linelist("/data/misfit/benlowe/xshooter/p115/linelist/vald_3300-9000_linelist_metal-poor.lin", format="vald", isotopic_abundances=Korg.isotopic_abundances)

# el_linelist_path = "/data/misfit/benlowe/xshooter/p113/linelist/abund_lines/v3/"
el_linelist_path = "/data/misfit/benlowe/xshooter/p115/linelist/abund_lines/v3/"


# Loading in the stellar parameters
# emp_params = CSV.read("/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter9/stellar_params.csv", DataFrame)
# emp_abunds_path = "/data/misfit/benlowe/xshooter/p113/korg_data/abunds/final_iter9/"
emp_params = CSV.read("/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final_iter9/stellar_params.csv", DataFrame)
emp_abunds_path = "/data/misfit/benlowe/xshooter/p115/korg_data/abunds_v2/final_iter9/"

# Loading in the non-detections list
# non_detections = CSV.read("/data/misfit/benlowe/xshooter/p113/korg_data/abund_detections.csv", DataFrame)
non_detections = CSV.read("/data/misfit/benlowe/xshooter/p115/korg_data/abund_detections.csv", DataFrame)

# Loading in the p-value chisquare determination regions for each element
# pvalue_regions = CSV.read("/data/misfit/benlowe/xshooter/p113/linelist/pval_reg.csv", DataFrame)
pvalue_regions = CSV.read("/data/misfit/benlowe/xshooter/p115/linelist/pval_reg.csv", DataFrame)


# Save path
# savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/tests/sc_abunds/ra_1752-4300_s214"
# savepath = "/data/misfit/benlowe/xshooter/p113/korg_data/upper_limits_v6/"
savepath = "/data/misfit/benlowe/xshooter/p115/korg_data/upper_limits_v2/"


# ---RUNNING KORG--- #
# @everywhere include("korg_code/upper_limit.jl")
# pmap(path -> run_upperlimit(path, vald_linelist, el_linelist_path, emp_params, emp_abunds_path, non_detections, pvalue_regions, savepath), star_paths)

@everywhere include("korg_code/upper_limit_iter.jl")
pmap(path -> run_upperlimit(path, vald_linelist, el_linelist_path, emp_params, emp_abunds_path, non_detections, pvalue_regions, savepath), star_paths)


# for path in star_paths
#     run_upperlimit(path, vald_linelist, el_linelist_path, emp_params, emp_abunds_path, non_detections, pvalue_regions, savepath)
# end