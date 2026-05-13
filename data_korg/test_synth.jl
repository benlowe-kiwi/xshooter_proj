using Korg
using CSV
using DataFrames
using PyPlot
using PyCall
using Distributed
fits = pyimport("astropy.io.fits")
plt = pyimport("matplotlib.pyplot")

# Loading in linelist
function linelist_load(path)
    """
    Taking in a given linelist and converting it into a suitable format for Korg. All
    wavelengths will be converted from air to vacuum.

    The linelist should have the following columns at a minimum:
    1. species
    2. wavelengths (in A and in air)
    3. elow (in eV)
    4. loggf 

    Parameters
    ----------
    path : Str
        Path to the linelist.
            
    Returns
    -------
    linelist_korg : Vector
        The linelist in a suitable format for Korg.
    """

    # Loading in the linelist, then sorting by wavelength
    linelist_df = CSV.read(path, DataFrame)
    linelist_df = sort(linelist_df, :wave)

    # Getting the linelist and windows into suitable format for Korg
    linelist_korg = Korg.Line.(Korg.air_to_vacuum.(linelist_df.wave),
                               linelist_df.loggf,
                               Korg.Species.(linelist_df.species),
                               linelist_df.elow)

    return linelist_korg
end

# linelist_path = "/data/misfit/benlowe/xshooter/linelist/lines.csv"
# linelist_korg = linelist_load(linelist_path)
linelist_korg = Korg.get_VALD_solar_linelist()

# Interpolating MARCS model atmosphere
teff = 5350
logg = 1.5
feh = -3.0 
R = 8900

name = "/data/misfit/benlowe/xshooter/p115/korg_data/test/stars_with_odd_stel_params/halpha_look/change_logg/t5350_logg150_feh-300"

# Elemental abundances
A_x = Korg.format_A_X(feh) # Dict("Sc" => cfe + feh, "Fe" => feh)

# waves = []
# fluxes = []
# teffs = []
# for t in teff
#     atm = Korg.interpolate_marcs(t, logg, A_x)

#     # Synthesizing spectrum
#     @time synth_spec = Korg.synthesize(atm, linelist_korg, A_x, Korg.Wavelengths(4240, 4253, 0.74, air_wavelengths=true))
#     convolved_flux = Korg.apply_LSF(synth_spec.flux, synth_spec.wavelengths, R)
#     norm_flux = convolved_flux ./ synth_spec.cntm

#     push!(fluxes, norm_flux)
#     push!(teffs, t)
#     push!(waves, synth_spec.wavelengths)

#     # # Saving the data
#     # spec_data = DataFrame(wave=synth_spec.wavelengths,
#     #                     flux=convolved_flux ./ synth_spec.cntm)
#     # CSV.write("/data/misfit/benlowe/1p2p_gcs/1P/rgb_base_teff$(teff)_logg$(split(string(logg), ".")[1])$(split(string(logg), ".")[2])_cfe$(split(string(cfe), ".")[1])$(split(string(cfe), ".")[2]).csv", spec_data)

# end

# # Plotting the spectrum
# plt.figure(figsize=(10, 5))

# for (i, t) in enumerate(teffs)

#     plt.plot(waves[i], fluxes[i], label=string(t))
#     plt.xlabel("Wavelength (Å)")
#     plt.ylabel("Normalized flux")
#     plt.tight_layout()
#     plt.savefig("/data/misfit/benlowe/xshooter/korg_data/tests/sc_abunds/ra_1752-4300_s214/all_teff.png", dpi=300, transparent=false, facecolor="white")
#     plt.close()
# end

atm = Korg.interpolate_marcs(teff, logg, A_x)

# Synthesizing spectrum
@time synth_spec = Korg.synthesize(atm, linelist_korg, A_x, Korg.Wavelengths(6555, 6575, air_wavelengths=true))
convolved_flux = Korg.apply_LSF(synth_spec.flux, synth_spec.wavelengths, R)
norm_flux = convolved_flux ./ synth_spec.cntm

# Plotting the spectrum
plt.figure(figsize=(10, 5))
plt.plot(synth_spec.wavelengths, norm_flux)
plt.xlabel("Wavelength (Å)")
plt.ylabel("Normalized flux")
plt.tight_layout()
plt.savefig("$(name).png", dpi=300, transparent=false, facecolor="white")
plt.close()

# Saving the data
spec_data = DataFrame(wave=synth_spec.wavelengths,
                      flux=norm_flux)
CSV.write("$(name).csv", spec_data)

# A_x = Korg.format_A_X(Dict("C" => cfe + feh, "Fe" => feh))
# @time synth_spec = synth(teff, logg, feh, A_x, linelist_korg, (4260, 4360), true, 1300)

