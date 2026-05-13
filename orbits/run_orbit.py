import os
import numpy as np
import pandas as pd
from astropy.io import fits
from orbit_code import orbit_calc
from multiprocessing import Pool
from functools import partial
import astropy.units as u
from astropy.table import Table


# ---LOADING IN THE STARS--- #
# Loading in the sample stars
# Must have the following columns in the dataframe:
# star (star name),
# ra (ra in degrees),
# dec (dec in degrees),
# plx (parallax in mas) or dist (distance in kpc),
# plx_err (parallax error in mas) or dist_err (distance error in kpc),
# pmra (ra proper motion in mas/yr),
# pmra_err (ra proper motion error in mas/yr),
# pmdec (dec proper motion in mas/yr),
# pmdec_err (dec proper motion error in mas/yr),
# rv (heliocentric-corrected radial velocity in km/s),
# rv_err (radial velocity error in km/s)

# # ---LITERATURE STARS--- #
# # lit_df = pd.read_csv('/data/misfit/benlowe/xshooter/lit_abund_data/all_lit_combined_data_v2.csv')
# lit_df = pd.read_csv('/data/misfit/benlowe/xshooter/lit_abund_data/all_lit_combined_data_leftover.csv')
# lit_df = lit_df[['star_id', 'ra', 'dec', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'rv', 'rv_err']]
# lit_df.rename(columns={'star_id': 'star', 'pmra_error': 'pmra_err', 'pmdec_error': 'pmdec_err'}, inplace=True)

# # lit_dist_df = pd.read_csv('/data/misfit/benlowe/xshooter/lit_abund_data/distances/all_dist_res.csv')
# lit_dist_df = pd.read_csv('/data/misfit/benlowe/xshooter/lit_abund_data/distances/all_dist_res_leftover.csv')
# lit_dist_df = lit_dist_df[['star_id', 'dist_mean', 'dist_std']]
# lit_dist_df.rename(columns={'star_id': 'star', 'dist_mean': 'dist', 'dist_std': 'dist_err'}, inplace=True)

# comb_df = lit_df.merge(lit_dist_df, on='star', how='left')

# ---X-SHOOTER STARS--- #
xshooter_df = pd.read_csv('/data/misfit/benlowe/xshooter/p115/stellar_params/emp_orb_inputs_ALL.csv')


# stars_ran = [str(star) for star in os.listdir('/data/misfit/benlowe/xshooter/lit_abund_data/orbits/')]
# comb_df = comb_df[~comb_df['star'].isin(stars_ran)]

# Save file path for the orbits
save_type = 'fits'  # Choose either 'csv' or 'fits'
orbit_savepath = '/data/misfit/benlowe/xshooter/p115/orbits_v3/'
orbit_filename = f'xshooter_orbits.{save_type}'

# Calculating the orbits
norbits = 100

# Determining the time steps for the orbit calculation
ts = np.arange(-1, 1, 0.01) * u.Gyr # ±4 Gyr with 10 million time steps


# ---CALCULATING THE ORBITS--- #
# Running with parallel processing    
num_cores = 15
data_split = np.array_split(xshooter_df, num_cores)  # Split DataFrame into equal parts
run_partial = partial(orbit_calc.calc_orbit_mcmillian, 
                      savepath=orbit_savepath, 
                      ts=ts, 
                      nrand=norbits, 
                      use_plx=False, 
                      save_type=save_type,
                      plot=False)

with Pool(processes=num_cores) as pool:  
    orb_df = pool.map(run_partial, data_split)

# Combining
final_info_df = pd.concat(orb_df, ignore_index=True)


# ---SAVING THE FINAL RESULTS--- #
# Saving the final results into a CSV or FITS file
print("Saving the final results of all orbits")
if save_type == 'csv':
    final_info_df.to_csv(orbit_savepath + orbit_filename, index=False) 
if save_type == 'fits':
    final_info_table = Table.from_pandas(final_info_df)
    final_info_table.write(orbit_savepath + orbit_filename.replace('.csv', '.fits'), overwrite=True)
else:
    print("Invalid save_type. Please choose either 'csv' or 'fits'. Will save as .csv file")
    final_info_df.to_csv(orbit_savepath + orbit_filename, index=False) 

print("Done!")


# ---GALAH ORBITS--- #
# galah_df = pd.read_csv('/data/misfit/benlowe/4most/stars/galah_dr4_data_cleaned.csv')
# galah_df['r_med_err'] = galah_df['r_hi'] - galah_df['r_med']
# galah_df['r_med'] = galah_df['r_med']/1000 # Convert to kpc
# galah_df['r_med_err'] = galah_df['r_med_err']/1000 # Convert to kpc
# galah_df = galah_df[['sobject_id', 'ra', 'dec', 'r_med', 'r_med_err',
#                  'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
#                  'rv_comp_1', 'e_rv_comp_1']]
# galah_df = galah_df.head(100)

# # Renaming columns to match the expected format
# galah_df.columns = ['star', 'ra', 'dec', 'dist', 'dist_err',
#                   'pmra', 'pmra_err', 'pmdec', 'pmdec_err',
#                   'rv', 'rv_err']

# # Save file path for the orbits
# orbit_savepath = '/data/misfit/benlowe/4most/orbits/galah/time_step_tests/1myr/'
# orbit_filename = 'galah_1myr_orbits.csv'
# # orbit_savepath = '/data/misfit/benlowe/4most/orbits/galah/time_step_tests/5myr/'
# # orbit_filename = 'galah_5myr_orbits.csv'
# # orbit_savepath = '/data/misfit/benlowe/4most/orbits/galah/time_step_tests/10myr/'
# # orbit_filename = 'galah_10myr_orbits.csv'
# # orbit_savepath = '/data/misfit/benlowe/4most/orbits/galah/time_step_tests/50myr/'
# # orbit_filename = 'galah_50myr_orbits.csv'
# # orbit_savepath = '/data/misfit/benlowe/4most/orbits/galah/time_step_tests/100myr/'
# # orbit_filename = 'galah_100myr_orbits.csv'
# # orbit_savepath = '/data/misfit/benlowe/4most/orbits/galah/time_step_tests/500myr/'
# # orbit_filename = 'galah_500myr_orbits.csv'

# # Calculating the orbits
# norbits = 50

# # Determining the time steps for the orbit calculation
# ts = np.arange(-4, 4, 0.001) * u.Gyr # ±4 Gyr with 1 million time steps
# # ts = np.arange(-4, 4, 0.005) * u.Gyr # ±4 Gyr with 5 million time steps
# # ts = np.arange(-4, 4, 0.01) * u.Gyr # ±4 Gyr with 10 million time steps
# # ts = np.arange(-4, 4, 0.05) * u.Gyr # ±4 Gyr with 50 million time steps
# # ts = np.arange(-4, 4, 0.1) * u.Gyr # ±4 Gyr with 100 million time steps
# # ts = np.arange(-4, 4, 0.5) * u.Gyr # ±4 Gyr with 500 million time steps