import numpy as np
import pandas as pd
from chisqr_code import calc_chisqr
from multiprocessing import Pool
from functools import partial

# Loading in the masks DataFrame
masks_df = pd.read_csv('/data/misfit/benlowe/xshooter/korg_data/nh_ch_masks.csv')

# Element to run chi-square on
el = 'N_I' # 'C_I'

# Path to the synthetic spectra
synth_path = '/data/misfit/benlowe/xshooter/korg_data/upper_limits_iter/'

# Save path for the chi-square results
savepath = "/data/misfit/benlowe/xshooter/korg_data/upper_limits_v4/"


# ---RUNNING CHI-SQUARE CALCULATION ROUTINE--- #
# Running with parallel processing    
# num_cores = 16
# data_split = np.array_split(masks_df, num_cores)  
# run_partial = partial(calc_chisqr.get_chisqr, 
#                       el=el,
#                       synth_path=synth_path, 
#                       savepath=savepath)

# with Pool(processes=num_cores) as pool:  
#     pool.map(run_partial, data_split)
    
    
# ---CALCULATING DELTA CHI-SQUARE--- #
# Getting list of paths to each star's chisqrs.csv file
paths = [f"{savepath}{star}" for star in masks_df['star_id'].values]

calc_chisqr.delta_chisqr(paths, el)
    
    

