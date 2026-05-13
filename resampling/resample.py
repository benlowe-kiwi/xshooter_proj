import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import os
from multiprocessing import Pool
from functools import partial
from resample_code import run_resample

# Path to directory containing the data and chisquares
element = "scmg"
data_dir = f"/data/misfit/benlowe/xshooter/p115/chisqrs_resample/{element}_mcmillan17/"
chisqr_dir = "/data/misfit/benlowe/xshooter/p115/results/abund_comparision/delta_chisqr_mcmillan17/"

# Number of iterations for resampling
n_iterations = 1000000

# Running parallel processing
num_cores = 40
iters_split = np.array_split(range(n_iterations), num_cores)  # Split iterations into equal parts

run_partial = partial(run_resample.run_resample,
                      data_dir = data_dir,
                      chisqr_dir = chisqr_dir,
                      element = element)

with Pool(num_cores) as pool:
    results = pool.map(run_partial, iters_split)

# Unpacking the results and combining into final DataFrames
prodisk, gse = zip(*results)
final_prodisk_df = pd.concat(prodisk, ignore_index=True)
final_gse_df = pd.concat(gse, ignore_index=True)

# Saving the final results into CSV files
final_prodisk_df.to_csv(data_dir + f"{element}_prodisk_deltachisqrs_resample.csv", index=False)
final_gse_df.to_csv(data_dir + f"{element}_gse_deltachisqrs_resample.csv", index=False)