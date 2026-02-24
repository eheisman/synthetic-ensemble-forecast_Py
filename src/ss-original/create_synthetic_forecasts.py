import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import gc

import importnb
#import syn_gen function
with importnb.imports("ipynb"):
    from syn_gen import syn_gen 


processed_data_dir = Path('../Data/processed_daily_data')
out_dir = Path('../Data/simulated_data')

keysite_label = "WSDC1"

num_samples = 20

# --------------------- Read in key inputs ----------------------------
ixx_hefs = np.load(processed_data_dir / "ixx_hefs.npy",allow_pickle=True)               # the initialization dates for HEFS
ixx_obs = np.load(processed_data_dir / "ixx_obs.npy",allow_pickle=True)                 # the dates for the obs
ixx_obs_forward = np.load(processed_data_dir / "ixx_obs_forward.npy",allow_pickle=True) # the dates for forward looking obs
hefs_forward = np.load(processed_data_dir / "hefs_forward.npy",allow_pickle=True)       # the forward-looking HEFS forecasts for all sites
obs_forward = np.load(processed_data_dir / "obs_forward.npy",allow_pickle=True)         # the forward-looking obs flows for all sites
obs_flows = pd.read_csv(processed_data_dir / "observed_flows.csv")    # matrix of observed flows

#####--- Set Parameters for Model ------######
cur_seed = 1
kk      = 20        # 20
knn_pwr = -0.2         # 0
scale_pwr = 0.397   # 0.381
hi      = 10        # 10
lo      = 1.4       # 1.4 
sig_a   = 2.79      # 2.79
sig_b   = -2.83     # -2.83

gen_start = ixx_obs_forward[0]
gen_end = ixx_obs_forward[-1]  #to make sure we have days with obs_forward
fit_start = ixx_hefs[0]
fit_end = ixx_hefs[-1]

ixx_gen = pd.date_range(gen_start, gen_end, freq="D", tz="UTC").to_numpy(dtype="datetime64[ns]")
ixx_gen= pd.to_datetime(ixx_gen,utc=True)
ixx_gen = ixx_gen.normalize()

#convert to datetime index format
ixx_obs = pd.to_datetime(ixx_obs)
ixx_obs_forward = pd.to_datetime(ixx_obs_forward)
ixx_hefs = pd.to_datetime(ixx_hefs)
ixx_gen = pd.to_datetime(ixx_gen)

#first we empty out_dir
for filename in os.listdir(out_dir):
    file_path = os.path.join(out_dir, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)


for i in range(0,num_samples):

    out = syn_gen(
        seed=i,                          # random seed
        kk=kk,                                  # number of options for resampling of hefs
        keysite_label=keysite_label,            # label of the keysite to use
        knn_pwr=knn_pwr,                                # 
        scale_pwr=scale_pwr,                              #
        hi=hi,                                     #
        lo=lo,                                     #
        sig_a=sig_a,                                  #
        sig_b=sig_b,                                  #
        fit_start=fit_start,                              # e.g. "1980-10-01" or pd.Timestamp
        fit_end=fit_end,                                #
        gen_start=gen_start,                              #
        gen_end=gen_end,                                #
        obs_flows=obs_flows,                              #
        obs_forward = obs_forward,
        hefs_forward=hefs_forward,                           # shape: (n_sites, n_ens, n_hefs_time, leads)
        ixx_hefs=ixx_hefs,                               # 1D datetime-like, len = n_hefs_time
        ixx_obs=ixx_obs,                                 # 1D datetime-like, len = n_obs_forward
        ixx_obs_forward = ixx_obs_forward
    )

    syn_forecast = out[0]
    resampled_dates = out[1]
    hefs_scaling_factor = out[2]

    file_path = out_dir / f"syn_forecast_{i}.npz"
    #np.save(file_path,syn_forecast)
    np.savez_compressed(file_path, syn_forecast)

    file_path = out_dir / f"resampled_dates_{i}.npz"
    #np.save(file_path, resampled_dates)
    np.savez_compressed(file_path, resampled_dates)

    file_path = out_dir / f"hefs_scaling_factor_{i}.npz"
    #np.save(file_path, hefs_scaling_factor)
    np.savez_compressed(file_path, hefs_scaling_factor)

#also save the dates for the synthetic forecast generation, in the same way ixx_hefs and ixx_obs were created
np.save(out_dir / "ixx_gen.npy", ixx_gen)                         # the initialization dates for synthetic forecasts
