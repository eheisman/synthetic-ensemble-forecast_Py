import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


in_dir = Path('./raw_data')
out_dir = Path('./processed_daily_data')

# file name for obs hourly CSV
obs_csv_file = "./Hindcasts/russian_river_POR_historical_sim.csv"

# --------------------- Process the (already daily aggregated) CBPR HEFS hindcasts ----------------------------

data = np.load(in_dir  / "./Hindcasts/inp_rr_hefs_cbpr_daily_por.npz")
ixx_hefs = data['dt_hcst_init']     #initialization dates of the HEFS forecasts (daily)

#each element (site) in data has dimension (10957,31,44) 
#   10957 days
#   31 day forecast (horizon of 31 days), but the first day is all nan, so really 30 day forecast
#       note1: the original hourly forecasts have 721 hours, starting at 12pm GMT and ending at 12pm GMT. 
#       note2: (721-1)/24 = 30 --> the daily forecast data is from 1pm GMT to 12pm GMT for each day
#       note3: By comparing the daily data in inp_rr_hefs_cbpr_daily_por.npz to the hourly forecasts in 2019093012_RussianNapa_hefs_gefs.csv, 
#              I confirmed that you skip the first hour when converting to daily. That is, you skip the first 12pm GMT value, 
#              so each day is the average of 1pm GMT to 12pm GMT on the next day
#       note4: IMPORTANT - this implies that 'lead1' forecasts are actually lead0, i.e., they align with
#              the target day being predicted. SO ==> we need to make sure obs_forward is the same     
#   44 ensemble members

date_key = "dt_hcst_init"
all_keys = data.files
site_keys = [k for k in all_keys if k != date_key]

# (Optional) sort site_keys for consistent ordering
site_keys = sorted(site_keys)

# 2. For each site, slice lead times to keep indices 1:16 (i.e., 1..15)
site_arrays = []
for key in site_keys:
    arr = data[key]          # shape: (10957, 31, 44)
    arr = arr[:, 1:16, :]    # keep leads 1–15 (drop 0 and last 15)
    site_arrays.append(arr)

# 3. Stack into a single array
#    shape: (#sites, 10957, 15, 44)
hefs_forward = np.stack(site_arrays, axis=0)

print("hefs_forward shape:", hefs_forward.shape)
print(ixx_hefs[0])

# --------------------- Produce the daily observations for each site ----------------------------

#read in and covert hourly observed flow data to daily observed flow data
#Important: for the hourly observations, each day should be averaged between 1pm GMT and 12pm GMT the next day. See code block above for logic behind this

#GOAL = produce 'observed_flows.csv'
# REQUIREMENTS FOR 'observed_flows.csv'
# 1) The observed flow matrix represents daily flows
# 2) The first column is named "Date" and has dates formatted as yyyy-mm-dd
# 3) The remaining columns each have a different site, and are named using the site ID (e.g., LAMC1F)
# 4) The units of flow are cfs

# 1. Read the file
#    - First row = header
#    - Second row = secondary labels -> skip
raw_obs = pd.read_csv(in_dir / obs_csv_file,header=0, skiprows=[1])

# 2. Parse the GMT datetime column (first column)
time_col = raw_obs.columns[0]          # should be "GMT"
raw_obs[time_col] = pd.to_datetime(raw_obs[time_col], utc=True)

# 3. Replace missing-code (-999) with NaN
raw_obs = raw_obs.replace(-999, np.nan)

# 4. Sort by time (just in case) and drop the first row (12:00 with -999, partial day)
raw_obs = raw_obs.sort_values(time_col).reset_index(drop=True)
raw_obs = raw_obs.iloc[1:]   # drop the first data row (12:00 on the first day)

# 5. Set datetime as index
raw_obs = raw_obs.set_index(time_col)

# 6. Shift the index by 13 hours so that:
#    - A "day" is 13:00 (1pm) to 12:00 (noon) next day
#    - After shifting, those 24 hours map to a single calendar day
raw_obs_shift = raw_obs.copy()
raw_obs_shift.index = raw_obs_shift.index - pd.Timedelta(hours=13)
print(raw_obs.index[0])
print(raw_obs_shift.index[0])

# 7. Resample to daily means (still UTC/GMT)
daily = raw_obs_shift.resample("D").mean()

# 8. Add in the Date column as an index
daily_out = daily.copy()
daily_out.index.name = "Date"

# 9. Only keep the columns with associated forecasts
key_to_col = {              # Map the special HEFS-style keys to the actual column names in daily_out
    "LAMC1F": "LAMC1",
    "HOPC1L": "HOPC1",
}
# Build the list of column names we want to keep
cols_to_keep = {}
for key in site_keys:
    col_name = key_to_col.get(key, key)    # map special names, else unchanged
    if col_name in daily_out.columns:
        cols_to_keep[col_name] = key       # remember mapping: dataframe_col → desired_name

# Step 2: Subset the dataframe to only these columns
obs_flows = daily_out[list(cols_to_keep.keys())].copy()

# Subset dataframe and rename columns so they exactly match the forecast labels
obs_flows = obs_flows.rename(columns=cols_to_keep)

#the observed dates
ixx_obs = obs_flows.index
print(obs_flows.columns)
print(site_keys)

# --------------------- Create the forward-looking observed flow object ----------------------------

#convert to datetime objects
ixx_obs = pd.to_datetime(ixx_obs)
ixx_hefs = pd.to_datetime(ixx_hefs)

n_obs, n_sites = obs_flows.shape
site_names = obs_flows.columns
n_hefs_sites, n_hefs, leads, n_ens = hefs_forward.shape       # (# sites, # HEFS dates, # leads, # ensemble members)
n_time_forward = n_obs - leads      #we have to drop 'leads' observations
obs_forward = np.full((n_sites, n_time_forward, leads), np.nan, dtype=float)

#IMPORTANT - because 'lead1' in hefs_forward is actually lead0 (i.e., it aligns with
#            the target day being predicted), we need to make sure 'lead1' in obs_forward
#            also aligns with the current date of interest
               
for j in range(n_sites):                  # 0..n_sites-1
    for i in range(n_time_forward):       # 0..(n_obs - leads - 1)
        # rows i+1 .. i+1+leads (remember, exclusive of upper index)
        obs_forward[j, i, :] = obs_flows.iloc[i:(i+leads), j].to_numpy()    #used to be obs_flows.iloc[(i+1):(i+1+leads), j].to_numpy(), but I think this was a mistake that caused misalignment with hefs_forward

# Dates associated with "lead0", i.e., the first lead entry in obs_forward:
# on date t, this is the sequence of obs flows over the NEXT `leads` days
ixx_obs_forward = ixx_obs[:n_time_forward]

print(ixx_obs_forward)
print(obs_forward[-1,0,:])

#ensure all dates are set as datetime objects
#also make sure hour is at 12am, so all the datetime objects are consistent and can be mached
ixx_obs= pd.to_datetime(ixx_obs,utc=True) 
ixx_hefs= pd.to_datetime(ixx_hefs,utc=True)
ixx_obs_forward = pd.to_datetime(ixx_obs_forward,utc=True)

# here we add one day to everything, because the USACE convention is to
# label the days based on the ending date of the 24 hour period, where
# the days go from 1pm GMT on one day to 12pm GMT the next day. 
# Based on the way we've coded the dates above (using the beginning date at the 1pm GMT), 
# we need to add 1 day here so we use instead the ending date at 12pm GMT the next day
ixx_obs = ixx_obs.normalize() + pd.Timedelta(days=1)
ixx_hefs = ixx_hefs.normalize() + pd.Timedelta(days=1)
ixx_obs_forward = ixx_obs_forward.normalize() + pd.Timedelta(days=1)
#also need to update the index of obs_flows
obs_flows.index = ixx_obs

print(ixx_obs[0])
print(ixx_hefs[0])
print(ixx_obs_forward[0])
print(obs_flows.index[0])

# --------------------- Save key outputs for later use  ----------------------------
np.save(out_dir / "ixx_hefs.npy", ixx_hefs)                         # the initialization dates for HEFS
np.save(out_dir / "ixx_obs.npy", ixx_obs)                           # the dates for obs flows
np.save(out_dir / "ixx_obs_forward.npy", ixx_obs_forward)           # the dates for vector of forward-looking obs flows
np.save(out_dir / "hefs_forward.npy", hefs_forward)                 # the forward-looking HEFS forecasts for all sites
np.save(out_dir / "obs_forward.npy", obs_forward)                   # the forward-looking obs flows for all sites
obs_flows.to_csv(out_dir / "observed_flows.csv",index=False)    # matrix of observed flows