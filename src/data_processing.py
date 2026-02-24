# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 15:20:34 2026

@author: zpb4
"""

import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

max_leads = 15

loc = 'ADO'
sites = ['ADOC1']

in_dir = Path('./raw_data/')
out_dir = Path('./data/')

def split_return(x,match):
    spl_tex = x.split('.')
    out = False
    if spl_tex[0] == match:
        out = True
    return out
#---------------------- Read and process obs data ------------------------------
obs_file = 'observed_flows.csv'
obs = pd.read_csv(in_dir / obs_file,header=0,index_col=0) 
sites = obs.columns
obs_dtg = obs.index

#---------------------- calculate obs_forward array ------------------------------
obs_dtg = pd.to_datetime(obs_dtg)

n_obs, n_sites = len(obs_dtg), len(sites)

n_time_forward = n_obs - max_leads      #we have to drop 'leads' observations
obs_forward = np.full((n_sites, n_time_forward, max_leads+1), np.nan, dtype=float)

#IMPORTANT - because 'lead1' in hefs_forward is actually lead0 (i.e., it aligns with
#            the target day being predicted), we need to make sure 'lead1' in obs_forward
#            also aligns with the current date of interest
               
for j in range(n_sites):                  # 0..n_sites-1
    for i in range(n_time_forward):       # 0..(n_obs - leads - 1)
        # rows i+1 .. i+1+leads (remember, exclusive of upper index)
        obs_forward[j, i, :] = obs.iloc[i:(i+max_leads+1), j].to_numpy()    #used to be obs_flows.iloc[(i+1):(i+1+leads), j].to_numpy(), but I think this was a mistake that caused misalignment with hefs_forward

# Dates associated with "lead0", i.e., the first lead entry in obs_forward:
# on date t, this is the sequence of obs flows over the NEXT `leads` days
obs_fwd_dtg = obs_dtg[:n_time_forward]

print(obs_dtg[0],obs_dtg[-1])
print(obs_fwd_dtg[0],obs_fwd_dtg[-1])

#---------------------- process GEFSv12 1989-2023 ------------------------------
hefs_path = './%s/HEFS/%s_hefs_gefs_hourly' %(loc,sites[0])
file_names = os.listdir(in_dir / hefs_path)
st = '%s-%s-%s' %(file_names[0][:4],file_names[0][4:6],file_names[0][6:8])
en = '%s-%s-%s' %(file_names[-1][:4],file_names[-1][4:6],file_names[-1][6:8])

hefs_template = pd.read_csv(in_dir / ('%s/%s' %(hefs_path,file_names[0])),header=0, skiprows=[1], index_col=0)
ens_names = list(hefs_template.columns)
ens_names_split = [split_return(ens_names[x],sites[0]) for x in range(len(ens_names))]

site_in_hourly = hefs_template.iloc[1:, ens_names_split]
site_in_hourly_shift = pd.to_datetime(hefs_template.index[:(-1)]) + pd.Timedelta(hours=12)
site_in_hourly.index = site_in_hourly_shift
site_daily = site_in_hourly.resample("D").mean()

ixx_out = pd.date_range(st,en,freq='D')
out_arr = np.zeros((len(sites),len(ixx_out),max_leads,np.shape(site_daily)[1]))

fname_temp = file_names[0]

def process_hourly_forecasts(k):
    new_raw_dir = './%s/HEFS/%s_hefs_gefs_daily' %(loc,sites[k])
    os.makedirs(in_dir / new_raw_dir,exist_ok=True)
    site_arr = np.zeros((len(ixx_out),max_leads,np.shape(out_arr)[3]))
    missing_dates = []
    for i in range(len(ixx_out)):
        d_idx = ixx_out[i].strftime('%Y%m%d')
        fname_out = d_idx + fname_temp[8:]
        try:
            hefs_path = './%s/HEFS/%s_hefs_gefs_hourly/%s' %(loc,sites[k],fname_out)
            hefs_in_hourly = pd.read_csv(in_dir / hefs_path,header=0, skiprows=[1], index_col=0)
        except FileNotFoundError: 
            #if no file, record it as a missing date and replace it with a dummy file from a different year (1992 leap year to include all possible dates)
            try:
                missing_dates.append(ixx_out[i])
                d_idx_mday = ixx_out[i].strftime('%m%d')
                new_d_idx = '1992'+d_idx_mday
                fname_out = new_d_idx + fname_temp[8:]
            
                hefs_path = './%s/HEFS/%s_hefs_gefs_hourly/%s' %(loc,sites[k],fname_out)
                hefs_in_hourly = pd.read_csv(in_dir / hefs_path,header=0, skiprows=[1], index_col=0)
            #if 1992 is not avail, try 1996
            except FileNotFoundError:
                missing_dates.append(ixx_out[i])
                d_idx_mday = ixx_out[i].strftime('%m%d')
                fname_out = new_d_idx + fname_temp[8:]
            
                hefs_path = './%s/HEFS/%s_hefs_gefs_hourly/%s' %(loc,sites[k],fname_out)
                hefs_in_hourly = pd.read_csv(in_dir / hefs_path,header=0, skiprows=[1], index_col=0)
        
        ens_names = list(hefs_in_hourly.columns)
        ens_names_split = [split_return(ens_names[x],sites[k]) for x in range(len(ens_names))]

        site_in_hourly = hefs_in_hourly.iloc[1:, ens_names_split]
        site_in_hourly_shift = pd.to_datetime(hefs_in_hourly.index[:(-1)]) + pd.Timedelta(hours=12)
        site_in_hourly.index = site_in_hourly_shift
        site_daily = site_in_hourly.resample("D").mean()
        site_daily.index = site_daily.index + pd.Timedelta(hours=12)
        site_daily_out = site_daily.iloc[:max_leads,:]
        
        site_arr[i,:,:] = site_daily_out
        
        outfile = new_raw_dir + '/%s12_%s_hefs_gefs_daily.csv' %(d_idx,sites[k])
        site_daily_out.to_csv(in_dir / outfile,index=True)

    return site_arr,missing_dates

par_out = Parallel(n_jobs=len(sites))(delayed(process_hourly_forecasts)(i) for i in range(len(sites)))

missing_dates = []

new_data_dir = './%s' %(loc)
os.makedirs(out_dir / new_data_dir,exist_ok=True)
    
for k in range(len(sites)):
    site_arr = par_out[0][0]
    missing_dates.append(par_out[0][1])
    out_arr[k,:,:,:] = site_arr

missing_dates = np.unique(missing_dates)

outfile_npz = '%s_hefs_gefs_generation-dataset_daily.npz' %(loc)
np.savez(out_dir / outfile_npz,obs=obs,obs_fwd=obs_forward,hefs=out_arr,sites=sites,obs_dtg=obs_dtg,obs_fwd_dtg=obs_fwd_dtg,hefs_dtg=ixx_out,missing_dates=missing_dates)


#//////////////////////////////////////////////////////////////////////////////////////////////
#Simple plot verification of synchronization
#//////////////////////////////////////////////////////////////////////////////////////////////
data = np.load(out_dir / outfile_npz, allow_pickle=True)#,hefs=hefs_rfc_out,obs_fwd=obs_fwd_hefs_rfc,sites=sites_hefs_rfc,date_idx=out_idx,missing_dates=missing_dates_hefs_rfc)
#hefs array [n_sites x n_obs x n_leads x n_ens]
hefs = data['hefs']
#obs forward (perfect forecast array) [n_sites x n_obs x n_leads]  **note: n_leads is 1 longer than hefs_array because col 0 is day t observations in obs fwd
obs_fwd = data['obs_fwd']
#site index  **note: uses 'LAMC1F' and 'HOPC1L' in place of 'LAMC1' and 'HOPC1'
sites = data['sites']
#date/time vector
obs_fwd_dtg = data['obs_fwd_dtg']
hefs_dtg = data['hefs_dtg']
#bad forecast days
bad_forcs = data['missing_dates']

print(np.shape(hefs))
print(np.shape(obs_fwd))
print(sites)
print(obs_fwd_dtg[0],obs_fwd_dtg[-1])
print(hefs_dtg[0],hefs_dtg[-1])
print(bad_forcs)

###############################################################
#plot timeseries comparison
#start and end dates likely to be in all runs
st_cmn = '1990-10-01'
en_cmn = '2018-09-30'

dtg_cmn = pd.date_range(st_cmn,en_cmn,freq='D')

site = 'ADOC1'
site_idx = np.where((sites == site))[0][0]

obs_fwd_slice = np.arange(len(obs_fwd_dtg))
hefs_slice = np.arange(len(hefs_dtg))
sset_idx_hefs = hefs_slice[(hefs_dtg >= np.datetime64(st_cmn)) & (hefs_dtg<= np.datetime64(en_cmn))]
sset_idx_obs_fwd = obs_fwd_slice[(obs_fwd_dtg >= np.datetime64(st_cmn)) & (obs_fwd_dtg<= np.datetime64(en_cmn))]

obs_fwd_ver = obs_fwd[site_idx,sset_idx_obs_fwd,:]
hefs_ver = hefs[site_idx,sset_idx_hefs,:,:]

#get indices of largest events
ext_idx = np.argsort(obs_fwd_ver[:,0])[::-1]

#plot the top 12 sorted events at a 3-d lead
fig = plt.figure(layout='constrained',figsize=(10,8))
gs0 = fig.add_gridspec(4,3)

ld = 1
evt_indices = ext_idx[:12] - ld

for i in range(len(evt_indices)):
    ax1 = fig.add_subplot(gs0[i])
    ax1.plot(np.arange(max_leads+1),obs_fwd_ver[evt_indices[i],:],linewidth=2,c='black')
    ax1.axvline(ld,c='gray',linewidth=0.5,linestyle='--',alpha=0.5)
    ylm = max(obs_fwd_ver[evt_indices[0],:])*1.1
    print(ylm)
    ax1.set_ylim([0,ylm])
    ax1.text(4,0.9*ylm,str(dtg_cmn[evt_indices[i]+ld])[:10])
    for k in range(np.shape(hefs_ver)[2]):
        ens_out = obs_fwd_ver[evt_indices[i],:].copy()
        ens_out[1:] = hefs_ver[evt_indices[i],:,k].copy()
        ax1.plot(np.arange(max_leads+1),ens_out,c='gray',linewidth=1,alpha=0.5)

plt.show()

#plots check good for obs fwd and hefs alignment

###############################################################END###########################################################################