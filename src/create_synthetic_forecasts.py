import os
import sys
sys.path.insert(0, os.path.abspath('./src'))
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed

#record complete time 
now=datetime.now()
print('gen start',now.strftime("%H:%M:%S"))

#numba compatible function import 
from syn_gen import syn_gen

data_dir = Path('./data')

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#key user defined specifications
loc = 'ADO'
max_lds = 15
opt_pct = 0.99
fit_gen_strategy = 'default'  #set to 'default' to fit all available paired hindcast/obs_fwd and gen all available obs_fwd; set to 'specify' to set yourself
n_samples = 10
workers = 10        #number of cores to utilize in parallel; 50 works on Hopper w/ADO test case; reduce as needed to not overload memory

# site & dates to optimize on:
keysite_label = "ADOC1" #keysite for synthetic algorithm

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#create directory for output
out_dir = Path('./out/%s/keysite=%s_optpct=%s' %(loc,keysite_label,opt_pct))
os.makedirs(out_dir,exist_ok=True)

#Specify dates settings (only used if fit_gen_strategy is set to 'specify')
#Note: fit period has to include both hefs and obs_fwd data; will error if not inclusive
st_fit = '1990-10-01'
en_fit = '2018-09-30'
#Note: fit period has to include both hefs and obs_fwd data; will error if not inclusive
st_gen = '1940-10-01'
en_gen = '2018-09-30'
    

# --------------------- Read in key inputs ----------------------------
outfile_npz = '%s_hefs_gefs_daily.npz' %(loc)
data = np.load(data_dir / outfile_npz, allow_pickle=True)
#hefs array [n_sites x n_obs x n_leads x n_ens]
hcst = data['hefs']
#obs forward (perfect forecast array) [n_sites x n_obs x n_leads]  **note: n_leads is 1 longer than hefs_array because col 0 is day t observations in obs fwd
obs_fwd = data['obs_fwd']
#site index  
sites = data['sites']
#date/time vectors
obs_fwd_dtg = data['obs_fwd_dtg']
hcst_dtg = data['hefs_dtg']
#bad forecast days
bad_forcs = data['missing_dates']

outfile = './optimized-parameters_keysite=%s_opt-pct=%s.pkl' %(keysite_label,opt_pct)
opt_pars = pickle.load(open(out_dir / outfile,'rb'),encoding='latin1')

#####--- Set Parameters for Model ------######
cur_seed    = 1
kk          = opt_pars['kk']       
knn_pwr     = opt_pars['knn_pwr'] 
scale_pwr   = opt_pars['scale_pwr'] 
hi          = opt_pars['hi'] 
lo          = opt_pars['lo']       # 1.4 
sig_a       = opt_pars['sig_a'] 
sig_b       = opt_pars['sig_b'] 

keysite_idx = np.where(sites==keysite_label)[0][0]   #set keysite index for syn_gen code

# --------------------- Read in key inputs and format for generation ----------------------------
#1. set datetime array for fit and gen periods
#ensure fit dataset includes all dates in both hefs and obs_fwd date/time groups
fit_start = max(hcst_dtg[0],obs_fwd_dtg[0])
fit_end = min(hcst_dtg[-1],obs_fwd_dtg[-1])

ixx_fit = pd.date_range(fit_start, fit_end, freq="D").to_numpy(dtype="datetime64[us]")
ixx_gen = np.array(obs_fwd_dtg,dtype='datetime64[us]')

if fit_gen_strategy == 'specify':
    fit_dates = pd.date_range(st_fit, en_fit, freq="D").to_numpy(dtype="datetime64[us]")
    gen_dates = pd.date_range(st_gen, en_gen, freq="D").to_numpy(dtype="datetime64[us]")
    if fit_dates[0] < ixx_fit[0] or fit_dates[-1] > ixx_fit[-1]:
        raise ValueError("Specified fit dates outside available hindcast/obs_fwd dataset")
    if gen_dates[0] < ixx_gen[0] or gen_dates[-1] > ixx_gen[-1]:
        raise ValueError("Specified gen dates outside available obs_fwd dataset")
    ixx_fit = fit_dates
    ixx_gen = gen_dates

#2. index synthetic generation arrays
#fit arrays
obs_fwd_fit = obs_fwd[:,np.isin(obs_fwd_dtg,ixx_fit),:]
hcst_fit = hcst[:,np.isin(hcst_dtg,ixx_fit),:,:]
#gen array
obs_fwd_gen = obs_fwd[:,np.isin(obs_fwd_dtg,ixx_gen),:]

#to prevent instabilities in generation (e.g. divide by zero), set all obs and obs_fwd zeros to min non-zero value
for i in range(np.shape(obs_fwd)[0]):
    site_sset = obs_fwd[i,:,:]
    min_nzero_vec = site_sset[site_sset>0.0]
    min_nzero = min(min_nzero_vec)
    fit_sset = obs_fwd_fit[i,:,:]
    gen_sset = obs_fwd_gen[i,:,:]
    fit_sset[fit_sset==0.0] = min_nzero
    gen_sset[gen_sset==0.0] = min_nzero
    obs_fwd_fit[i,:,:] = fit_sset
    obs_fwd_gen[i,:,:] = gen_sset

#----------------------------------------------------------------------------------------------------------------------
#function to generate synthetic forecast samples and save an output .npz file for each sample
def syn_gen_par(i):
    out = syn_gen(
        seed=i, 
        keysite_idx=keysite_idx,                     
        kk=kk,                                  
        knn_pwr=knn_pwr,                              
        scale_pwr=scale_pwr,                            
        hi=hi,                                     
        lo=lo,                                     
        sig_a=sig_a,                                  
        sig_b=sig_b,                                 
        ixx_fit=ixx_fit,
        obs_fwd_fit=obs_fwd_fit,
        obs_fwd_gen=obs_fwd_gen,
        hcst_fit=hcst_fit
    )
    
    syn_fcst = out[0]               #synthetic forecast array
    
    outfile = './syn-forecast_keysite=%s_optpct=%s_samp=%s.npz' %(keysite_label,opt_pct,i+1)
    np.savez(out_dir/outfile,syn_fcst=syn_fcst)

    return out

#run synthetic generation code in parallel
par_out = Parallel(n_jobs=n_samples)(delayed(syn_gen_par)(i) for i in range(n_samples))

#generate a single combined .npz outfile file for all samples
n_sites, n_gen, n_leads, n_ens = np.shape(par_out[0][0])
syn_fcst_arr = np.full((n_samples,n_sites,n_gen,n_leads,n_ens),np.nan,np.float32)
resamp_date_arr = np.full((n_samples,n_gen),np.nan,np.float32)
hcst_scale_arr = np.full((n_samples,n_sites,n_gen,n_leads),np.nan,np.float32)

for i in range(n_samples):
    syn_fcst_arr[i,:,:,:,:] = par_out[i][0]     #synthetic forecast array
    resamp_date_arr[i,:] = par_out[i][1]        #resampled date vector for diagnosis
    hcst_scale_arr[i,:,:,:] = par_out[i][2]     #scaling vector for diagnosis
   

#save key data elements separately
data_outfile = './syn-forecast-data_keysite=%s_optpct=%s.npz' %(keysite_label,opt_pct)
np.savez(out_dir/data_outfile,hcst_fit=hcst_fit,obs_fwd_fit=obs_fwd_fit,obs_fwd_gen=obs_fwd_gen,ixx_fit=ixx_fit,ixx_gen=ixx_gen)

#save aggregated outfile
agg_outfile = './syn-forecast-aggregated_keysite=%s_optpct=%s_samps=%s.npz' %(keysite_label,opt_pct,n_samples)
np.savez(out_dir/agg_outfile,syn_fcst_arr=syn_fcst_arr,resamp_date_arr=resamp_date_arr,hcst_scale_arr=hcst_scale_arr)


#record complete time 
now=datetime.now()
print('gen end',now.strftime("%H:%M:%S"))

######################################################END#################################################################################
