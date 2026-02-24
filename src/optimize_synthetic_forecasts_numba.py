import os
import sys
sys.path.insert(0, os.path.abspath('./src'))
import re
from pathlib import Path
import numpy as np
import pandas as pd
import properscoring as ps
from scipy.optimize import differential_evolution  # or minimize, if you prefer
from datetime import datetime
import gc
import multiprocess as mp
from itertools import chain
from numba import njit
from datetime import datetime
import pickle

#numba compatible function import 
from syn_gen_opt_numba import syn_gen_opt_numba,compute_mean_crps_opt,compute_cumul_rankhist_opt

data_dir = Path('./data')
out_dir = Path('./out')

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#user defined inputs (expand in future to include hard settings of optimization params [e.g. kk or knn_pwr])
loc = 'ADO'
max_lds = 15
opt_pct = 0.99

# site & dates to optimize on:
keysite_label = "ADOC1" #keysite for synthetic algorithm, for now these two are the same; could be differentiated
#site_label = "ADOC1" #site to target optimization to
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# --------------------- Read in key inputs ----------------------------
outfile_npz = '%s_hefs_gefs_daily.npz' %(loc)
data = np.load(data_dir / outfile_npz, allow_pickle=True)
#hefs array [n_sites x n_obs x n_leads x n_ens]
hefs = data['hefs']
#obs forward (perfect forecast array) [n_sites x n_obs x n_leads]  **note: n_leads is 1 longer than hefs_array because col 0 is day t observations in obs fwd
obs_fwd = data['obs_fwd']
#site index  
sites = data['sites']
#date/time vectors
obs_fwd_dtg = data['obs_fwd_dtg']
hefs_dtg = data['hefs_dtg']
#bad forecast days
bad_forcs = data['missing_dates']

#ensure fit dataset includes all dates in both hefs and obs_fwd date/time groups
opt_start = max(hefs_dtg[0],obs_fwd_dtg[0])
opt_end = min(hefs_dtg[-1],obs_fwd_dtg[-1])

ixx_opt = pd.date_range(opt_start, opt_end, freq="D").to_numpy(dtype="datetime64[us]")
ixx_opt= pd.to_datetime(ixx_opt)

#convert to datetime index format
ixx_obs_fwd = pd.to_datetime(obs_fwd_dtg)
ixx_hefs = pd.to_datetime(hefs_dtg)

#reduce arrays to minimum required elements in the fit period
site_idx = np.where(sites==keysite_label)[0][0]
obs_opt = obs_fwd[site_idx,np.isin(ixx_obs_fwd,ixx_opt),0]
obs_fwd_opt = obs_fwd[site_idx,np.isin(ixx_obs_fwd,ixx_opt),:]
hefs_opt = hefs[site_idx,np.isin(ixx_hefs,ixx_opt),:,:]

#to prevent instabilities in generation (e.g. divide by zero), set all obs and obs_fwd zeros to min non-zero value
min_nzero_vec = obs_opt[obs_opt>0.0]
min_nzero = min(min_nzero_vec)
obs_opt[obs_opt==0.0] = min_nzero
obs_fwd_opt[obs_fwd_opt==0.0] = min_nzero

#select the desired number of date indices based on specified optimization percent value
num_top_dates = np.int64((1-opt_pct)*len(ixx_opt))
opt_dates = pd.Series(obs_opt, index=ixx_opt).nlargest(num_top_dates).index
opt_obs = pd.Series(obs_opt, index=ixx_opt).nlargest(num_top_dates).values
opt_dates = pd.to_datetime(opt_dates)
seed = 1

#compute a subsetted matrix of indices including offsets for lead times
opt_idx_mat = np.full((max_lds,num_top_dates),np.nan,dtype=np.int64)
for i in range(max_lds):
    date_sset = opt_dates - pd.Timedelta(days=(i+1))
    slice_idx = np.arange(len(ixx_opt),dtype=np.int64)
    date_idx = slice_idx[np.isin(ixx_opt,date_sset)]
    opt_idx_mat[i,:] = date_idx

#flattened version of matrix above for integration with synthetic generation code
opt_idx_flat = opt_idx_mat.flatten()
    

# --------------------- Function for defining objective function of synthetics ----------------------------
#Note: all intermediary functions are numba compatible @njit functions
@njit
def calc_objective(theta,
                   hefs_mean_crps,
                   hefs_rank_hist,
                   opt_idx_mat,
                   opt_idx_flat,
                   seed,
                   obs,
                   obs_forward,
                   hefs_forward):

    """
    theta = [kk, knn_pwr, scale_pwr, hi, lo, sig_a, sig_b]
    Returns mean squared difference in CRPS/cumul rank histogram between hefs and synthetics over all verifying dates and all leads.
    """

    kk      = max(5,int(round(theta[0])))  # KNN neighbors (integer)
    knn_pwr = theta[1]
    scale_pwr = theta[2]
    hi      = theta[3]
    lo      = theta[4]
    sig_a   = theta[5]
    sig_b   = theta[6]
    
    """
    #dummy values for code testing
    kk = 10
    knn_pwr = 0
    scale_pwr = .05
    hi = 10
    lo = 1.5
    sig_a = 1
    sig_b = 0
    
    obs_forward = obs_fwd_opt
    hefs_forward = hefs_opt
    """
    
    # --- run synthetic generator for one sample using only optimization dates for each lead time---
    syn_fcst = syn_gen_opt_numba(
        seed=seed,                          
        kk=kk,                                  
        knn_pwr=knn_pwr,                               
        scale_pwr=scale_pwr,                              
        hi=hi,                                     
        lo=lo,                                    
        sig_a=sig_a,                                 
        sig_b=sig_b,                                  
        opt_date_indices=opt_idx_flat,     #flattened vector of                        
        obs_forward = obs_fwd_opt,
        hefs_forward=hefs_opt                          
    )
    
    #compute the mean CRPS values for optimized dates across lead times
    syn_mean_crps = compute_mean_crps_opt(
        forecasts = syn_fcst,
        obs = opt_obs,
        forc_idx = opt_idx_mat,
        sset_forecast = True   #not using a forecast index because syn-forecast is only generated against selected dates
    )

    #compute the cumulative rank histograms for optimized dates across lead times
    syn_rank_hist = compute_cumul_rankhist_opt(
        forecasts = syn_fcst,
        obs = opt_obs,
        forc_idx = opt_idx_mat,
        sset_forecast = True   #not using a forecast index because syn-forecast is only generated against selected dates
    )

    #objective function for CRPS
    diff1 = syn_mean_crps - hefs_mean_crps  #differences between mean CRPS scores for each lead
    obj_value1 = np.mean(diff1**2)          #take mean of squared differences across lead times

    #objective function for rank histogram
    diff2 = np.abs(syn_rank_hist - hefs_rank_hist)  #absolute differences between cumul rank histograms for each lead time
    diff2_mn = np.full(max_lds,np.nan,np.float64)   
    #loop to calculate mean of abs differences for each lead (numba can't do apply functions)
    for i in range(max_lds):
        diff2_mn[i] = np.mean(diff2[:,i])
    obj_value2 = np.mean(diff2_mn**2) #take the mean of the squared mean differences across lead times

    #final objective (sum of crps and rank hist deviations)
    obj_value = obj_value1 + obj_value2

    return obj_value

#pre-calculate hefs crps and rank hist for calc_objective function
#CRPS
hefs_mean_crps = compute_mean_crps_opt(
    forecasts = hefs_opt,
    obs = opt_obs,
    forc_idx = opt_idx_mat, 
    sset_forecast = False   #using a forecast index to determine optimization dates
)
#Rank Histogram
hefs_rank_hist = compute_cumul_rankhist_opt(
    forecasts = hefs_opt,
    obs = opt_obs,
    forc_idx = opt_idx_mat,
    sset_forecast = False   #using a forecast index to determine optimization dates 
)

#check to see if objective function runs (important for diagnosing numba compatibility issues!)
calc_objective(
    theta=([5,0,3,10,1,1,0]),
    hefs_mean_crps=hefs_mean_crps,
    hefs_rank_hist=hefs_rank_hist,
    opt_idx_mat=opt_idx_mat,
    opt_idx_flat=opt_idx_flat,
    seed=seed,
    obs=opt_obs,
    obs_forward=obs_fwd_opt,
    hefs_forward=hefs_opt)

# --------------------- Helper function for optimization print-out ----------------------------
def de_callback(xk, convergence):
    """
    xk: current best parameter vector
    best_obj['value']: best objective function
    convergence: float describing convergence (lower = more converged)
    """
    print(f"Best params = {xk}")
    print(f"Convergence     = {convergence:.4g}")
    return False

# bounds for parameters
bounds = [
    (5, 50), # (5, 50),      # kk (will be rounded to int)
    (-3.0, 0.0), # (-3.0, 0.0),   # knn_pwr - this range covers equal weights for all leads (value of 0) to almost all weight on lead 1 (value of 3)
    (0.01, 3.0),   # (0.01, 3.0) scale_pwr - this range covers a linear decline from lead-1 to lead-15 (value of 0.01), all the way to a rapid decline from lead 1 to lead 2 (value of 3)
    (2.0, 30.0),   # hi - for now, we constrain this one to be some large number (i.e., lead-1 can always scale a lot)
    (1.0, 1.5),   # lo - have upper bound be below hi
    (0.0, 10.0),   # sig_a - should be positive, so that larger flows get more of the scaling, and small flows get reduced scaling
    (-10.0, 10.0),  # sig_b
]

#start time for optimization
now=datetime.now()
print('opt start',now.strftime("%H:%M:%S"))

if __name__ == "__main__":
    result = differential_evolution(
        calc_objective,
        bounds=bounds,
        args=(
            hefs_mean_crps,
            hefs_rank_hist,
            opt_idx_mat,
            opt_idx_flat,
            seed,
            opt_obs,
            obs_fwd_opt,
            hefs_opt
        ),
        callback=de_callback,
        maxiter=3,   # tweak as needed
        polish=False,
        workers=10
    )

    print("Finished!")
    print("Optimized parameters:", result.x)
    print("Minimum mean CRPS:", result.fun)
    print("NFE:", result.nfev)
    print('No_Iterations:', result.nit)


    best_kk      = int(round(result.x[0]))
    best_knn_pwr = result.x[1]
    best_scale_pwr = result.x[2]
    best_hi      = result.x[3]
    best_lo      = result.x[4]
    best_sig_a   = result.x[5]
    best_sig_b   = result.x[6]
    
#save best paremeter set
params = {'kk': best_kk, 'knn_pwr': best_knn_pwr, 'scale_pwr': best_scale_pwr, 'hi': best_hi,'lo': best_lo, 'sig_a': best_sig_a, 'sig_b': best_sig_b}
outfile = './%s/optimized-parameters_keysite=%s_opt-pct=%s.pkl' %(loc,keysite_label,opt_pct)

pickle.dump(params,open(out_dir / outfile,'wb'))

#record complete time 
now=datetime.now()
print('opt end',now.strftime("%H:%M:%S"))