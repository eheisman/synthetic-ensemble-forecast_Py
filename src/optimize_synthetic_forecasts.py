import os
import sys
sys.path.insert(0, os.path.abspath('./src'))
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution  # or minimize, if you prefer
from datetime import datetime
from numba import njit
import pickle


#numba compatible function import 
from syn_gen_opt import syn_gen_opt,compute_mean_crps_opt,compute_cumul_rankhist_opt

data_dir = Path('./data')

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#user defined inputs 
#location specifics
loc             = 'ADO'
keysite_label   = 'ADOC1'   #keysite for synthetic algorithm optimization and sampling

#basic algorithmic settings
max_lds         = 15        #number of daily lead times to optimize to (default is total number of leads in hindcast dataset)
opt_pct         = 0.99      #percentile of data to optimize to (e.g., 0.99 = optimize to top 1% of events by flow magnitude)
fixed_kk        = True      #use fixed k value for knn sampling?
fixed_knn_pwr   = True      #use a fixed knn_pwr value for knn sampling?
fix_kk          = 20        #if fixed_kk = True, what value to use (default: 20)
fix_knn_pwr     = -0.5      #if fixed_knn_pwr = True, what value to use (default: -0.5)

#NOTE: the optimization parameter .pkl output file always includes a value for 'fix_kk' and 'fix_knn_pwr', even if 'fixed_kk' and 'fixed_knn_pwr' are set to False
#If using the fixed_kk and fixed_knn_pwr set to 'False' in optimization, the 'fix_kk' and 'fix_knn_pwr' values are meaningless, but should be set to defaults for consistency

#optimization settings
maxiter         = 100       #maximum number of iterations in DE optimizer (note: iterations != nfe; an iteration often has an nfe ~ 100); typical optimizations converge <100 iter
polish          = False     #whether or not to polish DE opt result with local gradient based optimization
tol             = 0.01      #convergence tolerance; default is 0.01, make this number smaller if you want the optimization to run longer
workers         = -1        #number of workers to use for parallel DE optimization (-1 sets algorithm to use all available cores)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#create directory for output
out_dir = './out/%s/keysite=%s' %(loc,keysite_label)
os.makedirs(out_dir,exist_ok=True)

# --------------------- Read in key inputs ----------------------------
outfile_npz = './%s/%s_hefs_gefs_daily.npz' %(loc,loc)
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

# --------------------- Format and subset datasets for optimization ----------------------------
#ensure fit dataset includes all dates in both hefs and obs_fwd date/time groups
opt_start = max(hcst_dtg[0],obs_fwd_dtg[0])
opt_end = min(hcst_dtg[-1],obs_fwd_dtg[-1])

ixx_opt = pd.date_range(opt_start, opt_end, freq="D").to_numpy(dtype="datetime64[us]")
ixx_opt= pd.to_datetime(ixx_opt)

#convert to datetime index format
ixx_obs_fwd = pd.to_datetime(obs_fwd_dtg)
ixx_hcst = pd.to_datetime(hcst_dtg)

#reduce arrays to minimum required elements in the fit period
site_idx = np.where(sites==keysite_label)[0][0]
obs_opt = obs_fwd[site_idx,np.isin(ixx_obs_fwd,ixx_opt),0]
obs_fwd_opt = obs_fwd[site_idx,np.isin(ixx_obs_fwd,ixx_opt),:]
hcst_opt = hcst[site_idx,np.isin(ixx_hcst,ixx_opt),:,:]

#remove bad forecast days from the optimizer input datasets
rmv_idx = np.isin(ixx_opt,bad_forcs)
obs_opt = np.delete(obs_opt,rmv_idx)
obs_fwd_opt = np.delete(obs_fwd_opt,rmv_idx,axis=0)
hcst_opt = np.delete(hcst_opt,rmv_idx,axis=0)

#to prevent instabilities in generation (e.g. divide by zero), set all obs and obs_fwd zeros to min non-zero value
min_nzero_vec = obs_opt[obs_opt>0.0]
min_nzero = min(min_nzero_vec)
obs_opt[obs_opt==0.0] = min_nzero
obs_fwd_opt[obs_fwd_opt==0.0] = min_nzero

#select the desired number of date indices based on specified optimization percent value
num_top_dates = np.int64((1-opt_pct)*len(ixx_opt))
opt_sset_idx = np.argsort(obs_opt)[::-1][:num_top_dates]
opt_obs = obs_opt[opt_sset_idx]
seed = 1

#compute a subsetted matrix of indices including offsets for lead times
opt_idx_mat = np.full((max_lds,num_top_dates),np.nan,dtype=np.int64)
for i in range(max_lds):
    opt_idx_mat[i,:] = opt_sset_idx - (i+1)

#flattened version of matrix above for integration with synthetic generation code
opt_idx_flat = opt_idx_mat.flatten()
    
# --------------------- Function for defining objective function of synthetics ----------------------------
#Note: all intermediary functions are numba compatible @njit functions
@njit
def calc_objective(theta,
                   hcst_mean_crps,
                   hcst_rank_hist,
                   opt_idx_mat,
                   opt_idx_flat,
                   seed,
                   obs,
                   obs_forward,
                   hcst_forward):

    """
    theta = [kk, knn_pwr, scale_pwr, hi_dif, lo, sig_a, sig_b]
    Returns mean squared difference in CRPS/cumul rank histogram between hefs and synthetics over all verifying dates and all leads.
    """

    kk      = int(round(theta[0]))  # KNN neighbors (integer)
    knn_pwr = theta[1]
    scale_pwr = theta[2]
    hi_dif  = theta[3]
    lo      = theta[4]
    sig_a   = theta[5]
    sig_b   = theta[6]
    
    #run synthetic generator for one sample using only optimization dates for each lead time
    syn_fcst = syn_gen_opt(
        seed=seed,                          
        kk=kk,                                  
        knn_pwr=knn_pwr,                               
        scale_pwr=scale_pwr,                              
        hi_dif=hi_dif,                                     
        lo=lo,                                    
        sig_a=sig_a,                                 
        sig_b=sig_b,                                  
        opt_date_indices=opt_idx_flat,     #flattened vector of indices for the optimization subset                       
        obs_forward = obs_fwd_opt,
        hcst_forward=hcst_opt                          
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
    diff1 = syn_mean_crps - hcst_mean_crps  #differences between mean CRPS scores for each lead
    obj_value1 = np.mean(diff1**2)          #take mean of squared differences across lead times

    #objective function for rank histogram
    diff2 = np.abs(syn_rank_hist - hcst_rank_hist)  #absolute differences between cumul rank histograms for each lead time
    diff2_mn = np.full(max_lds,np.nan,np.float64)   
    #loop to calculate mean of abs differences for each lead (numba can't do apply functions)
    for i in range(max_lds):
        diff2_mn[i] = np.mean(diff2[:,i])
    obj_value2 = np.mean(diff2_mn**2) #take the mean of the squared mean differences across lead times

    #final objective (sum of crps and rank hist deviations)
    obj_value = obj_value1 + obj_value2

    return obj_value

#Helper function for optimization print-out 
def de_callback(xk, convergence):
    """
    xk: current best parameter vector
    best_obj['value']: best objective function
    convergence: float describing convergence (lower = more converged)
    """
    print(f"Best params = {xk}")
    print(f"Convergence     = {convergence:.4g}")
    return False

#--------------------------------------------------------------------------------------------
#pre-calculate hefs crps and rank hist for calc_objective function
#CRPS
hcst_mean_crps = compute_mean_crps_opt(
    forecasts = hcst_opt,
    obs = opt_obs,
    forc_idx = opt_idx_mat, 
    sset_forecast = False   #using a forecast index to determine optimization dates
)
#Rank Histogram
hcst_rank_hist = compute_cumul_rankhist_opt(
    forecasts = hcst_opt,
    obs = opt_obs,
    forc_idx = opt_idx_mat,
    sset_forecast = False   #using a forecast index to determine optimization dates 
)

#--------------------------------------------------------------------------------------------
#check to see if objective function runs (important for diagnosing numba compatibility issues!)
calc_objective(
    theta=([5,0,3,10,1,1,0]),
    hcst_mean_crps=hcst_mean_crps,
    hcst_rank_hist=hcst_rank_hist,
    opt_idx_mat=opt_idx_mat,
    opt_idx_flat=opt_idx_flat,
    seed=seed,
    obs=opt_obs,
    obs_forward=obs_fwd_opt,
    hcst_forward=hcst_opt)

#--------------------------------------------------------------------------------------------
# main optimization routines
# set bounds for parameters
bounds = [
    (5, 50), # (5, 50),      # kk (will be rounded to int)
    (-3.0, 0.0), # (-3.0, 0.0),   # knn_pwr - this range covers equal weights for all leads (value of 0) to almost all weight on lead 1 (value of -3)
    (0.01, 3.0),   # (0.01, 3.0) scale_pwr - this range covers a linear decline from lead-1 to lead-15 (value of 0.01), all the way to a rapid decline from lead 1 to lead 2 (value of 3)
    (0.1, 30.0),   # hi_dif - difference between hi end of threshold and lo; formulated as a difference to prevent artificial constraints on the hi and lo parameters
    (1.0, 10.0),   # lo - lower scaling threshold (minimum of 1)
    (0.01, 10.0),   # sig_a - scale parameter of the sigmoid function; positive so small flows activate less of threshold space than large flows
    (-10.0, 10.0),  # sig_b - location parameter of sigmoid function
]

#if fixed kk or knn_pwr parameters, constrain bounds 
if fixed_kk == True:
    bounds[0] = (fix_kk,fix_kk)
    
if fixed_knn_pwr == True:
    bounds[1] = (fix_knn_pwr,fix_knn_pwr)

#start time for optimization
now=datetime.now()
print('opt start',now.strftime("%H:%M:%S"))

#main optimization routine
if __name__ == "__main__":
    result = differential_evolution(
        calc_objective,
        bounds=bounds,
        args=(
            hcst_mean_crps,
            hcst_rank_hist,
            opt_idx_mat,
            opt_idx_flat,
            seed,
            opt_obs,
            obs_fwd_opt,
            hcst_opt
        ),
        callback=de_callback,
        maxiter=maxiter,
        tol=tol,
        polish=polish,
        workers=workers
    )

    print("Finished!")
    print("Optimized parameters:", result.x)
    print("Minimum obj function value:", result.fun)
    print("NFE:", result.nfev)
    print('No_Iterations:', result.nit)


    best_kk      = int(round(result.x[0]))
    best_knn_pwr = result.x[1]
    best_scale_pwr = result.x[2]
    best_hi      = result.x[3] + result.x[4]    #output actual hi threshold to generation scripts
    best_lo      = result.x[4]
    best_sig_a   = result.x[5]
    best_sig_b   = result.x[6]
    
#--------------------------------------------------------------------------------------------
#save best parameter set
#if parameters are fixed, ensure the fixed parameter is output from the optimizer
if fixed_kk == True:
    if best_kk != fix_kk:
        raise ValueError("fixed kk does not equal optimized kk value")
if fixed_knn_pwr == True:
    knn_str = str(fix_knn_pwr)
    dec_cnt = len(knn_str.split('.')[1])
    best_knn_pwr = np.round(best_knn_pwr,decimals=dec_cnt)
    if best_knn_pwr != fix_knn_pwr:
        raise ValueError("fixed knn_pwr does not equal optimized knn_pwr value")

#save parameters as a dictionary with key specifications in the outfile name for reloading
params = {'kk': best_kk, 'knn_pwr': best_knn_pwr, 'scale_pwr': best_scale_pwr, 'hi': best_hi,'lo': best_lo, 'sig_a': best_sig_a, 'sig_b': best_sig_b}
outfile = '/optimized-parameters_keysite=%s_opt-pct=%s_fixed-kk=%s_kk=%s_fixed-knn-pwr=%s_knn-pwr=%s.pkl' %(keysite_label,opt_pct,fixed_kk,best_kk,fixed_knn_pwr,best_knn_pwr)

pickle.dump(params,open(out_dir + outfile,'wb'))

#record complete time 
now=datetime.now()
print('opt end',now.strftime("%H:%M:%S"))



##############################################################END#############################################################################