import numpy as np
from numba import njit
import random

# ------------------------------------------------------------------
# function to calculate obs_fwd array
# ------------------------------------------------------------------
@njit
def obs_fwd_fun(obs,lds):
    n_obs = len(obs)
    n_time_forward = n_obs - lds     
    obs_fwd = np.full((n_time_forward, lds+1), np.nan, dtype=np.float64)

    for i in range(n_time_forward):       
        obs_fwd[i, :] = obs[i:(i+lds+1)]
        
    return obs_fwd
        
# ------------------------------------------------------------------
# Scale-decay function (matches R scale_decay_fun)
# ------------------------------------------------------------------
@njit
def scale_decay_fun(hi, lo, pwr, lds):
    w = np.arange(1, lds + 1)
    if pwr != 0:
        win = w[::-1]
        num = np.exp(pwr * win) - np.exp(pwr)
        den = np.exp(2*pwr) - np.exp(pwr)
        dcy = num / den
        dcy_out = dcy / dcy.max() * (hi - lo) + lo
    else:
        # linear from hi down to lo
        step = (hi - lo) / (len(w) - 1)
        dcy_out = hi - np.arange(len(w)) * step
    return dcy_out

# sigmoid function
@njit
def sigmoid_fun(x, a, b):
    return 1.0 / (1.0 + np.exp(-(x * a + b)))

# ------------------------------------------------------------------
# Simple Gaussian-like smoothing (approximate ksmooth with bandwidth=1)
# ------------------------------------------------------------------
@njit
def ksmooth_1d(x, bandwidth=1.0):
    if bandwidth <= 0:
        return x
    # discrete Gaussian kernel
    radius = int(3 * bandwidth)
    idx = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (idx / bandwidth) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same")


#------------------------------------------------------------------
# Numba compatible replacement for np.random.weighted_choice
#-----------------------------------------------------------------
@njit
def numba_weighted_choice(arr, prob):
    """
    Selects a random sample from 'arr' with probabilities 'prob' in Numba.
    
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
                 Must sum to 1 (or you can normalize it first).
    :return: A random sample (single value) from the given array with a given probability.
    """
    # Calculate the cumulative distribution function (CDF)
    # Note: prob should already be normalized so it sums to 1
    cdf = np.cumsum(prob) 
    
    # Generate a single random float between 0.0 and 1.0
    random_val = np.random.random() # Or random.random()

    # Find the index where the random value would be inserted to maintain order
    # 'right' side ensures that a value of 1.0 (if it occurred) would map to the last index
    idx = np.searchsorted(cdf, random_val, side="right")
    
    # Return the element at the found index
    return arr[idx]

#------------------------------------------------------------------
# Synthetic generation script for single site HEC-WAT FRA inputs
#-----------------------------------------------------------------
@njit
def syn_gen_hec_wat_fra(
    seed,
    kk,                     # 
    knn_pwr,                # 
    scale_pwr,              #
    hi,                     #
    lo,                     #
    sig_a,                  #
    sig_b,                  #
    ixx_fit,
    obs_fwd_fit,
    obs_fwd_gen,                    # obs forward array subsetted to same fit/gen period
    hcst_fit,                 # shape: (n_sites, n_ens, n_hefs_time, leads)
):
    """
    Returns
    -------
    all_leads_gen : np.ndarray
        Synthetic ensemble forecasts, shape (n_sites, n_ens, n_gen, leads)
    hefs_resamp_vec : pd.DatetimeIndex
        Resampled HEFS initialization dates (length n_gen)
    HEFS_scale_out : np.ndarray
        Final scaling factors, shape (n_sites, n_gen, leads)
    """

    random.seed(seed)

    # ------------------------------------------------------------------
    # Get other variables needed for algorithm
    # ------------------------------------------------------------------
    n_fit, n_leads, n_ens = hcst_fit.shape         # (# sites, # HEFS dates, # leads, # ensemble members)
    n_gen = np.shape(obs_fwd_gen)[0]
    
    ixx_fit = ixx_fit                                       #numpy date vector for fit period
    # ------------------------------------------------------------------
    # Error checks
    # ------------------------------------------------------------------
    if kk < 1 or int(kk) != kk:
        raise ValueError("kk is not a valid positive integer")


    # ------------------------------------------------------------------
    # Subset obs_forward arrays to simulation and fitting periods
    # ------------------------------------------------------------------
    # obs_forward for simulation dates (intersection with ixx_gen)
    obs_forward_gen = obs_fwd_gen               # (n_gen, leads)
    # obs_forward for simulation dates (intersection with ixx_gen)
    obs_forward_fit = obs_fwd_fit               # (n_fit, leads)

    # ------------------------------------------------------------------
    # KNN setup
    # ------------------------------------------------------------------
    # weights for neighbor ranks 1..kk
    wts_raw = np.array([1.0 / k for k in range(1, kk + 1)], dtype=np.float64)
    wts = wts_raw / wts_raw.sum()

    # decay weights across leads
    w_leads = np.arange(1, n_leads + 2, dtype=np.float64)
    decay = (w_leads ** knn_pwr) / np.sum(w_leads ** knn_pwr)


    # ------------------------------------------------------------------
    # KNN distances (based on keysite only)
    # ------------------------------------------------------------------
    gen_knn_data = obs_forward_gen[:,:]
    fit_knn_data = obs_forward_fit[:,:]
    
    gen_knn_data = np.transpose(gen_knn_data)
    fit_knn_data = np.transpose(fit_knn_data)

    knn_dist = np.empty((n_fit, n_gen), dtype=np.float64)
    for j in range(n_gen):
        diff = gen_knn_data[:, j][:, None] - fit_knn_data        # (leads, n_fit)
        knn_dist[:, j] = np.sqrt(np.sum(decay[:, None] * (diff ** 2), axis=0))

    # Resampled locations via KNN
    # For each gen day, sample 1 index from nearest kk fits (0-based indices into n_fit)
    knn_lst = np.empty(n_gen, dtype=np.int64)
    for j in range(n_gen):
        d = knn_dist[:, j].copy()
        # prevent zero-distance reuse
        d[d == 0.0] = np.nan
        valid = np.where(~np.isnan(d))[0]
        if valid.size < kk:
            raise ValueError("Not enough valid neighbors for KNN")
        sorted_valid = valid[np.argsort(d[valid])]
        neighbors = sorted_valid[:kk]
        # sample one neighbor with probabilities wts
        np.random.seed(seed)
        knn_samp = numba_weighted_choice(neighbors, wts)
        knn_lst[j] = np.int64(knn_samp)
    
    # ------------------------------------------------------------------
    # Scale-decay function for optimized threshold
    # ------------------------------------------------------------------
    dcy = scale_decay_fun(hi, lo, scale_pwr, n_leads)

    # ------------------------------------------------------------------
    # Main scaling loop over sites
    # ------------------------------------------------------------------
    
    # Final array for synthetic ensemble forecasts
    final_synthetic_forecasts = np.full((n_gen,n_leads+1,n_ens), np.nan, dtype=np.float64)
    
    # gen_scale, fit_scale: (n_gen, n_leads)
    gen_scale = obs_forward_gen[:, 1:].copy()
    fit_scale = obs_forward_fit[knn_lst, 1:].copy()
    
    #calculate scaling array
    HEFS_scale = gen_scale / fit_scale  # (n_gen, n_leads)

    for k in range(n_leads):
        col = HEFS_scale[:, k].copy()
        # handle NaN, Inf, and zeros
        invalid = np.isnan(col) | np.isinf(col) | (col == 0.0)
        col[invalid] = 1.0

        # obs_sc for scaling thresholds
        obs_sc = obs_forward_gen[:,k].copy()
        fit_sc = obs_forward_fit[:,k].copy()
        obs_sc = np.log(obs_sc)
        fit_sc = np.log(fit_sc)

        # standardized obs
        mu = np.mean(fit_sc)
        sd = np.std(fit_sc)
        
        #scale observations
        obs_scale = (obs_sc - mu) / sd
        
        #relax scaling threshold proportional to any obs in an obs_fwd row that is greater than the max fit period obs
        ext_scale_mat = gen_scale / np.max(obs_forward_fit)
        ext_scale_vec = np.ones(np.shape(gen_scale)[0],dtype=np.float64)
        for l in range(len(ext_scale_vec)):
            ext_scale_vec[l] = max(1.0,np.max(ext_scale_mat[l,:]))
        dcy_vec = dcy[k] * ext_scale_vec
        
        #calculate activated region of threshold for each row based on the obs value w/relaxed scaling 
        ratio_threshold = sigmoid_fun(obs_scale,sig_a,sig_b) * (dcy_vec-1) + 1

        # cap scaling ratios at threshold
        exceed = col > ratio_threshold
        col[exceed] = ratio_threshold[exceed]

        HEFS_scale[:, k] = col

    # Smooth across leads for each generation time if so desired
    HEFS_scale_sm = np.empty_like(HEFS_scale)
    #currently, this will lead to no smoothing, but it could be adjusted if desired
    base = [
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        ]
    repeat_arr = np.array([0.0, 1.0, 0.0])
    kernel = base + [repeat_arr] * (n_leads - len(base))

    for t in range(n_gen):
        # version with smoothing 
        HEFS_scale_sm[t, :] = np.array([np.convolve(HEFS_scale[t, :], kernel[l], mode='same')[l] for l in np.arange(0,n_leads)]) 

    # Apply scaling to each ensemble member
    for e in range(n_ens):
        hefs_fwd_sset = hcst_fit[knn_lst,:,:]
        final_synthetic_forecasts[:, 1:, e] = hefs_fwd_sset[:,:,e] * HEFS_scale_sm
        final_synthetic_forecasts[:, 0, e] = obs_forward_gen[:, 0]

    return final_synthetic_forecasts




##########################################################END#################################################################