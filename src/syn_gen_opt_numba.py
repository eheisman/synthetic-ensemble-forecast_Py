import numpy as np
import pandas as pd
import gc
from numba import njit
import numba

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
# Numba compatible ensemble crps calculation function (not compatible with properscoring package)
#-----------------------------------------------------------------
@njit
def ensemble_crps(ensemble,tgt):
    ne = len(ensemble)
    term1 = (1/ne) * np.sum(np.abs(ensemble - tgt))
    
    #for loop to calculate second ecrps term per Wilks 2019
    term2 = np.zeros((ne,ne))
    for i in range(ne-1):
        for j in range(i+1,ne):
            term2[i,j] = np.abs(ensemble[i]-ensemble[j])
    
    term2_result = (1 / (ne * (ne-1))) * np.sum(term2)
    out = term1 - term2_result
    
    return out

#------------------------------------------------------------------
# Function to compute mean crps across lead times for a single forecast sample
#-----------------------------------------------------------------
@njit
def compute_mean_crps_opt(
    forecasts,
    obs,
    forc_idx,
    sset_forecast
):
    """
    Compute mean CRPS by lead time for forecasts at a single site
    Requires pre-indexed optimization obs and forecast index
    Forecasts need to be for the same slice period as the pre-indexed obs
    The n_opt_obs is the subset of observations (generally upper quantile) being used for optimization
    
    Inputs
    ------
    forecasts : an n_obs x n_leads x n_ens array
    obs : a vector repeated for each lead time
    forc_idx : an array n_leads x n_opt_obs (only used if forecast is not pre subsetted)
    sset_forecast : if True, forecast is already subsetted to a flattened vector of forecast indices for the n_opt_obs subset (i.e. syn_gen_opt function)
                    if False, forecast is not subsetted (i.e. parent hindcast dataset)

    Returns
    -------
    hefs_mean_crps : np.ndarray of shape (n_leads,)
        Mean CRPS at each lead (1..n_leads).
    """
    if sset_forecast == False:
        n_dates, n_leads, n_ens = np.shape(forecasts)
        n_opt_obs = len(obs)
    
        crps_mat = np.full((n_opt_obs,n_leads),np.nan,dtype=np.float64)
    
        for i in range(n_leads):
            forc_eval = forecasts[forc_idx[i,:],i,:]
            for j in range(n_opt_obs):
                #crps_mat[j,i] = ps.crps_ensemble(obs[j],forc_eval[j,:])
                crps_mat[j,i] = ensemble_crps(forc_eval[j,:],obs[j])
    
        #mean_ecrps = np.apply_along_axis(np.mean,0,crps_mat)
        mean_ecrps = np.full(n_leads,np.nan,np.float64)
        for i in range(n_leads):
            mean_ecrps[i] = np.mean(crps_mat[:,i])
        
    else:
        n_dates, n_leads, n_ens = np.shape(forecasts)
        n_opt_obs = len(obs)
        forc_sset_idx = np.arange(n_dates)
        lead_id = np.repeat(np.arange(n_leads),n_opt_obs)
    
        crps_mat = np.full((n_opt_obs,n_leads),np.nan,dtype=np.float64)
    
        for i in range(n_leads):
            sset_idx = forc_sset_idx[np.where(lead_id==i)]
            forc_eval = forecasts[sset_idx,i,:]
            for j in range(n_opt_obs):
                #crps_mat[j,i] = ps.crps_ensemble(obs[j],forc_eval[j,:])
                crps_mat[j,i] = ensemble_crps(forc_eval[j,:],obs[j])
    
        #mean_ecrps = np.apply_along_axis(np.mean,0,crps_mat)
        mean_ecrps = np.full(n_leads,np.nan,np.float64)
        for i in range(n_leads):
            mean_ecrps[i] = np.mean(crps_mat[:,i])
        
    return mean_ecrps

#------------------------------------------------------------------
# Ensemble ranking helper function
#-----------------------------------------------------------------
@njit
def ensemble_rank(obs, ens):
    """
    Return the rank of obs relative to ensemble ens.
    Rank = number of ensemble members <= obs, in [0, K].
    """
    ens = np.asarray(ens)
    if not np.isfinite(obs):
        return np.nan
    ens = ens[np.isfinite(ens)]
    if ens.size == 0:
        return np.nan
    return np.sum(ens <= obs)

#------------------------------------------------------------------
# Function to compute cumulative rank histogram across lead times 
#-----------------------------------------------------------------
@njit
def compute_cumul_rankhist_opt(
    forecasts,
    obs,
    forc_idx,
    sset_forecast
):
    """
    Compute mean CRPS by lead time for forecasts at a single site
    Requires pre-indexed optimization obs and forecast index
    Forecasts need to be for the same slice period as the pre-indexed obs
    The n_opt_obs is the subset of observations (generally upper quantile) being used for optimization
    
    Inputs
    ------
    forecasts : an n_obs x n_leads x n_ens array
    obs : a vector repeated for each lead time
    forc_idx : an array n_leads x n_opt_obs (only used if forecast is not pre subsetted)
    sset_forecast : if True, forecast is already subsetted to a flattened vector of forecast indices for the n_opt_obs subset (i.e. syn_gen_opt function)
                    if False, forecast is not subsetted (i.e. parent hindcast dataset)

    Returns
    -------
    hefs_mean_crps : np.ndarray of shape (n_leads,)
        Mean CRPS at each lead (1..n_leads).
    """
    if sset_forecast == False:
        n_dates, n_leads, n_ens = np.shape(forecasts)
        n_opt_obs = len(obs)
        num_bins = n_ens + 1
    
        rank_mat = np.full((num_bins,n_leads),np.nan,dtype=np.float64)
        rank_vec = np.full(n_opt_obs,np.nan,dtype=np.int64)
    
        for i in range(n_leads):
            forc_eval = forecasts[forc_idx[i,:],i,:]
            for j in range(n_opt_obs):
                rank_vec[j] = np.int64(ensemble_rank(obs[j],forc_eval[j,:]))
            rankhist = np.histogram(rank_vec,bins=np.arange(-0.5,num_bins))[0]
            cumul_rankhist = np.cumsum(rankhist) / np.sum(rankhist)
            rank_mat[:,i] = cumul_rankhist
    
    else:
        n_dates, n_leads, n_ens = np.shape(forecasts)
        n_opt_obs = len(obs)
        num_bins = n_ens + 1
        forc_sset_idx = np.arange(n_dates)
        lead_id = np.repeat(np.arange(n_leads),n_opt_obs)
    
        rank_mat = np.full((num_bins,n_leads),np.nan,dtype=np.float64)
        rank_vec = np.full(n_opt_obs,np.nan,dtype=np.int64)
    
        for i in range(n_leads):
            sset_idx = forc_sset_idx[np.where(lead_id==i)]
            forc_eval = forecasts[sset_idx,i,:]
            for j in range(n_opt_obs):
                rank_vec[j] = np.int64(ensemble_rank(obs[j],forc_eval[j,:]))
            rankhist = np.histogram(rank_vec,bins=np.arange(-0.5,num_bins))[0]
            cumul_rankhist = np.cumsum(rankhist) / np.sum(rankhist)
            rank_mat[:,i] = cumul_rankhist
        
    return rank_mat

#------------------------------------------------------------------
# Synthetic generation script for optimization
#-----------------------------------------------------------------
@njit
def syn_gen_opt_numba(
    seed,                   # random seed
    kk,                     # 
    knn_pwr,                # 
    scale_pwr,              #
    hi,                     #
    lo,                     #
    sig_a,                  #
    sig_b,                  #
    opt_date_indices,              #flattened list of indices for optimization dates
    obs_forward,                    # obs forward array subsetted to same fit/gen period
    hefs_forward,                 # shape: (n_sites, n_ens, n_hefs_time, leads)
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

    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Get other variables needed for algorithm
    # ------------------------------------------------------------------
    n_hefs, n_leads, n_ens = hefs_forward.shape       # (# sites, # HEFS dates, # leads, # ensemble members)
    n_gen = len(opt_date_indices)
    n_fit = np.shape(hefs_forward)[0]

    # ------------------------------------------------------------------
    # Error checks
    # ------------------------------------------------------------------
    if kk < 1 or int(kk) != kk:
        raise ValueError("kk is not a valid positive integer")


    # ------------------------------------------------------------------
    # Subset obs_forward arrays to simulation and fitting periods
    # ------------------------------------------------------------------
    # obs_forward for simulation dates (intersection with ixx_gen)
    obs_forward_gen = obs_forward[opt_date_indices, :]  # (n_opt_gen, leads)
    # obs_forward for simulation dates (intersection with ixx_gen)
    obs_forward_fit = obs_forward

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
    gen_knn_data = np.transpose(obs_forward_gen) # (leads, n_gen)
    fit_knn_data = np.transpose(obs_forward_fit) # (leads, n_fit)

    # knn_dist: shape (n_fit, n_gen)
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
    # Scale-decay function (matches R scale_decay_fun)
    # ------------------------------------------------------------------
    dcy = scale_decay_fun(hi, lo, scale_pwr, n_leads)

    # ------------------------------------------------------------------
    # Main scaling loop over sites
    # ------------------------------------------------------------------
    
    # Final array for synthetic ensemble forecasts
    final_synthetic_forecasts = np.full((n_gen,n_leads,n_ens), np.nan, dtype=np.float64)

    # gen_scale, fit_scale: (n_gen, leads)
    gen_scale = obs_forward_gen[:, 1:].copy()
    fit_scale = obs_forward_fit[knn_lst, 1:].copy()
    
    #calculate scaling array
    HEFS_scale = gen_scale / fit_scale  # (n_gen, leads)

    for k in range(n_leads):
        col = HEFS_scale[:, k].copy()
        # handle NaN, Inf, and zeros
        invalid = np.isnan(col) | np.isinf(col) | (col == 0.0)
        col[invalid] = 1.0

        # obs_sc for scaling thresholds
        obs_sc = obs_forward_gen[:,k].copy()
        pos_obs = obs_sc[obs_sc > 0.0]
        if pos_obs.size == 0:
            min_obs = 1.0
        else:
            min_obs = pos_obs.min()
        obs_sc[obs_sc <= 0.0] = min_obs
        obs_sc = np.log(obs_sc)

        # standardized obs
        mu = np.mean(obs_sc)
        sd = np.std(obs_sc)
        
        #scale observations
        obs_scale = (obs_sc - mu) / sd
        
        #calculate activate region of threshold for each row based on the obs value
        ratio_threshold = sigmoid_fun(obs_scale, sig_a, sig_b) * (dcy[k] - 1.0) + 1.0

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
        hefs_fwd_sset = hefs_forward[knn_lst,:,:]
        final_synthetic_forecasts[:, :, e] = hefs_fwd_sset[:,:,e] * HEFS_scale_sm

    #reduce size of final synthetic forecast array
    final_synthetic_forecasts = final_synthetic_forecasts.astype(np.float32)

    # Clean up big intermediates (optional; not compatible with numba)
    ##del hefs_forward_resamp, hefs_forward_resamp_sub
    ##del obs_forward_fit, obs_forward_gen
    ##gc.collect()

    return final_synthetic_forecasts




##########################################################END#################################################################