import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import properscoring as ps
from scipy.optimize import differential_evolution  # or minimize, if you prefer
from datetime import datetime
import gc

import importnb
#import syn_gen function
with importnb.imports("ipynb"):
    from syn_gen import syn_gen 

processed_data_dir = Path('../Data/processed_daily_data')
out_dir = Path('../Data/simulated_data')


# --------------------- Read in key inputs ----------------------------
ixx_hefs = np.load(processed_data_dir / "ixx_hefs.npy",allow_pickle=True)               # the initialization dates for HEFS
ixx_obs = np.load(processed_data_dir / "ixx_obs.npy",allow_pickle=True)                 # the dates for the obs
ixx_obs_forward = np.load(processed_data_dir / "ixx_obs_forward.npy",allow_pickle=True) # the dates for forward looking obs
hefs_forward = np.load(processed_data_dir / "hefs_forward.npy",allow_pickle=True)       # the forward-looking HEFS forecasts for all sites
obs_forward = np.load(processed_data_dir / "obs_forward.npy",allow_pickle=True)         # the forward-looking obs flows for all sites
obs_flows = pd.read_csv(processed_data_dir / "observed_flows.csv")    # matrix of observed flows


gen_start = ixx_hefs[0]
gen_end = ixx_hefs[-1] - pd.Timedelta(days=15)  #to ensure generation over all days with hefs
fit_start = ixx_hefs[0]
fit_end = ixx_hefs[-1] - pd.Timedelta(days=15)  #to ensure generation over all days with hefs

ixx_gen = pd.date_range(gen_start, gen_end, freq="D", tz="UTC").to_numpy(dtype="datetime64[ns]")
ixx_gen= pd.to_datetime(ixx_gen,utc=True)
ixx_gen = ixx_gen.normalize()

#convert to datetime index format
ixx_obs = pd.to_datetime(ixx_obs)
ixx_obs_forward = pd.to_datetime(ixx_obs_forward)
ixx_hefs = pd.to_datetime(ixx_hefs)
ixx_gen = pd.to_datetime(ixx_gen)


# site & dates to optimize on:
keysite_label = "WSDC1" #keysite for synthetic algorithm
site_label = "WSDC1" #site to target optimization to
num_top_dates = 100
verify_dates = pd.Series(obs_flows[site_label].values, index=ixx_obs).nlargest(num_top_dates).index
seed = 1

# --------------------- Function to get average crps ----------------------------
def compute_mean_crps_by_lead(
    site_label,
    verify_dates,
    obs_flows,
    ixx_obs,
    forecasts_forward, #hefs_forward for hefs, syn_fcst for synthetics
    ixx, #ixx_hefs for hefs, ixx_gen for synthetics
):
    """
    Compute mean CRPS by lead time for HEFS forecasts at a single site.

    Returns
    -------
    hefs_mean_crps : np.ndarray of shape (n_leads,)
        Mean CRPS at each lead (1..n_leads).
    """

    verify_dates = pd.to_datetime(verify_dates)

    # basic dims
    n_sites, n_dates, n_leads, n_ens = forecasts_forward.shape

    # site index and obs series
    site_labels = list(obs_flows.columns)
    site_idx = site_labels.index(site_label)
    obs_series = pd.Series(obs_flows[site_label].values, index=ixx_obs)

    crps_sums = np.zeros(n_leads, dtype=float)
    crps_counts = np.zeros(n_leads, dtype=int)

    for date in verify_dates:
        if date not in obs_series.index:
            continue
        obs_val = obs_series[date]
        if not np.isfinite(obs_val):
            continue

        # need enough HEFS history before 'date' to support all leads
        if (date not in ixx) or ((date - pd.Timedelta(days=n_leads)) not in ixx):
            continue

        anchor_idx = np.where(ixx == date)[0][0]

        for l in range(n_leads):  # l = 0..(n_leads-1) => lead l+1 days
            d_idx = anchor_idx - l  # init index for this lead
            if d_idx < 0 or d_idx >= n_dates:
                continue

            ens = forecasts_forward[site_idx, d_idx, l, :]
            score = ps.crps_ensemble(obs_val, ens)
            if np.isfinite(score):
                crps_sums[l] += score
                crps_counts[l] += 1

    # mean CRPS for each lead; np.nan where no data
    mean_crps = np.full(n_leads, np.nan, dtype=float)
    valid = crps_counts > 0
    mean_crps[valid] = crps_sums[valid] / crps_counts[valid]

    return mean_crps

def ensemble_rank(obs, ens):
    """
    Return the rank of obs relative to ensemble ens.
    Rank = number of ensemble members <= obs, in [0, K].
    """
    ens = np.asarray(ens, dtype=float)
    if not np.isfinite(obs):
        return np.nan
    ens = ens[np.isfinite(ens)]
    if ens.size == 0:
        return np.nan
    return np.sum(ens <= obs)


def compute_cumul_rank_histogram_by_lead(
    site_label,
    verify_dates,
    obs_flows,
    ixx_obs,
    forecasts_forward, #hefs_forward for hefs, syn_fcst for synthetics
    ixx, #ixx_hefs for hefs, ixx_gen for synthetics
):
    """
    Compute cumulative rank histogram

    Returns
    -------
    hefs_mean_crps : np.ndarray of shape (n_leads,)
        Mean CRPS at each lead (1..n_leads).
    """
    # --- Basic setup ---
    n_sites, _, n_leads, n_ens = forecasts_forward.shape
    site_labels = list(obs_flows.columns)
    site_idx = site_labels.index(site_label)

    obs_series = pd.Series(obs_flows[site_label].values, index=ixx_obs)

    # Rank bins: 0..n_ens (inclusive)
    num_bins = n_ens + 1
    hs = [i for i in np.arange(1,n_leads+1)]
    rank_bins = np.arange(num_bins)

    # Storage for cumulative counts
    forecast_cumul = {h: np.zeros(num_bins, dtype=float) for h in hs}
    forecast_counts = {h: 0 for h in hs}

    # --- Accumulate rank counts over all verification dates ---
    for date in verify_dates:
        # observation at 'date'
        if date not in obs_series.index:
            continue
        obs_val = obs_series[date]
        if not np.isfinite(obs_val):
            continue

        # ---------- HEFS ----------
        # Ensure we have enough HEFS history before 'date'
        # (you were using (date - n_leads) in ixx_hefs as a guard)
        if (date in ixx) and ((date - pd.Timedelta(days=n_leads)) in ixx):
            anchor_idx = np.where(ixx == date)[0][0]  # index of 'date' in HEFS time grid

            for h in hs:
                l = h - 1  # lead index 0..(n_leads-1)
                if l < 0 or l >= n_leads:
                    continue

                d_idx = anchor_idx - l  # init index = date - l days
                if d_idx < 0:
                    continue

                ens = forecasts_forward[site_idx, d_idx, l, :]
                r = ensemble_rank(obs_val, ens)
                if np.isfinite(r):
                    r = int(r)
                    if 0 <= r < num_bins:
                        forecast_cumul[h][r] += 1
                        forecast_counts[h] += 1


    for h in hs:
        total = forecast_counts[h]
        if total > 0:
            freqs = forecast_cumul[h] / total
            forecast_cumul[h] = np.cumsum(freqs)
        else:
            forecast_cumul[h] = np.full(num_bins, np.nan)

    return forecast_cumul




# --------------------- Function for defining objective function of synthetics ----------------------------

def calc_objective(theta,
                   hefs_mean_crps,
                   hefs_rank_hist,
                   site_label,
                   verify_dates,
                   ixx_gen,
                   # fixed syn_gen args:
                   seed,
                   keysite_label,
                   fit_start,
                   fit_end,
                   gen_start,
                   gen_end,
                   obs_flows,
                   obs_forward,
                   hefs_forward,
                   ixx_hefs,
                   ixx_obs,
                   ixx_obs_forward):

    """
    theta = [kk, knn_pwr, scale_pwr, hi, lo, sig_a, sig_b]
    Returns mean squared difference in CRPS between hefs and synthetics over all verifying dates and all leads.
    """

    kk      = max(5,int(round(theta[0])))  # KNN neighbors (integer)
    knn_pwr = theta[1]
    scale_pwr = theta[2]
    hi      = theta[3]
    lo      = theta[4]
    sig_a   = theta[5]
    sig_b   = theta[6]

    # --- run synthetic generator ---
    syn_fcst, _, _ = syn_gen(
        seed=seed,                          # random seed
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

    syn_mean_crps = compute_mean_crps_by_lead(
        site_label=site_label,
        verify_dates=verify_dates,
        obs_flows=obs_flows,
        ixx_obs=ixx_obs,
        forecasts_forward=syn_fcst, #hefs_forward for hefs, syn_fcst for synthetics
        ixx=ixx_gen, #ixx_hefs for hefs, ixx_gen for synthetics
    )

    syn_rank_hist = compute_cumul_rank_histogram_by_lead(
    site_label=site_label,
    verify_dates=verify_dates,
    obs_flows=obs_flows,
    ixx_obs=ixx_obs,
    forecasts_forward=syn_fcst, #hefs_forward for hefs, syn_fcst for synthetics
    ixx=ixx_gen, #ixx_hefs for hefs, ixx_gen for synthetics
    )

    # objective for crps: mean squared difference between HEFS and synthetic CRPS
    syn_mean_crps = np.asarray(syn_mean_crps, dtype=float)
    hefs_mean_crps = np.asarray(hefs_mean_crps, dtype=float)
    # only compare where both have finite values
    mask = np.isfinite(hefs_mean_crps) & np.isfinite(syn_mean_crps)
    if not np.any(mask):
        return np.inf
    #objective function
    diff = syn_mean_crps[mask] - hefs_mean_crps[mask]
    obj_value1 = np.mean(diff**2)     #optimize crps 


    # objective for rank hist: mean squared difference between HEFS and synthetic rank hist, across all leads
    # keys shared by both dictionaries
    keys = hefs_rank_hist.keys()  # or: hefs_rank_hist.keys() & syn_rank_hist.keys()
    # MSE for each key
    mses_rank_hist = [
        np.mean((hefs_rank_hist[k] - syn_rank_hist[k])**2)
        for k in keys
    ]
    # final metric: average MSE across keys
    obj_value2 = np.mean(mses_rank_hist)

    #final objective (sum of crps and rank hist deviations)
    obj_value = obj_value1 + obj_value2*(10**6)  #reweighting because values are on different scales

    return obj_value

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




#to save time, calculate hefs crps and rank hist, and reach into optimization
hefs_mean_crps = compute_mean_crps_by_lead(
    site_label=site_label,
    verify_dates=verify_dates,
    obs_flows=obs_flows,
    ixx_obs=ixx_obs,
    forecasts_forward=hefs_forward,
    ixx=ixx_hefs,
)

hefs_rank_hist = compute_cumul_rank_histogram_by_lead(
    site_label=site_label,
    verify_dates=verify_dates,
    obs_flows=obs_flows,
    ixx_obs=ixx_obs,
    forecasts_forward=hefs_forward,
    ixx=ixx_hefs,
)

# bounds for parameters
bounds = [
    (19.6, 20), # (5, 50),      # kk (will be rounded to int)
    (-0.01, 0.0), # (-3.0, 0.0),   # knn_pwr - this range covers equal weights for all leads (value of 0) to almost all weight on lead 1 (value of 3)
    (0.1, 0.4),   # (0.01, 3.0) scale_pwr - this range covers a linear decline from lead-1 to lead-15 (value of 0.01), all the way to a rapid decline from lead 1 to lead 2 (value of 3)
    (9.99, 10.0),   # hi - for now, we constrain this one to be some large number (i.e., lead-1 can always scale a lot)
    (1.0, 1.5),   # lo - have upper bound be below hi
    (0.0, 10.0),   # sig_a - should be positive, so that larger flows get more of the scaling, and small flows get reduced scaling
    (-10.0, 10.0),  # sig_b
]

if __name__ == "__main__":
    result = differential_evolution(
        calc_objective,
        bounds=bounds,
        args=(
            hefs_mean_crps,
            hefs_rank_hist,
            site_label,
            verify_dates,
            ixx_gen,
            seed,
            keysite_label,
            fit_start,
            fit_end,
            gen_start,
            gen_end,
            obs_flows,
            obs_forward,
            hefs_forward,
            ixx_hefs,
            ixx_obs,
            ixx_obs_forward
        ),
        callback=de_callback,
        maxiter=7,   # tweak as needed
        popsize=7,
        polish=True,
        workers=-1
    )

    print("Finished!")
    print("Optimized parameters:", result.x)
    print("Minimum mean CRPS:", result.fun)


    best_kk      = int(round(result.x[0]))
    best_knn_pwr = result.x[1]
    best_scale_pwr = result.x[2]
    best_hi      = result.x[3]
    best_lo      = result.x[4]
    best_sig_a   = result.x[5]
    best_sig_b   = result.x[6]