import numpy as np
import pandas as pd
import gc


def syn_gen(
    seed,                   # random seed
    kk,                     # 
    keysite_label,          # label of the keysite to use
    knn_pwr,                # 
    scale_pwr,              #
    hi,                     #
    lo,                     #
    sig_a,                  #
    sig_b,                  #
    fit_start,              # e.g. "1980-10-01" or pd.Timestamp
    fit_end,                #
    gen_start,              #
    gen_end,                #
    obs_flows,
    obs_forward,
    hefs_forward,                 # shape: (n_sites, n_ens, n_hefs_time, leads)
    ixx_hefs,                     # 1D datetime-like, len = n_hefs_time
    ixx_obs,                       # 1D datetime-like, len = n_obs_forward
    ixx_obs_forward
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
    # Build continuous daily date ranges (UTC)
    # ------------------------------------------------------------------
    gen_start = pd.to_datetime(gen_start, utc=True)
    gen_end = pd.to_datetime(gen_end, utc=True)
    fit_start = pd.to_datetime(fit_start, utc=True)
    fit_end = pd.to_datetime(fit_end, utc=True)

    # generation period
    ixx_gen = pd.date_range(gen_start, gen_end, freq="D", tz="UTC")
    n_gen = len(ixx_gen)

    # fit period
    ixx_fit = pd.date_range(fit_start, fit_end, freq="D", tz="UTC")
    n_fit = len(ixx_fit)

    # ------------------------------------------------------------------
    # Get other variables needed for algorithm
    # ------------------------------------------------------------------

    #convert to datetime objects
    ixx_obs = pd.to_datetime(ixx_obs)
    ixx_obs_forward = pd.to_datetime(ixx_obs_forward)
    ixx_hefs = pd.to_datetime(ixx_hefs)
    
    n_obs, n_sites = obs_flows.shape
    site_names = obs_flows.columns
    n_hefs_sites, n_hefs, leads, n_ens = hefs_forward.shape       # (# sites, # HEFS dates, # leads, # ensemble members)

    # Ensure fit dates lie within available obs_forward index
    ixx_fit = ixx_fit[ixx_fit.isin(ixx_obs_forward)]

    # HEFS dates that also have observed forward-looking data
    mask_hefs_obs_fwd = ixx_hefs.isin(ixx_obs_forward)
    ixx_hefs_obs_fwd = ixx_hefs[mask_hefs_obs_fwd]


    # ------------------------------------------------------------------
    # Error checks
    # ------------------------------------------------------------------
    if kk < 1 or int(kk) != kk:
        raise ValueError("kk is not a valid positive integer")

    if (ixx_gen[0] < ixx_obs_forward[0]) or (ixx_gen[-1] > ixx_obs_forward[-1]):
        raise ValueError("simulation period outside available observational period")

    if (n_hefs_sites != n_sites):
        raise ValueError("# of hefs sites doesn't equal number of obs sites")


    # ------------------------------------------------------------------
    # Subset obs_forward arrays to simulation and fitting periods
    # ------------------------------------------------------------------
    # obs_forward for simulation dates (intersection with ixx_gen)
    mask_gen = ixx_obs_forward.isin(ixx_gen)
    obs_forward_gen = obs_forward[:, mask_gen, :]  # (n_sites, n_gen, leads)
    n_gen_check = obs_forward_gen.shape[1]
    if n_gen_check != n_gen:
        raise ValueError("obs_forward_gen time dimension does not match ixx_gen length")

    # obs_forward for simulation dates (intersection with ixx_gen)
    mask_fit = ixx_obs_forward.isin(ixx_fit)
    obs_forward_fit = obs_forward[:,mask_fit,:]
    n_fit_check = obs_forward_fit.shape[1]
    if n_fit_check != n_fit:
        raise ValueError("obs_forward_gen time dimension does not match ixx_gen length")


    # ------------------------------------------------------------------
    # Subset hefs_forward array for resampling
    # ------------------------------------------------------------------
    # hefs_forward_resamp_sub: subset hefs_forward where ixx_hefs in ixx_obs_forward
    hefs_forward_resamp_sub = hefs_forward[:,mask_hefs_obs_fwd,:, :]  # (n_sites,len(ixx_hefs_obs_fwd),leads,n_ens)

    # Now subset further where ixx_hefs_obs_fwd in ixx_fit
    mask_hefs_fit = np.isin(ixx_hefs_obs_fwd, ixx_fit)
    hefs_forward_resamp = hefs_forward_resamp_sub[:,mask_hefs_fit,:, :]  # (n_sites,n_fit,leads,n_ens)


    # ------------------------------------------------------------------
    # KNN setup
    # ------------------------------------------------------------------
    # weights for neighbor ranks 1..kk
    wts_raw = np.array([1.0 / k for k in range(1, kk + 1)], dtype=float)
    wts = wts_raw / wts_raw.sum()

    # decay weights across leads
    w_leads = np.arange(1, leads + 1, dtype=float)
    decay = (w_leads ** knn_pwr) / np.sum(w_leads ** knn_pwr)


    # ------------------------------------------------------------------
    # KNN distances (based on keysite only)
    # ------------------------------------------------------------------
    keysite = site_names == keysite_label
    if np.sum(keysite)==0:
        raise ValueError("no site name matches keysite_label")

    gen_knn_data = obs_forward_gen[keysite, :, :][-1,:,:].T  # (leads, n_gen)
    fit_knn_data = obs_forward_fit[keysite, :, :][-1,:,:].T  # (leads, n_fit)

    # knn_dist: shape (n_fit, n_gen)
    knn_dist = np.empty((n_fit, n_gen), dtype=float)
    for j in range(n_gen):
        diff = gen_knn_data[:, j][:, None] - fit_knn_data        # (leads, n_fit)
        knn_dist[:, j] = np.sqrt(np.sum(decay[:, None] * (diff ** 2), axis=0))

    # Resampled locations via KNN
    # For each gen day, sample 1 index from nearest kk fits (0-based indices into n_fit)
    knn_lst = np.empty(n_gen, dtype=int)
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
        knn_lst[j] = np.random.choice(neighbors, p=wts)

    # Resampled HEFS dates
    hefs_resamp_vec = ixx_fit[knn_lst]


    # ------------------------------------------------------------------
    # Scale-decay function (matches R scale_decay_fun)
    # ------------------------------------------------------------------
    def scale_decay_fun(hi, lo, pwr, lds):
        w = np.arange(1, lds + 1, dtype=float)
        if pwr != 0:
            win = w[::-1]
            num = np.exp(pwr * win) - np.exp(pwr)
            den = np.exp(2*pwr) - np.exp(pwr)
            dcy = num / den
            dcy_out = dcy / dcy.max() * (hi - lo) + lo
        else:
            # linear from hi down to lo
            step = (hi - lo) / (len(w) - 1)
            dcy_out = hi - np.arange(len(w), dtype=float) * step
        return dcy_out

    dcy = scale_decay_fun(hi, lo, scale_pwr, leads)

    # sigmoid function
    def sigmoid_fun(x, a, b):
        return 1.0 / (1.0 + np.exp(-(x * a + b)))


    # ------------------------------------------------------------------
    # Simple Gaussian-like smoothing (approximate ksmooth with bandwidth=1)
    # ------------------------------------------------------------------
    def ksmooth_1d(x, bandwidth=1.0):
        if bandwidth <= 0:
            return x
        # discrete Gaussian kernel
        radius = int(3 * bandwidth)
        idx = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (idx / bandwidth) ** 2)
        kernel /= kernel.sum()
        return np.convolve(x, kernel, mode="same")


    # ------------------------------------------------------------------
    # Main scaling loop over sites
    # ------------------------------------------------------------------
    
    # Final array for synthetic ensemble forecasts
    final_synthetic_forecasts = np.full((n_sites, n_gen, leads,n_ens), np.nan, dtype=float)
    HEFS_scale_out = np.full((n_sites, n_gen, leads), np.nan, dtype=float)

    for j in range(n_sites):

        # gen_scale, fit_scale: (n_gen, leads)
        gen_scale = obs_forward_gen[j, :, :].copy()
        fit_scale = obs_forward_fit[j, knn_lst, :].copy()

        # Replace zeros with smallest positive in gen_scale
        pos_gen = gen_scale[gen_scale > 0.0]
        if pos_gen.size == 0:
            min_pos = 1.0
        else:
            min_pos = pos_gen.min()

        gen_scale[gen_scale == 0.0] = min_pos
        fit_scale[fit_scale == 0.0] = min_pos

        HEFS_scale = gen_scale / fit_scale  # (n_gen, leads)

        for k in range(leads):
            col = HEFS_scale[:, k].copy()
            # handle NaN, Inf, and zeros
            invalid = np.isnan(col) | np.isinf(col) | (col == 0.0)
            col[invalid] = 1.0

            # obs_sc for scaling thresholds
            obs_sc = obs_forward_gen[j, :, k].copy()
            pos_obs = obs_sc[obs_sc > 0.0]
            if pos_obs.size == 0:
                min_obs = 1.0
            else:
                min_obs = pos_obs.min()
            obs_sc[obs_sc <= 0.0] = min_obs
            obs_sc = np.log(obs_sc)

            # standardized obs
            mu = obs_sc.mean()
            sd = obs_sc.std(ddof=1)
            if sd == 0.0 or np.isnan(sd):
                obs_scale = np.zeros_like(obs_sc)
            else:
                obs_scale = (obs_sc - mu) / sd

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
        kernel = base + [repeat_arr] * (leads - len(base))

        for t in range(n_gen):
            # version with smoothing 
            HEFS_scale_sm[t, :] = np.array([np.convolve(HEFS_scale[t, :], kernel[l], mode='same')[l] for l in np.arange(0,leads)]) 
            #version without smoothing
            #HEFS_scale_sm[t, :] = HEFS_scale[t, :]

        # Apply scaling to each ensemble member
        # hefs_forward_resamp[j, knn_lst, :, e] has shape (n_gen, leads)
        for e in range(n_ens):
            final_synthetic_forecasts[j, :, :, e] = hefs_forward_resamp[j, knn_lst, :, e] * HEFS_scale_sm

        HEFS_scale_out[j, :, :] = HEFS_scale_sm


    #reduce size of final synthetic forecast array
    final_synthetic_forecasts = final_synthetic_forecasts.astype(np.float32)
    HEFS_scale_sm = HEFS_scale_sm.astype(np.float32)

    # Clean up big intermediates (optional)
    del hefs_forward_resamp, hefs_forward_resamp_sub
    del obs_forward_fit, obs_forward_gen
    gc.collect()

    return final_synthetic_forecasts, hefs_resamp_vec, HEFS_scale_out