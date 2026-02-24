import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import properscoring as ps
from itertools import combinations
from math import ceil

processed_data_dir = Path('../Data/processed_daily_data')
simulation_dir = Path('../Data/simulated_data')
figure_dir = Path('../Figures')

#sites for time series plots
selected_sites = ["LAMC1F","WSDC1"]
# --------------------- Read in key inputs ----------------------------
ixx_hefs = np.load(processed_data_dir / "ixx_hefs.npy",allow_pickle=True)             # the initialization dates for HEFS
ixx_obs = np.load(processed_data_dir / "ixx_obs.npy",allow_pickle=True)              # the initialization dates for HEFS
ixx_obs_forward = np.load(processed_data_dir / "ixx_obs_forward.npy",allow_pickle=True)              # the initialization dates for HEFS
hefs_forward = np.load(processed_data_dir / "hefs_forward.npy",allow_pickle=True)         # the forward-looking HEFS forecasts for all sites
obs_forward = np.load(processed_data_dir / "obs_forward.npy",allow_pickle=True)
obs_flows = pd.read_csv(processed_data_dir / "observed_flows.csv")    # matrix of observed flows

#load in synthetic forecasts and associated dates
ixx_gen = np.load(simulation_dir / 'ixx_gen.npy',allow_pickle=True)

#convert to datetime index format
ixx_obs = pd.to_datetime(ixx_obs)
ixx_obs_forward = pd.to_datetime(ixx_obs_forward)
ixx_hefs = pd.to_datetime(ixx_hefs)
ixx_gen = pd.to_datetime(ixx_gen)

# these are the forward_looking observations that are available during the hefs period
ixx_obs_forward_in_hefs = ixx_obs_forward[ixx_obs_forward.isin(ixx_hefs)]
obs_forward_in_hefs = obs_forward[:,ixx_obs_forward.isin(ixx_hefs),:]

#read in synthetic forecasts
pattern = "syn_forecast_*.npz"
# Sort files numerically by the index at the end
files = sorted(
    simulation_dir.glob(pattern),
    key=lambda p: int(p.stem.split("_")[-1])
)
synthetic_forecasts = []
for f in files:
    with np.load(f) as data:
        key = data.files[0]       # get the only array in the npz
        synthetic_forecasts.append(data[key])

#read in scaling factors
pattern = "hefs_scaling_factor_*.npz"
# Sort files numerically by the index at the end
files = sorted(
    simulation_dir.glob(pattern),
    key=lambda p: int(p.stem.split("_")[-1])
)
hefs_scaling_factor = []
for f in files:
    with np.load(f) as data:
        key = data.files[0]       # get the only array in the npz
        hefs_scaling_factor.append(data[key])

#read in resampled dates for hefs
pattern = "resampled_dates_*.npz"
# Sort files numerically by the index at the end
files = sorted(
    simulation_dir.glob(pattern),
    key=lambda p: int(p.stem.split("_")[-1])
)
resampled_dates = []
for f in files:
    with np.load(f,allow_pickle=True) as data:
        key = data.files[0]       # get the only array in the npz
        resampled_dates.append(data[key])

### ------------- Time series plots of HEFS and synthetic forecasts for selected dates ------------- ###

def plot_obs_vs_multiple_forecasts(
    date,
    site_label,
    y_label,
    hs,
    obs_flows,
    ixx_obs,
    hefs_forward,
    ixx_hefs,
    synthetic_forecasts=None,   # list of arrays, or None
    synthetic_dates=None,        # list of DatetimeIndex, or None
    figure_dir = None
):
    """
    Plot observed flows and ensemble forecasts for multiple lead times (hs)
    and multiple forecast systems (HEFS + synthetic sets).

    Rows:  1st row = HEFS, subsequent rows = synthetic_forecasts entries
    Cols:  one column per lead time h in hs
    """

    # --- normalize inputs ---
    hs = list(hs)
    if synthetic_forecasts is None:
        synthetic_forecasts = []
    if synthetic_dates is None:
        synthetic_dates = [None] * len(synthetic_forecasts)

    date = pd.to_datetime(date)
    # assume all indices share the same tz as ixx_obs
    if ixx_obs.tz is not None and date.tzinfo is None:
        date = date.tz_localize(ixx_obs.tz)

    #convert synthetic_dates to matching list of dates
    if isinstance(synthetic_dates, pd.DatetimeIndex):
        synthetic_dates = [synthetic_dates] * len(synthetic_forecasts)

    # site index
    site_labels = list(obs_flows.columns)
    site_idx = site_labels.index(site_label)

    # observed series for that site
    obs_series = pd.Series(obs_flows[site_label].values, index=ixx_obs)

    # number of leads (assume same across all)
    n_leads = hefs_forward.shape[2]

    # total rows = 1 (HEFS) + number of synthetic systems
    n_rows = 1 + len(synthetic_forecasts)
    n_cols = len(hs)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=False,
        sharey=True,
        squeeze=False,
    )

    def plot_one_system(ax, fc_arr, fc_dates, row_label, h):
        """
        fc_arr: forecast array (n_sites, n_dates, n_leads, n_ens) or None
        fc_dates: DatetimeIndex of init dates, or None
        row_label: string for panel title prefix
        h: integer (# of days to go back before init date for plot start)
        """
        # number of leads (assume same across all systems)
        n_leads = hefs_forward.shape[2]

        # 1) Define the plotting window
        #    start at (date - h), long enough to include all leads
        start_date = date - pd.Timedelta(days=h)
        valid_dates = start_date + pd.to_timedelta(np.arange(n_leads), unit="D")
        # 2) Observations over full plotting window
        obs_seg = obs_series.reindex(valid_dates)

        # 3) Plot ensemble forecasts ONLY where they exist
        if fc_arr is not None and fc_dates is not None and date in fc_dates:
            date_idx = np.where(fc_dates == start_date)[0][0]
            # all leads for this init
            fc = fc_arr[site_idx, date_idx, :, :]  # (n_leads, n_ens)
            # valid times for each lead: date+1, ..., date+n_leads --> we need to shift forward 1, since the forecasts are always forward looking
            fc_dates_valid = start_date + pd.to_timedelta(np.arange(0, (n_leads)), unit="D")
            for m in range(fc.shape[1]):
                ax.plot(fc_dates_valid, fc[:, m], alpha=0.15)
        
            #if h==1:
            #    print('start_date')
            #    print(start_date)
            #    print('valid_dates')
            #    print(valid_dates)
            #    print('obs_seg')
            #    print(obs_seg)
            #    print('fc_dates_valid')
            #    print(fc_dates_valid)

        # 4) Plot observations across the full window
        ax.plot(valid_dates, obs_seg.values, lw=2, color="k")

        ax.set_title(f"{row_label}, h={h}d", fontsize=9)

    # --- first row: HEFS ---
    for j, h in enumerate(hs):
        ax = axs[0, j]
        plot_one_system(ax, hefs_forward, ixx_hefs, "HEFS", h)
        if j == 0:
            ax.set_ylabel(y_label)


    # --- subsequent rows: synthetic forecasts ---
    for row, (fc_arr, fc_dates) in enumerate(zip(synthetic_forecasts, synthetic_dates), start=1):
        row_label = f"Synthetic {row}"
        for j, h in enumerate(hs):
            ax = axs[row, j]
            plot_one_system(ax, fc_arr, fc_dates, row_label, h)
            if j == 0:
                ax.set_ylabel(y_label)

    for ax in axs[-1, :]:
        ax.set_xlabel("Date")

    for ax in axs.ravel():
        ax.tick_params(axis="x", rotation=45)

    # Turn off x tick labels for all rows except the last
    for r in range(n_rows - 1):
        for c in range(n_cols):
            axs[r, c].set_xticklabels([])
            axs[r, c].tick_params(axis="x", length=0)


    fig.suptitle(f"{site_label} - peak {date.date()}", y=0.98)
    plt.tight_layout()

    # build filename (customize as desired)
    date_str = date.strftime("%Y%m%d")
    fname = f"{site_label}_syn_forecast_init{date_str}.png"
    fig.savefig(figure_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()

#make all plots for selected sites
for selected_site in selected_sites:
    top10_dates = pd.Series(obs_flows[selected_site].values, index=ixx_obs).nlargest(10).index

    for cur_date in top10_dates:
        plot_obs_vs_multiple_forecasts(
            date=cur_date,
            site_label=selected_site,
            hs=[1, 4, 7],
            y_label = 'flow (cfs)',
            obs_flows=obs_flows,
            ixx_obs=ixx_obs,
            hefs_forward=hefs_forward,
            ixx_hefs=ixx_hefs,
            synthetic_forecasts=[synthetic_forecasts[0], synthetic_forecasts[1]],
            synthetic_dates=ixx_gen,
            figure_dir=figure_dir
        )

### ------------- Time series plots of scaling factors used for synthetic forecasts for selected dates ------------- ###

def plot_scaling_factors(
    date,
    site_label,
    y_label,
    hs,
    obs_flows,
    ixx_obs,
    hefs_forward,
    ixx_hefs,
    hefs_scaling_factor=None,   # list of arrays, each # sites, # dates, # lead times
    resampled_dates=None,
    synthetic_dates=None,        # list of DatetimeIndex, or None
    figure_dir = None
):
    """
    Plot hefs scaling factors for multiple lead times (hs) used by synthetic forecasts.

    Rows:  1st row = HEFS, subsequent rows = hefs_scaling_factor entries
    Cols:  one column per lead time h in hs
    """

    # --- normalize inputs ---
    hs = list(hs)
    if hefs_scaling_factor is None:
        hefs_scaling_factor = []
    if resampled_dates is None:
        resampled_dates = []
    if synthetic_dates is None:
        synthetic_dates = [None] * len(hefs_scaling_factor)

    for i in range(0,len(resampled_dates)):
        resampled_dates[i] = pd.to_datetime(resampled_dates[i])

    date = pd.to_datetime(date)
    # assume all indices share the same tz as ixx_obs
    if ixx_obs.tz is not None and date.tzinfo is None:
        date = date.tz_localize(ixx_obs.tz)

    #convert synthetic_dates to matching list of dates
    if isinstance(synthetic_dates, pd.DatetimeIndex):
        synthetic_dates = [synthetic_dates] * len(hefs_scaling_factor)

    #create list of resampled hefs forecasts
    # Precompute a lookup: datetime → index in hefs_forward and obs_forward_in_hefs
    date_to_idx1 = {d: i for i, d in enumerate(ixx_hefs)}
    date_to_idx2 = {d: i for i, d in enumerate(ixx_obs_forward_in_hefs)}
    # Allocate output list
    hefs_forward_resample = []
    obs_forward_resample = []
    for date_vec in resampled_dates:      # each is length n_gen
        # Convert to integer indices into hefs_forward
        idx1 = np.array([date_to_idx1[d] for d in date_vec], dtype=int)
        idx2 = np.array([date_to_idx2[d] for d in date_vec], dtype=int)
        # Reorder the HEFS array along the date dimension
        # hefs_forward shape: (n_sites, n_dates, n_leads, n_ens)
        out1 = hefs_forward[:, idx1, :, :]
        out2 = obs_forward_in_hefs[:, idx2, :]
        hefs_forward_resample.append(out1)
        obs_forward_resample.append(out2)

    # site index
    site_labels = list(obs_flows.columns)
    site_idx = site_labels.index(site_label)

    # observed series for that site
    obs_series = pd.Series(obs_flows[site_label].values, index=ixx_obs)

    # number of leads (assume same across all)
    n_leads = hefs_forward.shape[2]

    # total rows = 1 (HEFS) + number of synthetic systems
    n_rows = 1 + len(hefs_scaling_factor)
    n_cols = len(hs)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=False,
        sharey=True,
        squeeze=False,
    )

    def plot_one_system(ax, fc_arr, fc_dates, row_label, h, obs_arr=None, scaling_factor=None):
        """
        fc_arr: forecast array (n_sites, n_dates, n_leads, n_ens) or None
        fc_dates: DatetimeIndex of init dates, or None
        row_label: string for panel title prefix
        h: integer (# of days to go back before init date for plot start)
        """
        # number of leads (assume same across all systems)
        n_leads = hefs_forward.shape[2]

        # 1) Define the plotting window
        #    start at (date - h), long enough to include all leads
        start_date = date - pd.Timedelta(days=h)
        valid_dates = start_date + pd.to_timedelta(np.arange(n_leads), unit="D")
        # 2) Observations over full plotting window
        obs_seg = obs_series.reindex(valid_dates)
        # 3) Plot ensemble forecasts ONLY where they exist
        if fc_arr is not None and fc_dates is not None and date in fc_dates:
            date_idx = np.where(fc_dates == start_date)[0][0]
            # all leads for this init
            fc = fc_arr[site_idx, date_idx, :, :]  # (n_leads, n_ens)
            # valid times for each lead: date+1, ..., date+n_leads
            fc_dates_valid = start_date + pd.to_timedelta(np.arange(0, n_leads), unit="D")
            for m in range(fc.shape[1]):
                ax.plot(fc_dates_valid, fc[:, m], alpha=0.15)
        # 4) Plot observations across the full window
        ax.plot(valid_dates, obs_seg.values, lw=2, color="k")
        ax.set_title(f"{row_label}, h={h}d", fontsize=9)
        # 5) plot resampled observations
        if obs_arr is not None:
            ob = obs_arr[site_idx, date_idx, :]  # (n_leads)
            ax.plot(fc_dates_valid, ob, lw=1, color="k")
        #6) plot scaling factors
        if scaling_factor is not None:
            scal_fac = scaling_factor[site_idx,date_idx,:]
            ax2 = ax.twinx()
            ax2.plot(fc_dates_valid, scal_fac, lw=1, color="r", label="Scale Factor")
            ax2.set_ylabel("Scaling Factor", color="r")
            ax2.set_ylim(bottom=0)
            ax2.tick_params(axis="y", labelcolor="r")

    # --- first row: HEFS ---
    for j, h in enumerate(hs):
        ax = axs[0, j]
        plot_one_system(ax, hefs_forward, ixx_hefs, "HEFS", h,)
        if j == 0:
            ax.set_ylabel(y_label)


    # --- subsequent rows: resampled hefs ---
    for row, (fc_arr, fc_dates, obs_arr, scaling_factor) in enumerate(zip(hefs_forward_resample, synthetic_dates, obs_forward_resample, hefs_scaling_factor), start=1):
        row_label = f"Synthetic {row}"
        for j, h in enumerate(hs):
            ax = axs[row, j]
            plot_one_system(ax, fc_arr, fc_dates, row_label, h, obs_arr, scaling_factor)
            if j == 0:
                ax.set_ylabel(y_label)

    for ax in axs[-1, :]:
        ax.set_xlabel("Date")

    for ax in axs.ravel():
        ax.tick_params(axis="x", rotation=45)

    # Turn off x tick labels for all rows except the last
    for r in range(n_rows - 1):
        for c in range(n_cols):
            axs[r, c].set_xticklabels([])
            axs[r, c].tick_params(axis="x", length=0)


    fig.suptitle(f"{site_label} - peak {date.date()}", y=0.98)
    plt.tight_layout()

    # build filename (customize as desired)
    date_str = date.strftime("%Y%m%d")
    fname = f"{site_label}_scale_fac_init{date_str}.png"
    fig.savefig(figure_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()

#make all plots for selected sites
for selected_site in selected_sites:
    top10_dates = pd.Series(obs_flows[selected_site].values, index=ixx_obs).nlargest(10).index

    for cur_date in top10_dates:
        plot_scaling_factors(
            date=cur_date,
            site_label=selected_site,
            hs=[1, 4, 7],
            y_label = 'flow (cfs)',
            obs_flows=obs_flows,
            ixx_obs=ixx_obs,
            hefs_forward=hefs_forward,
            ixx_hefs=ixx_hefs,
            hefs_scaling_factor=[hefs_scaling_factor[0], hefs_scaling_factor[1]],   # list of arrays, each # sites, # dates, # lead times
            resampled_dates=[resampled_dates[0], resampled_dates[1]],    
            synthetic_dates=ixx_gen,
            figure_dir=figure_dir
        )

### ------------- Calculate and plot CRPS for select dates at all lead times ------------- ###

def plot_crps_boxplots(
    dates,
    site_label,
    obs_flows,
    ixx_obs,
    hefs_forward,
    ixx_hefs,
    synthetic_forecasts,
    synthetic_dates,
    figure_dir=figure_dir
):
    """
    Compute and plot boxplots of CRPS vs lead time (1..15 days)
    for HEFS and pooled synthetic forecasts.

    Parameters
    ----------
    dates : iterable of date-like
        Init dates to evaluate (e.g., top-10 event dates).
    site_label : str
        Name of the site (column in obs_flows).
    obs_flows : pd.DataFrame
        Observed flows, columns = site labels, index or accompanied by ixx_obs.
    ixx_obs : DatetimeIndex
        Observation dates, same length as obs_flows.
    hefs_forward : np.ndarray
        Shape (n_sites, n_dates_hefs, n_leads, n_ens).
    ixx_hefs : DatetimeIndex
        Forecast initialization dates for HEFS, length = n_dates_hefs.
    synthetic_forecasts : list of np.ndarray
        Each array has shape (n_sites, n_dates_gen_k, n_leads, n_ens).
    synthetic_dates : list of DatetimeIndex
        One DatetimeIndex per synthetic array, giving its init dates.
    """

    # Normalize inputs
    dates = pd.to_datetime(dates)
    ixx_obs = pd.DatetimeIndex(ixx_obs)
    ixx_hefs = pd.DatetimeIndex(ixx_hefs)
    if isinstance(synthetic_dates, pd.DatetimeIndex):
        synthetic_dates = [synthetic_dates] * len(synthetic_forecasts)
    else:
        synthetic_dates = [pd.DatetimeIndex(ix) for ix in synthetic_dates]

    # Align tz for dates with obs index if needed
    if ixx_obs.tz is not None:
        dates = dates.tz_localize(ixx_obs.tz) if dates.tz is None else dates.tz_convert(ixx_obs.tz)

    # Basic setup
    n_sites, _, n_leads, _ = hefs_forward.shape
    site_labels = list(obs_flows.columns)
    site_idx = site_labels.index(site_label)

    # Observed series for this site
    obs_series = pd.Series(obs_flows[site_label].values, index=ixx_obs)

    # Storage for CRPS: one list per lead
    crps_hefs = [[] for _ in range(n_leads)]
    crps_syn = [[] for _ in range(n_leads)]

    # Loop over selected dates
    for date in dates:
        # --- HEFS system ---
        if (date - pd.Timedelta(days=n_leads)) in ixx_hefs:
            for l in range(n_leads):  # l = 0..14 corresponds to lead 1..15
                d_idx = np.where(ixx_hefs == date)[0][0] - l
                if date in obs_series.index:
                    obs_val = obs_series[date]
                    if np.isfinite(obs_val):
                        ens = hefs_forward[site_idx, d_idx, l, :]
                        score = ps.crps_ensemble(obs_val, ens)
                        if np.isfinite(score):
                            crps_hefs[l].append(score)

        # --- pooled synthetic systems ---
        for fc_arr, fc_dates in zip(synthetic_forecasts, synthetic_dates):
            if (date - pd.Timedelta(days=n_leads)) in fc_dates:
                for l in range(n_leads):
                    d_idx = np.where(fc_dates == date)[0][0] - l
                    if date in obs_series.index:
                        obs_val = obs_series[date]
                        if np.isfinite(obs_val):
                            ens = fc_arr[site_idx, d_idx, l, :]
                            score = ps.crps_ensemble(obs_val, ens)
                            if np.isfinite(score):
                                crps_syn[l].append(score)

    # Convert any empty lists to np.nan so boxplot doesn't choke
    crps_hefs = [vals if len(vals) > 0 else [np.nan] for vals in crps_hefs]
    crps_syn = [vals if len(vals) > 0 else [np.nan] for vals in crps_syn]

    # --- Plot boxplots ---
    leads = np.arange(1, n_leads + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # offset positions so HEFS and synthetic appear side-by-side
    pos_hefs = leads - 0.15
    pos_syn = leads + 0.15

    bp1 = ax.boxplot(crps_hefs, positions=pos_hefs, widths=0.25, patch_artist=False)
    bp2 = ax.boxplot(crps_syn, positions=pos_syn, widths=0.25, patch_artist=False)

    # Choose your colors
    color_hefs = "C0"      # blue
    color_syn  = "C1"      # orange

    # --- Color HEFS boxplots ---
    for element in ["boxes", "whiskers", "caps", "medians"]:
        for artist in bp1[element]:
            artist.set_color(color_hefs)
            if element == "medians":
                artist.set_linewidth(2)

    # --- Color synthetic boxplots ---
    for element in ["boxes", "whiskers", "caps", "medians"]:
        for artist in bp2[element]:
            artist.set_color(color_syn)
            if element == "medians":
                artist.set_linewidth(2)

    ax.plot([], [], color=color_hefs, lw=2, label="HEFS")
    ax.plot([], [], color=color_syn,  lw=2, label="Synthetic")


    # --- Add mean markers for HEFS ---
    means_hefs = [np.mean(arr) for arr in crps_hefs]
    ax.scatter(
        pos_hefs, means_hefs,
        color=color_hefs, marker='o', zorder=3, s=20, label='HEFS mean'
    )
    # --- Add mean markers for SYN ---
    means_syn = [np.mean(arr) for arr in crps_syn]
    ax.scatter(
        pos_syn, means_syn,
        color=color_syn, marker='o', zorder=3, s=20, label='Synthetic mean'
    )

    ax.legend()

    ax.set_xticks(leads)
    ax.set_xticklabels(leads)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("CRPS")
    ax.set_title(f"CRPS vs lead time – {site_label}")

    plt.tight_layout()

    # build filename (customize as desired)
    fname = f"{site_label}_CRPS.png"
    fig.savefig(figure_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()

# dates (e.g., top-100 events)
num_top_dates = 100
event_dates = pd.Series(obs_flows[selected_site].values, index=ixx_obs).nlargest(num_top_dates).index

for selected_site_iter in obs_flows.columns:
    plot_crps_boxplots(
        dates=event_dates,
        site_label=selected_site_iter,
        obs_flows=obs_flows,
        ixx_obs=ixx_obs,
        hefs_forward=hefs_forward,
        ixx_hefs=ixx_hefs,
        synthetic_forecasts=synthetic_forecasts,   # list of arrays
        synthetic_dates=ixx_gen,
        figure_dir=figure_dir    # matching init-date indices
    )


### ------------- Calculate and plot cumul. rank histograms for select dates at all lead times ------------- ###

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


def plot_cumulative_rank_histograms(
    dates,
    site_label,
    hs,
    obs_flows,
    ixx_obs,
    hefs_forward,
    ixx_hefs,
    synthetic_forecasts,
    synthetic_dates,
):
    """
    Plot cumulative rank histograms for selected lead times (hs)
    for HEFS and a set of synthetic forecast systems.

    Alignment logic (like your CRPS fix):
      For each verification date 'date' and lead l (0-based):
        - use forecast initialized at date - l
        - use lead index l
        - verify against obs at 'date'
    """

    # --- Normalize inputs ---
    dates = pd.to_datetime(dates)
    ixx_obs = pd.DatetimeIndex(ixx_obs)
    ixx_hefs = pd.DatetimeIndex(ixx_hefs)
    hs = list(hs)

    # Allow a single DatetimeIndex for all synthetic systems
    if isinstance(synthetic_dates, (pd.DatetimeIndex, np.ndarray)):
        synthetic_dates = [pd.DatetimeIndex(synthetic_dates)] * len(synthetic_forecasts)
    else:
        synthetic_dates = [pd.DatetimeIndex(ix) for ix in synthetic_dates]

    # Align tz for dates with obs index if needed
    if ixx_obs.tz is not None:
        if dates.tz is None:
            dates = dates.tz_localize(ixx_obs.tz)
        else:
            dates = dates.tz_convert(ixx_obs.tz)

    # --- Basic setup ---
    n_sites, _, n_leads, n_ens = hefs_forward.shape
    site_labels = list(obs_flows.columns)
    site_idx = site_labels.index(site_label)

    obs_series = pd.Series(obs_flows[site_label].values, index=ixx_obs)

    # Rank bins: 0..n_ens (inclusive)
    num_bins = n_ens + 1
    rank_bins = np.arange(num_bins)

    # Storage for cumulative counts
    hefs_cum = {h: np.zeros(num_bins, dtype=float) for h in hs}
    syn_cum_all = [
        {h: np.zeros(num_bins, dtype=float) for h in hs}
        for _ in synthetic_forecasts
    ]

    hefs_counts = {h: 0 for h in hs}
    syn_counts = [{h: 0 for h in hs} for _ in synthetic_forecasts]

    # --- Accumulate rank counts over all verification dates ---
    for date in dates:
        # observation at 'date'
        if date not in obs_series.index:
            continue
        obs_val = obs_series[date]
        if not np.isfinite(obs_val):
            continue

        # ---------- HEFS ----------
        # Ensure we have enough HEFS history before 'date'
        # (you were using (date - n_leads) in ixx_hefs as a guard)
        if (date in ixx_hefs) and ((date - pd.Timedelta(days=n_leads)) in ixx_hefs):
            anchor_idx = np.where(ixx_hefs == date)[0][0]  # index of 'date' in HEFS time grid

            for h in hs:
                l = h - 1  # lead index 0..(n_leads-1)
                if l < 0 or l >= n_leads:
                    continue

                d_idx = anchor_idx - l  # init index = date - l days
                if d_idx < 0:
                    continue

                ens = hefs_forward[site_idx, d_idx, l, :]
                r = ensemble_rank(obs_val, ens)
                if np.isfinite(r):
                    r = int(r)
                    if 0 <= r < num_bins:
                        hefs_cum[h][r] += 1
                        hefs_counts[h] += 1

        # ---------- Synthetic systems ----------
        for sys_idx, (fc_arr, fc_dates) in enumerate(zip(synthetic_forecasts, synthetic_dates)):
            fc_dates = pd.DatetimeIndex(fc_dates)
            if (date in fc_dates) and ((date - pd.Timedelta(days=n_leads)) in fc_dates):
                anchor_idx = np.where(fc_dates == date)[0][0]

                for h in hs:
                    l = h - 1
                    if l < 0 or l >= fc_arr.shape[2]:
                        continue

                    d_idx = anchor_idx - l  # init index = date - l days
                    if d_idx < 0:
                        continue

                    ens = fc_arr[site_idx, d_idx, l, :]
                    r = ensemble_rank(obs_val, ens)
                    if np.isfinite(r):
                        r = int(r)
                        if 0 <= r < num_bins:
                            syn_cum_all[sys_idx][h][r] += 1
                            syn_counts[sys_idx][h] += 1

    # --- Convert counts -> cumulative relative frequencies ---
    # HEFS
    for h in hs:
        total = hefs_counts[h]
        if total > 0:
            freqs = hefs_cum[h] / total
            hefs_cum[h] = np.cumsum(freqs)
        else:
            hefs_cum[h] = np.full(num_bins, np.nan)

    # Synthetic systems
    syn_cum_arrays = {h: [] for h in hs}  # h -> list of cum arrays (one per system)
    for sys_idx in range(len(synthetic_forecasts)):
        for h in hs:
            total = syn_counts[sys_idx][h]
            if total > 0:
                freqs = syn_cum_all[sys_idx][h] / total
                cum = np.cumsum(freqs)
            else:
                cum = np.full(num_bins, np.nan)
            syn_cum_arrays[h].append(cum)

    # --- Plotting ---
    n_cols = len(hs)
    fig, axs = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=True, squeeze=False)

    color_hefs = "C0"
    color_syn = "C1"

    for j, h in enumerate(hs):
        ax = axs[0, j]

        # HEFS line
        ax.plot(rank_bins, hefs_cum[h], color=color_hefs, lw=2, label="HEFS")

        # Synthetic band: min–max across systems
        syn_cums = np.array(syn_cum_arrays[h])  # shape (n_sys, num_bins)
        if np.isfinite(syn_cums).any():
            syn_min = np.nanmin(syn_cums, axis=0)
            syn_max = np.nanmax(syn_cums, axis=0)
            ax.fill_between(rank_bins, syn_min, syn_max, color=color_syn, alpha=0.3, label="Synthetic range")

        # --- 1–1 reference line ---
        K = rank_bins[-1]   # number of ensemble members
        one2one = rank_bins / K
        ax.plot(rank_bins, one2one, "--", color="gray", linewidth=1.2, alpha=0.7)

        ax.set_xlabel("Rank")
        if j == 0:
            ax.set_ylabel("Cumulative probability")
        ax.set_title(f"Lead {h} day(s)")
        ax.set_ylim(0, 1)

    # One legend for the whole figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # build filename (customize as desired)
    fname = f"{site_label}_rank_hist.png"
    fig.savefig(figure_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()



# dates (e.g., top-100 events) and lead times:
num_top_dates = 100
event_dates = pd.Series(obs_flows[selected_site].values, index=ixx_obs).nlargest(num_top_dates).index
hs = [1, 4, 7]

for selected_site_iter in obs_flows.columns:
    plot_cumulative_rank_histograms(
        dates=event_dates,
        site_label=selected_site_iter,
        hs=hs,
        obs_flows=obs_flows,
        ixx_obs=ixx_obs,
        hefs_forward=hefs_forward,
        ixx_hefs=ixx_hefs,
        synthetic_forecasts=synthetic_forecasts,  # list of arrays
        synthetic_dates=ixx_gen                   # single DatetimeIndex used for all
    )



### ------------- Calculate and plot pairwise site correlations of CRPS values for all lead times ------------- ###

def plot_crps_site_corr_panels(
    dates,
    obs_flows,
    ixx_obs,
    hefs_forward,
    ixx_hefs,
    synthetic_forecasts,
    synthetic_dates,
    figure_dir=figure_dir,
    fname="CRPS_site_corr_panels.png"
):
    """
    For each lead time, compute CRPS at all sites (HEFS and pooled synthetic),
    then compute pairwise correlations of CRPS across sites separately for
    HEFS and synthetics, and plot HEFS vs Synthetic pairwise correlations.

    Parameters
    ----------
    dates : iterable of date-like
        Verification dates to evaluate (same role as in your original function).
    obs_flows : pd.DataFrame
        Observed flows, columns = site labels, index or accompanied by ixx_obs.
    ixx_obs : DatetimeIndex
        Observation dates, same length as obs_flows.
    hefs_forward : np.ndarray
        Shape (n_sites, n_dates_hefs, n_leads, n_ens).
    ixx_hefs : DatetimeIndex
        Forecast initialization dates for HEFS, length = n_dates_hefs.
    synthetic_forecasts : list of np.ndarray
        Each array has shape (n_sites, n_dates_gen_k, n_leads, n_ens).
    synthetic_dates : list of DatetimeIndex or a single DatetimeIndex
        One DatetimeIndex per synthetic array, giving its init dates. If a single
        DatetimeIndex is passed, it is reused for all synthetic arrays.
    figure_dir : pathlib.Path
        Directory to save the figure.
    fname : str
        Filename for the figure.
    """

    # --- Normalize inputs ---
    dates = pd.to_datetime(dates)
    ixx_obs = pd.DatetimeIndex(ixx_obs)
    ixx_hefs = pd.DatetimeIndex(ixx_hefs)

    # Make sure obs_flows has a proper datetime index
    obs_df = obs_flows.copy()
    if not isinstance(obs_df.index, pd.DatetimeIndex) or not obs_df.index.equals(ixx_obs):
        obs_df.index = ixx_obs

    # Normalize synthetic dates
    if isinstance(synthetic_dates, pd.DatetimeIndex):
        synthetic_dates = [synthetic_dates] * len(synthetic_forecasts)
    else:
        synthetic_dates = [pd.DatetimeIndex(ix) for ix in synthetic_dates]

    # Align tz for dates with obs index if needed
    if ixx_obs.tz is not None:
        if dates.tz is None:
            dates = dates.tz_localize(ixx_obs.tz)
        else:
            dates = dates.tz_convert(ixx_obs.tz)

    # --- Basic setup ---
    n_sites, _, n_leads, _ = hefs_forward.shape
    site_labels = list(obs_df.columns)

    # Storage: for each lead, keep a dict date -> CRPS vector (length n_sites)
    hefs_scores_by_lead = [dict() for _ in range(n_leads)]
    syn_scores_by_lead  = [dict() for _ in range(n_leads)]

    # --- Main loop over verification dates ---
    for date in dates:
        # Skip if we don't have observations for this date
        if date not in obs_df.index:
            continue

        obs_row = obs_df.loc[date].values  # shape (n_sites,)

        # Skip if all obs are NaN
        if not np.any(np.isfinite(obs_row)):
            continue

        # For each lead l (0..n_leads-1 corresponds to lead 1..n_leads)
        for l in range(n_leads):
            # HEFS availability check
            if (date - pd.Timedelta(days=n_leads)) not in ixx_hefs:
                # no HEFS forecasts that far back
                continue

            # find the initialization date index whose forecast verifies on "date" at lead l
            # ixx_hefs[d_idx] is the init date, and ixx_hefs[d_idx] + l days = date
            # equivalent to d_idx = index of date, then minus l
            idx_matches = np.where(ixx_hefs == date)[0]
            if len(idx_matches) == 0:
                continue
            d_idx_hefs = idx_matches[0] - l
            if d_idx_hefs < 0:
                continue

            # --- HEFS CRPS for all sites ---
            ens_hefs = hefs_forward[:, d_idx_hefs, l, :]  # shape (n_sites, n_ens)
            crps_hefs_vec = np.full(n_sites, np.nan)

            for s in range(n_sites):
                obs_val = obs_row[s]
                if np.isfinite(obs_val):
                    ens = ens_hefs[s, :]
                    score = ps.crps_ensemble(obs_val, ens)
                    if np.isfinite(score):
                        crps_hefs_vec[s] = score

            # Only keep if at least one site has a finite CRPS
            if np.any(np.isfinite(crps_hefs_vec)):
                hefs_scores_by_lead[l][date] = crps_hefs_vec

            # --- Synthetic CRPS (pooled ensembles across generators) ---
            syn_ens_list = []
            for fc_arr, fc_dates in zip(synthetic_forecasts, synthetic_dates):
                # Check availability for this synthetic system at this verification date
                if (date - pd.Timedelta(days=n_leads)) not in fc_dates:
                    continue
                idx_syn = np.where(fc_dates == date)[0]
                if len(idx_syn) == 0:
                    continue
                d_idx_syn = idx_syn[0] - l
                if d_idx_syn < 0:
                    continue

                syn_ens_list.append(fc_arr[:, d_idx_syn, l, :])  # (n_sites, n_ens_k)

            if len(syn_ens_list) > 0:
                # Pooled synthetic ensembles: concatenate along ensemble dimension
                pooled_syn_ens = np.concatenate(syn_ens_list, axis=-1)  # (n_sites, n_ens_total)
                crps_syn_vec = np.full(n_sites, np.nan)
                for s in range(n_sites):
                    obs_val = obs_row[s]
                    if np.isfinite(obs_val):
                        ens = pooled_syn_ens[s, :]
                        score = ps.crps_ensemble(obs_val, ens)
                        if np.isfinite(score):
                            crps_syn_vec[s] = score

                if np.any(np.isfinite(crps_syn_vec)):
                    syn_scores_by_lead[l][date] = crps_syn_vec

    # --- Compute pairwise correlations for each lead ---
    pair_corr_hefs = []   # list of arrays, one per lead
    pair_corr_syn  = []

    for l in range(n_leads):
        # Build DataFrames: rows = dates, columns = sites
        if len(hefs_scores_by_lead[l]) == 0 or len(syn_scores_by_lead[l]) == 0:
            # no data at this lead
            pair_corr_hefs.append(np.array([]))
            pair_corr_syn.append(np.array([]))
            continue

        hefs_df = pd.DataFrame.from_dict(
            hefs_scores_by_lead[l],
            orient="index",
            columns=site_labels
        )

        syn_df = pd.DataFrame.from_dict(
            syn_scores_by_lead[l],
            orient="index",
            columns=site_labels
        )

        # Correlation matrices across sites (Pearson, pairwise complete)
        corr_hefs = hefs_df.corr()
        corr_syn  = syn_df.corr()

        # Extract upper triangle (site pairs)
        hefs_vals = []
        syn_vals  = []
        for i, j in combinations(range(n_sites), 2):
            c_h = corr_hefs.iloc[i, j]
            c_s = corr_syn.iloc[i, j]
            if np.isfinite(c_h) and np.isfinite(c_s):
                hefs_vals.append(c_h)
                syn_vals.append(c_s)

        pair_corr_hefs.append(np.array(hefs_vals))
        pair_corr_syn.append(np.array(syn_vals))

    # --- Plot HEFS vs Synthetic pairwise correlations for each lead ---
    # Determine a reasonable panel layout (e.g., 3 columns)
    n_leads_effective = n_leads
    n_cols = 5
    n_rows = ceil(n_leads_effective / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for l in range(n_leads_effective):
        ax = axes[l // n_cols, l % n_cols]

        x = pair_corr_hefs[l]
        y = pair_corr_syn[l]

        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"Lead {l+1}")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.axhline(0, color="0.8", lw=1)
            ax.axvline(0, color="0.8", lw=1)
            continue

        ax.scatter(x, y, alpha=0.7, s=20)

        # 1:1 line
        ax.plot([-1, 1], [-1, 1], ls="--", lw=1, color="0.3")

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("HEFS pairwise corr (CRPS across sites)")
        ax.set_ylabel("Synthetic pairwise corr (CRPS across sites)")
        ax.set_title(f"Lead {l+1} (n={len(x)})")

    # Turn off any unused subplots
    for k in range(n_leads_effective, n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(figure_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Optionally return the raw correlation arrays if you want to analyze them
    return pair_corr_hefs, pair_corr_syn



plot_crps_site_corr_panels(
    dates=event_dates,
    obs_flows=obs_flows,
    ixx_obs=ixx_obs,
    hefs_forward=hefs_forward,
    ixx_hefs=ixx_hefs,
    synthetic_forecasts=synthetic_forecasts,
    synthetic_dates=ixx_gen,
    figure_dir=figure_dir,
    fname="CRPS_site_corr_panels.png"
)

