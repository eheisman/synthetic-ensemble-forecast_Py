import os
import sys
sys.path.insert(0, os.path.abspath('./src'))
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed
from hecdss import HecDss
from hecdss import RegularTimeSeries
from util import water_day
import json
import random
import calendar

#record complete time 
now=datetime.now()
print('gen start',now.strftime("%H:%M:%S"))

#numba compatible function import 
from syn_gen_hec_wat_fra import syn_gen_hec_wat_fra,obs_fwd_fun

data_dir = Path('./data')

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# User defined specifications
# site & dates to optimize on:
loc                 = 'ADO'                 #system level location descriptor (e.g. YRS for Yuba-Feather system, etc)
keysite_label       = 'ADOC1'               #keysite for synthetic generation; sets indexing across sites to preserve spatial correlations 
gen_site            = 'ADOC1'               #specific site being generated for HEC-WAT input events

#optimization parameters (set these to match the desired optimization run from the 'optimize_synthetic_forecasts.py' script)
max_lds             = 15            #number of daily lead times to optimize to (default is total number of leads in hindcast dataset)
opt_pct             = 0.99          #percentile of data to optimize to (e.g., 0.9 = optimize to top 1% of events by flow magnitude)
fixed_kk            = True          #use fixed k value for knn sampling?
fixed_knn_pwr       = True          #use a fixed knn_pwr value for knn sampling?
fix_kk              = 20            #if fixed_kk = True, what value to use (default: 20)
fix_knn_pwr         = -0.5          #if fixed_knn_pwr = True, what value to use (default: -0.5)

#NOTE: the optimization parameter .pkl file always includes a value for 'fix_kk' and 'fix_knn_pwr', even if 'fixed_kk' and 'fixed_knn_pwr' are set to False
#If using the fixed_kk and fixed_knn_pwr set to 'False' in optimization, the 'fix_kk' and 'fix_knn_pwr' values are meaningless, but should match what was set in the optimization script

#generation settings
fit_gen_strategy    = 'default'     #set to 'default' to fit all available paired hindcast/obs_fwd and gen all available obs_fwd; set to 'specify' to set yourself
workers             = 10            #number of cores to utilize in parallel; 50 works on Hopper w/ADO test case; reduce as needed to not overload memory
n_events            = 50            #the number of simulated events for each lifecycle from HEC-WAT; could be added to json file for consistency

#Specify dates settings (only used if fit_gen_strategy is set to 'specify')
#Note: fit period has to include both hefs and obs_fwd data; will error if not inclusive
st_fit              = '1990-10-01'
en_fit              = '2018-09-30'
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1. Open and read JSON configuration file
#json_path = 'e:/CW3E/projects/HEC-WAT_synthetic-forecasts/runs/HFO/FRA_50yr/realization 1/lifecycle 1/event 1/rScriptConfig.json'

"""
#current json path for testing; would likely come as input from the 'callSynForecast' function in HEC-WAT scripting or something like that
"""
json_path = data_dir / ('./%s/SynForecastConfig.json' %(loc)) 
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
print(data) # Outputs the parsed Python object
"""
#current json path for testing; would likely come as input from the 'callSynForecast' function in HEC-WAT scripting or something like that
"""

#retrieve key json config elements
realization = data['Indices']['Realization Number']
lifecycle = data['Indices']['Lifecycle Number']
event = data['Indices']['Event Number']

realization_seed = data['Randoms']['Realization Random']
lifecycle_seed = data['Randoms']['Lifecycle Random']
event_seed = data['Randoms']['Event Random']

dss_file = data['Outputs']['DSS File']
watershed_dir = data['Outputs']['Watershed Directory']
F_part = data['Outputs']['F Part']
run_dir = data['Outputs']['Run Directory']

#set random seed for event
random.seed(event_seed)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 2. Setup file paths
#path to optimized parameters
opt_dir = Path('./out/%s/keysite=%s' %(loc,keysite_label))

#create directory for output
out_dir = Path('./out/%s/site=%s_hec-wat-fra_realization=%s_lifecycle=%s' %(loc,gen_site,realization,lifecycle))
os.makedirs(out_dir,exist_ok=True)
    
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 3. Download and extract key data elements for the synthetic forecast code
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

outfile = './optimized-parameters_keysite=%s_opt-pct=%s_fixed-kk=%s_kk=%s_fixed-knn-pwr=%s_knn-pwr=%s.pkl' %(keysite_label,opt_pct,fixed_kk,fix_kk,fixed_knn_pwr,fix_knn_pwr)
opt_pars = pickle.load(open(opt_dir / outfile,'rb'),encoding='latin1')

# Optimized parameters from the optimize_synthetic_forecasts script
cur_seed    = event_seed
kk          = opt_pars['kk']       
knn_pwr     = opt_pars['knn_pwr'] 
scale_pwr   = opt_pars['scale_pwr'] 
hi          = opt_pars['hi'] 
lo          = opt_pars['lo']       # 1.4 
sig_a       = opt_pars['sig_a'] 
sig_b       = opt_pars['sig_b'] 

keysite_idx = np.where(sites==keysite_label)[0][0]      #set keysite index for syn_gen code (indexing site to preserve spatial correlations)
site_idx = np.where(sites==gen_site)[0][0]              #set site index for syn_gen code    (site where synthetic forecasts are being generated)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 4. Set datetime indices and clean data inputs for synthetic generation

#set datetime array for fit and gen periods
#ensure fit dataset includes all dates in both hefs and obs_fwd date/time groups
fit_start = max(hcst_dtg[0],obs_fwd_dtg[0])
fit_end = min(hcst_dtg[-1],obs_fwd_dtg[-1])

ixx_fit = pd.date_range(fit_start, fit_end, freq="D").to_numpy(dtype="datetime64[us]")

if fit_gen_strategy == 'specify':
    fit_dates = pd.date_range(st_fit, en_fit, freq="D").to_numpy(dtype="datetime64[us]")
    if fit_dates[0] < ixx_fit[0] or fit_dates[-1] > ixx_fit[-1]:
        raise ValueError("Specified fit dates outside available hindcast/obs_fwd dataset")
    ixx_fit = fit_dates

#index synthetic generation arrays
#arrays for calibration (fit) dataset observation and hindcast pairs
obs_fwd_fit = obs_fwd[site_idx,np.isin(obs_fwd_dtg,ixx_fit),:]
hcst_fit = hcst[site_idx,np.isin(hcst_dtg,ixx_fit),:,:]

#calculate a daily mean across obs_data (used to concatenate additional days to the synthetic events in section 5 below)
obs_dtg = pd.to_datetime(obs_fwd_dtg)
dowy_obs = np.array([water_day(d,calendar.isleap(d.year)) for d in obs_dtg])
dly_mean = np.zeros(np.max(dowy_obs))
for i in range(len(dly_mean)):
    dowy_idx = np.where(dowy_obs == i)
    dly_mean[i] = np.mean(obs_fwd[site_idx,dowy_idx,0])

#remove any bad forecast days from the calibration (fit) datasets
rmv_idx = np.isin(ixx_fit,bad_forcs)
obs_fwd_fit = np.delete(obs_fwd_fit,rmv_idx,axis=0)
hcst_fit = np.delete(hcst_fit,rmv_idx,axis=0)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 5. Read the HEC-WAT hydrologic sample DSS output file, extract 'n_events' synthetic hydrologic event, and format for synthetic forecast generation

#read in data
"""
#current dss file path and data for testing; these data should come from the json file or be provided as a script input from the 'callSynForecast' function
"""
##dss_file = 'E:/CW3E/projects/HEC-WAT_synthetic-forecasts/runs/HFO/FRA_50yr/realization 1/lifecycle 1/HFO-FRA_50yr.dss' #this was the original path to the run
dss_file = './data/%s/HFO-FRA_50yr.dss' %(loc) #shortened path for the repo
dss_data_file = '//Prado_IN-combined/Flow//15Minute/C:000001|HFO:FRA_50yr:Scripting-Hydrograph_unscaler/'
##dss_data_file = '//Prado IN-SAR/FLOW//15Minute/C:000001|HFO:FRA_50yr:HydroSampl-PradoHS/'
"""
#current dss file path and data for testing; these data should come from the json file or be provided as a script input from the 'callSynForecast' function
"""

#load the DSS file of synthetic events for first event to extract key info
dss = HecDss(dss_file)

#extract information elements from the DSS file (note: much of this is redundant with the procedure in the loop below, but left for now for demonstration purposes)
parts = dss_data_file.split('/')[1:]
alphabet = [chr(i) for i in range(ord('a'), ord('z') + 1)][:len(parts)]
part_dict = {alphabet[i]:parts[i] for i in range(len(parts))}
data = dss.get(dss_data_file)
print(data.start_date,data.units,data.data_type)
dtg = pd.to_datetime(data.times)
flow_kcfs = data.values / 1000

#configure data to daily timeseries aggregated across 12 - 12 UTC
dtg_shift = dtg + pd.Timedelta(hours=12)                #shifting forward by 12 hours allows aggregation function to aggregate from 12-12 GMT
flow_series = pd.Series(flow_kcfs, index=dtg_shift)
flow_daily = flow_series.resample("D").mean()           #aggregate to daily mean values
reindex = flow_daily.index - pd.Timedelta(hours=12)     #shifting 12 hours back to original timing to ensure output file is consistent with HEC-WAT input
final_flow_daily = pd.Series(flow_daily.values,index=reindex)

#extract each of the 50 synthetic events from the DSS file and save to an 'obs_fwd' configured array
#array to store obs_fwd arrays
obs_fwd_gen_mat = np.full((n_events,len(flow_daily),max_lds+1),np.nan,dtype=np.float64)
#dictionary to store the date/time sequence for each synthetic event
fcst_issue_dates = {}
for i in range(n_events):
    i_idx = i+1
    evt_num = f'{i_idx:02}'
    data = dss.get('//Prado_IN-combined/Flow//15Minute/C:0000%s|HFO:FRA_50yr:Scripting-Hydrograph_unscaler/' %(evt_num))
    dtg = pd.to_datetime(data.times)
    flow_kcfs = data.values / 1000

    #configure data to daily timeseries aggregated across 12 - 12 UTC
    dtg_shift = dtg + pd.Timedelta(hours=12)
    flow_series = pd.Series(flow_kcfs, index=dtg_shift)
    flow_daily = flow_series.resample("D").mean()
    reindex = flow_daily.index - pd.Timedelta(hours=12)
    
    #add 'max_lds' number of days of daily mean to each synthetic obs sequence to support 'obs_fwd' generation across the entire synthetic sequence
    ext_dates = pd.date_range(pd.to_datetime(flow_daily.index[-1]) + pd.Timedelta(hours=12),pd.to_datetime(flow_daily.index[-1]) + pd.Timedelta(hours=(max_lds-1)*24+12),freq='D')
    dowy_concat = np.array([water_day(d,calendar.isleap(d.year)) for d in ext_dates])
    concat_flows = dly_mean[dowy_concat] #
    flow_daily_concat = np.concat((flow_daily.values,concat_flows))
                                  
    fcst_issue_dates[evt_num] = reindex
    obs_fwd_gen_mat[i,:,:] = obs_fwd_fun(flow_daily_concat,max_lds)

dss.close()

#to prevent instabilities in generation (e.g. divide by zero), set all obs and obs_fwd zeros to min non-zero value
min_nzero_vec = obs_fwd[obs_fwd>0.0]
min_nzero = min(min_nzero_vec)
obs_fwd_fit[obs_fwd_fit==0.0] = min_nzero
obs_fwd_gen_mat[obs_fwd_gen_mat==0.0] = min_nzero

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 6. Setup the synthetic forecast generation function and run in parallel to generate synthetic forecasts for each HEC-WAT synthetic event

#function to generate synthetic forecast samples and save an output .npz file for each sample
def syn_gen_par(i):
    out = syn_gen_hec_wat_fra(
        seed=i,                    
        kk=kk,                                  
        knn_pwr=knn_pwr,                              
        scale_pwr=scale_pwr,                            
        hi=hi,                                     
        lo=lo,                                     
        sig_a=sig_a,                                  
        sig_b=sig_b,                                 
        ixx_fit=ixx_fit,
        obs_fwd_fit=obs_fwd_fit,
        obs_fwd_gen=obs_fwd_gen_mat[i,:,:],
        hcst_fit=hcst_fit
    )
    
    outfile = './syn-forecast_site=%s_lifecycle=%s_event=%s.npz' %(gen_site,lifecycle,i+1)
    np.savez(out_dir/outfile,syn_fcst=out)

    return out

#run synthetic generation code in parallel; par_out is a list of length 'n_events' where each list element is a n_obs x n_leads+1 x n_ens array
#Note: the array is configured to have index 0 in dimension 2 as the day t observation for compatibility with HEC-WAT; this is why dimension 2 is n_leads+1
par_out = Parallel(n_jobs=workers)(delayed(syn_gen_par)(i) for i in range(n_events))


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 7. Output both the aggregated daily obs file and the synthetic forecast sequence to a DSS file for each HEC-WAT synthetic event

#extract the ensemble number value from the hindcast dataset
n_ens = np.shape(hcst)[3]

#define a function for Parallel processing 
def dss_out_par(i):
    i_idx = i+1
    evt_num = f'{i_idx:02}' #0-padded event number
    #outfiles for the synthetic forecast and daily synthetic obs
    dss_outfile = '/syn-forecast_site=%s_lifecycle=%s_records=%s.dss' %(gen_site,lifecycle,evt_num)
    dss_obs_outfile = '/syn-obs_site=%s_lifecycle=%s_records=%s.dss' %(gen_site,lifecycle,evt_num)
    #set directories for the outfiles
    odir = './out/%s/site=%s_hec-wat-fra_realization=%s_lifecycle=%s' %(loc,gen_site,realization,lifecycle)
    outdss = odir + dss_outfile
    outdss_obs = odir + dss_obs_outfile
    
    #output each daily-aggregated synthetic observation to a DSS file
    obs_outpath = '//Prado_IN-combined/Flow//1Day/C:0000%s|HFO:FRA_50yr:Scripting-Hydrograph_unscaler/' %(evt_num)
    times = fcst_issue_dates[evt_num]
    obs_values = obs_fwd_gen_mat[i,:,0] * 1000      #reset to cfs
    obsValuesAsList = obs_values.tolist()
    outTimeSeriesForThisTrace = RegularTimeSeries.create(obsValuesAsList, data_type='PER-AVE',times=times, start_date=fcst_issue_dates[evt_num][0], interval='1Day', units="cfs", path=obs_outpath)   # this assumes traceValuesAsList is a list of flows, start_date is the date fo the first timestep in the sequence 
    with HecDss(outdss_obs) as outDssObs:
        outDssObs.put(outTimeSeriesForThisTrace)
        
    #output the synthetic forecast sequence for each synthetic event to a DSS file for HEC-WAT ingest
    with HecDss(outdss) as outDss:
        for j in range(len(fcst_issue_dates[evt_num])):
            #format F_part datetime groups
            fcstIssueDate_short = fcst_issue_dates[evt_num][j].strftime("%Y%m%d-%H%M")
            fcstIssueDate = fcst_issue_dates[evt_num][j].strftime("%Y%m%d-%H%M%S")
            #time vector for the RegularTimeSeries file
            times = pd.date_range(fcst_issue_dates[evt_num][j],fcst_issue_dates[evt_num][j] + pd.Timedelta(days=max_lds),freq='D')
            for k in range(n_ens):
                #format the required elements of the output record name for DSS
                ensembleMemberID = k+1
                # Do the following for _each_ ensemble member in _each_ forecast issued.   Example pathname we are looking to create from TSEnsemble library "//Kanektok.BCAC1/flow/01Nov2013/1Hour/C:000007|T:20131103-1200|V:20131103-120000|/"
                a = part_dict['a']
                b = part_dict['b']
                c = part_dict['c']
                d = fcst_issue_dates[evt_num][0].strftime("%d%b%Y")
                ePart = "1Day" # for daily timestep forecast, "6Hour" for 6 hour steps, "1Hour", for hourly, "15Min" for 15Min data... there's a list but this must match.
                fPart = "C:%06d|T:%s|V:%s|%s" % (
                    ensembleMemberID,       # ensemble member number
                    fcstIssueDate_short, 
                    fcstIssueDate,          # for a study, the T/V values are identical.
                    F_part[9:]              # F part label from json file
                    )
                dssOutPath = "/".join(["",a,b,c,d,ePart,fPart,""])          #combine all part labels for the record name
                traceValues = par_out[i][j,:,k] * 1000                      #reset values to kcfs
                traceValuesAsList = traceValues.tolist()                    #RegularTimeSeries requires the flow values as a list
                outTimeSeriesForThisTrace = RegularTimeSeries.create(traceValuesAsList, data_type='PER-AVE',times=times, start_date=fcst_issue_dates[evt_num][j], interval=ePart, units="cfs", path=dssOutPath)   # this assumes traceValuesAsList is a list of flows, start_date is the date fo the first timestep in the sequence 
                outDss.put(outTimeSeriesForThisTrace)
    outDss.close()

#Parallelize the output of each synthetic observation and synthetic forecast file
Parallel(n_jobs=workers)(delayed(dss_out_par)(i) for i in range(n_events))

#--------------------------------------------------------------------------------
#check dss files
dss_infile = '/syn-forecast_site=%s_lifecycle=%s_records=%s.dss' %(gen_site,lifecycle,'50')
indir = './out/%s/site=%s_hec-wat-fra_realization=%s_lifecycle=%s' %(loc,gen_site,realization,lifecycle)
indss = indir + dss_infile
dss_chk = HecDss(indss)
print(f" record_count = {dss_chk.record_count()}")
n_records = 50
dss_event = '//Prado_IN-combined/FLOW//1Day/C:000033|T:19991113-1200|V:19991113-120000|HFO:FRA_50yr:Scripting-GenFcsts/'
data = dss_chk.get(dss_event)
print(data.start_date,data.units)
dtg = pd.to_datetime(data.times)
flow_kcfs = data.values / 1000

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

now=datetime.now()
print('gen end',now.strftime("%H:%M:%S"))


"""
Plotting routine below for checking output; will be ported to a different script as code is developed
"""
#########################################################################################################################
# Ensemble plots for a cursory verification of outputs
#########################################################################################################################
#plot timeseries comparison
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

out_dir = '../figs/%s/%s/realization=%s_lifecycle=%s' %(loc,gen_site,realization,lifecycle)
os.makedirs(out_dir,exist_ok=True)

#select event number from DSS files (generally arranged sequentially smallest to largest)
evt = 25
evt_num = f'{evt:02}'

dates = fcst_issue_dates[evt_num]
#get indices of largest events
hefs_rec = par_out[evt-1]
obs_fwd_rec = obs_fwd_gen_mat[(evt-1),:,:]
ext_idx = np.argsort(obs_fwd_rec[:,0])[::-1][0]

ext_idx_fit = np.argsort(obs_fwd_fit[:,0])[::-1][0]

#plot the top 12 sorted events at a 3-d lead
fig = plt.figure(layout='constrained',figsize=(7,3))
gs0 = fig.add_gridspec(1,2)

ld = 7
evt_ind = ext_idx - ld
evt_ind_fit = ext_idx_fit - ld

ax1 = fig.add_subplot(gs0[0])
l1, = ax1.plot(np.arange(max_lds+1),obs_fwd_rec[evt_ind,:],linewidth=2,c='black')
l2, = ax1.plot(np.arange(max_lds+1),obs_fwd_fit[evt_ind_fit,:],linewidth=2,c='blue',alpha=0.25)
ax1.axvline(ld,c='gray',linewidth=0.5,linestyle='--',alpha=0.5)
ylm = max(max(obs_fwd_rec[evt_ind,:]),max(obs_fwd_fit[evt_ind_fit,:]))*1.25
print(ylm)
ax1.set_ylim([0,ylm])
ax1.set_xlim([0,max_lds])
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_ylabel('Flow (kcfs)')
ax1.set_xlabel('Lead (days)')
ax1.text(ld,max(obs_fwd_rec[evt_ind,:]),str(dates[evt_ind+ld])[:10])
ax1.text(ld,max(obs_fwd_fit[evt_ind_fit,:]),str(ixx_fit[evt_ind_fit+ld])[:10],color='blue')
for k in range(np.shape(hefs_rec)[2]):
    ens_out = hefs_rec[evt_ind,:,k].copy()
    l3, = ax1.plot(np.arange(max_lds+1),ens_out,c='gray',linewidth=1,alpha=0.5)
if ld > 3:
    leg = ax1.legend([l1,l2,l3],['HEC-WAT','POR','syn-fcst'],loc='upper left',fontsize='medium',frameon=False)
    ax1.text(10,0.9*ylm,'Event = %s' %(evt_num))
elif ld <= 3:
    leg = ax1.legend([l1,l2,l3],['HEC-WAT','POR','syn-fcst'],loc='upper right',fontsize='medium',frameon=False)
    ax1.text(3,0.9*ylm,'Event = %s' %(evt_num))
ax1.add_artist(leg)
    
ax1 = fig.add_subplot(gs0[1])
l1, = ax1.plot(np.arange(max_lds+1),obs_fwd_rec[evt_ind,:],linewidth=2,c='black',alpha=0.15)
l2, = ax1.plot(np.arange(max_lds+1),obs_fwd_fit[evt_ind_fit,:],linewidth=2,c='blue')
ax1.axvline(ld,c='gray',linewidth=0.5,linestyle='--',alpha=0.5)
ylm = max(max(obs_fwd_rec[evt_ind,:]),max(obs_fwd_fit[evt_ind_fit,:]))*1.25
print(ylm)
ax1.set_ylim([0,ylm])
ax1.set_xlim([0,max_lds])
ax1.yaxis.set_ticklabels([])
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_xlabel('Lead (days)')
ax1.text(ld,max(obs_fwd_rec[evt_ind,:]),str(dates[evt_ind+ld])[:10])
ax1.text(ld,max(obs_fwd_fit[evt_ind_fit,:]),str(ixx_fit[evt_ind_fit+ld])[:10],color='blue')
for k in range(np.shape(hefs_rec)[2]):
    ens_out_fit = obs_fwd_fit[evt_ind_fit,:].copy()
    ens_out_fit[1:] = hcst_fit[evt_ind_fit,:,k].copy()
    l3, = ax1.plot(np.arange(max_lds+1),ens_out_fit,c='blue',linewidth=1,alpha=0.1)
if ld > 3:
    leg = ax1.legend([l1,l2,l3],['HEC-WAT','POR','HEFS'],loc='upper left',fontsize='medium',frameon=False)
    ax1.text(10,0.9*ylm,'Event = %s' %(evt_num))
elif ld <= 3:
    leg = ax1.legend([l1,l2,l3],['HEC-WAT','POR','HEFS'],loc='upper right',fontsize='medium',frameon=False)
    ax1.text(3,0.9*ylm,'Event = %s' %(evt_num))
ax1.add_artist(leg)

plt.show()
fig_path = '../figs/%s/%s/realization=%s_lifecycle=%s/syn-fcst_ensemble-plot_event=%s_evt-date=%s_ld=%s.png' %(loc,gen_site,realization,lifecycle,evt_num,str(dates[ext_idx])[:10],ld)
fig.savefig(fig_path,dpi=300,bbox_inches='tight')


######################################################END#################################################################################
