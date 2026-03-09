# synthetic-ensemble-forecast_Py
Repository for Python-based synthetic ensemble forecasting codebase that is derived from the R-codebase linked below. This synthetic ensemble forecast generation approach was developed in Brodeur et al. (2025) and relies on a resampling and scaling approach to generate ensemble forecasts for any plausible streamflow timeseries (e.g. historical observations, simulations, projections) for a given forecast site, given a sufficiently long hindcast dataset for fitting of the model. Fitting of the model to the available hindcast data involves the calibration of a curve that modulates the sampling and rescaling procedure across lead times. This fitting procedures uses a loss function that seeks to minimize the skill differences between the generated synthetic forecasts and the original hindcasts across lead times, ensuring that the synthetic forecasts accurately represent the skill attributes of the original hindcasts

R-based synthetic ensemble forecast model: [https://github.com/zpb4/Synthetic-Forecast-v2-FIRO-DISES](https://github.com/zpb4/Synthetic-Forecast-v2-FIRO-DISES)    

Brodeur, Z. P., Taylor, W., Herman, J. D., & Steinschneider, S. (2025). Synthetic Ensemble Forecasts: Operations‐Based Evaluation and Inter‐Model Comparison for Reservoir Systems Across California. Water Resources Research, 61(e2024WR039324). https://doi.org/10.1029/%25202024WR039324

As currently configured, the synthetic ensemble forecasting routine in this repository is setup to support two generation scenarios:
1. Generation of synthetic forecasts against an observational record of streamflow that extends beyond the available hindcast dataset
2. Generation of synthetic forecasts agains synthetic streamflow events produced by the hydrologic sampler routine in the USACE HEC-WAT software suite

The examples provided in this repository are based on NOAA/NWS Hydrologic Ensemble Forecast Service (HEFS) ensemble forecasts for the Prado Dam location (ADO) in southern California, where there is a single inflow forecast site for the dam (ADOC1). 

## Primary dependencies
- Pandas
- Numpy
- Numba*
- hecdss**
- matplotlib
- joblib

*The main optimization and generation scripts require minimal external libraries due to the use of Numba @njit (no-python) compilers used to speed up the code. These compilers can be turned off by simply commenting out the '@njit' lines in the scripts. These compilers are relatively limited in their capability to handle anything besides basic Python built-in functions and Numpy functions/data structures and may cause problems if any changes to the scripts are made.   
    
**This USACE library is required for reading and writing .dss files from HEC-WAT software. This library is not required for basic operation of the synthetic ensemble forecast generation code

## Workflow   
The raw data for HEFS forecasts are typically provided in the form of .csv files for each day of forecast initialization across the hindcast periods (1990 - 2019 for the Prado Dam example). However, hindcasts may come in many different raw data formats. The data processing script in this repository describes the processing steps to configure HEFS data into the Numpy array format required for the follow on optimization and generation routines but will need to be tailored to other data sources.

Aggregated spatial labeling of a watershed 'location' ('loc' variable) with a number of forecast 'sites' ('site' variable) is the basic approach used in these scripts, for example:  
ADO/ADOC1 - Prado Dam location / Prado Dam inflow forecast site   

Time estimates are shown in parantheses for each script element for the single site example (more sites would require more processing time generally speaking). Overall, the entire workflow can be run in 2 - 2.5 hours on a typical desktop PC for the example case. The code employs internal parallelization through the joblib library for computationally intensive tasks, which can significantly reduce computation time on larger HPC resources.

### 1. Data Processing
Example script to process the raw observation and HEFS .csv files into the Numpy array format required for optimization and generation
A. ./src/data_processing.py - data processing script example for ADO/ADOC1 example (<30 min)

### 2.  Synthetic forecast model optimization
Optimize the synthetic forecast model to the hindcast calibration data. By default, only the primary 'keysite' is optimized, meaning that watershed locations with multiple sites will not require additional optimization time.
- ./src/optimize_synthetic_forecasts.py - main optimization routine w/several user-defined settings; calls the syn_gen_opt.py script (~1 - 1.5 hours)
- ./src/syn_gen_opt.py - functions for the synthetic forecast model optimization; generates a single synthetic sample for each iteration of the optimization

### 3. Synthetic ensemble forecast generation
Generates stochastic ensemble forecast samples against the provided streamflow sequence (observed streamflows in the ADO/ADOC1 example) using the fitted parameters from the optimization. User defines how many samples to generate and must ensure that specifications match the desired optimization run from step 2.
- ./src/create_synthetic_forecasts.py - main synthetic ensemble forecast generation script; calls the syn_gen.py script (10-15 min; single site)
- ./src/syn_gen.py - functions to support the synthetic ensemble forecast generation procedure
  
### 4. USACE HEC-WAT synthetic forecast generation 
Employs a similar synthetic generation routine to Step 3, but setup to ingest .json configuration files from HEC-WAT hydrologic sampler routine and read/write .dss file formats for input/output. Generates a single synthetic forecast sequence for each synthetic streamflow event provided in the HEC-WAT .dss file and outputs a daily streamflow and synthetic forecast .dss file for each events. In the provided example for ADO/ADOC1, the input .dss file includes 50 events.
- ./src/create_synthetic_forecasts_hec-wat_fra.py - main synthetic ensemble forecast generation script that leverages HEC-WAT .dss file I/O; calls the syn_gen_hec_wat_fra.py script (1-2 min for the 50 event ADO/ADOC1 sample from HEC-WAT)
- ./src/syn_gen_hec_wat_fra.py - functions to support the main generation

### 5. Miscellaneous
- ./src/util.py - various helper functions 

### Contact
Zach Brodeur, zpbrodeur@ucsd.edu
