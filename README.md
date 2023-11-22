# Reemergence
Scripts for analysis of re-emergence in sea surface temperature (SST) and salinity (SSS).

# Table of Contents

==preprocessing==
preproc_glorys12v1.py : Merge daily glorys12v1 output to monthly means
preprocess_data : Preprocess data for re-emergence calculations
preprocess_data_byens : Copy of \[preprocess_data\] but for CESM2
npz_to_nc : Convert \[pointwise_autocorreation\] output to netcdf

==stochastic_model==
point case studies
- stochmod_point : Test stochastic SSS and SST model at a point
- stochmod_point_dampingest : Try different ways of estimating damping parameter at a point
- get_point_data_stormtrack : Get data for point for stochastic model on stormtrack
- point_case_study : early test script for SSS stochastic model
basinwide analysis
- map_flux_ratios : Identify where Latent Heat Flux plays a major role relative to Sensible
- map_TS_relationship : Look at T-S Lag Correlation relationship in CESM1-LENs
- calc_SSS_feedback_CESM1 : Calculate SSS feedback
Td calculations
- calc_Td_decay  : Compute decay rate of anomalies below the mixed layer (using get_Td output)
- get_Td         : Retrieve timeseries at base of deepest mixed-layer in the seasonal cycle

==calculations==
pointwise_autocorrelation_lens : Compute pointwise ACF for CMIP6 LENs
estimate_damping_fit : Estimate damping parameter from exponential fit to ACF
calc_ac_point : Examine autocorrelation for a given point
correlation_experiments : Look at AR(1) timeseries and compute T2 and ACF to see how they vary...?

==depth_analysis==
calc_ac_depth   : Get UOHC data for a single point
viz_ac_depth    : Visualize autocorrelation (Depth vs. Lag)
rem_depth-v-lag : Compute correlation (Depth vs Lag)

==scrap==
exponential_decay_scrap : Visualize and test fitting of exponential decay to ACF...
stochastic_salinity_test : Seems like a dummy script, has some reference constants
test_ac_depth : Seems incomplete, probably will delete
viz_threshold_comparisons : Also seems incomplete, will delete
make_landmask : Seems to be WIP script to make a landmask. It doesn't actually save anything...

==analysis==
cluster_reemergence_maps : Perform k-means clustering on SST/SSS ACFs
quick_remeemergence_plots : Make quick plots of lag vs correlation for ACF at a point...
REM_Autocor_CESM5deg : Read in 5-deg CESM1 LENS and make ACF/re-emergence plots (probably from 12.860)

==notebooks==
viz_timescales_interactive
viz_timeseries_interactive
