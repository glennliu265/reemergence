#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Postprocess Stochastic Model Output from SSS basinwide Integrations
Currently a working draft, will copy the essential functions once I have finalized things..

Created on Wed Feb  7 17:23:00 2024

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob
import time
import os

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine    = "stormtrack"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
sys.path.append(pathdict['scmpath'] + "../")
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx
import stochmod_params as sparams

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']

procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']
lipath      = pathdict['lipath']

# Make Needed Paths
proc.makedir(figpath)

#%% User Edits ================================================================

# Indicate experiments to process

# # Testing effect of detrainment damping
# expnames = ["SST_EOF_LbddEnsMean","SSS_EOF_Qek_LbddEnsMean",
#              "SST_EOF_NoLbdd","SSS_EOF_NoLbdd",#
#              "SSS_CESM","SST_CESM"]

expnames = [
    #"SST_CESM",
    #"SSS_CESM",
    # "SST_EOF_LbddCorr_Rerun",
    # "SST_EOF_LbddCorr_Rerun_NoLbdd",
    "SSS_EOF_LbddCorr_Rerun_lbdE_neg",
    #"SSS_EOF_LbddCorr_Rerun",
    # "SSS_EOF_LbddCorr_Rerun_NoLbdd", 
    ]



# Region Setting (see re-emergence parameters, rdict[selnames])
regionset              = "SSSCSU"#"TCMPi24" 

# Analysis Settings (Toggle On/Off)

varthres               = 10   # Variance threshold above which values will be masked for AMV computation
compute_variance       = False # Set to True to compute pointwise variance
regional_analysis      = True
calc_amv               = False

# Settings for CESM (assumes CESM output is located at rawpath)
anom_cesm  = False                          # Set to false to anomalize CESM data
bbox_sim   = np.array([-80,   0,  20,  65]) # BBOX of stochastic model simulations, to crop CESM output

#%% Main Script below =========================================================
    
for expname in expnames:
    if "SSS" in expname:
        varname = "SSS"
    elif "SST" in expname:
        varname = "SST"
    
    print("Performing Postprocessing for %s" % expname)
    print("\tSearching for output in %s"     % output_path)
    
    # --------------------------------------------------------
    #%% Load output (copied from analyze_basinwide_output_SSS)
    # --------------------------------------------------------
    # Takes 16.23s for the a standard stochastic model run (10 runs, 12k months)
    print("Loading output...")
    st          = time.time()
    
    if "CESM" in expname:
        # Load NC files
        ncname    = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % varname
        
        # Load DS
        ds_cesm   = xr.open_dataset(rawpath+ncname).squeeze()
        
        # Slice to region
        ds_cesm   = proc.sel_region_xr(ds_cesm,bbox_sim)
        
        # Correct Start time
        ds_cesm   = proc.fix_febstart(ds_cesm)
        ds_cesm   = ds_cesm.sel(time=slice('1920-01-01','2005-12-31')).load()
        
        # Anomalize if necessary
        if anom_cesm is False:
            print("Detrending and deseasonalizing variable!")
            ds_cesm = proc.xrdeseason(ds_cesm) # Deseason
            ds_cesm = ds_cesm[varname] - ds_cesm[varname].mean('ensemble')
        else:
            ds_cesm = ds_cesm[varname]
        
        # Rename to "RUN" to fit the formatting
        ds_cesm = ds_cesm.rename(dict(ensemble='run'))
        ds_sm   = ds_cesm
        
    else:
        
        # Load NC Files
        expdir      = output_path + expname + "/Output/"
        expmetrics  = output_path + expname + "/Metrics/"
        nclist      = glob.glob(expdir +"*.nc")
        nclist.sort()
        
        # Load DS, deseason and detrend to be sure
        ds_all      = xr.open_mfdataset(nclist,concat_dim="run",combine='nested').load()
        ds_sm       = proc.xrdeseason(ds_all[varname])
        ds_sm       = ds_sm - ds_sm.mean('run')
        
        # Load Param Dictionary
        dictpath    = output_path + expname + "/Input/expparams.npz"
        expdict     = np.load(dictpath,allow_pickle=True)
        
    # Set Postprocess Output Path
    metrics_path = output_path + expname + "/Metrics/" 
    
    print("\tOutput loaded in %.2fs" % (time.time()-st))
    print("\tMetrics will be saved to %s" % metrics_path)
    
    # -----------------------------------------------
    #%% Load masks for regional average computations
    # -----------------------------------------------
    stice     = time.time()
    
    # Load the Land Ice Mask
    liname    = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"
    ds_mask   = xr.open_dataset(lipath+liname).MASK.squeeze().load()
    ds_mask   = proc.sel_region_xr(ds_mask,bbox_sim)
    
    
    # Roll in All directions to remove coastal points
    maskcoast = ds_mask.values.copy()
    maskcoast = np.roll(maskcoast,1,axis=0) * np.roll(maskcoast,-1,axis=0) * np.roll(maskcoast,1,axis=1) * np.roll(maskcoast,-1,axis=1)
    
    ds_maskcoast = xr.DataArray(maskcoast,coords=ds_mask.coords,dims=ds_mask.dims).rename('mask')
    
    # Save Coast Mask
    savename_mask = "%sLand_Ice_Coast_Mask.nc" % (metrics_path)
    ds_maskcoast.to_netcdf(savename_mask)
    
    print("Loaded Land Ice Mask in %.2fs" % (time.time()-stice))
    
    # --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> 
    # Data Loading Complete... 
    # --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> 
    
    # ----------------------------------------------------------------------
    # %% 1) Compute Overall Pointwise Variance and Seasonal Average Variance
    # ----------------------------------------------------------------------
    if compute_variance:
        print("Computing Pointwise Variances...")
        st1            = time.time()
        
        # Copy over the DataSet
        ds             = ds_sm.copy()
        
        # Compute Variances
        dsvar_byens    = ds.std('time')
        dsvar_seasonal = ds.groupby('time.season').std('time')
        
        # Save output
        # >> Save Overall Variance
        savenamevar    = "%sPointwise_Variance.nc" % (metrics_path)#(run: 10, lat: 48, lon: 65)
        edict          = {varname:{'zlib':True}}
        dsvar_byens.to_netcdf(savenamevar,encoding=edict)
        
        # >> Save Seasonal Variance
        savenamevar    = "%sPointwise_Variance_Seasonal.nc" % (metrics_path)#(run: 10, lat: 48, lon: 65)
        dsvar_seasonal.to_netcdf(savenamevar,encoding=edict)
        
        print("\tSaved Pointwise and Seasonal Variances in %.2fs " % (time.time()-st1))
        
        # End Pointwise Variance <--
    
    # ---------------------------------------------------
    #%% Part (2) Regional Analysis
    # ---------------------------------------------------
    if regional_analysis:
        print("Computing Regional Averages...")
        st2         = time.time()
        ds          = ds_sm.copy() * ds_maskcoast
        
        #% Pull Parameters for regional analysis
        rdict       = rparams.region_sets[regionset]
        regions_sel = rdict['regions']
        bboxes      = rdict['bboxes']
        rcols       = rdict['rcols']
        nregs       = len(regions_sel)
        
        #% Compute Regional Averages ----
        
        # Calculate Regional Average Over each selected bounding box
        ssts_reg = []
        for r in range(len(regions_sel)):
            bbin      = bboxes[r]
            rsst      = proc.sel_region_xr(ds,bbin)     # (run: 10, time: 12000, lat: 22, lon: 37)
            ravg      = proc.area_avg_cosweight(rsst)   # (run: 10, time: 12000)
            
            ssts_reg.append(ravg)
        ssts_reg      = xr.concat(ssts_reg,dim="r")     # (region: 4, run: 10, time: 12000)
        
        # Place into new xr.DataArray
        coords_new    = dict(regions=regions_sel,run=ssts_reg.run,time=ssts_reg.time)
        coords_reg    = dict(regions=regions_sel,bounds=["W","E","S","N"])
        da_rsst       = xr.DataArray(ssts_reg.values,coords=coords_new,dims=coords_new,name=varname)
        da_reg        = xr.DataArray(np.array(bboxes),coords=coords_reg,dims=coords_reg,name='bboxes')
        rsst_out      = xr.merge([da_rsst,da_reg])
        
        # Save the Output
        edict         = proc.make_encoding_dict(rsst_out)
        savename_rsst = "%sRegional_Averages_%s.nc" % (metrics_path,regionset) 
        rsst_out.to_netcdf(savename_rsst,encoding=edict)
        print("\tSaved Regional Averages in %.2fs" % (time.time() - st2))
        
        # -------------------------------------
        #%% Part (3) Compute Timeseries Metrics
        # -------------------------------------
        print("Computing Metrics for Regional Averages")
        st3 = time.time()
        
        # Set Metrics Options (Move this to the top)
        nsmooth     = 150
        lags        = np.arange(37)
        pct         = 0.10
        metrics_str = "nsmooth%03i_pct%03i_lagmax%02i" % (nsmooth,pct*100,lags[-1])
        
        tsm_regs = {}
        for r in tqdm(range(len(regions_sel))):
            
            
            rsst_in = ssts_reg.isel(r=r)
            rsst_in = np.where((np.abs(rsst_in)==np.inf) | np.isnan(rsst_in),0.,rsst_in)
            
            tsm = scm.compute_sm_metrics(rsst_in,nsmooth=nsmooth,lags=lags,pct=pct)
            tsm_regs[regions_sel[r]] = tsm
              
        savename_tsm = "%sRegional_Averages_Metrics_%s.npz" % (metrics_path,regionset) 
        np.savez(savename_tsm,**tsm_regs,allow_pickle=True)
        print("\n\tSaved Metrics for Regional Averages in %.2fs" % (time.time() - st3))
        
        # End Regional Analysis <--
    
    # ------------------------------------------
    # %%  Part (4) Compute AMV Pattern and Index
    # ------------------------------------------
    if calc_amv:
        st4    = time.time()
        amvidx_allreg = []
        amvpat_allreg = []
        idx_unsmooth  = []
        nruns  = len(ssts_reg.run)
        
        lon = ds_sm.lon.values
        lat = ds_sm.lat.values
        
        # Make a mask where values are very large
        savenamevar    = "%sPointwise_Variance.nc" % (metrics_path)#(run: 10, lat: 48, lon: 65)
        ds_var         = xr.open_dataset(savenamevar).load()[varname].mean('run')
        ds_varmask     = xr.where(ds_var>varthres,np.nan,1) # (lat x lon)
        amv_mask       = (ds_varmask.values * maskcoast).T # Transpose to [lon x lat]
        
        for ireg in range(nregs):
            bbxreg = bboxes[ireg]
            
            amvidx = []
            amvpat = []
            idxusm = []
            
            for irun in range(nruns):
                
                #ts_in  = ssts_reg.isel(run=irun,r=ireg).values # Time
                sst_in = ds_sm.isel(run=irun).transpose('lon','lat','time').values #* maskcoast.T[...,None]
                
                idx,pat,usm=proc.calc_AMVquick(sst_in,
                                           lon,lat,
                                           bbxreg,dropedge=5,mask=amv_mask,return_unsmoothed=True,verbose=False)
                amvidx.append(idx)
                amvpat.append(pat)
                idxusm.append(usm)
                # End Run Loop ---
            amvidx_allreg.append(amvidx)
            amvpat_allreg.append(amvpat)
            idx_unsmooth.append(idxusm)
        
        # List to Numpy
        amvidx_allreg = np.array(amvidx_allreg) # (2, 42, 86)
        amvpat_allreg = np.array(amvpat_allreg) # (2, 42, 65, 48)
        idx_unsmooth  = np.array(idx_unsmooth)  # (2, 42, 86)
        
        # Make in DataArray
        years         = np.arange(amvidx_allreg.shape[-1]) + 1920
        coords_idx = dict(region=regions_sel,run=ds_sm.run,year=years)
        coords_pat = dict(region=regions_sel,run=ds_sm.run,lon=lon,lat=lat)
        da_pat = xr.DataArray(amvpat_allreg,coords=coords_pat,dims=coords_pat,name="amv_pattern").transpose('region','run','lat','lon')
        da_idx = xr.DataArray(amvidx_allreg,coords=coords_idx,dims=coords_idx,name="amv_index")
        da_usm = xr.DataArray(idx_unsmooth,coords=coords_idx,dims=coords_idx,name="amv_index_unsmoothed")
        ds_amv = xr.merge([da_pat,da_idx,da_usm])
        
        # Save Output
        edict    = proc.make_encoding_dict(ds_amv)
        savename = "%sAMV_Patterns_%s.nc" % (metrics_path,regionset) 
        ds_amv.to_netcdf(savename,encoding=edict)
        
        # End Calc AMV <--
    
            
            
            
    
    
    

