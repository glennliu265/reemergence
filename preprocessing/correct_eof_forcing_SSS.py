#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Correct EOF Forcing for SSS (Evaporation, Precipitation)
Also Do this for Qek? (SST and SSS)

Copied correct_eof_forcing on 2024.03.01
(started) Adding Qek Corrections 2024.08.26

Perform EOF filtering based on a variance threshold.
Compute the Required Variance needed to correct back to 100% (monthly std(E') or std(P')) at each month.

Currently works on Astraeus but need to make this machine adaptable

Inputs:
------------------------
    varname             : dims                              - units                 - Full Name
    LHFLX               : (month,ens,lat,lon)
    PRECTOT             : (mon,ens,lat,lon)
#    Qek_SST             : (mon,ens,lat,lon)                 [degC / sec]
#    Qek_SSS             : (mon,ens,lat,lon)                 [degC/sec]
    

Outputs: 
------------------------

    varname             : dims                              - units                 - Full Name
    correction_factor   : (mon, lat, lon)                   [W/m2]
    eofs                : (mode, mon, lat, lon)             [W/m2 per std(pc)]
    
    
    

What does this script do?
------------------------
(1) Load in EOF Output and Fprime and take ensemble mean of variance explained, patterns, and std(F')
(2) Apply EOF Filtering, retaining only modes explaining up to N %
(3) Take difference of std(F') and std(EOF_filtered) to get pointwise variance correction factor
                                                                                                 
                                                                                                 

Note that correct ion is performed on the ENSEMBLE MEAN forcing and Fstd!!


Created on Tue Feb 13 20:21:00 2024

@author: gliu
"""


import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt


# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "stormtrack"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']


#%% Indicate Paths

#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240411/"
datpath   = pathdict['raw_path']
outpath   = pathdict['input_path']+"forcing/"
rawpath   = pathdict['raw_path']


#%% User Edits


# Indicate Filtering OPtions
eof_thres = 0.90
bbox_crop = [-90,0,0,90]
# Indicate dataset name
dataset = "CESM1_HTR"
concat_ens=True


if dataset == "CESM1_HTR":
    
    datpath   = outpath
    # Indicate Forcing Options
    dampstr   = "nomasklag1"
    rollstr   = "nroll0"

    # Load EOF results
    nceof     = rawpath + "CESM1_HTR_EOF_Monthly_NAO_EAP_Fprime_%s_%s_NAtl_concatEns.nc" % (dampstr,rollstr)
    
    # Load Fprime
    ncfprime  = "CESM1_HTR_FULL_Fprime_timeseries_%s_%s_NAtl.nc" % (dampstr,rollstr)
    
    # Load Evap, Precip
    ncevap    = rawpath + "CESM1_HTR_FULL_Eprime_timeseries_LHFLXnomasklag1_nroll0_NAtl.nc"#"CESM1_HTR_FULL_Eprime_nroll0_NAtl.nc"
    ncprec    = rawpath + "PRECTOT_HTR_FULL.nc"#"CESM1_HTR_FULL_PRECTOT_NAtl.nc"

    # Load EOF Regression output
    ncprec_eof = outpath + "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_concatEns.nc"
    #ncevap_eof = "CESM1_HTR_FULL_LHFLX_EOF_nomasklag1_nroll0_NAtl.nc"
    ncevap_eof = outpath + "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_concatEns.nc"
    
elif dataset == "cesm1le_htr_5degbilinear":
    
    datpath         = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/proc/"
    
    # Indicate Forcing Options
    dampstr         = "cesm1le5degLHFLX"
    dampstr_qnet    = "cesm1le5degqnet"  
    rollstr         = "nroll0"
    regstr          = "Global"
    
    # Load EOF Results (NHFLX_EOF_Monthly)
    nceof     = datpath + "%s_EOF_Monthly_NAO_EAP_Fprime_%s_%s_%s.nc" % (dataset,dampstr_qnet,rollstr,regstr)
    
    # Load Fprime (Qnet)
    ncfprime  = "%s_Fprime_timeseries_%s_%s_%s.nc" %  (dataset,dampstr_qnet,rollstr,regstr)
    
    # Load Evap, Precip (to compute Monthly Standard Deviations)
    # Evap: hfcalc/Main/calc_Fprime
    # Prec: reemergence/preprocesing/combine_precip)
    ncevap    = datpath + "%s_Eprime_timeseries_%s_%s_%s.nc" % (dataset,dampstr,rollstr,regstr)
    ncprec    = datpath + "cesm1_htr_5degbilinear_PRECTOT_%s_1920to2005.nc" % (regstr)
    
    # Load Evap, Precip Regressions computed in /regress_EOF_forcing/
    ncevap_eof    = outpath + "cesm1le_htr_5degbilinear_Eprime_EOF_cesm1le5degLHFLX_nroll0_Global.nc"#
    ncprec_eof    = outpath + "cesm1le_htr_5degbilinear_PRECTOT_EOF_cesm1le5degLHFLX_nroll0_Global.nc" #
    

    
    
    
    
    
    
    

# Load Ekman Forcing

#fp1  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
#ncek = "CESM1_HTR_FULL_Qek_SSS_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc"
#dsek = xr.open_dataset(fp1+ncek)

# Other things
crop_sm     = True # Save Cropped Version
regstr_crop = "NAtl"
bbox_crop   = [-90,0,0,90]
bbox        = [-80,0,0,65]
debug       = True


def anomalize(ds):
    ds = ds - ds.mean('ens')
    ds = proc.xrdeseason(ds)
    return ds

def reshape_concat_ens(ds):
    ds = ds.transpose('time','ens','lat','lon')
    ntime,nens,nlat,nlon=ds.shape
    dsout     = ds.data.reshape(ntime*nens,1,nlat,nlon)
    timefake  = proc.get_xryear('0000',nmon=ntime*nens)
    coordsout = dict(time=timefake,ens=np.arange(1,2),lat=ds.lat,lon=ds.lon)
    dsout     = xr.DataArray(dsout,coords=coordsout,dims=coordsout)
    return dsout
    
    

#%% Procedure

# (1) Load EOF Results, compute variance explained
dseof    = xr.open_dataset(nceof).load()
eofs     = dseof.eofs.mean('ens')   # (mode: 86, mon: 12, lat: 96, lon: 89)
varexp   = dseof.varexp.mean('ens') # (mode: 86, mon: 12)

# # (2) Load Fprime, compute std(F') at each point
# dsfp     = xr.open_dataset(datpath+ncfprime).load()
# monvarfp = dsfp.Fprime.groupby('time.month').std('time')#.mean('ens') # (month: 12, lat: 96, lon: 89)

# (2) Load  Evap, Precip to compute Monthly Stdev 
dsevap    = xr.open_dataset(ncevap).load().LHFLX # (time, ens, lat, lon)
dsevap    = proc.fix_febstart(dsevap)
dsevap    = anomalize(dsevap)
if concat_ens:
    dsevap      = reshape_concat_ens(dsevap)
monvarE   = dsevap.groupby('time.month').std('time')
monvarE   = monvarE.rename({'month':'mon'})

#dsevap    = dsevap.rename({'month':'mon'})
dsprec    = xr.open_dataset(ncprec).load().PRECTOT
dsprec    = proc.fix_febstart(dsprec)
dsprec    = anomalize(dsprec)
if concat_ens:
    dsprec    = reshape_concat_ens(dsprec)
monvarP   = dsprec.groupby('time.month').std('time')
monvarP   = monvarP.rename({'month':'mon'})

# (3) Load EOF regressions of Evap, Precip
dsevap_eof = xr.open_dataset(ncevap_eof).load().LHFLX # (mode, ens, mon, lat, lon)
dsprec_eof = xr.open_dataset(ncprec_eof).load().PRECTOT # (mode, ens, mon, lat, lon)

# Make dsure variables are the same
dsevap_eof,monvarE = proc.resize_ds([dsevap_eof,monvarE])
dsprec_eof,monvarP = proc.resize_ds([dsprec_eof,monvarP])

#%% 3. Perform EOF filtering (retain enough modes to explain [eof_thres]% of variance for each month)

# Inputs
#eofs_std   = dseof.eofs
varexp_in  = varexp.values           # Variance explained (for Fprime Analysis) [mode, mon]
vnames     = ["LHFLX","PRECTOT"]     # names of variables
ds_eof_raw = [dsevap_eof,dsprec_eof] # EOF regressions    (mode, ens, mon, lat, lon)
ds_std     = [monvarE,monvarP]         # Monthly standard deviation (ens, mon , lat, lon)
ncnames    = [ncevap_eof,ncprec_eof]
nvars      = len(vnames)

ensavg_first = True




# By Default, use the longitude coordinates of the regressed variables
# CAUTION: This script DOES NOT check if longitude is same across all variables
# Need to write a check for this..
lon_out    = dsprec_eof.lon
lat_out    = dsprec_eof.lat

for v in range(nvars): # Loop by Variable
    

    if ensavg_first:
        # Index variables (take ensemble average)
        eofvar_in   = np.nanmean(ds_eof_raw[v].transpose('mode','ens','mon','lat','lon').values,1)
        monvarfp    = np.nanmean(ds_std[v].transpose('ens','mon','lat','lon').data,0)
        varexp_eavg = varexp_in#np.nanmean(varexp_in,1) # 
        
        # Perform Filtering
        eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt=proc.eof_filter(eofvar_in,varexp_eavg,
                                                            eof_thres,axis=0,return_all=True)
        
        
        # Compute Stdev of EOFs
        eofs_std = np.sqrt(np.sum(eofs_filtered**2,0)) # [Mon x Lat x Lon]
        
        lon_out    = ds_eof_raw[v].lon
        lat_out    = ds_eof_raw[v].lat
        
        # Compute pointwise correction
        correction_diff = monvarfp - eofs_std
        
        # Prepare to Save -------------------------
        
        corcoords     = dict(mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        eofcoords     = dict(mode=ds_eof_raw[0].mode,mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        
        da_correction = xr.DataArray(correction_diff,coords=corcoords,dims=corcoords,name="correction_factor")
        da_eofs_filt  = xr.DataArray(eofs_filtered,coords=eofcoords,dims=eofcoords  ,name=vnames[v])
    
        ds_out        = xr.merge([da_correction,da_eofs_filt])
        edict         = proc.make_encoding_dict(ds_out)
        
        # Save for all ensemble members
        savename       = proc.addstrtoext(ncnames[v],"_corrected",adjust=-1)
        savename       = proc.addstrtoext(savename,"_EnsAvgFirst",adjust=-1)
        ds_out.to_netcdf(savename,encoding=edict)
        print("Saved output to %s" % savename)
        
    else:
            
            
                
        # Index variables
        eofvar_in = ds_eof_raw[v].transpose('mode','ens','mon','lat','lon').values
        monvarfp  = ds_std[v].transpose('ens','mon','lat','lon').values
            
        # Perform Filtering
        eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt=proc.eof_filter(eofvar_in,varexp_in,
                                                           eof_thres,axis=0,return_all=True)
        
        # Compute Stdev of EOFs
        eofs_std = np.sqrt(np.sum(eofs_filtered**2,0)) # [Ens x Mon x Lat x Lon]
        
        
        lon_out    = ds_eof_raw[v].lon
        lat_out    = ds_eof_raw[v].lat
        
        
    
            
            
        # Compute pointwise correction
        correction_diff = monvarfp - eofs_std
        
        # Prepare for output -----
        corcoords     = dict(ens=ds_std[0].ens,mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        eofcoords     = dict(mode=dseof.mode,ens=ds_std[0].ens,mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        
        da_correction = xr.DataArray(correction_diff,coords=corcoords,dims=corcoords,name="correction_factor")
        da_eofs_filt  = xr.DataArray(eofs_filtered,coords=eofcoords,dims=eofcoords  ,name=vnames[v])
        
        ds_out        = xr.merge([da_correction,da_eofs_filt])
        edict         = proc.make_encoding_dict(ds_out)
        
        # Save for all ensemble members
        savename       = proc.addstrtoext(ncnames[v],"_corrected",adjust=-1)
        ds_out.to_netcdf(savename,encoding=edict)
        
        savename_emean = proc.addstrtoext(savename,"_EnsAvg",adjust=-1)
        ds_out_ensavg  = ds_out.mean('ens')
        ds_out_ensavg.to_netcdf(savename_emean,encoding=edict)
        
        # if crop_sm:
        #     print("Cropping to region %s" % (regstr_crop))
        #     ds_out = proc.lon360to180_xr(ds_out)
            
        #     ds_out_reg = proc.sel_region_xr(ds_out,bbox_crop)
        #     savename_reg = proc.addstrtoext(ncnames[v],"_corrected",adjust=-1).replace(regstr,regstr_crop)
        #     ds_out_reg.to_netcdf(savename_reg,encoding=edict)
            
        #     savename_emean = proc.addstrtoext(savename_reg,"_EnsAvg",adjust=-1)
        #     ds_out_reg_ensavg  = ds_out_reg.mean('ens')
        #     ds_out_reg_ensavg.to_netcdf(savename_emean,encoding=edict)
        #     print("Saved Ens Avg. Cropped Output to %s" % savename_emean)

# Check values
if debug:
    mons3   = proc.get_monstr()
    fig,ax  = viz.init_monplot(1,1)
    ax.bar(mons3,nmodes_needed,alpha=0.5,color='darkred')
    ax.set_xlim([-1,12])
    ax.set_title("Number of Modes Needed to Explain %.2f" % (eof_thres*100) + "% of Variance")
    ax.set_ylabel("Number of Modes")
    savename = "%sNAO_EAP_Fprime_Forcing_NumModes_thres%03i.png" % (figpath,eof_thres*100)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    fig,ax  = viz.init_monplot(1,1)
    ax.plot(np.sum(varexp,0),label="Raw")
    ax.plot(np.sum(varexps_filt,0),label="Post-Filtering")
    ax.set_ylabel("Total Variance Explained")
    ax.legend()