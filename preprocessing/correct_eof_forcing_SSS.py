#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Correct EOF Forcing for SSS (Evaporation, Precipitation)

Copied correct_eof_forcing on 2024.03.01

Perform EOF filtering based on a variance threshold.
Compute the Required Variance needed to correct back to 100% (monthly std(E') or std(P')) at each month.

Currently works on Astraeus but need to make this machine adaptable

Inputs:
------------------------
    varname             : dims                              - units                 - Full Name
    LHFLX               : (month,ens,lat,lon)
    PRECTOT             : (mon,ens,lat,lon)
    

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

figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240411/"
datpath   = pathdict['raw_path']
outpath   = pathdict['input_path']+"forcing/"
proc.makedir(figpath)

#%% User Edits
# Indicate Filtering OPtions
eof_thres = 0.90

# Indicate Forcing Options
dampstr   = "nomasklag1"
rollstr   = "nroll0"

# Load EOF results
nceof     = "EOF_Monthly_NAO_EAP_Fprime_%s_%s_NAtl.nc" % (dampstr,rollstr)

# Load Fprime
ncfprime  = "CESM1_HTR_FULL_Fprime_timeseries_%s_%s_NAtl.nc" % (dampstr,rollstr)

# Load Evap, Precip
ncevap    = "CESM1_HTR_FULL_Eprime_nroll0_NAtl.nc"
ncprec    = "CESM1_HTR_FULL_PRECTOT_NAtl.nc"

# Load EOF Regression output
ncprec_eof = "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl.nc"
#ncevap_eof = "CESM1_HTR_FULL_LHFLX_EOF_nomasklag1_nroll0_NAtl.nc"
ncevap_eof = "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl.nc"

# Load Ekman Forcing

#fp1  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
#ncek = "CESM1_HTR_FULL_Qek_SSS_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc"
#dsek = xr.open_dataset(fp1+ncek)

# Other things

bbox     = [-80,0,0,65]
debug    = True
#%% Procedure

# (1) Load EOF Results, compute variance explained

dseof    = xr.open_dataset(datpath+nceof).load()
eofs     = dseof.eofs.mean('ens')   # (mode: 86, mon: 12, lat: 96, lon: 89)
varexp   = dseof.varexp.mean('ens') # (mode: 86, mon: 12)

# (2) Load Fprime, compute std(F') at each point
#dsfp     = xr.open_dataset(datpath+ncfprime).load()
#monvarfp = dsfp.Fprime.groupby('time.month').std('time').mean('ens') # (month: 12, lat: 96, lon: 89)

# (2) Load Monthly Stdev of Evap, Precip
dsevap    = xr.open_dataset(outpath+ncevap).load() # (month, ens, lat, lon)
dsevap    = dsevap.rename({'month':'mon'})
dsprec    = xr.open_dataset(outpath+ncprec).load() # c

# (3) Load EOF regressions of Evap, Precip
dsevap_eof = xr.open_dataset(outpath+ncevap_eof).load() # (mode, ens, mon, lat, lon)
dsprec_eof = xr.open_dataset(outpath+ncprec_eof).load() # (mode, ens, mon, lat, lon)

#%% 3. Perform EOF filtering (retain enough modes to explain [eof_thres]% of variance for each month)

# Inputs
eofs_std   = dseof.eofs
varexp_in  = varexp.values           # Variance explained (for Fprime Analysis) [mode, mon]
vnames     = ["LHFLX","PRECTOT"]     # names of variables
ds_eof_raw = [dsevap_eof,dsprec_eof] # EOF regressions    (mode, ens, mon, lat, lon)
ds_std     = [dsevap,dsprec]         # Monthly standard deviation (mon , ens, lat, lon)
ncnames    = [ncevap_eof,ncprec_eof]

nvars      = len(vnames)

for v in range(nvars): # Loop by Variable
    
    # Index variables
    eofvar_in = ds_eof_raw[v][vnames[v]].values
    monvarfp  = ds_std[v][vnames[v]].transpose('ens','mon','lat','lon').values
    
    # Perform Filtering
    eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt=proc.eof_filter(eofvar_in,varexp_in,
                                                       eof_thres,axis=0,return_all=True)
    
    # Compute Stdev of EOFs
    eofs_std = np.sqrt(np.sum(eofs_filtered**2,0)) # [Ens x Mon x Lat x Lon]
    
    # Compute pointwise correction
    correction_diff = monvarfp - eofs_std
    
    # Prepare for output -----
    corcoords     = dict(ens=ds_std[0].ens,mon=np.arange(1,13,1),lat=dseof.lat,lon=dseof.lon)
    eofcoords     = dict(mode=dseof.mode,ens=ds_std[0].ens,mon=np.arange(1,13,1),lat=dseof.lat,lon=dseof.lon)
    
    da_correction = xr.DataArray(correction_diff,coords=corcoords,dims=corcoords,name="correction_factor")
    da_eofs_filt  = xr.DataArray(eofs_filtered,coords=eofcoords,dims=eofcoords  ,name=vnames[v])
    
    ds_out        = xr.merge([da_correction,da_eofs_filt])
    edict         = proc.make_encoding_dict(ds_out)
    
    # Save for all ensemble members
    savename       = proc.addstrtoext(outpath+ncnames[v],"_corrected",adjust=-1)
    ds_out.to_netcdf(savename,encoding=edict)
    
    savename_emean = proc.addstrtoext(savename,"_EnsAvg",adjust=-1)
    ds_out_ensavg  = ds_out.mean('ens')
    ds_out_ensavg.to_netcdf(savename_emean,encoding=edict)

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
