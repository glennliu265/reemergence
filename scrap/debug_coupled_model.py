#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Debug the Coupled Model

Copy upper section from viz_SST_SSS_coupling

Created on Wed May 29 09:29:16 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']

#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm

#%% Indicate some parameters

lonf = -30
latf = 50

lags = np.arange(37)

#%% Make sone functions


def load_sm_pt(expname,lonf,latf,output_path):
    # Function to Load Basinwide Stochastic Model Output and subset to a point
    # assumes only the basinwide output is in the output folder (glob searches for *run*.nc...)
    expdir  = "%s%s/Output/" % (output_path,expname)
    nclist  = glob.glob(expdir + "*run*.nc")
    nclist.sort()
    nruns   = len(nclist)
    ds_byrun = []
    for rr in range(nruns):
        ds = proc.selpt_ds(xr.open_dataset(nclist[rr]),lonf,latf).load()
        ds_byrun.append(ds)
    ds_byrun = xr.concat(ds_byrun,dim='run')
    return ds_byrun


def load_expdict(expname,output_path):
    expdir  = "%s%s/Input/" % (output_path,expname)
    expdict = np.load(expdir + "expparams.npz",allow_pickle=True)
    
    # Convert all inputs into string, if needed
    keys = expdict.files
    expdict_out = {}
    for key in keys:
        if type(expdict[key]) == np.ndarray and (len(expdict[key].shape) < 1): #  Check if size is zero, andnumpy array is present
            expdict_out[key] = expdict[key].item()
        else:
            expdict_out[key] = expdict[key]
    
    return expdict_out


def load_params(expparams,debug=False):
    # Copied run_SSS_pointmode_coupled
    # Loads and checks for all necessary inputs of the stochastic model
    
    # To Do: Return eof_flag...
    
    # First, Check if there is EOF-based forcing (remove this if I eventually redo it)
    if expparams['eof_forcing']:
        print("EOF Forcing Detected.")
        eof_flag = True
    else:
        eof_flag = False
    
    # Indicate the Parameter Names (sorry, it's all hard coded...)
    if expparams['varname']== "SSS": # Check for LHFLX, PRECTOT, Sbar
        chk_params = ["h","LHFLX","PRECTOT","Sbar","lbd_d","beta","kprev","lbd_a","Qek"]
        param_type = ["mld","forcing","forcing","forcing","damping","mld","mld","damping",'forcing']
    elif expparams['varname'] == "SST": # Check for Fprime
        chk_params = ["h","Fprime","lbd_d","beta","kprev","lbd_a","Qek"]
        param_type = ["mld","forcing","damping","mld","mld","damping",'forcing']
    
    # Check the params
    ninputs       = len(chk_params)
    inputs_ds     = {}
    inputs        = {}
    inputs_type   = {}
    missing_input = []
    for nn in range(ninputs):
        # Get Parameter Name and Type
        pname = chk_params[nn]
        ptype = param_type[nn]
        
        # Check for Exceptions (Can Fix this in the preprocessing stage)
        if pname == 'lbd_a':
            da_varname = 'damping'
        else:
            da_varname = pname
        
        # Load DataArray
        if type(expparams[pname])==str: # If String, Load from input folder
            
            # Load ds
            if (expparams['varname'] == "SST") and (pname =="Fprime") and "Fprime" not in expparams[pname]:
                
                da_varname   = "LHFLX" # Swap to LHFLX for now
                varname_swap = True # True so "Fprime" is input as da_varname later
                swapname     = "Fprime" 
            
            else:
                varname_swap = False
                
                #inputs_type['Fprime'] = 'forcing' # Add extra Fprime variable
            ds = xr.open_dataset(input_path + ptype + "/" + expparams[pname])[da_varname]
            
            
            # Crop to region
            
            # Load dataarrays for debugging
            dsreg            = proc.selpt_ds(ds,lonf,latf)#proc.sel_region_xr(ds,expparams['bbox_sim']).load()
            inputs_ds[pname] = dsreg.copy() 
            
            
            # Load to numpy arrays 
            varout           = dsreg.values
            varout           = varout[...,None,None]
            if debug:
                print(pname) # Name of variable
                print("\t%s" % str(ds.shape)) # Original Shape
                print("\t%s" % str(varout.shape)) # Point Shape
            inputs[pname]    = varout.copy()
            
            if ((da_varname == "Fprime") and (eof_flag)) or ("corrected" in expparams[pname]):
                print("Loading %s correction factor for EOF forcing..." % pname)
                ds_corr                          = xr.open_dataset(input_path + ptype + "/" + expparams[pname])['correction_factor']
                ds_corr_reg                      = proc.selpt_ds(ds_corr,lonf,latf).load()
                
                
                if varname_swap == True:
                    da_varname = pname # Swap from LHFLX back to Fprime for SST Integration
                    
                # set key based on variable type
                if da_varname == "Fprime":
                    keyname = "correction_factor"
                elif da_varname == "LHFLX":
                    keyname = "correction_factor_evap"
                elif da_varname == "PRECTOT":
                    keyname = "correction_factor_prec"
                
                inputs_ds[keyname]   = ds_corr_reg.copy()
                inputs[keyname]      = ds_corr_reg.values.copy()[...,None,None]
                if debug:
                    print(da_varname + " Corr") # Variable Name
                    print("\t%s" % str(ds_corr.shape))
                    print("\t%s" % str(inputs[keyname].shape)) # Corrected Shape
                inputs_type[keyname] = "forcing"
            
        else:
            print("Did not find %s" % pname)
            missing_input.append(pname)
        
        inputs_type[pname] = ptype
    
    #% Detect and Process Missing Inputs
    
    _,nlat,nlon=inputs['h'].shape
    
    for pname in missing_input:
        if type(expparams[pname]) == float:
            print("Float detected for <%s>. Making array with the repeated value %f" % (pname,expparams[pname]))
            inputs[pname] = np.ones((12,nlat,nlon)) * expparams[pname]
        else:
            print("No value found for <%s>. Setting to zero." % pname)
            inputs[pname] = np.zeros((12,nlat,nlon))
    
    # Get number of modes
    if eof_flag:
        if expparams['varname'] == "SST":
            nmode = inputs['Fprime'].shape[0]
        elif expparams['varname'] == "SSS":
            nmode = inputs['LHFLX'].shape[0]
            
    # Unpack things from dictionary (added after for debugging)
    ninputs     = len(inputs_ds)
    param_names = list(inputs_ds.keys())
    params_vv   = [] # Unpack from dictionary
    for ni in range(ninputs):
        pname = param_names[ni]
        dsin  = inputs_ds[pname]
        params_vv.append(dsin.copy())
    return inputs,inputs_ds,inputs_type,params_vv


#%% Load Output from a model (full basin run)

"""

Notes: TBegin by check ACFs for the full basin run for SST/SSS
Is there actually re-emergence in these runs? Timestep is [12000 months x 10 runs]

"""

expname      = "SSS_EOF_LbddCorr_Rerun_lbdE"
ds_sss_basin = load_sm_pt(expname,lonf,latf,output_path)


expname      = "SST_EOF_LbddCorr_Rerun"
ds_sst_basin = load_sm_pt(expname,lonf,latf,output_path)

ds_basin = [ds_sst_basin,ds_sss_basin]
vnames   = ["SST","SSS"]
vcolors  = ["hotpink","navy"]

expnames_basin = ["SST_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_lbdE",]

#%% Compute Re-emergence (basinwide script)

acfs_basin = np.zeros((2,10,12,len(lags))) * np.nan
for vv in range(2):
    
    acfs_byrun = []
    
    for rr in range(10):
        
        tsin = ds_basin[vv].isel(run=rr)[vnames[vv]].values
        
        acf  = scm.calc_autocorr_mon(tsin,lags,verbose=False,return_da=False)
        acfs_basin[vv,rr,:,:] = acf.copy()
#ax.set_title("")


        
#%% Load corresponding timeseries from pointwise run

expname_pt         = "SST_SSS_LHFLX"

# Copied below from [viz_SSS_SST_coupling.py] ---------------------------------

metrics_path = output_path + expname_pt + "/Metrics/" 
exp_output   = output_path + expname_pt + "/Output/" 

# For some reason, 2 lat values are saved for SSS (50 and 50.42). 
# Need to fix this
ds_all  = []
var_all = []
for vv in range(2):
    
    globstr = "%s%s_runid*.nc" % (exp_output,vnames[vv])
    nclist  = glob.glob(globstr)
    nclist.sort()
    ds      = xr.open_mfdataset(nclist,combine='nested',concat_dim="run").load()
    
    if len(ds.lat) > 1: # Related to SSS error...
        print("Fixing latitude index for SSS")
        remake_ds = []
        for nr in range(len(ds.run)):
            invar = ds.isel(run=nr)[vnames[vv]]
            
            if np.all(np.isnan(invar.isel(lat=0))): 
                klat = 1
            if np.all(np.isnan(invar.isel(lat=1))):
                klat = 0
            print("Non-NaN Latitude Index was %i for run %i" % (klat,nr))
            invar = invar.isel(lat=klat)
            #invar['lat'] = 50.
            remake_ds.append(invar.values.copy())
        coords = dict(run=ds.run,time=ds.time)
        ds     = xr.DataArray(np.array(remake_ds).squeeze(),coords=coords,name=vnames[vv])
    else:
        ds = ds[vnames[vv]]
    
    #.sel(lat=50.42,method='nearest')
    ds_all.append(ds)
    var_all.append(ds.values.squeeze()) # [Run x Time]
    

# End Copy from [viz_SSS_SST_coupling.py] -------------------------------------

# Compute Re-emergence (for pointwise script)
acfs_pt = np.zeros((2,10,12,len(lags))) * np.nan
for vv in range(2):
    
    acfs_byrun = []
    
    for rr in range(10):
        
        tsin = var_all[vv][rr,:]#ds_basin[vv].isel(run=rr)[vnames[vv]].values
        
        acf  = scm.calc_autocorr_mon(tsin,lags,verbose=False,return_da=False)
        acfs_pt[vv,rr,:,:] = acf.copy()

#%% Plot ACFs
    
kmonth = 1

fig,ax  = plt.subplots(1,1,figsize=(14.5,6),constrained_layout=True)
ax,_    = viz.init_acplot(kmonth,lags,lags,ax=ax)


for ex in range(2):
    
    if ex == 0:
        in_acfs = acfs_basin
        ls      = 'solid'
    else:
        in_acfs = acfs_pt
        ls      = 'dashed'
        
    for vv in range(2):
        
        for rr in range(10):
            
            ax.plot(lags,in_acfs[vv,rr,kmonth,:],c=vcolors[vv],lw=1.5,alpha=0.5,ls=ls)


#%%



"""
Notes: There seems to be re-emergence in the basinwide runs at the SPG point.
Why isn't it happening for the coupled simulation? 


Let's first check the parameters...

    If the parameters are different, this might be the culprit
    If not, well it's probably the code...

"""


#%% Get the parameters from the full run

expdict_basin     = []
params_basin      = []
params_basin_name = []

for vv in range(2):
    
    # Get Experiment Dictionary
    expname = expnames_basin[vv]
    expdict = load_expdict(expname,output_path)
    
    # Load Parameters
    inputs,inputs_ds,inputs_type,params_vv = load_params(expdict)
    
    # Silly Fix (for double correction factor)
    pnames = list(inputs_ds.keys())
    if vv == 1:
        idprec = pnames.index('correction_factor_prec')
        params_vv[idprec] = params_vv[idprec].rename("correction_factor_prec")
    params_vv = xr.merge(params_vv)
    
    
    #Append and Save
    expdict_basin.append(expdict)
    params_basin.append(params_vv)
    params_basin_name.append(pnames)

    

#%% Load and compare the coupled experiment parameters

expname_cpl = "SST_SSS_LHFLX"
params_cpl  = []
for vv in range(2):
    cplpath = output_path + "/" + expname_cpl +"/Input/"
    ncname = "%s%s_params.nc" % (cplpath,vnames[vv])
    dsparam = xr.open_dataset(ncname).load()
    params_cpl.append(dsparam)

#%% Now, compare everything

mons3      = proc.get_monstr()
param_sets = []
skip_vars  = ["lon","lat","mode",'mon']

mmcol      = ["navy","orange"]
mmls         = ["solid","dashed"]

mmname    = ["SST-SSS couple run","basinwide"]
for vv in range(2):
    
    paramin_cpl   = params_cpl[vv]
    paramin_basin = params_basin[vv] 
    compare_vars  = list(paramin_cpl.variables)
    
    mm_in         = [paramin_cpl,paramin_basin]
    
    nvars = len(compare_vars)
    for nv in range(nvars):
        pname = compare_vars[nv]
        if pname in skip_vars:
            continue
        
        fig,ax = viz.init_monplot(1,1,figsize=(6.5,4))
        
        
        for mm in range(2):
            if (vv == 0) and (mm == 1) and (pname == "LHFLX"):
                pname = "Fprime"
            
            plotvar = mm_in[mm][pname]
            if 'mode' in list(plotvar.dims):
                plotvar = np.sqrt((plotvar**2).sum('mode'))
                
            ax.plot(mons3,plotvar,label=mmname[mm],c=mmcol[mm],ls=mmls[mm])
        ax.legend()
        ax.set_title("%s (%s)" % (vnames[vv],pname))

#%%



