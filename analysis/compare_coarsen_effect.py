#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:24:32 2024

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
import cartopy.crs as ccrs



# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine    = "Astraeus"

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


#%% Add some functions


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
            ds = ds.drop_duplicates('lon')
            
            # Crop to region
            
            # Load dataarrays for debugging
            dsreg            = proc.sel_region_xr(ds,expparams['bbox_sim']).load()
            inputs_ds[pname] = dsreg.copy() 
            
            
            # Load to numpy arrays 
            varout           = dsreg.values
            #varout           = varout[...,None,None]
            if debug:
                print(pname) # Name of variable
                print("\t%s" % str(ds.shape)) # Original Shape
                print("\t%s" % str(varout.shape)) # Point Shape
            inputs[pname]    = varout.copy()
            
            if ((da_varname == "Fprime") and (eof_flag)) or ("corrected" in expparams[pname]):
                print("Loading %s correction factor for EOF forcing..." % pname)
                ds_corr                          = xr.open_dataset(input_path + ptype + "/" + expparams[pname])['correction_factor']
                ds_corr                          = ds_corr.drop_duplicates('lon')
                ds_corr_reg                      = proc.sel_region_xr(ds_corr,expparams['bbox_sim']).load()#.selpt_ds(ds_corr,lonf,latf).load()
                
                
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
    print(inputs['h'].shape)
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
            
    # New Section: Check for SST-Evaporation Feedback ------------------------
    if 'lbd_e' in expparams.keys() and expparams['varname'] == "SSS":
        if expparams['lbd_e'] is not None: 
            print("Adding SST-Evaporation Forcing on SSS!")
            # Load lbd_e
            lbd_e   = xr.open_dataset(input_path + "forcing/" + expparams['lbd_e']).lbd_e.load() # [mon x lat x lon]
            lbd_e   = proc.sel_region_xr(lbd_e,bbox=expparams['bbox_sim'])
            
            inputs['lbd_e'] = lbd_e.data
            inputs_ds['lbd_e'] = lbd_e
            inputs_type['lbd_e']='damping'
            
            #
            #inputs_type
            
            
    # Unpack things from dictionary (added after for debugging)
    ninputs     = len(inputs_ds)
    param_names = list(inputs_ds.keys())
    params_vv   = [] # Unpack from dictionary
    for ni in range(ninputs):
        pname = param_names[ni]
        dsin  = inputs_ds[pname]
        params_vv.append(dsin.copy())
        
    
    return inputs,inputs_ds,inputs_type,params_vv



#%% sm Edits

sm_set1 = ["SST_CESM1_5deg_lbddcoarsen_rerun","SSS_CESM1_5deg_lbddcoarsen"]
sm_set2 = ["SST_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_lbdE_neg"]
vnames  = ["SST","SSS"]
ds_ref    = []
ds_coarse = []

for ii in range(2):
    print(ii)
    
    dref        = dl.load_smoutput(sm_set2[ii],output_path).load()
    ds_ref.append(dref[vnames[ii]])
    
    dcoar       = dl.load_smoutput(sm_set1[ii],output_path).load()
    ds_coarse.append(dcoar[vnames[ii]])
    
    
#%% Load Experimenet Dictionaries and parameters

# Get the experiment dictionaries for each
expnames        = sm_set1 + sm_set2
expdicts        = [load_expdict(exn,output_path) for exn in expnames]

expparams       = [load_params(exn) for exn in expdicts]
expparams_ds    = [expd[-1] for expd in expparams]

#%% Compare Timeseries at a point

ds_all  = ds_ref + ds_coarse
dsnames = ["SST","SSS","SST_coarse","SSS_coarse"] 
ds_all  = [ds.drop_duplicates('lon') for ds in ds_all]


lonf    = -30
latf    = 50
locfn,loctitle=proc.make_locstring(lonf,latf)
dspt    = [proc.selpt_ds(ds,lonf,latf) for ds in ds_all]


lags    =   np.arange(37)
pct     =   0.10
nsmooth =   110

tsm_all = []
for ii in range(4):
    
    dsin    = dspt[ii].data
    tsin    = [dsin[rr,:] for rr in range(10)]
    tsm     = scm.compute_sm_metrics(tsin,nsmooth=nsmooth,lags=lags,pct=pct)
    tsm_all.append(tsm)

sstm    = tsm_all[0]
sssm    = tsm_all[1]
csstm   = tsm_all[2]
csssm   = tsm_all[3]


#%% Compute Metrics

# ---------------------
#%% Spatial Comparisons
# ---------------------





#%% Compare Overall Variance


ds_vars = [ds.var('time').mean('run') for ds in ds_all]


#%%

bboxplot        = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3           = proc.get_monstr(nletters=3)

fsz_tick        = 18
fsz_axis        = 20
fsz_title       = 16

#rhocrit = proc.ttest_rho(0.05,2,86)
proj        = ccrs.PlateCarree()
vcolors     = "gray"

#%% Compare Pointwise Variance
fig,axs,_ = viz.init_orthomap(2,2,bboxplot,figsize=(20,14.5))

for ii in range(4):
    
    if ii%2 == 0:
        vlims = [0,1]
        cmap  = 'cmo.thermal'
    else:
        vlims = [0,0.010]
        cmap  = 'cmo.haline'
    ax  = axs.flatten()[ii]
    ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    
    ax.set_title(dsnames[ii],fontsize=fsz_title)
    pv = ds_vars[ii]
    pcm = ax.pcolormesh(pv.lon,pv.lat,pv.data,
                        vmin=vlims[0],vmax=vlims[1],
                        cmap=cmap,transform=proj)
    cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
    
savename = "%sPointwise_Variance_Coarse_v_Original.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% compare monthly variance

ds_monvar = [ds.groupby('time.month').var('time') for ds in dspt]

fig,axs   = viz.init_monplot(2,1)
for ii in range(4):
    aa = ii%2
    ax = axs[aa]
    plotvar = ds_monvar[ii]
    mu      = plotvar.mean('run')
    for rr in range(10):
        ax.plot(mons3,plotvar.isel(run=rr),alpha=0.1)
    ax.plot(mons3,mu,label=dsnames[ii])    
    ax.legend()
ax.set_title("Monthly Variance at 50N, 30W")

savename = "%sMonthlyVarianceSPGPoint.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Compare ACF

im = 1
lags = np.arange(37)
xtks = lags[::1]

fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(10.5,4.5))
ax,_    = viz.init_acplot(im,xtks,lags,ax=ax,title="")

for ii in range(4):
    acf_exp = np.array(tsm_all[ii]['acfs'])[im,...] # [mon x run x lags]
    
    for rr in range(10):
        ax.plot(lags,acf_exp[rr,:],alpha=0.1)
    mu = acf_exp.mean(0)
    ax.plot(lags,mu,label=dsnames[ii])
    
ax.legend()


savename = "%sFebACFSPGPoint_%s.png" % (figpath,locfn)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Compare ACF Maps

vlms = [-.25,.25]

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(10.5,8.5))

# Plot SST
ax = axs[0] #SST
acf_origin =  np.array(sstm['acfs']).mean(1)
acf_coarse =  np.array(csstm['acfs']).mean(1)
pv         = acf_origin - acf_coarse
pcm = ax.pcolormesh(lags,mons3,pv,
                    cmap='cmo.balance',vmin=vlms[0],vmax=vlms[1],
                    edgecolors="lightgray")
ax.set_title("SST")

# Plot SSS
ax = axs[1] #SST
acf_origin =  np.array(sssm['acfs']).mean(1)
acf_coarse =  np.array(csssm['acfs']).mean(1)
pv         = acf_origin - acf_coarse
pcm = ax.pcolormesh(lags,mons3,pv,
                    cmap='cmo.balance',vmin=vlms[0],vmax=vlms[1],
                    edgecolors="lightgray")

cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.set_label("Correlation Difference (Original - Coarse)")
ax.set_title("SSS")

for ax in axs:
    ax.set_xticks(xtks)
    ax.tick_params(labelsize=14)
    ax.set_aspect('equal')
    ax.invert_yaxis()
plt.suptitle("Correlation Difference @ %s" % (loctitle))
    
savename = "%sFebACFMapPoint_%s.png" % (figpath,locfn)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# -----------------------------------------------------------------------------
#%% Compare Inputs at a point
# -----------------------------------------------------------------------------
"""
h
Fprime
correction_factor
lbd_d
kprev
damping
Qek

"""

# Indicate the parameter dictionaries
sstp    = xr.merge(expparams_ds[2],join='override')
sssp    = xr.merge(expparams_ds[3],compat='override',join='override') # Note Correction Factors are combined...
csstp   = xr.merge(expparams_ds[0],join='override')
csssp   = xr.merge(expparams_ds[1],compat='override',join='override')

def printep(ep):
    [print(ds.name) for ds in ep]
    
#%% Compare Damping (and mld)

dt   = 3600*24*30


fig,axs = viz.init_monplot(5,1,figsize=(6,12))

ax = axs[0]
ax.set_title("Net Heat Flux Feedback [W/m2/degC]")
ax.plot(mons3,proc.selpt_ds(sstp,lonf,latf,).damping.data,label="Original")
ax.plot(mons3,proc.selpt_ds(csstp,lonf,latf,).damping.data,label="Coarsened")
ax.legend()

ax = axs[1]
ax.set_title("$\lambda^d$ (SST) [corr(detrain,entrain)]")
ax.plot(mons3,proc.selpt_ds(sstp,lonf,latf,).lbd_d.data,label="Original")
ax.plot(mons3,proc.selpt_ds(csstp,lonf,latf,).lbd_d.data,label="Coarsened")
ax.legend()

ax = axs[2]
ax.set_title("$\lambda^d$ (SSS) [corr(detrain,entrain)]")
ax.plot(mons3,proc.selpt_ds(sssp,lonf,latf,).lbd_d.data,label="Original")
ax.plot(mons3,proc.selpt_ds(csssp,lonf,latf,).lbd_d.data,label="Coarsened")
ax.legend()

ax = axs[3]
ax.set_title("SST-Evaporation Feedback [$psu \, per \, \degree C$]")
ax.plot(mons3,proc.selpt_ds(sssp,lonf,latf,).lbd_e.data*dt,label="Original")
ax.plot(mons3,proc.selpt_ds(csssp,lonf,latf,).lbd_e.data*dt,label="Coarsened")
ax.legend()

ax = axs[4]
ax.set_title("Mixed Layer Depth (HMXL) [h]")
ax.plot(mons3,proc.selpt_ds(sstp,lonf,latf,).h.data,label="Original")
ax.plot(mons3,proc.selpt_ds(csstp,lonf,latf,).h.data,label="Coarsened")
ax.legend()

plt.suptitle("Damping Comparison (Original vs. Coarsened)\n%s" % loctitle)

savename = "%sComparison_Damp_v_Coarsened.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Compare Forcing

def rmse(ds):
    return ((ds**2).sum('mode'))**(0.5)

fig,axs = viz.init_monplot(5,1,figsize=(6,12))

ax = axs[0]
ax.set_title("Stochastic Heat Flux (F') [W/m2]")
ax.plot(mons3,rmse(proc.selpt_ds(sstp,lonf,latf,).Fprime).data,label="Original")
ax.plot(mons3,rmse(proc.selpt_ds(csstp,lonf,latf,).Fprime).data,label="Coarsened")
ax.legend()

ax = axs[1]
ax.set_title("$Q_{ek}$ (SST) (F') [W/m2]")
ax.plot(mons3,rmse(proc.selpt_ds(sstp,lonf,latf,).Qek).data,label="Original")
ax.plot(mons3,rmse(proc.selpt_ds(csstp,lonf,latf,).Qek).data,label="Coarsened")
ax.legend()

ax = axs[2]
ax.set_title("$Q_{ek}$ (SSS) (F') [psu/mon]")
ax.plot(mons3,rmse(proc.selpt_ds(sssp,lonf,latf,).Qek).data*dt,label="Original")
ax.plot(mons3,rmse(proc.selpt_ds(csssp,lonf,latf,).Qek).data*dt,label="Coarsened")
ax.legend()

ax = axs[3]
ax.set_title("Precipitation (P') [psu/mon]")
ax.plot(mons3,rmse(proc.selpt_ds(sssp,lonf,latf,).PRECTOT).data*dt,label="Original")
ax.plot(mons3,rmse(proc.selpt_ds(csssp,lonf,latf,).PRECTOT).data*dt,label="Coarsened")
ax.legend()

ax = axs[4]
ax.set_title("Stochastic LHFLX (E') [W/m2]")
ax.plot(mons3,rmse(proc.selpt_ds(sssp,lonf,latf,).LHFLX).data,label="Original")
ax.plot(mons3,rmse(proc.selpt_ds(csssp,lonf,latf,).LHFLX).data,label="Coarsened")
ax.legend()


plt.suptitle("Forcing Comparison (Original vs. Coarsened)\n%s" % loctitle)

savename = "%sComparison_Force_v_Coarsened.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Compare Seasonal Average Maps of each parameter

# Function to Initialize Map
def init_smap():
    fig,axs,_ = viz.init_orthomap(2,4,bboxplot,figsize=(24,10))
    
    #for a,ax in enumerate(axs.flatten()):
    
    fsz_title = 24
    fsz_ticks = 14
    
    snames = ["DJF","MAM","JJA","SON"]
    vnames = ["Original","Coarsened"]
    for vv in range(2):
        for sid in range(4):
            ax           = axs[vv,sid]
            ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                                            fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
            #cb = viz.hcbar(pcm,ax=ax,fraction = 0.045)
            
            #cb.ax.tick_params(labelsize=fsz_ticks)
            if sid == 0:
                viz.add_ylabel(vnames[vv],ax=ax,fontsize=fsz_title,x=-0.15)
            if vv == 0:
                ax.set_title(snames[sid],fontsize=fsz_title)
    return fig,axs



#%% Plot net heat Flux feedback 
fsz_ticks   = 14
vname       = "damping"
title       = "Net Heat Flux Feedback [W/m2/degC]"
vlms        = [-40,40]
ds_in       = [sstp,csstp]
v_savg      = [proc.calc_savg_mon(ds[vname]) for ds in ds_in] # Take Seasonal Average
fig,axs     = init_smap()

for vv in range(2):
    for sid in range(4):
        ax  = axs[vv,sid]
        pv  = v_savg[vv].isel(season=sid)
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,cmap='cmo.balance',
                            vmin=vlms[0],vmax=vlms[1],
                            transform=proj)
        cb = viz.hcbar(pcm,ax=ax,fraction = 0.045)
        cb.ax.tick_params(labelsize=fsz_ticks)
        
dpath = input_path + "damping/"
test = xr.open_dataset(dpath + 'CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc')
plt.suptitle(title,fontsize=32)

savename = "%sCoarsen_Comparison_%s.png" % (figpath,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Detrainment Damping (SST)
fsz_ticks   = 14
vname       = "lbd_d"
vname_save  = "SSTlbdd"
title       = "SST Detrainment Damping [corr(detrain,entrain)]"
vlms        = [0,1]
cmap        = 'cmo.tempo'
ds_in       = [sstp,csstp]
v_savg      = [proc.calc_savg_mon(ds[vname]) for ds in ds_in] # Take Seasonal Average
fig,axs     = init_smap()

for vv in range(2):
    for sid in range(4):
        ax  = axs[vv,sid]
        pv  = v_savg[vv].isel(season=sid)
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,cmap=cmap,
                            vmin=vlms[0],vmax=vlms[1],
                            transform=proj)
        cb = viz.hcbar(pcm,ax=ax,fraction = 0.045)
        cb.ax.tick_params(labelsize=fsz_ticks)
        

plt.suptitle(title,fontsize=32)

savename = "%sCoarsen_Comparison_%s.png" % (figpath,vname_save)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Detrainment Damping (SSS)
fsz_ticks   = 14
vname       = "lbd_d"
vname_save  = "SSSlbdd"
title       = "SSS Detrainment Damping [corr(detrain,entrain)]"
vlms        = [0,1]
cmap        = 'cmo.tempo'
ds_in       = [sssp,csssp]
v_savg      = [proc.calc_savg_mon(ds[vname]) for ds in ds_in] # Take Seasonal Average
fig,axs     = init_smap()

for vv in range(2):
    for sid in range(4):
        ax  = axs[vv,sid]
        pv  = v_savg[vv].isel(season=sid)
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,cmap=cmap,
                            vmin=vlms[0],vmax=vlms[1],
                            transform=proj)
        cb = viz.hcbar(pcm,ax=ax,fraction = 0.045)
        cb.ax.tick_params(labelsize=fsz_ticks)
        

plt.suptitle(title,fontsize=32)

savename = "%sCoarsen_Comparison_%s.png" % (figpath,vname_save)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% SST Evaporation Feedback

fsz_ticks   = 14
vname       = "lbd_e"
vname_save  = "lbd_e"
title       = "SST-Evaporation Feedback [psu/degC/mon]"
vlms        = [0,0.02]
cmap        = 'cmo.matter'
ds_in       = [sssp,csssp]
v_savg      = [proc.calc_savg_mon(ds[vname]) * dt for ds in ds_in] # Take Seasonal Average
fig,axs     = init_smap()

for vv in range(2):
    for sid in range(4):
        ax  = axs[vv,sid]
        pv  = v_savg[vv].isel(season=sid)
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,cmap=cmap,
                            vmin=vlms[0],vmax=vlms[1],
                            transform=proj)
        cb = viz.hcbar(pcm,ax=ax,fraction = 0.045)
        cb.ax.tick_params(labelsize=fsz_ticks)
        

plt.suptitle(title,fontsize=32)

savename = "%sCoarsen_Comparison_%s.png" % (figpath,vname_save)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Mixed Layer Depth

fsz_ticks   = 14
vname       = "h"
vname_save  = "h"
title       = "Mixed-layer Depth [meters]"
vlms        = [0,1000]
cmap        = 'cmo.deep'
ds_in       = [sssp,csssp]
v_savg      = [proc.calc_savg_mon(ds[vname]) for ds in ds_in] # Take Seasonal Average
fig,axs     = init_smap()

for vv in range(2):
    for sid in range(4):
        ax  = axs[vv,sid]
        pv  = v_savg[vv].isel(season=sid)
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,cmap=cmap,
                            vmin=vlms[0],vmax=vlms[1],
                            transform=proj)
        cb = viz.hcbar(pcm,ax=ax,fraction = 0.045)
        cb.ax.tick_params(labelsize=fsz_ticks)
        

plt.suptitle(title,fontsize=32)

savename = "%sCoarsen_Comparison_%s.png" % (figpath,vname_save)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Fprime

fsz_ticks   = 14
vname       = "Fprime"
vname_save  = "Fprime"
title       = "Stochastic Heat Flux Forcing [W/m2]"
vlms        = [0,50]
cmap        = 'inferno'
ds_in       = [sstp,csstp]
v_savg      = [rmse(proc.calc_savg_mon(ds[vname])) for ds in ds_in] # Take Seasonal Average
fig,axs     = init_smap()

for vv in range(2):
    for sid in range(4):
        ax  = axs[vv,sid]
        pv  = v_savg[vv].isel(season=sid)
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,cmap=cmap,
                            vmin=vlms[0],vmax=vlms[1],
                            transform=proj)
        cb = viz.hcbar(pcm,ax=ax,fraction = 0.045)
        cb.ax.tick_params(labelsize=fsz_ticks)
        

plt.suptitle(title,fontsize=32)

savename = "%sCoarsen_Comparison_%s.png" % (figpath,vname_save)
plt.savefig(savename,dpi=150,bbox_inches='tight') 



#%% Qek (SST_)

fsz_ticks   = 14
vname       = "Qek"
vname_save  = "QekSST"
title       = "Ekman Forcing (SST) [W/m2]"
vlms        = [0,50]
cmap        = 'cmo.thermal'
ds_in       = [sstp,csstp]
v_savg      = [rmse(proc.calc_savg_mon(ds[vname])) for ds in ds_in] # Take Seasonal Average
fig,axs     = init_smap()

for vv in range(2):
    for sid in range(4):
        ax  = axs[vv,sid]
        pv  = v_savg[vv].isel(season=sid)
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,cmap=cmap,
                            vmin=vlms[0],vmax=vlms[1],
                            transform=proj)
        cb = viz.hcbar(pcm,ax=ax,fraction = 0.045)
        cb.ax.tick_params(labelsize=fsz_ticks)
        

plt.suptitle(title,fontsize=32)

savename = "%sCoarsen_Comparison_%s.png" % (figpath,vname_save)
plt.savefig(savename,dpi=150,bbox_inches='tight') 
 


#%% Qek (SSS_)

fsz_ticks   = 14
vname       = "Qek"
vname_save  = "QekSSS"
title       = "Ekman Forcing (SST) [W/m2]"
vlms        = [0,0.05]
cmap        = 'cmo.haline'
ds_in       = [sssp,csssp]
v_savg      = [rmse(proc.calc_savg_mon(ds[vname]))*dt for ds in ds_in] # Take Seasonal Average
fig,axs     = init_smap()

for vv in range(2):
    for sid in range(4):
        ax  = axs[vv,sid]
        pv  = v_savg[vv].isel(season=sid)
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,cmap=cmap,
                            vmin=vlms[0],vmax=vlms[1],
                            transform=proj)
        cb = viz.hcbar(pcm,ax=ax,fraction = 0.045)
        cb.ax.tick_params(labelsize=fsz_ticks)
        

plt.suptitle(title,fontsize=32)

savename = "%sCoarsen_Comparison_%s.png" % (figpath,vname_save)
plt.savefig(savename,dpi=150,bbox_inches='tight') 



#%% LHFLX

fsz_ticks   = 14
vname       = "LHFLX"
vname_save  = "LHFLX"
title       = "Evaporation (LHFLX) [W/m2]"
vlms        = [0,50]
cmap        = 'cmo.amp'
ds_in       = [sssp,csssp]
v_savg      = [rmse(proc.calc_savg_mon(ds[vname])) for ds in ds_in] # Take Seasonal Average
fig,axs     = init_smap()

for vv in range(2):
    for sid in range(4):
        ax  = axs[vv,sid]
        pv  = v_savg[vv].isel(season=sid)
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,cmap=cmap,
                            vmin=vlms[0],vmax=vlms[1],
                            transform=proj)
        cb = viz.hcbar(pcm,ax=ax,fraction = 0.045)
        cb.ax.tick_params(labelsize=fsz_ticks)
        

plt.suptitle(title,fontsize=32)

savename = "%sCoarsen_Comparison_%s.png" % (figpath,vname_save)
plt.savefig(savename,dpi=150,bbox_inches='tight') 

#%% Precip

fsz_ticks   = 14
vname       = "PRECTOT"
vname_save  = "PRECTOT"
title       = "Precipitation Forcing (P') [psu/mon]"
vlms        = [0,0.025]
cmap        = 'cmo.rain'
ds_in       = [sssp,csssp]
v_savg      = [rmse(proc.calc_savg_mon(ds[vname]))*dt for ds in ds_in] # Take Seasonal Average
fig,axs     = init_smap()

for vv in range(2):
    for sid in range(4):
        ax  = axs[vv,sid]
        pv  = v_savg[vv].isel(season=sid)
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,cmap=cmap,
                            vmin=vlms[0],vmax=vlms[1],
                            transform=proj)
        cb = viz.hcbar(pcm,ax=ax,fraction = 0.045)
        cb.ax.tick_params(labelsize=fsz_ticks)
        

plt.suptitle(title,fontsize=32)

savename = "%sCoarsen_Comparison_%s.png" % (figpath,vname_save)
plt.savefig(savename,dpi=150,bbox_inches='tight') 



