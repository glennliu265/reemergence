#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Prepare Parameter Inputs for SSS Stochastic Model

Works with CESM1-HTR LENS variables [ens x time x lat x lon] processed with 


combine_precip.py (PRECTOT)
prep_data_byvariable_monthly (Other variables)


Steps:

(1) Load and process HMXL

(2) Load and process Precip Forcing

(3) Load and compute stochastic evaporation forcing (HFF, Temp, etc)

Created on Thu Feb  1 09:03:41 2024

@author: gliu
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

import tqdm
import time

#%% Import Custom Modules
amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl


#%% Import stochastic model parameters (delete this eventually)

#sys.path.append(scmpath + "../") # Not this is not working
#import stochmod_params as spm

#%%


def rename_var(da):
    format_dict = {
        'ensemble':'ens',
        'month':'mon',
        }
    da = da.rename(format_dict)
    return da

# Convenience function
def save_ens_all_avg(ds,savename,edict,adjust=-1):
    """
    Parameters
    ----------
    ds       : Target Data Array with 'ens' dimension
    savename : STR Name to save to 
    edict    : TEncoding dictionary
    
    """
    
    # First, save for all members
    ds.to_netcdf(savename,encoding=edict)
    print("Saved variable to %s!" % savename)
    
    # Then, save for ens avg.
    ds_ensavg    = ds.mean('ens')
    sname_ensavg = proc.addstrtoext(savename,"_EnsAvg",adjust=adjust)
    ds_ensavg.to_netcdf(sname_ensavg,encoding=edict)
    print("Saved Ens Avg to %s!" % sname_ensavg)
    
    return None


#%% Set Paths

# Path to variables processed by prep_data_byvariable_monthly
rawpath1 = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
ncstr1   = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc"

# Path to variables processed by combine_precip
rawpath2 = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/PRECIP/HTR_FULL/"
ncstr2   = "%s_HTR_FULL.nc"

# Output paths
mldpath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/mld/"
fpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/forcing/"
dpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/damping/"


# Bounding Boxes
bbox_crop     = [-90,20,0,90]  # Preprocessing box

# HFF Processing Options
tails      = 2
p          = 0.05
mode       = 5 # 1 = No Mask; 2 = SST autocorr; 3 --> SST-FLX cross corr; 4 = Both, 5 = Replace SLAB
sellags    = [0]
lagstr     = "lag1"

hffname    = "method%0i_%s_p%03i_tails%i" % (mode,lagstr,p,tails)

#%% Part 1: Load and process HMXL

# Out: h [ mon x <ens> x lat x lon]
# Load variable
ncmld = ncstr1 % "HMXL" 
dsh   = xr.open_dataset(rawpath1+ncmld).load()

# Double check first month
dsh   = proc.fix_febstart(dsh)

# Compute mean seasonal cycle
dsh_scycle = dsh.groupby('time.month').mean('time')

# Convert cm --> m
dsh_scycle = dsh_scycle/100

# Make encoding dict and rename dimensions
#edict   = proc.make_encoding_dict(dsh_scycle)
dsh_out = rename_var(dsh_scycle.HMXL) 
dsh_out = dsh_out.rename("h") 
edict   = {"h":{"zlib":True}}

# Save All Ensembles
savename = "%sCESM1_HTR_FULL_HMXL_NAtl.nc" % mldpath
dsh_out.to_netcdf(savename,encoding=edict) # h [ mon x ens x lat x lon]


# Save an ensemble mean versio
dsh_ensavg = dsh_out.mean('ens')
savename = "%sCESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc" % mldpath
dsh_ensavg.to_netcdf(savename,encoding=edict) # h [ mon x lat x lon]

#%% Compute Kprev

# Just Compute Kprev for the Ens. Avg Pattern
hcycle      = dsh_ensavg.values
_,nlat,nlon = hcycle.shape
kprev = np.zeros((12,nlat,nlon)) * np.nan
for o in range(nlon):
    for a in range(nlat):
        kpt,_ = scm.find_kprev(hcycle[:,a,o])
        kprev[:,a,o] = kpt.copy()

# Save the Output
cdict_mon = {
    'mon':np.arange(1,13,1),
    'lat':dsh_ensavg.lat.values,
    'lon':dsh_ensavg.lon.values,
    }
da_kprev = xr.DataArray(kprev,coords=cdict_mon,dims=cdict_mon,name="kprev")
edict   = {"kprev":{"zlib":True}}
savename = "%sCESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc" % mldpath
da_kprev.to_netcdf(savename,encoding=edict) # h [ mon x lat x lon]

#%% Part 2: Load and process Precipitation Forcing

# Load the files
ncprec = ncstr2 % "PRECTOT" 
dsp   = xr.open_dataset(rawpath2+ncprec).load()

# Remove seasonal cycle
dsp_anom = proc.xrdeseason(dsp)

# Detrend by removing ensemble average
dsp_dt   = dsp_anom - dsp_anom.mean('ens')

# Compute monthly stdev
dsp_monstd = dsp_dt.groupby('time.month').std('time')

# Get encoding dict (ens  dimensions already renamed)
dsp_out = dsp_monstd.PRECTOT.rename({'month':'mon'})
#dsp_out = dsp_out.rename("PREC") 
edict   = {"PRECTOT":{"zlib":True}}


# Save the output (All ensembles) # PRECTOT [ mon x ens x lat x lon]
savename = "%sCESM1_HTR_FULL_PRECTOT_NAtl.nc" % fpath
dsp_out.to_netcdf(savename,encoding=edict) # h [ mon x ens x lat x lon]

# Save the Ensemble Average
dsp_ensavg = dsp_out.mean('ens')
savename = "%sCESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc" % fpath
dsp_ensavg.to_netcdf(savename,encoding=edict) # h [ mon x lat x lon]

#%% Part 3: Compute Stochastic Evaporation Forcing

# Load TS, LHFLX
varnames =['SST','LHFLX']
ds_load = [xr.open_dataset(rawpath1+ ncstr1 % vn).load() for vn in varnames]

# Anomalize
ds_anom = [proc.xrdeseason(ds) for ds in ds_load]

# Detrend
ds_dt   = [ds-ds.mean('ensemble') for ds in ds_anom] # [ens x time x lat x lon]

# Transpose to [mon x ens x lat x lon]
ds_dt    = [ds.transpose('time','ensemble','lat','lon') for ds in ds_dt]

# Load LHFF
hff_nc   = "CESM1_HTR_FULL_LHFLX_damping_nomasklag1.nc"
hff_path = dpath
dshff    = xr.open_dataset(hff_path + hff_nc) # [mon x ens x lat x lon']

# Output to numpy
hff = dshff['damping'].values
sst = ds_dt[0].SST.values
ql = ds_dt[1].LHFLX.values

# Get dimension and tile
ntime,nens,nlat,nlon=ql.shape
nyrs = int(ntime/12)
hfftile=np.tile(hff.transpose(1,2,3,0),nyrs)
hfftile= hfftile.transpose(3,0,1,2)
# Check plt.pcolormesh(hfftile[0,0,:,:]-hfftile[12,0,:,:]),plt.colorbar(),plt.show()


#%% Calculate E' Convert
nroll    = 0
rollstr  = "nroll%0i"  % nroll

Eprime   = ql + hfftile*np.roll(sst,nroll)

#%% Check results

klon,klat = proc.find_latlon(-30,50,dshff.lon.values,dshff.lat.values)

#%% CHECK SPECTRA
checkts = [ql[:,0,klat,klon],Eprime[:,0,klat,klon]]
dtplot=3600*24*30
tsmetrics = scm.compute_sm_metrics(checkts)
fig,ax=plt.subplots(1,1,figsize=(12,4))
ax.plot(tsmetrics['freqs'][0]*dtplot,tsmetrics['specs'][0]/dtplot,label="QL")
ax.plot(tsmetrics['freqs'][1]*dtplot,tsmetrics['specs'][1]/dtplot,label="E'")
ax.legend()

plt.show()

# Check Monvar
fig,ax = plt.subplots(1,1)
ax.plot(np.arange(12)+1,tsmetrics['monvars'][0],label="QL")
ax.plot(np.arange(12)+1,tsmetrics['monvars'][1],label="E")
ax.legend()
plt.show()

#%% Save the Output

cdict_time = {
    'time':ds_dt[0].time.values,
    'ens':np.arange(1,43,1),
    'lat':dshff.lat.values,
    'lon':dshff.lon.values,
    }
edict = {"LHFLX":{"zlib":True}}

Eprime_da  = xr.DataArray(Eprime,coords=cdict_time,dims=cdict_time)


Eprime_std = Eprime_da.groupby('time.month').std('time')
Eprime_std = Eprime_std.rename("LHFLX")


# Save the output (All ensembles) # PRECTOT [ mon x ens x lat x lon]
savename = "%sCESM1_HTR_FULL_Eprime_%s_NAtl.nc" % (fpath,rollstr)
Eprime_std.to_netcdf(savename,encoding=edict) # h [ mon x ens x lat x lon]

# Save the Ensemble Average
Eprime_ensavg = Eprime_std.mean('ens')
savename = "%sCESM1_HTR_FULL_Eprime_%s_NAtl_EnsAvg.nc" % (fpath,rollstr)
Eprime_ensavg.to_netcdf(savename,encoding=edict) # h [ mon x lat x lon]

#%% Repeat for Latent Heat Flux Forcing


edict = {"LHFLX":{"zlib":True}}

ql_da  = xr.DataArray(ql,coords=cdict_time,dims=cdict_time)

ql_std = ql_da.groupby('time.month').std('time')
ql_std = ql_std.rename("LHFLX")

# Save the output (All ensembles) # PRECTOT [ mon x ens x lat x lon]
savename = "%sCESM1_HTR_FULL_QL_NAtl.nc" % (fpath)
ql_std.to_netcdf(savename,encoding=edict) # h [ mon x ens x lat x lon]

# Save the Ensemble Average
ql_ensavg = ql_std.mean('ens')
savename = "%sCESM1_HTR_FULL_QL_NAtl_EnsAvg.nc" % (fpath)
ql_ensavg.to_netcdf(savename,encoding=edict) # h [ mon x lat x lon]

#%% Calculate Sbar ----------------------------------------------------------

# Out: h [ mon x <ens> x lat x lon]
# Load variable
ncsss = ncstr1 % "SSS" 
dss   = xr.open_dataset(rawpath1+ncsss).load()

# # Double check first month
dss   = proc.fix_febstart(dss)

# Compute mean seasonal cycle
dss_scycle = dss.groupby('time.month').mean('time')

# # Convert cm --> m
# dsh_scycle = dsh_scycle/100

# Make encoding dict and rename dimensions
#edict   = proc.make_encoding_dict(dsh_scycle)
dss_out = rename_var(dss_scycle.SSS) 
dss_out = dss_out.rename("Sbar") 
edict   = {"Sbar":{"zlib":True}}

# # Save All Ensembles
savename = "%sCESM1_HTR_FULL_Sbar_NAtl.nc" % fpath
dss_out.to_netcdf(savename,encoding=edict) # h [ mon x ens x lat x lon]


# # Save an ensemble mean versio
dss_ensavg = dss_out.mean('ens')
savename = "%sCESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc" % fpath
dss_ensavg.to_netcdf(savename,encoding=edict) # h [ mon x lat x lon]


#%% Compute Stochastic Heat FLux Forcing Fprime
# Copied section from stochastic evaporation forcing above

# Step 1 (Load Qnet)

# def get_bbox(ds):
#     bbox = [ds.lon.values[0],
#             ds.lon.values[-1],
#             ds.lat.values[0],
#             ds.lat.values[-1]]
#     return bbox

# # def resize_ds(ds_list):
# #     # Note this was made to work with degrees west, have not handeled crrossing dateline
#     bboxes  = np.array([proc.get_bbox(ds) for ds in ds_list]) # [ds.bound]
    
#     bbxsmall = np.zeros(4)
#     bbxsmall[0] = np.max(bboxes[:,0]) # Easternmost Westbound
#     bbxsmall[1] = np.min(bboxes[:,1]) # Westernmost Eastbound
#     bbxsmall[2] = np.max(bboxes[:,2]) # Northerhmost Southbound
#     bbxsmall[3] = np.min(bboxes[:,3]) # Southernmost Northbound
    
#     ds_resize = [proc.sel_region_xr(ds,bbxsmall) for ds in ds_list]
# #     return ds_resize
    
    
    
    
    


# Load TS, f;ux
varnames =['SST','qnet']
ds_load = [xr.open_dataset(rawpath1+ ncstr1 % vn).load() for vn in varnames]



# Anomalize
ds_anom = [proc.xrdeseason(ds) for ds in ds_load]

# Detrend
ds_dt   = [ds-ds.mean('ensemble') for ds in ds_anom] # [ens x time x lat x lon]

# Transpose to [mon x ens x lat x lon]
ds_dt    = [ds.transpose('time','ensemble','lat','lon') for ds in ds_dt]

# Load LHFF
dampstr = "ExpfitSST123"
if dampstr == "Expfitlbda123":
    convert_wm2=True
    hff_nc   = "CESM1_HTR_FULL_Expfit_lbda_damping_lagsfit123.nc"#"CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"
elif dampstr == None:
    convert_wm2=False
    hff_nc = "CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"
elif dampstr == "ExpfitSST123":
    convert_wm2=True
    hff_nc   = "CESM1_HTR_FULL_Expfit_SST_damping_lagsfit123.nc"#"CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"

    
hff_path = dpath
dshff    = xr.open_dataset(hff_path + hff_nc) # [mon x ens x lat x lon']

# Load h
mldnc  = "%sCESM1_HTR_FULL_HMXL_NAtl.nc" % mldpath
ds_mld = xr.open_dataset(mldnc)

# Check sizes
ds_list = ds_dt + [dshff,ds_mld]
ds_rsz  = proc.resize_ds(ds_list)
ds_dt = ds_rsz[:2]
dshff = ds_rsz[2]
ds_mld = ds_rsz[3]

# Convert if need
if convert_wm2:
    dt  = 3600*24*30
    cp0 = 3996
    rho = 1026
    dshff = dshff.damping * (rho*cp0*ds_mld.h) / dt  *-1 #need to do a check for - value!!
else:
    dshff= dshff.damping

# Output to numpy
hff = dshff.values
sst = ds_dt[0].SST.values
qnet = ds_dt[1].qnet.values

# Check sizes
# Get dimension and tile
ntime,nens,nlat,nlon=qnet.shape
ntimeh,nensh,nlath,nlonh=hff.shape

# Tile
nyrs = int(ntime/12)
hfftile=np.tile(hff.transpose(1,2,3,0),nyrs)
hfftile= hfftile.transpose(3,0,1,2)
# Check plt.pcolormesh(hfftile[0,0,:,:]-hfftile[12,0,:,:]),plt.colorbar(),plt.show()


#%% Calculate F' Convert
nroll    = 0
rollstr  = "nroll%0i"  % nroll

Fprime   = qnet + hfftile*np.roll(sst,nroll)

#%% Check results

klon,klat = proc.find_latlon(-30,50,dshff.lon.values,dshff.lat.values)

#%% CHECK SPECTRA
checkts = [qnet[:,0,klat,klon],Fprime[:,0,klat,klon]]
dtplot=3600*24*30
tsmetrics = scm.compute_sm_metrics(checkts)
fig,ax=plt.subplots(1,1,figsize=(12,4))
ax.plot(tsmetrics['freqs'][0]*dtplot,tsmetrics['specs'][0]/dtplot,label="Qnet")
ax.plot(tsmetrics['freqs'][1]*dtplot,tsmetrics['specs'][1]/dtplot,label="F'")
ax.legend()

plt.show()

# Check Monvar
fig,ax = plt.subplots(1,1)
ax.plot(np.arange(12)+1,np.sqrt(tsmetrics['monvars'][0]),label="Qnet")
ax.plot(np.arange(12)+1,np.sqrt(tsmetrics['monvars'][1]),label="F'")
ax.legend()
plt.show()

#%% Save the Output

cdict_time = {
    'time':ds_dt[0].time.values,
    'ens':np.arange(1,43,1),
    'lat':dshff.lat.values,
    'lon':dshff.lon.values,
    }
edict = {"Fprime":{"zlib":True}}

Fprime_da  = xr.DataArray(Fprime,coords=cdict_time,dims=cdict_time)


Fprime_std = Fprime_da.groupby('time.month').std('time')
Fprime_std = Fprime_std.rename("Fprime")


# Save the output (All ensembles) # PRECTOT [ mon x ens x lat x lon]
savename = "%sCESM1_HTR_FULL_Fprime_%s_%s_NAtl.nc" % (fpath,dampstr,rollstr)
Fprime_std.to_netcdf(savename,encoding=edict) # h [ mon x ens x lat x lon]

# Save the Ensemble Average
Fprime_ensavg = Fprime_std.mean('ens')
savename = "%sCESM1_HTR_FULL_Fprime_%s_%s_NAtl_EnsAvg.nc" % (fpath,dampstr,rollstr)
Fprime_ensavg.to_netcdf(savename,encoding=edict) # h [ mon x lat x lon]

#%% Repeat for Latent Heat Flux Forcing


edict = {"Fprime":{"zlib":True}}

qnet_da  = xr.DataArray(qnet,coords=cdict_time,dims=cdict_time)

qnet_std = qnet_da.groupby('time.month').std('time')
qnet_std = qnet_std.rename("Fprime")

# Save the output (All ensembles) # PRECTOT [ mon x ens x lat x lon]
savename = "%sCESM1_HTR_FULL_qnet_NAtl.nc" % (fpath)
qnet_std.to_netcdf(savename,encoding=edict) # h [ mon x ens x lat x lon]

# Save the Ensemble Average
qnet_ensavg = qnet_std.mean('ens')
savename = "%sCESM1_HTR_FULL_qnet_NAtl_EnsAvg.nc" % (fpath)
qnet_ensavg.to_netcdf(savename,encoding=edict) # h [ mon x lat x lon]


#%%

# #%% 

# sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
# import yo_box as ybx
# nsmooth=20
# pct=0.10
# opt=1
# dt=3600*24*30
# outp = scm.quick_spectrum(checkts,nsmooth,pct,opt=opt,dt=dt,return_dict=True) 


#%%


                    
                    
                    #%%

# Preprocess LHFF
# dshff    = proc.lon360to180_xr(dshff)
# dshff_reg = proc.sel_region_xr(dshff,bbox_crop)



