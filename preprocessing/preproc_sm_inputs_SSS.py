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
    NOTE: I have moved a lot of this to NHFLX_EOF_monthly_lens..
    Need to clean and delete sections of the script. 

(4) Load and subset detrainment damping [from regrid_detrainment_damping]

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
from amv import proc
import scm

# Get needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
rawpath     = pathdict['raw_path']
rawpath_3d  = rawpath + "ocn_var_3d/"

# Set input parameter paths
mpath     = input_path + "mld/"
dpath     = input_path + "damping/"
fpath     = input_path + "forcing/"

vnames      = ["SALT","TEMP"]

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

stormtrack = 1
if machine == "stormtrack":
    stormtrack = 1

if stormtrack:
    # Path to variables processed by prep_data_byvariable_monthly
    rawpath1   = rawpath
    ncstr1     = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc"
    
    # Path to variables processed by combine_precip
    rawpath2   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/PRECIP/HTR_FULL/"
    ncstr2     = "%s_HTR_FULL.nc"
    
    # # Output paths
    # input_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input"

else:
    # Path to variables processed by prep_data_byvariable_monthly
    rawpath1   = rawpath
    ncstr1     = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc"
    
    # Path to variables processed by combine_precip
    rawpath2   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/PRECIP/HTR_FULL/"
    ncstr2     = "%s_HTR_FULL.nc"
    
    # # Output paths
    # input_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input"


# Bounding Boxes
bbox_crop     = [-90,20,0,90]  # Preprocessing box

# HFF Processing Options
tails      = 2
p          = 0.05
mode       = 5 # 1 = No Mask; 2 = SST autocorr; 3 --> SST-FLX cross corr; 4 = Both, 5 = Replace SLAB
sellags    = [0]
lagstr     = "lag1"

hffname    = "method%0i_%s_p%03i_tails%i" % (mode,lagstr,p,tails)

# ============================================================
# %% HMXL Section
# ============================================================

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
savename = "%sCESM1_HTR_FULL_HMXL_NAtl.nc" % mpath
dsh_out.to_netcdf(savename,encoding=edict) # h [ mon x ens x lat x lon]

# Save an ensemble mean versio
dsh_ensavg = dsh_out.mean('ens')
savename = "%sCESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc" % mpath
dsh_ensavg.to_netcdf(savename,encoding=edict) # h [ mon x lat x lon]

#%% Compute Kprev

# Indicate which file to load
mld_nc      = mpath + "cesm1_htr_5degbilinear_HMXL_NAtl_1920to2005.nc"#"cesm2_pic_HMXL_NAtl_0200to2000.nc"
savename    = mpath + "cesm1_htr_5degbilinear_kprev_NAtl_1920to2005_EnsAvg.nc"#"cesm2_pic_kprev_NAtl_0200to2000.nc"
dsh_ensavg  = xr.open_dataset(mld_nc).h


# If ens Avg is not available, make it
if 'ens' in list(dsh_ensavg.dims):
    print("Computing ensemble average")
    dsh_ensavg = dsh_ensavg.mean('ens')
    savename_new = proc.addstrtoext(mld_nc,"_EnsAvg",adjust=-1)
    dsh_ensavg.to_netcdf(savename_new)

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
#savename = "%sCESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc" % mldpath
da_kprev.to_netcdf(savename,encoding=edict) # h [ mon x lat x lon]
print("Saved kprev output to %s" % savename)

# ============================================================
# %% PRECIP FORCING
# ============================================================

# ------------------------------------------------
#%% Part 2: Load and process Precipitation Forcing
# ------------------------------------------------
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


# ============================================================
# %% Part 3: Stochastic Evaporation
# ============================================================


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


# ============================================================
# %% LHFLX Forcing
# ============================================================


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

# ============================================================
# %% SBAR
# ============================================================


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


# ============================================================
# %% Ekman Forcing (prepare for the region) --> works with out from [calc_ekman_advection_htr.py]
# ============================================================

# works with out from [calc_ekman_advection_htr.py]
# Flips Longitude
# Crops to North atlantic Region

# Processing for CESM1 LENS HTR 5 Degree Regridded
ncname      = "cesm1_htr_5degbilinear_Qek_TS_NAO_cesm1le5degqnet_nroll0_Global_EnsAvg.nc"
ncname_out  = ncname.replace('Global','NAtl')
ds          = xr.open_dataset(fpath + ncname).load()#.Qek.load() ('mode', 'mon', 'lat', 'lon')

if np.any(ds.lon > 180):
    print("Flipping Longitude")
    ds = proc.lon360to180_xr(ds)
dsreg       = proc.sel_region_xr(ds,bbox_crop)
edict       = proc.make_encoding_dict(dsreg)
dsreg.to_netcdf(fpath + ncname_out,encoding=edict)
print("Saved output to %s" % (fpath+ncname_out))

#% make a debugging plot
# dsreg.Qek.isel(mon=0,mode=0).plot(),plt.show()

    

# ============================================================
# %% Fprime
# ============================================================

# I started working with this and realized that the region was already cropped

ncname      = "cesm1le_htr_5degbilinear_Fprime_EOF_corrected_cesm1le5degqnet_nroll0_perc090_Global_EnsAvg.nc"
ncname_out  = ncname.replace('Global','NAtl')# "cesm1le_htr_5degbilinear_Fprime_EOF_corrected_cesm1le5degqnet_nroll0_perc090_NAtl_EnsAvg.nc"
# It seems things have already been cropped

#ds          = xr.open_dataset(fpath + ncname).load()#.Qek.load() ('mode', 'mon', 'lat', 'lon')



# ============================================================
# %% Land Ice Masks
# ============================================================

maskpath_in  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/masks/"
maskpath_out = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/masks/"
masknc       = "cesm1_htr_5degbilinear_limask_0.3p_0.05p_year1920to2005_enssum.nc"
masknc_out   = "cesm1_htr_5degbilinear_limask_0.3p_0.05p_year1920to2005_NAtl_enssum.nc"

# Load and flip longitude if needed
ds           = xr.open_dataset(maskpath_in + masknc).load() # lat x lon
if np.any(ds.lon > 180):
    print("Flipping Longitude")
    ds = proc.lon360to180_xr(ds)

# Crop to region and save
dsreg           = proc.sel_region_xr(ds,bbox_crop)
edict           = proc.make_encoding_dict(dsreg)
dsreg.to_netcdf(maskpath_out + masknc_out,encoding=edict)

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

#%% Repeat for qnet Forcing

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

# -----------------------------------------------------------------------------
#%% Load and Process Detrainment Damping (Ens Member 1)
# -----------------------------------------------------------------------------
# Works with output from regrid_detrainment_damping
# Works on Astraeus (NOT stormtrack)
# Loads in lbd_d, multiplies by -1

vnames_in  = ["SALT","TEMP"]
vnames_out = ["SSS","SST"]


inpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
dpath         = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"

searchstr     = "CESM1_HTR_FULL_lbd_d_params_%s_detrendensmean_lagmax3_ens01_regridNN.nc"

savenames_out = ["%sCESM1_HTR_FULL_%s_Expfit_lbdd_monvar_detrendensmean_lagmax3_Ens01.nc" % (dpath,vnames_out[v],) for v in range(2)]

for v in range(2):
    # Set Variable Na,e
    vn           = vnames_in[v]
    
    # Load lbd_d
    ncstr        = inpath + searchstr % vn
    ds           = xr.open_dataset(ncstr)
    lbd_d        = ds.lbd_d * -1 # [Mon x Lat x Lon] # Multiple by -1 since negative will be applied in formula
    
    savename_out = savenames_out[v]
    edict        = {'lbd_d':{'zlib':True}}
    lbd_d.to_netcdf(savename_out,encoding=edict)
    
    
# -----------------------------------------------------------------------------
#%% Same as above, but combine and merge for each ensemble member
# -----------------------------------------------------------------------------
# Works with output from regrid_detrainment_damping
# Loads in lbd_d, multiplies by -1
# Save for all ens. Take Ens mean, then save again.

nens          = 42
vnames_in     = ["TEMP",]#["TEMP",]#"SALT",
vnames_out    = ["SST",]#["SST",]#"SSS",

# for v in range(2):
#     vname = vnames_in[v]
    
#     for e in range():
        


#inpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
#dpath         = dpath

#searchstr     = "CESM1_HTR_FULL_lbd_d_params_%s_detrendensmean_lagmax3_ens01_regridNN.nc"

savenames_out = ["%sCESM1_HTR_FULL_%s_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAll.nc" % (dpath,vnames_out[v],) for v in range(len(vnames_in))]


for v in range(len(vnames_in)):
    
    # Set Variable Name
    vn           = vnames_in[v]
    ds_all       = []
    for e in range(nens):
        ncstr = "%sCESM1_HTR_FULL_lbd_d_params_%s_detrendensmean_lagmax3_ens%02i_regridNN.nc" % (rawpath_3d,vn,e+1)
        ds    = xr.open_dataset(ncstr).lbd_d.load()
        ds_all.append(ds)
    
    # Merge and Flip Sign
    ds_all = xr.concat(ds_all,dim="ens") 
    
    # FLip the Sign
    ds_all = ds_all * -1 # Multiply by -1 since negative will be applied in formula
    
    # Save output
    ds_all = ds_all.rename('lbd_d')
    savename_out = "%sCESM1_HTR_FULL_%s_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAll.nc" % (dpath,vnames_out[v],)
    edict        = {'lbd_d':{'zlib':True}}
    ds_all.to_netcdf(savename_out,encoding=edict)
    
    # Save ensemble average
    ds_ensavg    = ds_all.mean('ens')
    savename_ens = "%sCESM1_HTR_FULL_%s_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAvg.nc" % (dpath,vnames_out[v],)
    ds_ensavg.to_netcdf(savename_ens,encoding=edict)
    
    print("Saved output to %s" % savename_ens)
    
    
#%% Save and process correlation-base detrainment damping, as computed by [calc_detrainment_correlation_pointwise.py]

# Indicate paths and names 
inpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
ncname  = "CESM1_HTR_FULL_corr_d_%s_detrendensmean_lagmax3_interp1_ceil0_imshift1_dtdepth1_ensALL_regridNN.nc"
vnames  = ["SALT","TEMP"]
dpath   = input_path + "damping/"

# 
ds_all = []
for vv in range(2):
    vname=vnames[vv]
    nc   = inpath + ncname % vname
    ds   = xr.open_dataset(nc).lbd_d.load()
    ds_all.append(ds)
    
    # Save Ensemble Average
    ds_ensavg    = ds.mean('ens')
    savename_ens = "%sCESM1_HTR_FULL_corr_d_%s_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc" % (dpath,vname,)
    edict        = {'lbd_d':{'zlib':True}}
    ds_ensavg.to_netcdf(savename_ens,encoding=edict)

#%%






