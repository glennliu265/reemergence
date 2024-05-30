#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Ice Edge/Fraction
Also visualize vertical profiles of SST/SSS at the ice edge, over
target points (Depth vs Month)

Visualize Re-emergence (computed by calc_ac_depth) as well with SST/SSS contours



Copied upper section from visualize_rei_acf.py


Created on Fri May 10 10:26:45 2024

@author: gliu

"""


import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

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

#%% Load some information

# Indicate files containing ACFs
cesm_name   = "CESM1_1920to2005_%sACF_lag00to60_ALL_ensALL.nc"
vnames      = ["SST","SSS"]
#%% Load ACFs and REI

acfs_byvar  = []
rei_byvar   = []
for vv in range(2):
    ds = xr.open_dataset(procpath + cesm_name % vnames[vv]).acf.squeeze()
    acfs_byvar.append(ds)
    
    dsrei = xr.open_dataset("%s%s_CESM/Metrics/REI_Pointwise.nc" % (output_path,vnames[vv])).rei.load()
    rei_byvar.append(dsrei)
    
#%% Load mixed layer depth

ds_h    = xr.open_dataset(input_path + "mld/CESM1_HTR_FULL_HMXL_NAtl.nc").h.load()
#id_hmax = np.argmax(ds_h.mean(0),0)

cints_mld = np.arange(500,2100,100)
#%% Load BSF and Ice Mask (copied from compare_detrainment_damping)

bsf      = dl.load_bsf()

# Load Land Ice Mask
icemask  = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# Resize
#bsf,icemask,_    = proc.resize_ds([bsf,icemask,acfs_in_rsz[0]])
bsf_savg = proc.calc_savg_mon(bsf)

#
mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

#%% Load UVEL and VVEL [quick_scycle]

velpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
ds_uvel = xr.open_dataset(velpath + "UVEL/UVEL_NATL_ensALL_scycle.nc").UVEL.load()


ds_vvel = xr.open_dataset(velpath + "VVEL/VVEL_NATL_ensALL_scycle.nc").VVEL.load()

dsvel       = [ds_uvel,ds_vvel]
dsvel_upper = [ds.sel(z_t=slice(0,500*100)).mean('z_t').mean('ens') for ds in dsvel]


dsvel_ensm  = [ds.mean('ens') for ds in dsvel]
#dsvel_ensm  = [xr.merge([ds,dsvel[0].isel(ens=0).TLONG,dsvel[0].isel(ens=0).TLAT]) for ds in dsvel]

# for ii in range(2):
#     dsvel_ensm[ii]['TLONG'] = tlon
#     dsvel_ensm[ii]['TLAT'] = tlat
    
tlon = dsvel[0].isel(ens=0).TLONG.values
tlat = dsvel[0].isel(ens=0).TLAT.values

tlon_ds     = xr.DataArray(tlon,coords=dict(nlat=dsvel[0].nlat,nlon=dsvel[0].nlon),name="TLONG")
tlat_ds     = xr.DataArray(tlat,coords=dict(nlat=dsvel[0].nlat,nlon=dsvel[0].nlon),name="TLAT")
dsvel_ensm  = [xr.merge([ds,tlon_ds,tlat_ds]) for ds in dsvel_ensm]

#dsvel_ensm  = [ds.assign(dict(TLONG=tlon,TLAT=tlat)) for ds in dsvel_ensm]
#%% Indicate Plotting Parameters (taken from visualize_rem_cmip6)


bboxplot                    = [-80,0,10,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 14
fsz_title                   = 16
rhocrit                     = proc.ttest_rho(0.05,2,86)

proj                        = ccrs.PlateCarree()
#%% Load the ice fraction

# Ice Fraction
icenc = "%sCESM1LE_ICEFRAC_NAtl_19200101_20050101_bilinear.nc" % (rawpath)
iceds = xr.open_dataset(icenc).ICEFRAC.load()

# Compute mean seasonal cycle
icecycle = iceds.groupby('time.month').mean('time')

#%% Plot Re-emergence

rei_sss   = rei_byvar[1].isel(mon=[11,0,1,2]).mean('mon').mean('yr').mean('ens')
cints_rei = np.arange(0,0.55,0.05)
cints_rei_max = np.arange(0.40,0.62,0.02)

#%% Configure an ice edge plot

qint = 1
bboxice = [-70,-10,55,70]

for im in range(12):
    
    rei_sss   = rei_byvar[1].isel(mon=im).mean('yr').mean('ens')
    
    fig,ax,_ = viz.init_orthomap(1,1,bboxice,figsize=(12,4))
    ax       = viz.add_coast_grid(ax,bbox=bboxice,fill_color='lightgray')
    
    pv       =icecycle.isel(month=im).mean('ensemble')
    pcm      = ax.pcolormesh(pv.lon,pv.lat,pv,cmap='cmo.ice',transform=proj,zorder=-1)
    
    cb=viz.hcbar(pcm,ax=ax,fraction=0.045)
    cb.set_label("Ice Fraction (%)",fontsize=fsz_axis)
    ax.set_title("%s Ice Fraction" % (mons3[im]),fontsize=fsz_title)
    
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="red",linewidths=1.5,levels=[0,1],zorder=1,transform=proj,label="Ice Mask Edge")
    
    cl=ax.contour(rei_sss.lon,rei_sss.lat,rei_sss,transform=proj,levels=cints_rei)
    ax.clabel(cl)
    
    
    # Plot the quivers
    plotu = dsvel_upper[0].isel(month=im).values
    plotv = dsvel_upper[1].isel(month=im).values
    ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],color='hotpink',transform=proj)
    
    
    savename = "%sREI_SSS_mon%02i.png" % (figpath,im+1)
    plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Same as above but all in the same plot


fig,axs,_ = viz.init_orthomap(4,3,bboxice,figsize=(16,12))
for ii in range(12):
    ax        = axs.flatten()[ii]
    im        =  np.roll(np.arange(12),1)[ii]
    
    rei_sss   = rei_byvar[1].isel(mon=im).mean('yr').mean('ens')
    
    
    ax       = viz.add_coast_grid(ax,bbox=bboxice,fill_color='lightgray')
    
    pv        = icecycle.isel(month=im).mean('ensemble')
    pcm      = ax.pcolormesh(pv.lon,pv.lat,pv,cmap='cmo.ice',transform=proj,zorder=-1)
    
    ax.set_title("%s" % (mons3[im]),fontsize=fsz_title)
    
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="red",linewidths=1.5,levels=[0,1],zorder=1,transform=proj,label="Ice Mask Edge")
    
    cl=ax.contour(rei_sss.lon,rei_sss.lat,rei_sss,transform=proj,levels=cints_rei)
    ax.clabel(cl)
    
        
    # Plot the quivers
    plotu = dsvel_upper[0].isel(month=im).values
    plotv = dsvel_upper[1].isel(month=im).values
    ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],color='hotpink',transform=proj)
    
    
    
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.02)
savename = "%sREI_SSS_monALL.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Visualize the seasonal cycle of the mixed layer depth in these regions as well

fig,axs,_ = viz.init_orthomap(4,3,bboxice,figsize=(16,12))
for ii in range(12):
    ax       = axs.flatten()[ii]
    im       =  np.roll(np.arange(12),1)[ii]
    
    rei_sss  = rei_byvar[1].isel(mon=im).mean('yr').mean('ens')
    
    ax       = viz.add_coast_grid(ax,bbox=bboxice,fill_color='lightgray')
    
    # Plot Ice Concentration
    pv       = icecycle.isel(month=im).mean('ensemble')
    pcm      = ax.pcolormesh(pv.lon,pv.lat,pv,cmap='cmo.ice',transform=proj,zorder=-1)
    
    ax.set_title("%s" % (mons3[im]),fontsize=fsz_title)
    
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="red",linewidths=1.5,levels=[0,1],zorder=1,transform=proj,label="Ice Mask Edge")
    
    plotcl  = ds_h.isel(mon=im).mean('ens')
    cl      = ax.contour(plotcl.lon,plotcl.lat,plotcl,linewidths=0.75,
                         transform=proj,levels=cints_mld,colors="skyblue")
    ax.clabel(cl)
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.02)
savename = "%sMLD_SSS_monALL.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Plot Point selection

centerpoints = ([-37,62],[-55,59],)

sel_mons  = [1,2]

fig,axs,_ = viz.init_orthomap(1,2,bboxice,figsize=(12,6),constrained_layout=True)

for ii in range(2):
    ax       = axs[ii]
    im       = sel_mons[ii]
    rei_sss  = rei_byvar[1].isel(mon=im).mean('yr').mean('ens')
    
    ax       = viz.add_coast_grid(ax,bbox=bboxice,fill_color='lightgray')
    
    # Plot Ice Concentration
    pv       = icecycle.isel(month=im).mean('ensemble')
    pcm      = ax.pcolormesh(pv.lon,pv.lat,pv,cmap='cmo.ice',transform=proj,zorder=-1)
    
    # Plot RE Index
    cl=ax.contour(rei_sss.lon,rei_sss.lat,rei_sss,transform=proj,levels=cints_rei,linewidths=0.9,cmap='cmo.deep_r')
    ax.clabel(cl)
    
    # Pot ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="red",linewidths=1.5,linestyles='dotted',
               levels=[0,1],zorder=1,transform=proj,label="Ice Mask Edge")

    
    # Plot the points
    cp = centerpoints[ii]
    locfn,loctitle=proc.make_locstring(cp[0],cp[1])
    ax.plot(cp[0],cp[1],transform=proj,marker='o',markersize=5,color='yellow')
    ax.plot(cp[0]+3,cp[1],transform=proj,marker='o',markersize=3,color='yellow')
    ax.plot(cp[0]-3,cp[1],transform=proj,marker='o',markersize=3,color='yellow')
    ax.plot(cp[0],cp[1]+3,transform=proj,marker='o',markersize=3,color='yellow')
    ax.plot(cp[0],cp[1]-3,transform=proj,marker='o',markersize=3,color='yellow')

    ax.set_title("%s - %s" % (mons3[im],loctitle),fontsize=fsz_title)
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.02)
savename = "%sProfile_Analysis_locator.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Pull in profile data

profpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"
ncnames  = ["IrmingerEns01_CrossPoints.nc","LabradorEns01_CrossPoints.nc"]
cpnames  = ["Irminger","Labrador"]

vv = 0

ds_savgs = []
for vv in range(2):

    ds      = xr.open_dataset(profpath + ncnames[vv])
    ds_savg = ds.groupby('time.month').mean('time').sel(z_t=slice(0*100,1000*100))
    ds_savgs.append(ds_savg)
    
    

mon = np.arange(1,13,1)
z   = ds_savg.z_t


#%%



fig,axs = plt.subplots(3,3,constrained_layout=True,figsize=(4,16))

# North Grid
ax = axs[0,1]
dsplot = ds_savg.sel(dir="N").SALT.T
pcm = ax.pcolormesh(mon,z,dsplot)


# Center Grid
ax = axs[1,1]
dsplot = ds_savg.sel(dir="Center").SALT.T
pcm = ax.pcolormesh(mon,z,dsplot)

# West
ax = axs[1,0]
dsplot = ds_savg.sel(dir="W").SALT.T
pcm = ax.pcolormesh(mon,z,dsplot)


# East
ax = axs[1,2]
dsplot = ds_savg.sel(dir="E").SALT.T
pcm = ax.pcolormesh(mon,z,dsplot)

# South
ax = axs[2,1]
dsplot = ds_savg.sel(dir="S").SALT.T
pcm = ax.pcolormesh(mon,z,dsplot)

#%% Just Plot Profiles 1 Direction at a time

# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
cc   = 0
xtks = np.arange(1,13,1)

if cc == 0:
    cints_salt = np.arange(34,36,0.1)
    cints_temp = np.arange(0,10,0.4)
elif cc == 1:
    cints_salt = np.arange(34,35.6,0.1)
    cints_temp = np.arange(0,10,0.4)

di       = 'Center'
dirnames = ds_savgs[0].dir.values

for di in dirnames:
    
    
    
    fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(8,6))
    
    # Plot the Salinity
    pv      = ds_savgs[cc].sel(dir=di).SALT.T
    pcm     = ax.contourf(mon,z/100,pv,cmap='cmo.haline',levels=cints_salt,extend='both')
    
    # Plot the Temp
    pv      = ds_savgs[cc].sel(dir=di).TEMP.T
    cl      = ax.contour(mon,z/100,pv,cmap='cmo.thermal',levels=cints_temp,extend='both')
    ax.clabel(cl)
    
    # # Plot the Temp
    # pv2      = ds_savg.sel(dids_r=di).TEMP.T
    # pcm2     = ax.contourf(mon,z/100,pv,cmap='cmo.ter')
    
    ax.set_ylim([0,800])
    plt.gca().invert_yaxis()
    fig.colorbar(pcm,ax=ax)
    
    # Plot Mean Mixed Layer Depth
    plotmld = ds_h.isel(ens=0).sel(lon=pv.TLONG-360,lat=pv.TLAT,method='nearest')
    ax.plot(mon,plotmld,ls='dashed',color='violet',label="HMXL")
    ax.set_ylabel("Depth (meters)")
    ax.set_xticks(xtks,labels=mons3)
    ax.grid(True,ls='dotted',c='w',alpha=0.5)
    
    # Plot the ice fraction
    plotice = icecycle.isel(ensemble=0).sel(lon=pv.TLONG-360,lat=pv.TLAT,method='nearest')
    ax2 = ax.twinx()
    ax2.plot(mon,plotice,label="Ice Fraction",alpha=1,color="k",marker="x")
    ax2.set_ylim([0,0.5])
    ax2.set_ylabel("Ice Fraction")
    
    ax2.legend()
    
    ax.set_title("%s (%s) : Lon %.2f, Lat %.2f" % (cpnames[cc],di,pv.TLONG,pv.TLAT))
    savename = "%sVertical_Profiles_Scycle_%s_%s.png" % (figpath,cpnames[cc],di)
    
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    
# --------------------------------------------------------
#%% Load and format re-emergence computed by calc-ac-depth
# --------------------------------------------------------
# Center, North, South, East, West
ptcoords = ([-38,62],[-38,65],[-38,59],[-41,62],[-35,62], # Irminger
            [-57,59],[-57,62],[-57,56],[-60,59],[-54,59]) # Labrador

dirnames   = ds_savgs[0].dir
vnames2   = ["TEMP","SALT"]
rem3dpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/depthvlag/"

vv        = 0

rem_byvar = []
for vv in range(2):
    npts      = len(ptcoords)
    ds_all    = []
    
    for pt in range(npts):
        
        
        
        
        lonf = ptcoords[pt][0] + 360
        latf = ptcoords[pt][1] 
        
        #CESM1LE_UOSALT_lon322_lat62.nc
        ncstr = "%sCESM1LE_UO%s_lon%03i_lat%02i.nc" % (rem3dpath,vnames2[vv],lonf,latf)
        ds = xr.open_dataset(ncstr).load()
        
        ds_all.append(ds.copy())
    
    
    
    
    rem_irm = xr.concat(ds_all[:5],dim='dir')
    rem_lab = xr.concat(ds_all[5:],dim='dir')
    rem_irm['dir'] = dirnames
    rem_lab['dir'] = dirnames
    rem_byreg = [rem_irm,rem_lab] # Dir x Ens x Time x Z_t
    
    rem_byvar.append(rem_byreg)

# Save Output
outpath_rem = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"

# Save Irminger
rem_irm_all = xr.merge([rem_byvar[0][0],rem_byvar[1][0]])
edict       = proc.make_encoding_dict(rem_irm_all)
outname     = outpath_rem + "IrmingerAllEns_SALT_TEMP.nc"
rem_irm_all.to_netcdf(outname,encoding=edict)


# Save Labrador
rem_lab_all = xr.merge([rem_byvar[0][1],rem_byvar[1][1]])
edict       = proc.make_encoding_dict(rem_irm_all)
outname     = outpath_rem + "LabradorAllEns_SALT_TEMP.nc"
rem_lab_all.to_netcdf(outname,encoding=edict)

#%% Now visualize things...

plotvar = rem_irm_all.isel(dir=0,ensemble=0)
fig,ax = plt.subplots(1,1,constrained_layout=True)



#%% Load SST/TEMP data from above


# Load data
outpath_rem = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"
ncname      = "IrmingerAllEns_SALT_TEMP.nc"
ds          = xr.open_dataset(outpath_rem + ncname).load()

# Demean and Deseason, Fix February Start
dsvars = [ds.TEMP,ds.SALT]
dsanoms = [ds - ds.mean('ensemble') for ds in dsvars]
dsanoms = [proc.xrdeseason(ds) for ds in dsanoms]
dsanoms = [proc.fix_febstart(ds) for ds in dsanoms]

#%% Compute pointwise ACF


# based on scm.calc_autocorr_mon

def calc_autocorr_mon(ds3d,dspt,lags,verbose=False,return_da=True):
    # Given 2-D array (time x depth), compute monthly lag correlation relative to a depth level
    #ts        = ts.values
    tsyrmon   = proc.year2mon(ts)  # [mon x yr]
    #print(tsyrmon.shape)
    assert tsyrmon.shape[0] == 12,"Timeseries dims (%s) are wrong (should be mon x year)" % (str(tsyrmon.shape))
    
    # Deseason
    tsa = tsyrmon - np.mean(tsyrmon,1)[:,None]
    
    # Detrend
    tsa = signal.detrend(tsa,axis=1,type='linear')
    
    if ts1 is not None: # Repeat for above, but for ts1
        ts1_yrmon = proc.year2mon(ts1)
        assert ts1_yrmon.shape[0] == 12,"Timeseries dims (%s) are wrong (should be mon x year)" % (str(ts1_yrmon.shape))
        ts1a = ts1_yrmon - np.mean(ts1_yrmon,1)[:,None]  # Deseason
        ts1a = signal.detrend(ts1a,axis=1,type='linear') # Detrend
    
    # Compute Autocorrelation (or Cross Correlation)
    acf    = []
    for im in range(12):
        if ts1 is not None:
            tsa_lag = ts1a
        else:
            tsa_lag = tsa
        ac = proc.calc_lagcovar(tsa,tsa_lag,lags,im+1,0,debug=verbose)
        acf.append(ac)
    acf    = np.array(acf)
    if return_da:
        coords = dict(mon=np.arange(1,13,1),lag=lags)
        acf     = xr.DataArray(acf,coords=coords,dims=coords,name='acf')
        return acf
    return acf

from amv import xrfunc as xrf

vv = 0
e  = 0

ds_in     = dsanoms[vv].isel(dir=0).isel(ensemble=e)
rem_bymon = []
for im in range(12):
    
    dspt = ds_in.isel(z_t)
    
    

#%%

from scipy import signal


def calc_lagcovar_2d(ds3d,dspt,lags):
    #ds3d       = ds_in
    #dspt       = ds_in.isel(z_t=0)
    #lags       = np.arange(37)
    ntime,nother = ds3d.shape
    nyr = int(ntime/12)
    nlags = len(lags)
    # Separate to Mon x Year
    dsyrmon    = ds3d.values.reshape(nyr,12,nother).transpose(1,0,2) # [yr x mon x other] --> [mon x yr x other]
    dspt_yrmon = proc.year2mon(dspt.values)   # [mon x yr]
    
    # # # # Deaseason
    # dsyrmon    = dsyrmon - np.mean(dsyrmon,1,keepdims=True)
    # dspt_yrmon = dspt_yrmon - np.mean(dspt_yrmon,1,keepdims=True)
    
    # # # # Detrend
    # dsyrmon         = signal.detrend(dsyrmon,axis=1,type='linear')
    # dspt_yrmon      = signal.detrend(dspt_yrmon,axis=1,type='linear')
    
    # Compute ACF for each month (using xarray)
    acf             = np.zeros((12,nother,nlags)) * np.nan
    for im in range(12):
        for kk in range(nother):
            tsbase = dsyrmon[:,:,kk]
            tslag  = dspt_yrmon[:,:]
            if np.any(np.isnan(tsbase)) or np.any(np.isnan(tslag)):
                continue
            
            ac = proc.calc_lagcovar(tsbase,tslag,lags,im+1,0,debug=False,)
            acf[im,kk] = ac.copy()
    return acf
    #coords = dict(mon=np.arange(1,13,1),z_t=ds3d.z_t)


    
#%% 



for di in range(5):

    rem_byvar = []
    for vv in range(2):
        
        rem_byens = []
        
        for e in tqdm.tqdm(range(42)):
            
            ds_in      = dsanoms[vv].isel(dir=di).isel(ensemble=e)
            ds3d       = ds_in
            dspt       = ds_in.isel(z_t=0)
            ac         = calc_lagcovar_2d(ds3d,dspt,lags)
            
            rem_byens.append(ac)
        
        rem_byens = np.array(rem_byens)
        rem_byvar.append(rem_byens)
            
    coords      = dict(ens=np.arange(1,43,1),mon=np.arange(1,13,1),z_t=ds3d.z_t,lag=lags,)
    rem_byvar   = [xr.DataArray(rem_byvar[ii],coords=coords,dims=coords,name=vnames[ii]) for ii in range(2)]
     
    ds_out      = xr.merge(rem_byvar)
    outname     = "%sIrmingerAllEns_SALT_TEMP_REM_3DCorr_%s" % (outpath_rem,ds.dirnames[di])


#%% Save the output


def lag_tiler(scycle,lags,kmonth):
    # (1) Shift so that scycle begins with month index [kmonth]
    scycle_shift = np.roll(scycle,-kmonth)
    
    # (2) Tile so the scycle is the same as the number of lags
    nlags       = len(lags)
    ntile       = int(nlags/12)
    addm        = nlags%12
    scycle_tile = np.hstack([np.tile(scycle_shift,ntile),scycle_shift[:addm]])
    
    return scycle_tile
    


#%%
outpath_rem = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"
ncname      = "IrmingerAllEns_SALT_TEMP.nc"

#%% Load and plot re-emergence computed by [calc_rem_3d_crosspoint]

# Also load full profile data (for point inforation, whoops)
ncpath          = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"
ncname2         = "LabradorAllEns_SALT_TEMP.nc"
ds3d            = xr.open_dataset(ncpath + ncname2)
dirnames        = ds3d.dir.values

# Set Information
di              = 5
kmonth          = 2
lags            = np.arange(37)
vnames          = ["TEMP","SALT"]
lonf            = ds3d.TLONG[di].values.item() # Degrees East
lonfw           = lonf - 360
latf            = ds3d.TLAT[di].values.item() # Degrees West
locfn,loctitle  = proc.make_locstring(lonf,latf)

# Load Re-emergence Data
ncname          = "LabradorAllEns_SALT_TEMP_REM_3DCorr_%s.nc" % (dirnames[di])
dsrem           = xr.open_dataset(ncpath+ncname).load()

# Select data from above
mldpt    = ds_h.sel(lon=lonfw,lat=latf,method='nearest').mean('ens')
icept    = icecycle.sel(lon=lonfw,lat=latf,method='nearest').mean('ensemble')
dsvel_pt = [proc.find_tlatlon(ds,lonf,latf) for ds in dsvel_ensm]
u2_pt    = np.sqrt(dsvel_pt[0].UVEL**2 + dsvel_pt[1].VVEL**2)


# Plotting Selections
kmonth   = 2
mons3    = proc.get_monstr()
cints    = np.arange(-1,1.05,0.05)
xtks     = np.arange(0,37,1)
fig,axs  = plt.subplots(2,1,figsize=(14,6.5))

# Tile based on kmonth
mons3tile = lag_tiler(mons3,lags,kmonth)
mldtile   = lag_tiler(mldpt,lags,kmonth)
icetile   = lag_tiler(icept,lags,kmonth)
#veltile   = [lag_tiler(icept,)]

for vv in range(2):
    ax = axs[vv]
    # Plot Temp contours
    plotvar = dsrem.isel(mon=kmonth).mean('ens')[vnames[vv]]
    cf = ax.contourf(dsrem.lag,dsrem.z_t/100,plotvar,levels=cints,cmap='RdBu_r')

    # # Plot Salt on same plot
    # plotvar = dsrem.isel(mon=1).mean('ens').SALT
    # cl = ax.contour(dsrem.lag,dsrem.z_t/100,plotvar,levels=cints,colors="k",lw=0.75)
    # ax.clabel(cl)
    ax.set_ylim([0,1000])
    ax.invert_yaxis()
    ax.set_xticks(xtks,labels=mons3tile)
    
    ax.set_ylabel("Depth (m)")
    ax.set_title(vnames[vv])
    
    if vv == 1:
        ax.set_xlabel("Lag (months)")
    
    cb = fig.colorbar(cf,ax=ax,pad=0.01,fraction=0.02)
    
    # Plot Other Variables
    ax.plot(lags,mldtile,c='magenta',ls='dashed')
    #ax2 = ax.twinx()
    #ax.plot(lags,icetile)
    
plt.suptitle("Re-emergence @ %s" % loctitle,y=0.95)
savename = "%s" % (figpath)



#%%

fig,ax = viz.init_monplot(1,1,)

cf = ax.pcolormesh(np.arange(1,13,1),dsvel_pt[0].z_t/100,u2_pt.T)
ax.quiver(np.arange(1,13,1),dsvel_pt[0].z_t/100,dsvel_pt[0].UVEL.T,dsvel_pt[1].VVEL.T,scale=150)
ax.invert_yaxis()

fig.colorbar(cf,ax=ax)

#%%




