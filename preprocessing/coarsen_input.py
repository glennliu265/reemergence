#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Do a simple coarsen operation on a stochastic model input
(as an alternative to re-estimating everything)

Created on Mon Jul  8 17:20:24 2024

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
machine = "Astraeus"

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
maskpath  = input_path + "masks/" 

vnames      = ["SALT","TEMP"]

#%% Indicate the file name

# Load the Target Dataset
#target_nc   = "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc"
#varname     = "damping"
target_nc   = "CESM1_HTR_FULL_corr_d_SALT_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc"
varname     = "lbd_d"
target_path = dpath
dstarg      = xr.open_dataset(target_path + target_nc)[varname].load()
target_var  = dstarg.transpose('mon','lat','lon').data
latold      = dstarg.lat.data
lonold      = dstarg.lon.data

# Load and get the reference grid
ref_nc      = "cesm1le_htr_5degbilinear_Fprime_EOF_corrected_cesm1le5degqnet_nroll0_perc090_NAtl_EnsAvg.nc"
ref_path    = fpath
dsref       = xr.open_dataset(ref_path + ref_nc).load()

# Make it to the same bounding box
#dsref,dstarg = proc.resize_ds([dsref,dstarg])

lonnew      = dsref.lon.data
latnew      = dsref.lat.data

outnc       = dpath + "CESM1_HTR_FULL_qnet_damping_nomasklag1_coarsen5deg.nc"

print("Coarsening: %s" % target_nc)
print("\tTarget Grid from \t: %s" % ref_nc)
print("\tOutput saved to \t\t: %s" % outnc)

#%% Start to regrid (using coarsen bavg)

help(proc.coarsen_byavg)

dx = (lonnew[1:] - lonnew[:-1])[0] # Use this
#dy = (latnew[1:] - latnew[:-1])[0]

# Coarsen
outvar,latn,lonn=proc.coarsen_byavg(target_var,latold,lonold,deg=5,tol=dx/2,newlatlon=[lonnew,latnew])

# Save output
coords  = dict(mon=dstarg.mon,lat=latn,lon=lonn)
daout   = xr.DataArray(outvar,coords=coords,dims=coords,name=varname)
edict   = proc.make_encoding_dict(dstarg)
daout.to_netcdf(outnc,encoding=edict)
print("Saved output to %s" % outnc)

# --------------------------------- -------------------------------------------
# ---------------------------------- -----------------------------------------|
#%% For debugging, compare the two -- ----------------------------------------<
# ---------------------------------- -----------------------------------------|
# --------------------------------- -------------------------------------------
# Currently works for eat flux feedback
# Also load re-estimated output
nc_reest = "cesm1_htr_5degbilinear_qnet_damping_damping_cesm1le5degqnetDamp_EnsAvg.nc"
dsreest  = xr.open_dataset(dpath + nc_reest).damping.load()

#%%


for im in range(12):
    
    # 
    dsraw    = dstarg.isel(mon=im)
    dscoarse = daout.isel(mon=im)
    dsre     = dsreest.isel(mon=im)
    
    #% Plot Everything
    
    import matplotlib as mpl
    import cartopy.crs as ccrs
    from amv import viz
    
    bboxplot                        = [-80,0,10,65]
    mpl.rcParams['font.family']     = 'Avenir'
    mons3                           = proc.get_monstr(nletters=3)
    
    fsz_tick                        = 18
    fsz_axis                        = 20
    fsz_title                       = 16
    
    #rhocrit = proc.ttest_rho(0.05,2,86)
    proj                            = ccrs.PlateCarree()
    
    
    
    plotvars    = [dsraw,dscoarse,dsre]
    plotnames   = ["Raw","Coarsened","Re-estimated"]
    
    
    fig,axs,_ = viz.init_orthomap(1,3,bboxplot,figsize=(16,8),constrained_layout=True,centlat=45)
    
    for yy in range(3):
        ax  = axs.flatten()[yy]
        blb = viz.init_blabels()
        
        if yy !=0:
            blb['left']=False
        else:
            blb['left']=True
        blb['lower']=True
        
        ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                        fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        ax.set_title(plotnames[yy],fontsize=fsz_title)
        
        pv = plotvars[yy]
        
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv.data,
                            transform=proj,
                            cmap='cmo.balance',vmin=-35,vmax=35)
        
        
    cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.045)
    cb.set_label("Qnet Heat Flux Feedback")
    
    savename = "%sHFF_Coarsen_effects_mon%02i.png" % (figpath,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    #% Plot the Differences
    
    ## None of this is working
    #dscoarse.reset_index('lon')
    #dscoarse = dscoarse.drop_isel({'lon':-1})
    
    dsre_sel = dsre.sel(lon=slice(-90,-5))
    dsco_sel = dscoarse.sel(lon=slice(-90,-5))
    
    limask = xr.where(np.isnan(dsco_sel),np.nan,1)
    diff   = dsre_sel * limask - dsco_sel
    
    fig,ax,_ = viz.init_orthomap(1,1,bboxplot,figsize=(16,4.5),constrained_layout=True,centlat=45)
    
    
    ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    
    
    pv = diff
    pcm = ax.pcolormesh(pv.lon,pv.lat,pv.data,
                        transform=proj,
                        cmap='cmo.balance',vmin=-10,vmax=10)
    ax.set_title("Differences (Restimate - Coarsened)")
    cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
    cb.set_label("Qnet Heat Flux Feedback")
    
    savename = "%sHFF_Coarsen_effects_diff_mon%02i.png" % (figpath,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    
#%%





