#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Covariability between missing terms of SST and SSS
Created on Wed Sep 25 17:05:06 2024

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

from cmcrameri import cm

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd             = os.getcwd()
sys.path.append(cwd + "/..")

# Paths and Load Modules
import reemergence_params as rparams
pathdict        = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath         = pathdict['figpath']
input_path      = pathdict['input_path']
output_path     = pathdict['output_path']
procpath        = pathdict['procpath']
rawpath         = pathdict['raw_path']

# %% Import Custom Modules

from amv import proc, viz
import scm
import amv.xrfunc as xrf
import amv.loaders as dl

# Import stochastic model scripts
proc.makedir(figpath)

#%% Define some functions

# From viz_inputs_basinwide
def init_monplot(bboxplot=[-80,0,20,60],fsz_axis=24):
    mons3         = proc.get_monstr()
    plotmon       = np.roll(np.arange(12),1)
    fig,axs,mdict = viz.init_orthomap(4,3,bboxplot=bboxplot,figsize=(18,18))
    for a,ax in enumerate(axs.flatten()):
        im = plotmon[a]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        ax.set_title(mons3[im],fontsize=fsz_axis)
    return fig,axs

def anomalize(ds):
    ds = ds - ds.mean('ens')
    ds = proc.xrdeseason(ds)
    return ds

#%% Load Plotting Parameters
bboxplot        = [-80,0,20,60]

# Load Land Ice Mask
icemask         = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

mask            = icemask.MASK.squeeze()
mask_plot       = xr.where(np.isnan(mask),0,mask)#mask.copy()


mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub


mask_apply      = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

# Load Gulf Stream
ds_gs           = dl.load_gs()
ds_gs           = ds_gs.sel(lon=slice(-90,-50))
ds_gs2          = dl.load_gs(load_u2=True)


#%% Other Plotting Parameters

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28

proj                        = ccrs.PlateCarree()




#%% Load more infomation on the points

# Get Point Info
pointset    = "PaperDraft02"
ptdict      = rparams.point_sets[pointset]
ptcoords    = ptdict['bboxes']
ptnames     = ptdict['regions']
ptnames_long = ptdict['regions_long']
ptcols      = ptdict['rcols']
ptsty       = ptdict['rsty']

#%% Load SST/SSS

# Check Role in Damping
vnames = ["SST","SSS",]
ds_all = []
for vv in range(2):
    
    st      = time.time()
    
    ncname  = rawpath + "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % vnames[vv]
    ds      = xr.open_dataset(ncname)[vnames[vv]].load()
    ds_all.append(ds)
    print("Loaded %s in %.2fs" % (vnames[vv],time.time()-st))
    
#%% Anomalize and compute tendency

ds_anom = [anomalize(ds.rename(dict(ensemble='ens'))) for ds in ds_all]
ds_tendency = [ds.differentiate('time',datetime_unit='m') for ds in ds_anom]

# -----------------------------------------------------------------------------
#%% Load and visualize MLD Variance
# -----------------------------------------------------------------------------


# Load Tendencies
st      = time.time()
ds_mldvar = []
for vv in range(2):
    ncname = "%sCESM1LE_MLDvarTerm_%s_NAtl_19200101_20050101_bilinear.nc" % (rawpath,vnames[vv])
    ds = xr.open_dataset(ncname)[vnames[vv]].load()
    ds_mldvar.append(ds)
print("Loaded tendency Terms in %.2fs" % (time.time()-st))


#%% Compute the covariance (using xarray)

st      = time.time()
st             = time.time()
cov_mldvar     = []
for vv in range(2):
    covv = xr.cov(ds_anom[vv],ds_mldvar[vv],dim="time")
    cov_mldvar.append(covv)
print("Computed covariance with mldvar in %.2fs" % (time.time()-st))


#%% Visualize!

fsz_axis        = 24
fsz_title       = 28
fsz_tick        = 16

covstr = r"Cov($\frac{ \partial %s'}{\partial t}, \, \frac{h'}{\overline{h}} \,\frac{\partial \overline{%s}}{\partial t}$)" 
#[print(vname[0],vname[0]) for vname in vnames]
titles = [covstr % (vname[-1],vname[-1]) for vname in vnames] #["Subplot A"," Subplot B"]
vmaxes = [.05,.0005]
vunits = ["\degree C","psu"]

#cb_lab = "Colorbar Label [Units]"

fig,axs,_ = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))

ii = 0
for vv in range(2):
    
    ax      = axs[vv]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    ax.set_title(titles[vv],fontsize=fsz_title)
    
    cb_lab  = "$%s^{2} month^{-2}$" % (vunits[vv]) # titles[vv]
    
    plotvar = cov_mldvar[vv].mean('ens') * mask
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                            transform=proj,
                            vmin=-vmaxes[vv],vmax=vmaxes[vv],cmap='cmo.balance')
    
    cb = viz.hcbar(pcm,ax=ax,fraction=0.05,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label(cb_lab,fontsize=fsz_axis)
    
    viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,x=0.05)
    ii += 1


savename = "%sCovariability_MLDVar.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')  


# -----------------------------------------------------------------------------
#%% Load and visualize geostorphic advection
# -----------------------------------------------------------------------------

st          = time.time()
ugeoprime   = xr.open_dataset(rawpath + 'ugeoprime_gradT_gradS_NATL.nc').load()
ugeobar     = xr.open_dataset(rawpath + 'ugeobar_gradTprime_gradSprime_NATL.nc').load()
print("Loaded files in %.2fs" % (time.time()-st))

ugeos_sst   = [ugeobar.SST,ugeoprime.SST]
ugeos_sss   = [ugeobar.SSS,ugeoprime.SSS]

#%% Compute covariance with the tendency

dtmon = 3600*24*30

st               = time.time()
cov_ugeo_sst     = [xr.cov(ds_anom[0],ugeos_sst[ii] * dtmon,dim='time') for ii in range(2)]
cov_ugeo_sss     = [xr.cov(ds_anom[1],ugeos_sss[ii] * dtmon,dim='time') for ii in range(2)]
print("Computed covariance with ugeo terms in %.2fs" % (time.time()-st))


#%% Now Visualize it...

fig,axs,_   = viz.init_orthomap(2,2,bboxplot,figsize=(20,14.5))
ii          = 0

for vv in range(2):
    
    if vv == 0:
        cov_in = cov_ugeo_sst
        vname  = "SST"
        vmax   = 0.1
        #cblab  = 
    else:
        cov_in = cov_ugeo_sss
        vname  = "SSS"
        vmax   = 0.01
        
    
    for yy in range(2): # Loop for experinent
    
        if yy == 0:
            uname = "$\overline{u_{geo}}$"
        else:
            uname = "$u_{geo}'$"
            
    
        
        # Select Axis
        ax  = axs[vv,yy]
        
        # Set Labels
        blb = viz.init_blabels()
        if yy !=0:
            blb['left']=False
        else:
            blb['left']=True
            viz.add_ylabel(vname,ax=ax,rotation='horizontal',fontsize=fsz_title)
        if vv == 1:
            blb['lower'] =True
        
        
        if vv == 0:
            ax.set_title(uname,fontsize=fsz_title)
        
        #ax.set_title("%s\n$\sigma^2$=%.5f $%s^2$"% (plotnames[ii],plotstds[ii],vunit),fontsize=fsz_title)
        ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=fsz_tick,blabels=blb,
                                        fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        
        plotvar = cov_in[yy].mean('ens') * mask
        
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,
                                vmin=-vmax,vmax=vmax,
                                cmap='cmo.balance')
        
        
        
        if yy == 1:
            cb_lab  = "$%s^{2} month^{-2}$" % (vunits[vv]) # titles[vv]
            cb = fig.colorbar(pcm,ax=axs[vv,:].flatten(),fraction=0.015,pad=0.01)
            cb.ax.tick_params(labelsize=fsz_tick)
            cb.set_label(cb_lab,fontsize=fsz_axis)
            
        #cb.set_label(cb_lab,fontsize=fsz_axis)

savename = "%sCovariability_Ugeo.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')  


#%% Repeat with the other variables






#%%




#%%
# Compute dTbar dt [ for dT/dt(Jan) = T(Feb) - T(Jan) ], units are [degC/mon] and [psu/mon]
dTbar_dt       = Tbar.differentiate('month',datetime_unit='m') # (42, 12, 96, 89)
dSbar_dt       = Sbar.differentiate('month',datetime_unit='m')



#%%





#%%

#%%

