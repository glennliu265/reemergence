#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Inputs for basinwide SST/SSS stochastic model

Created on Mon Mar  4 13:00:38 2024

@author: gliu

"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
machine    = "Astraeus"
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

# ----------------------------------
#%% User Edits
# ----------------------------------

# Indicate the experiment
expname         = "SSS_EOF_Qek_LbddEnsMean"#"SSS_EOF_Qek_LbddEnsMean"

# Load the parameter dictionary
expparams_raw   = np.load("%s%s/Input/expparams.npz" % (output_path,expname),allow_pickle=True)

# Fix parameter dictionary (they are all 0-d arrays)
expkeys     = list(expparams_raw.files)
expparams   = {}
for k in range(len(expkeys)):
    kn     = expkeys[k]
    arrout = expparams_raw[kn]
    if arrout.shape == ():
        arrout = arrout.item()
    else:
        arrout = np.array(arrout)
    expparams[kn] = arrout

# Constants
dt    = 3600*24*30 # Timestep [s]
cp    = 3850       # 
rho   = 1026       # Density [kg/m3]
B     = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L     = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document


debug = False

#%% Check and Load Params (copied from run_SSS_basinwide.py on 2024-03-04)

print("Loading inputs for %s" % expname)

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
    
    #print(pname)
    if type(expparams[pname])==str: # If String, Load from input folder
        
        # Load ds
        ds = xr.open_dataset(input_path + ptype + "/" + expparams[pname])[da_varname]
        

        # Crop to region
        
        # Load dataarrays for debugging
        dsreg            = proc.sel_region_xr(ds,expparams['bbox_sim']).load()
        inputs_ds[pname] = dsreg.copy() 
        
        # Load to numpy arrays 
        varout           = dsreg.values
        inputs[pname]    = dsreg.values.copy()
        
        if ((da_varname == "Fprime") and (eof_flag)) or ("corrected" in expparams[pname]):
            print("Loading %s correction factor for EOF forcing..." % pname)
            ds_corr                          = xr.open_dataset(input_path + ptype + "/" + expparams[pname])['correction_factor']
            ds_corr_reg                      = proc.sel_region_xr(ds_corr,expparams['bbox_sim']).load()
            
            # set key based on variable type
            if da_varname == "Fprime":
                keyname = "correction_factor"
            elif da_varname == "LHFLX":
                keyname = "correction_factor_evap"
            elif da_varname == "PRECTOT":
                keyname = "correction_factor_prec"
                
            inputs_ds[keyname]   = ds_corr_reg.copy()
            inputs[keyname]      = ds_corr_reg.values.copy()
            inputs_type[keyname] = "forcing"
        
    else:
        print("Did not find %s" % pname)
        missing_input.append(pname)
    inputs_type[pname] = ptype


#%% Visualize some of the inputs

# Set up mapping template
# Plotting Params
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
lon                         = ds.lon.values
lat                         = ds.lat.values
mons3                       = proc.get_monstr()

plotmon                     = np.roll(np.arange(12),1)

fsz_title= 26
fsz_axis = 22
fsz_lbl  = 10
#%%


def init_monplot():
    plotmon       = np.roll(np.arange(12),1)
    fig,axs,mdict = viz.init_orthomap(4,3,bboxplot=bboxplot,figsize=(18,18))
    for a,ax in enumerate(axs.flatten()):
        im = plotmon[a]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        ax.set_title(mons3[im],fontsize=fsz_axis)
    return fig,axs


# def make_mask_xr(ds,var):
#     ds_masked = ds.where(ds != val)
#     return ds_masked
# # ds_mask = make_mask_xr(selvar,0.)
# # ds_mask = ds_mask.sum('mon')

# Make A Mask
ds_mask = ds.sum('mon')
ds_mask = xr.where( ds_mask!=0. , 1 , np.nan)
ds_mask.plot()

#%% Lets make the plot
print(inputs.keys())

# ---------------------------------
#%% Plot Mixed Layer Depth by Month
# ---------------------------------

# Set some parameters
vname       = 'h'
vname_long  = "Mixed-Layer Depth"
vlabel      = "HMXL (meters)"
plotcontour = False
vlms        = [0,200]# None
cints_sp    = np.arange(200,1500,100)# None
cmap        = 'cmo.dense'

# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Special plot for HMXL, month of maximum
hmax        = selvar.argmax('mon').values
hmin        = selvar.argmin('mon').values


fig,axs     = init_monplot()

for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) * ds_mask
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    
    # Do special contours
    if cints_sp is not None:
        cl = ax.contour(lon,lat,plotvar,transform=proj,
                        levels=cints_sp,colors="w",linewidths=0.75)
        ax.clabel(cl,fontsize=fsz_lbl)
        
        
    # Special plot for MLD (mark month of maximum)
    hmask_max  = (hmax == im) * ds_mask # Note quite a fix, as 0. points will be rerouted to april
    hmask_min  = (hmin == im) * ds_mask
    
    smap = viz.plot_mask(lon,lat,hmask_max.T,reverse=True,
                         color="yellow",markersize=0.75,marker="x",
                         ax=ax,proj=proj,geoaxes=True)
    
    smap = viz.plot_mask(lon,lat,hmask_min.T,reverse=True,
                         color="red",markersize=0.75,marker="o",
                         ax=ax,proj=proj,geoaxes=True)
    
    
    
if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.025)
    cb.set_label(vlabel)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow.png" % (figpath,expname,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ---------------------------------
#%% Plot Detrainment Damping
# ---------------------------------

# Set some parameters
vname          = 'lbd_d'
vname_long     = "Detrainment Damping"
vlabel         = "$\lambda_d$ ($months^{-1}$)"
plotcontour    = False
#vlms          = [0,48]#[0,0.2]# None


if "SSS" in expname:
    cints_sp       = np.arange(0,66,12)#None#np.arange(200,1500,100)# None
elif "SST" in expname:
    cints_sp       = np.arange(0,66,12)


plot_timescale = False

# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Preprocessing
ds_mask = xr.where( selvar != 0. , 1 , np.nan)
if plot_timescale:
    selvar = 1/selvar
    vlabel = "$\lambda^d^{-1}$ ($months$)"
    if "SSS" in expname:
        vlms           = [0,48]
    elif "SST" in expname:
        vlms           = [0,24]
    cmap           = 'inferno'
else:
    vlabel ="$\lambda^d$ ($months^{-1}$)"
    if "SSS" in expname:
        vlms           = [0,0.2]
    elif "SST" in expname:
        vlms           = [0,0.5]
    cmap           = 'inferno'
    
selvar = selvar * ds_mask

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    
    # Do special contours
    if cints_sp is not None:
        plotvar = 1/plotvar
        cl = ax.contour(lon,lat,plotvar,transform=proj,
                        levels=cints_sp,colors="lightgray",linewidths=0.75)
        ax.clabel(cl,fontsize=fsz_lbl)
        
    
    
if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.025)
    cb.set_label(vlabel)

plt.suptitle("%s (Colors) and Timescale (Contours), CESM1 Ensemble Average" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow_timescale%i.png" % (figpath,expname,vname,plot_timescale)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ----------------------------------------------------------
#%% Plot Wintertime and Summertime Mean Detrainment Damping
# ----------------------------------------------------------


# Set some parameters
vname          = 'lbd_d'
vname_long     = "Detrainment Damping"
vlabel         = "$\lambda_d$ ($months^{-1}$)"
plotcontour    = False
#vlms          = [0,48]#[0,0.2]# None


if "SSS" in expname:
    cints_sp       = np.arange(0,66,12)#None#np.arange(200,1500,100)# None
elif "SST" in expname:
    cints_sp       = np.arange(0,66,12)


plot_timescale = False

# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Preprocessing
ds_mask = xr.where( selvar != 0. , 1 , np.nan)
if plot_timescale:
    selvar = 1/selvar
    vlabel = "$\lambda^d^{-1}$ ($months$)"
    if "SSS" in expname:
        vlms           = [0,48]
    elif "SST" in expname:
        vlms           = [0,24]
    cmap           = 'inferno'
else:
    vlabel ="$\lambda^d$ ($months^{-1}$)"
    if "SSS" in expname:
        vlms           = [0,0.2]
    elif "SST" in expname:
        vlms           = [0,0.5]
    cmap           = 'inferno'
    
selvar = selvar * ds_mask

fig,axs,mdict = viz.init_orthomap(1,2,bboxplot=bboxplot,figsize=(12,6))
for a,ax in enumerate(axs.flatten()):
    im = plotmon[a]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    ax.set_title(mons3[im],fontsize=fsz_axis)
    
    if a == 0:
        selmon = [11,0,1,2]
        title  = "Winter (DJFM)"
        
        cints_sp       = np.arange(0,66,12)
    elif a == 1:
        selmon = [6,7,8,9]
        title  = "Summer (JJAS)"
        
        if "SST" in expname:
            cints_sp       = np.arange(0,25,1)
        elif "SSS" in expname:
            cints_sp       = np.arange(0,36,3)
    plotvar = selvar.isel(mon=selmon).mean('mon')
    
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    
    # Do special contours
    if cints_sp is not None:
        plotvar = 1/plotvar
        cl = ax.contour(lon,lat,plotvar,transform=proj,
                        levels=cints_sp,colors="lightgray",linewidths=0.75)
        ax.clabel(cl,fontsize=fsz_lbl)
    ax.set_title(title,fontsize=fsz_title-2)

# for aa in range(12):
#     ax      = axs.flatten()[aa]
#     im      = plotmon[aa]
#     plotvar = selvar.isel(mon=im) 
    
#     # Just Plot the contour with a colorbar for each one
#     if vlms is None:
#         pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap)
#         fig.colorbar(pcm,ax=ax)
#     else:
#         pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
#                             cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    
#     # Do special contours
#     if cints_sp is not None:
#         plotvar = 1/plotvar
#         cl = ax.contour(lon,lat,plotvar,transform=proj,
#                         levels=cints_sp,colors="lightgray",linewidths=0.75)
#         ax.clabel(cl,fontsize=fsz_lbl)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.05)
    cb.set_label(vlabel)

plt.suptitle("%s (Colors) and Timescale (Contours),\nCESM1 Ensemble Average" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_wintersummermean_timescale%i.png" % (figpath,expname,vname,plot_timescale)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

# -------------------
#%% Visualize Precip
# -------------------

# Set some parameters
vname          = 'PRECTOT'
vname_long     = "Total Precipitation"
vlabel         = "$P$ ($m/s$)"
plotcontour    = False
vlms           = np.array([0,1.5])*1e-8#None#[0,0.05]#[0,0.2]# None
cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.rain'

# For Precip, also get the correct factor

# Get variable, lat, lon
selvar      = inputs_ds[vname]
selvar      = np.sqrt((selvar**2).sum('mode'))
lon         = selvar.lon
lat         = selvar.lat

# Preprocessing
ds_mask     = xr.where( selvar != 0. , 1 , np.nan)
selvar      = selvar * ds_mask 

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)
    
    # plotvar2 = selvar2.isel(mon=im)
    # cl = ax.contour(lon,lat,plotvar2,transform=proj,
    #                 colors="k",linewidths=0.75)
    # ax.clabel(cl,fontsize=fsz_lbl)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.05)
    cb.set_label(vlabel)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow.png" % (figpath,expname,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# -------------------
#%% Visualize Evap
# -------------------

# Set some parameters
vname          = 'LHFLX'
vname_long     = "Latent Heat FLux"
vlabel         = "$LHFLX$ ($W/m^2$)"
plotcontour    = False
vlms           = [0,35]#[0,0.2]# None
cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.amp'

# Get variable, lat, lon
selvar      = inputs_ds[vname]
selvar      = np.sqrt((selvar**2).sum('mode'))
lon         = selvar.lon
lat         = selvar.lat

# Preprocessing
ds_mask     = xr.where( selvar != 0. , 1 , np.nan)
selvar      = selvar * ds_mask 

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)
    
    # # Do special contours
    # if cints_sp is not None:
    #     plotvar = 1/plotvar
    #     cl = ax.contour(lon,lat,plotvar,transform=proj,
    #                     levels=cints_sp,colors="lightgray",linewidths=0.75)
    #     ax.clabel(cl,fontsize=fsz_lbl)
        
    
    
if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.025)
    cb.set_label(vlabel)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow.png" % (figpath,expname,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')



# ----------------------------------------
#%% Visualize the correction factor (evap)
# ----------------------------------------

# Set some parameters
vname          = 'correction_factor_evap'
vname_long     = "Latent Heat FLux (Correction Factor)"
vlabel         = "$LHFLX$ ($W/m^2$)"
plotcontour    = False
vlms           = [0,15]#[0,0.2]# None
cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.amp'

# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Preprocessing
ds_mask     = xr.where( selvar != 0. , 1 , np.nan)
selvar      = selvar * ds_mask 

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.025)
    cb.set_label(vlabel)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow.png" % (figpath,expname,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ------------------------------------------
#%% Visualize the correction factor (precip)
# ------------------------------------------

# Set some parameters
vname          = 'correction_factor_prec'
vname_long     = "Precipitation (Correction Factor)"
vlabel         = "$P$ ($m/s$)"
plotcontour    = False
vlms           = np.array([0,3])*1e-9
cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.rain'


# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Preprocessing
ds_mask     = xr.where( selvar != 0. , 1 , np.nan)
selvar      = selvar * ds_mask 

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.025)
    cb.set_label(vlabel)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow.png" % (figpath,expname,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ------------------------------------------
#%% Visualize Total Ekman Forcing
# ------------------------------------------

# Set some parameters
vname          = 'Qek'
vname_long     = "Ekman Forcing"
vlabel         = "$Q_{ek}$ (psu/mon)"
plotcontour    = False
vlms           = np.array([0,3]) * 1e-8
cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.amp'

# Get variable, lat, lon
selvar      = inputs_ds[vname].sum('mode')
lon         = selvar.lon
lat         = selvar.lat

# Preprocessing
ds_mask     = xr.where( selvar != 0. , 1 , np.nan)
selvar      = selvar * ds_mask 

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.025)
    cb.set_label(vlabel)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow.png" % (figpath,expname,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')




#%%

#%%

#%%


