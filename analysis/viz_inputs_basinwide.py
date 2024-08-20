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
expname_sss         = "SSS_EOF_LbddCorr_Rerun_lbdE_neg" #"SSS_EOF_Qek_LbddEnsMean"#"SSS_EOF_Qek_LbddEnsMean"
expname_sst         = "SST_EOF_LbddCorr_Rerun"

# Load the parameter dictionary
expparams_raw_byvar = []
expkeys_byvar       = []
expparams_byvar     = []
for expname in [expname_sst,expname_sss]:
    expparams_raw   = np.load("%s%s/Input/expparams.npz" % (output_path,expname),allow_pickle=True)
    expparams_raw_byvar.append(expparams_raw)
    
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
    
    expkeys_byvar.append(expkeys)
    expparams_byvar.append(expparams)

# Constants
dt    = 3600*24*30 # Timestep [s]
cp    = 3850       # 
rho   = 1026       # Density [kg/m3]
B     = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L     = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

fsz_tick    = 18

debug       = False

#%% Add some functions to load (and convert) inputs

def stdsqsum(invar,dim):
    return np.sqrt(np.nansum(invar**2,dim))
    

def convert_ds(invar,lat,lon,):
    if len(invar.shape) == 4: # Include mode
        coords = dict(mode=np.arange(1,87),mon=np.arange(1,13,1),lat=lat,lon=lon)
    else:
        coords = dict(mon=np.arange(1,13,1),lat=lat,lon=lon)
    return xr.DataArray(invar,coords=coords,dims=coords)



#%% Check and Load Params (copied from run_SSS_basinwide.py on 2024-03-04)

# Load the parameters
print("Loading inputs for %s" % expname)

# Load Parameters
inputs,inputs_ds,inputs_type,params_vv = scm.load_params(expparams,input_path)

# Convert to the same units
convdict                               = scm.convert_inputs(expparams,inputs,return_sep=True)

# get lat.lon
ds = inputs_ds['h']
lat = ds.lat.data
lon = ds.lon.data

# 
varkeys = list(convdict.keys())
nk = len(varkeys)
conv_da = {}
for nn in range(nk):
    varkey = varkeys[nn]
    invar  = convdict[varkey]
    conv_da[varkey] =convert_ds(invar,lat,lon)

# # First, Check if there is EOF-based forcing (remove this if I eventually redo it)
# if expparams['eof_forcing']:
#     print("EOF Forcing Detected.")
#     eof_flag = True
# else:
#     eof_flag = False

# # Indicate the Parameter Names (sorry, it's all hard coded...)
# if expparams['varname']== "SSS": # Check for LHFLX, PRECTOT, Sbar
#     chk_params = ["h","LHFLX","PRECTOT","Sbar","lbd_d","beta","kprev","lbd_a","Qek"]
#     param_type = ["mld","forcing","forcing","forcing","damping","mld","mld","damping",'forcing']
# elif expparams['varname'] == "SST": # Check for Fprime
#     chk_params = ["h","Fprime","lbd_d","beta","kprev","lbd_a","Qek"]
#     param_type = ["mld","forcing","damping","mld","mld","damping",'forcing']

# # Check the params
# ninputs       = len(chk_params)
# inputs_ds     = {}
# inputs        = {}
# inputs_type   = {}
# missing_input = []
# for nn in range(ninputs):
#     # Get Parameter Name and Type
#     pname = chk_params[nn]
#     ptype = param_type[nn]
    
#     # Check for Exceptions (Can Fix this in the preprocessing stage)
#     if pname == 'lbd_a':
#         da_varname = 'damping'
#     else:
#         da_varname = pname
    
#     #print(pname)
#     if type(expparams[pname])==str: # If String, Load from input folder
        
#         # Load ds
#         ds = xr.open_dataset(input_path + ptype + "/" + expparams[pname])[da_varname]
        

#         # Crop to region
        
#         # Load dataarrays for debugging
#         dsreg            = proc.sel_region_xr(ds,expparams['bbox_sim']).load()
#         inputs_ds[pname] = dsreg.copy() 
        
#         # Load to numpy arrays 
#         varout           = dsreg.values
#         inputs[pname]    = dsreg.values.copy()
        
#         if ((da_varname == "Fprime") and (eof_flag)) or ("corrected" in expparams[pname]):
#             print("Loading %s correction factor for EOF forcing..." % pname)
#             ds_corr                          = xr.open_dataset(input_path + ptype + "/" + expparams[pname])['correction_factor']
#             ds_corr_reg                      = proc.sel_region_xr(ds_corr,expparams['bbox_sim']).load()
            
#             # set key based on variable type
#             if da_varname == "Fprime":
#                 keyname = "correction_factor"
#             elif da_varname == "LHFLX":
#                 keyname = "correction_factor_evap"
#             elif da_varname == "PRECTOT":
#                 keyname = "correction_factor_prec"
                
#             inputs_ds[keyname]   = ds_corr_reg.copy()
#             inputs[keyname]      = ds_corr_reg.values.copy()
#             inputs_type[keyname] = "forcing"
        
#     else:
#         print("Did not find %s" % pname)
#         missing_input.append(pname)
#     inputs_type[pname] = ptype



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

fsz_title   = 26
fsz_axis    = 22
fsz_lbl     = 10

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

#%% Load additional variables for gulf stream, land ice mask

# Load Land Ice Mask
icemask         = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

mask            = icemask.MASK.squeeze()
mask_plot       = xr.where(np.isnan(mask),0,mask)#mask.copy()


mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub

mask_apply  = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

# Load Gulf Stream
ds_gs           = dl.load_gs()
ds_gs           = ds_gs.sel(lon=slice(-90,-50))
ds_gs2          = dl.load_gs(load_u2=True)


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
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,cmap=cmap)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    
    # Do special contours
    if cints_sp is not None:
        cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        levels=cints_sp,colors="w",linewidths=0.75)
        ax.clabel(cl,fontsize=fsz_lbl)
        
        
    # Special plot for MLD (mark month of maximum)
    hmask_max  = (hmax == im) * ds_mask # Note quite a fix, as 0. points will be rerouted to april
    hmask_min  = (hmin == im) * ds_mask
    
    smap = viz.plot_mask(hmask_max.lon,hmask_max.lat,hmask_max.T,reverse=True,
                         color="yellow",markersize=0.75,marker="x",
                         ax=ax,proj=proj,geoaxes=True)
    
    smap = viz.plot_mask(hmask_min.lon,hmask_min.lat,hmask_min.T,reverse=True,
                         color="red",markersize=0.75,marker="o",
                         ax=ax,proj=proj,geoaxes=True)
    
    
    
if vlms is not None:
    cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    # cb = fig.colorbar(pcm,ax=axs.flatten(),
    #                   orientation='horizontal',pad=0.02,fraction=0.025)
    # cb.set_label(vlabel)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow.png" % (figpath,expname,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ------------------------------- -------------------
#%% Make plot of maximum wintertime mixed layer depth
# ------------------------------- -------------------

vlabel      = "Max Seasonal Mixed Layer Depth (meters)"

fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,14.5),)
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

# Plot maximum MLD
plotvar     = selvar.max('mon') * ds_mask


# Just Plot the contour with a colorbar for each one
if vlms is None:
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,cmap=cmap,zorder=-1)
    fig.colorbar(pcm,ax=ax)
else:
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-1)
    

# Do special contours
if cints_sp is not None:
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    levels=cints_sp,colors="w",linewidths=0.75)
    ax.clabel(cl,fontsize=fsz_tick)
    
# Special plot for MLD (mark month of maximum)
hmask_feb  = (hmax == 1) * ds_mask # Note quite a fix, as 0. points will be rerouted to april
hmask_mar  = (hmax == 2) * ds_mask

smap = viz.plot_mask(hmask_feb.lon,hmask_feb.lat,hmask_feb.T,reverse=True,
                     color="violet",markersize=5,marker="x",
                     ax=ax,proj=proj,geoaxes=True)

smap = viz.plot_mask(hmask_mar.lon,hmask_mar.lat,hmask_mar.T,reverse=True,
                     color="palegoldenrod",markersize=2.5,marker="o",
                     ax=ax,proj=proj,geoaxes=True)

        
if vlms is not None:
    cb = viz.hcbar(pcm,ax=ax,fraction=0.025)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
# Add additional features
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=4,c='firebrick',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=4,
           transform=proj,levels=[0,1],zorder=-1)
    


savename = "%sWintertime_MLD_CESM1_%s.png" % (figpath,expname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ---------------------------------
#%% Plot Detrainment Damping
# ---------------------------------

# Set some parameters
vname          = 'lbd_d'
vname_long     = "Deep Damping"
plotcontour    = False
#corrmode      = True # Using correlations rather than detrainment damping values
#vlms          = [0,48]#[0,0.2]# None
plot_timescale = False

if "corr" in expparams['lbd_d']:
    corrmode = True
else:
    corrmode = False

if corrmode:
    vlabel         = "Corr(Detrain,Entrain)"
else:
    vlabel         = "$\lambda_d$ ($months^{-1}$)"
    
    if "SSS" in expname:
        cints_sp       = np.arange(0,66,12)#None#np.arange(200,1500,100)# None
    elif "SST" in expname:
        cints_sp       = np.arange(0,66,12)

# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Preprocessing
ds_mask = xr.where( selvar != 0. , 1 , np.nan)
if corrmode is False:
    
    if plot_timescale:
        
        plotvar = 1/selvar
        vlabel = "$\lambda^d^{-1}$ ($months$)"
        if "SSS" in expname:
            vlms           = [0,48]
        elif "SST" in expname:
            vlms           = [0,24]
        cmap           = 'inferno'
    else:
        
        plotvar = selvar
        vlabel ="$\lambda^d$ ($months^{-1}$)"
        if "SSS" in expname:
            vlms           = [0,0.2]
        elif "SST" in expname:
            vlms           = [0,0.5]
        cmap           = 'inferno'
        
else:
    
    if plot_timescale:
        plotvar = -1/np.log(selvar)
        vlms    = [0,12]
        vlabel = "Deep Damping Timescale ($months$)"
    else:
        plotvar = selvar
        vlms    = [0,1]
        vlabel = "Corr(Detrain,Entrain)"
    
plotvar = plotvar * ds_mask

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    pv = plotvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,pv,transform=proj,cmap=cmap)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,pv,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    
    # Do special contours
    if cints_sp is not None:
        plotvar = 1/plotvar
        cl = ax.contour(lon,lat,pv,transform=proj,
                        levels=cints_sp,colors="lightgray",linewidths=0.75)
        ax.clabel(cl,fontsize=fsz_lbl)

if vlms is not None:
    
    cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    # cb = fig.colorbar(pcm,ax=axs.flatten(),
    #                   orientation='horizontal',pad=0.02,fraction=0.025)
    # cb.set_label(vlabel)

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
cints_prec     = np.arange(0,0.022,0.002)
plotcontour    = True

cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.rain'
convert_precip = True

# For Precip, also get the correct factor

# Get variable, lat, lon
selvar      = inputs_ds[vname]
selvar      = np.sqrt((selvar**2).sum('mode'))
lon         = selvar.lon
lat         = selvar.lat

# Convert Precipitation, if option is set
if convert_precip: # Adapted from ~line 559 of run_SSS_basinwide

    print("Converting precip to psu/mon")
    conversion_factor   = ( dt*inputs['Sbar'] / inputs['h'] )
    selvar              =  selvar * conversion_factor
    
    vlabel              = "$P$ ($psu/mon$)"
    vlms                = np.array([0,0.02])#None#[0,0.05]#[0,0.2]# None
    
else:
    
    vlabel         = "$P$ ($m/s$)"
    vlms           = np.array([0,1.5])*1e-8#None#[0,0.05]#[0,0.2]# None
    


# Preprocessing
ds_mask     = xr.where( selvar != 0. , 1 , np.nan)
selvar      = selvar * ds_mask 

fig,axs     = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        if plotcontour:
            pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                              cmap=cmap,levels=cints_prec,zorder=-3)
            
            cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                              levels=cints_prec,zorder=-3)
            ax.clabel(cl,fontsize=fsz_tick-4)
        else:
            pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                                cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)
    
    # plotvar2 = selvar2.isel(mon=im)
    # cl = ax.contour(lon,lat,plotvar2,transform=proj,
    #                 colors="k",linewidths=0.75)
    # ax.clabel(cl,fontsize=fsz_lbl)
    
    # Add additional features
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='cornflowerblue',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
    
if vlms is not None:
    cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow_convert%i.png" % (figpath,expname,vname,convert_precip)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot the mean Precip
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,14.5),)
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

plotvar = selvar.mean('mon')

# Just Plot the contour with a colorbar for each one
if vlms is None:
    pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
    fig.colorbar(pcm,ax=ax)
else:
    if plotcontour:
        pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                          cmap=cmap,levels=cints_prec,zorder=-3)
        
        cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                          levels=cints_prec,zorder=-3)
        ax.clabel(cl,fontsize=fsz_tick-4)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)

# plotvar2 = selvar2.isel(mon=im)
# cl = ax.contour(lon,lat,plotvar2,transform=proj,
#                 colors="k",linewidths=0.75)
# ax.clabel(cl,fontsize=fsz_lbl)

# Add additional features
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='cornflowerblue',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
           transform=proj,levels=[0,1],zorder=-1)

    
if vlms is not None:
    cb = viz.hcbar(pcm,ax=ax)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_Mean_convert%i.png" % (figpath,expname,vname,convert_precip)
plt.savefig(savename,dpi=150,bbox_inches='tight')


# -------------------
#%% Visualize Evap
# -------------------

# Set some parameters
vname          = 'LHFLX'
vname_long     = "Latent Heat FLux"
vlabel         = "$LHFLX$ ($W/m^2$)"
plotcontour    = True

cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.amp'
convert_lhflx  = True


# Get variable, lat, lon
selvar      = inputs_ds[vname]
selvar      = np.sqrt((selvar**2).sum('mode'))
lon         = selvar.lon
lat         = selvar.lat

# Do Conversion of Evaporation if option is set
if convert_lhflx:
    conversion_factor = ( dt*inputs['Sbar'] / (rho*L*inputs['h']))
    selvar_in         = selvar * conversion_factor # [Mon x Lat x Lon] * -1
    
    vlms           = [0,0.02]#[0,0.2]# None
    cints_evap     = np.arange(0,0.022,0.002)
    
else:
    selvar_in = selvar.copy()
    vlms           = [0,35]#[0,0.2]# None
    cints_evap     = np.arange(0,39,3)
    

# Preprocessing
ds_mask        = xr.where( selvar != 0. , 1 , np.nan)
selvar_in      = selvar_in * ds_mask 

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar_in.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        
        if plotcontour:
            
            pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                              cmap=cmap,levels=cints_evap,zorder=-3)
            
            cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                              levels=cints_prec,zorder=-3)
            ax.clabel(cl,fontsize=fsz_tick-4)
            
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
    cb = viz.hcbar(pcm,ax=axs.flatten())
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow_convert%i.png" % (figpath,expname,vname,convert_lhflx)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot the mean evaporation

fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,14.5),)
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

plotvar = selvar_in.mean('mon')


im      = plotmon[aa]
plotvar = selvar_in.isel(mon=im) 

# Just Plot the contour with a colorbar for each one
if vlms is None:
    pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
    fig.colorbar(pcm,ax=ax)
else:
    
    if plotcontour:
        
        pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                          cmap=cmap,levels=cints_evap,zorder=-3)
        
        cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                          levels=cints_prec,zorder=-3)
        ax.clabel(cl,fontsize=fsz_tick-4)
        
    else:
        
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)
    

    
    
if vlms is not None:
    cb = viz.hcbar(pcm,ax=ax)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_Mean_convert%i.png" % (figpath,expname,vname,convert_lhflx)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ----------------------------------------
#%% Visualize the correction factor (evap)
# ----------------------------------------

# Set some parameters
vname          = 'correction_factor_evap'
vname_long     = "Latent Heat FLux (Correction Factor)"
vlabel         = "$LHFLX$ ($W/m^2$)"
plotcontour    = True
vlms           = [0,15]#[0,0.2]# None
cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.amp'

convert_lhflx  = True


# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Do Conversion of Evaporation if option is set
if convert_lhflx:
    conversion_factor = ( dt*inputs['Sbar'] / (rho*L*inputs['h']))
    selvar_in         = selvar * conversion_factor # [Mon x Lat x Lon] * -1
    
    vlms           = [0,0.02]#[0,0.2]# None
    cints_evap     = np.arange(0,0.022,0.002)
    
else:
    selvar_in = selvar.copy()
    vlms           = [0,35]#[0,0.2]# None
    cints_evap     = np.arange(0,39,3)

# Preprocessing
ds_mask     = xr.where( selvar != 0. , 1 , np.nan)
selvar_in      = selvar_in * ds_mask 

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar_in.isel(mon=im) 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        
        if plotcontour:
            
            pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                              cmap=cmap,levels=cints_evap,zorder=-3)
            
            cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                              levels=cints_prec,zorder=-3)
            ax.clabel(cl,fontsize=fsz_tick-4)
            
        else:
            
            pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                                cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)
        
        


if vlms is not None:
    cb = viz.hcbar(pcm,ax=axs.flatten())
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow_convert%i.png" % (figpath,expname,vname,convert_lhflx)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ------------------------------------------
#%% Visualize the correction factor (precip)
# ------------------------------------------

# Set some parameters
vname          = 'correction_factor_prec'
vname_long     = "Precipitation (Correction Factor)"

plotcontour    = True
vlms           = np.array([0,3])*1e-9
cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.rain'
convert_precip = True

# Get variable, lat, lon
selvar      = inputs_ds[vname]
lon         = selvar.lon
lat         = selvar.lat

# Convert Precipitation, if option is set
if convert_precip: # Adapted from ~line 559 of run_SSS_basinwide
    print("Converting precip to psu/mon")
    conversion_factor   = ( dt*inputs['Sbar'] / inputs['h'] )
    selvar              =  selvar * conversion_factor
    
    vlms                = np.array([0,0.02])#None#[0,0.05]#[0,0.2]# None
    vlabel         = "$P$ ($psu/mon$)"
else:
    
    vlms           = np.array([0,1.5])*1e-8#None#[0,0.05]#[0,0.2]# None
    vlabel         = "$P$ ($m/s$)"




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
        
        if plotcontour:
            
            pcm = ax.contourf(lon,lat,plotvar,transform=proj,
                              cmap=cmap,levels=cints_evap,zorder=-3)
            
            cl = ax.contour(lon,lat,plotvar,transform=proj,colors="k",linewidths=0.75,
                              levels=cints_prec,zorder=-3)
            ax.clabel(cl,fontsize=fsz_tick-4)
        else:
            pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                                cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.025)
    cb.set_label(vlabel)

plt.suptitle("%s (CESM1 Ensemble Average)" % (vname_long),fontsize=fsz_title)
savename = "%s%s_Model_Inputs_%s_seasonrow_convert%i.png" % (figpath,expname,vname,convert_precip)
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

#%% Plot Monthly Values of each input
"""
For SSS

PRECIP
LHFL
Qek
Correction Factor(PRECIP)
Correction Factor(LHFLX)

For SST
 dict_keys(['correction_factor', 'Fprime', 'lbd_a', 'Qek', 'alpha'])
"""

vname_exp = expparams['varname']

if vname_exp == "SSS":
    
    prectot         = stdsqsum(convdict['PRECTOT'],0) # Precipitation  # Mon x Lat x Lon
    prectot_corr    = stdsqsum(convdict['correction_factor_prec'],0) # Precipitation 
    
    lhflx           = stdsqsum(convdict['LHFLX'],0)
    lhflx_corr      = stdsqsum(convdict['correction_factor_evap'],0)
    
    Qek             = stdsqsum(convdict['Qek'],0)
    
    alpha           = stdsqsum(convdict['alpha'],0)
    
    plotvars = [prectot,prectot_corr,lhflx,lhflx_corr,Qek,alpha]
    pvnames  = ["Precipitation" ,"Precipitation Correction" ,"Evaporation","Evaporation Correction","Ekman Forcing","Total Forcing"]
    pvcmaps  = ["cmo.rain"      ,"cmo.rain"                 ,"cmo.haline" ,"cmo.haline"            ,"cmo.amp"      ,"cmo.deep"]
    
elif vname_exp == "SST":
    
    
    fprime          = stdsqsum(convdict['Fprime'],0)
    corrfac         = stdsqsum(convdict['correction_factor'],0)
    
    Qek             = stdsqsum(convdict['Qek'],0)
    
    alpha           = stdsqsum(convdict['alpha'],0)
    
    plotvars = [fprime,corrfac,Qek,alpha]
    pvnames  = ["Stochastic Heat Flux"  ,"Heat Flux Correction" ,"Ekman Forcing"    ,"Total Forcing"]
    pvcmaps  = ["cmo.thermal"           ,"cmo.thermal"          ,"cmo.thermal"      ,"cmo.thermal"]

#% ['correction_factor_evap', 'LHFLX', 'correction_factor_prec', 'PRECTOT', 'lbd_a', 'Qek', 'Qfactor', 'alpha']

#%% First figure, visualize the total forcing for each season

plotfrac        = False
plotcontour     = True

if vname_exp == "SSS":
    nsp = 6
    figsize  = (28,4.75)
    vlim_in = [0,0.025]
    vstep = 0.002
    vunit = "$psu$"
elif vname_exp =="SST":
    figsize  = (18,4.75)
    nsp = 4
    vlim_in = [0,0.85]
    vstep = 0.05
    vunit = "$\degree C$"

fig,axs,mdict = viz.init_orthomap(1,nsp,bboxplot,figsize=figsize,constrained_layout=True,centlat=45)

for a,ax in enumerate(axs):
    
    # Plot Some Things
    ax = viz.add_coast_grid(ax,bbox=bboxplot,proj=proj,fill_color='lightgray')
    ax.set_title(pvnames[a],fontsize=fsz_title)
    
    plotvar = plotvars[a]
    if len(plotvar.shape)>2:
        plotvar = plotvar.mean(0)
    
    if plotfrac: # This is still giving funky answers
        alpha_denom = alpha.mean(0)
        alpha_denom[alpha_denom==0] = np.nan
        plotvar = plotvar/alpha_denom
        vlims = [0,1]
    else:
        vlims = vlim_in
        
        
    # Apply the mask
    coords  = dict(lat=lat,lon=lon)
    plotvar = xr.DataArray(plotvar,coords=coords)
    plotvar = plotvar * mask_reg
    
    if plotcontour:
        levels  = np.arange(vlims[0],vlims[1],vstep)
        pcm     = ax.contourf(lon,lat,plotvar,transform=proj,levels=levels,cmap=pvcmaps[a],extend="both")
        cl      = ax.contour(lon,lat,plotvar,transform=proj,levels=levels,colors="k",linewidths=0.75)#vmax=vlims[1],cmap=pvcmaps[a])
        ax.clabel(cl)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlims[0],vmax=vlims[1],cmap=pvcmaps[a])
    
    cb  = viz.hcbar(pcm,ax=ax,fraction=0.035)
    cb.ax.tick_params(labelsize=fsz_tick-3,rotation=45)
    
    # Plot Additional Features --- --- --- --- --- --- --- --- 
    # Plot Gulf Stream Position
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.5,c='red',ls='dashdot')
    
    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=1.5,
               transform=proj,levels=[0,1],zorder=2)
    
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    
plt.suptitle("Mean Std. Dev. for %s Forcing Terms [%s/mon]" % (vname_exp,vunit),fontsize=fsz_title)

savename = "%s%s_Model_Inputs_%s_MeanStdev.png" % (figpath,expname,vname_exp)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Look at regional averages of parameters

bbname    = "Sargasso Sea"
sel_box   = [-70,-55,35,40] # Sargasso Sea SSS CSU

# bbname     = "North Atlantic Current"
# sel_box    =  [-40,-30,40,50] # NAC

# bbname     = "Irminger Sea"
# sel_box    =  [-40,-25,50,60] # Irminger

varkeys = list(conv_da.keys())
nkeys   = len(varkeys)
da_reg  = []
for nk in range(nkeys):
    dar         = proc.sel_region_xr(conv_da[varkeys[nk]],sel_box)
    #dar_avg     = dar#.mean('lat').mean('lon')
    
    if len(dar.shape) > 3:
        dar = stdsqsum(dar,0)
    
    
    da_reg.append(dar)
    
da_reg.append(proc.sel_region_xr(inputs_ds['lbd_d'],sel_box))


# Set som consistent colors for plotting

if vname_exp == "SST":
    
    # varkeys: ['correction_factor', 'Fprime', 'lbd_a', 'Qek', 'alpha']
    vname_long_pt = ["Heat Flux Correction" ,"Stochastic Heat Flux"  ,"Atmospheric Heat Flux Feedback","Ekman Forcing" ,"Total Forcing"]
    vcolors       = ["plum"                 ,"darkviolet"            ,"red"                           ,"cornflowerblue","k"]
    vls           = ['dotted'               ,"dashed"                ,"solid"                         ,"dashed"        ,"solid"  ]
    
    
elif vname_exp == "SSS":
    
    skipvar       = ["lbd_a","Qfactor"]
    vname_long_pt = ["Evaporation Correction","Evaporation","Precipitation Correction" ,"Precipitation" ,"Atmospheric Heat Flux Feedback","Ekman Forcing","Correction Fator","Total Forcing"]
    vcolors       = ["khaki"                 ,"darkkhaki"  ,"darkturquoise"            ,"teal"          ,"gray"                          ,"cornflowerblue",'pink'           ,"k"]
    vls           = ["dotted"                ,"dashed"     ,"dotted"                   ,"dashed"        ,"solid"                         ,"dashed"        ,"dotted"         ,"solid"]
dcol = "navy"

    
#%% Plot the values



fig,ax = viz.init_monplot(1,1,figsize=(6,4))
for nk in range(nkeys):
    if varkeys[nk] in skipvar:
        print("skipping %s" % varkeys[nk] )
        continue
    plotvar = da_reg[nk]
    mu      = np.nanmean(plotvar,(1,2)) ##mean('lat').mean('lon')
    ax.plot(mons3,mu,label=vname_long_pt[nk],lw=2.5,c=vcolors[nk],ls=vls[nk])
    #ax.plot(mons3,mu,label=varkeys[nk],lw=2.5,)

# Plot deep damping
ax2 = ax.twinx()
ax2.plot(mons3,np.nanmean(da_reg[-1],(1,2)),c=dcol,lw=2.5)
ax2.set_ylabel("Detrainment Correlation")
ax2.spines['right'].set_color(dcol)
ax2.yaxis.label.set_color(dcol)
ax2.tick_params(axis='y', colors=dcol)

ax.legend(ncol=3,bbox_to_anchor=[1.1,1.15,0.1,0.1],fontsize=8)
ax.set_title("Monthly Variance for %s Forcing\nBounding Box: %s, %s" % (vname_exp,bbname,sel_box),y=1.25)
ax.set_ylabel("Forcing Amplitude (%s/month) \nor Damping (1/month)" % (vunit))
ax.set_xlabel("Month")


ax.tick_params(rotation=45)


savename = "%s%s_Model_Inputs_%s_region_%s.png" % (figpath,expname,vname,bbname)
print(savename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# da_regss = [stdsqsum(da,0) for da in da_reg]












#%% Plot as fraction of the total forcing

#%%

#%%


