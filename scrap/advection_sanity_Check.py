#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Do a sanity check on budget terms computed from the budget script and 
The terms computed via SSH/thermal wind


Copy Terms from each section

Created on Mon Oct 14 10:15:22 2024

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

from tqdm import tqdm

from cmcrameri import cm

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
rawpath     = pathdict['raw_path']

# ------------------------------
#%% Set some plotting parameters
# ------------------------------

bboxplot                        = [-80,0,20,65]
mpl.rcParams['font.family']     = 'Avenir'
mons3                           = proc.get_monstr(nletters=3)

fsz_tick                        = 18
fsz_axis                        = 20
fsz_title                       = 16

rhocrit                         = proc.ttest_rho(0.05,2,86)
proj                            = ccrs.PlateCarree()


dtmon   = 3600*24*30


# Get Point Info
pointset    = "PaperDraft02"
ptdict      = rparams.point_sets[pointset]
ptcoords    = ptdict['bboxes']
ptnames     = ptdict['regions']
ptnames_long = ptdict['regions_long']
ptcols      = ptdict['rcols']
ptsty       = ptdict['rsty']

#%% Load Mask and other plotting variables


# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()


mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub


mask_apply  = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

# Load Gulf Stream
ds_gs   = dl.load_gs()
ds_gs   = ds_gs.sel(lon=slice(-90,-50))
ds_gs2  = dl.load_gs(load_u2=True)


# ---------------------------------
#%% Load in UET/VNT/WTT
# ---------------------------------

# Load VNT UET
st          = time.time()
savename    = "%sCESM1_VNT_UET_MLSum_Budget_NAtl.nc" % rawpath
ds_vnt_uet  = xr.open_dataset(savename).load()
print("Loaded VNT, UET in %.2fs" % (time.time()-st))

# Load WTT
st          = time.time()
savename    = "%sCESM1_WTT_MLSum_Budget_NAtl.nc" % rawpath
ds_wtt  = xr.open_dataset(savename).load()
print("Loaded WTT in %.2fs" % (time.time()-st))


tlon = ds_vnt_uet.TLONG.data
tlat = ds_vnt_uet.TLAT.data

# ------------------------------------
#%%  Load Geostrophic Advection Terms
# ------------------------------------

vnames   = ["SST","SSS"]
ds_ugeos = []

for vv in range(2):
    vname           = vnames[vv]
    savename        = rawpath + "CESM1_HTR_FULL_Ugeo_%s_Transport_Full.nc" % (vname)
    ds              = xr.open_dataset(savename).load()
    ds_ugeos.append(ds)
    

# Just select 1 ensemble member
ds_ugeos_ens = [ds.isel(ens=1) for ds in ds_ugeos]






# ----------------------------
#%% Load Ekman transport
# ----------------------------

ds_ueks  = []
for vv in range(2):
    vname           = vnames[vv]
    savename        = rawpath + "CESM1_HTR_FULL_Uek_%s_Transport_Full.nc" % (vname)
    ds              = xr.open_dataset(savename).load()
    ds_ueks.append(ds)
    
    
    
    
#%% Compute Total Transport

ugeo_transport  = [ds.UET + ds.VNT for ds in ds_ugeos]
uek_transport   = [ds_ueks[0].SST, ds_ueks[1].SSS]
total_transport = ds_vnt_uet.UET + ds_vnt_uet.VNT

#%% Load additional variables to help

# Load the geostrophic currents
st               = time.time()
nc_ugeo          = "CESM1LE_ugeo_NAtl_19200101_20050101_bilinear.nc"
path_ugeo        = rawpath
ds_ugeo          = xr.open_dataset(path_ugeo + nc_ugeo).load()
print("Loaded ugeo in %.2fs" % (time.time()-st))

ds_ugeo_mean     = ds_ugeo.isel(ens=1).mean('time')

# Load time-mean SST and SSS
# Load data processed by [calc_monmean_CESM1.py]
ds_sss           = dl.load_monmean('SSS')
ds_sst           = dl.load_monmean('SST')


# ----------------------------------
#%% Compare Time Mean of the values (spatial pattern)
# ----------------------------------

vmax     = 1
ssize    = 50
plotvars = [ds_vnt_uet.UET.mean('time'),ds_vnt_uet.VNT.mean('time'),
            ds_ugeos_ens[0].UET.mean('time'),ds_ugeos_ens[0].VNT.mean('time')]


plotvnames  = ["UET","VNT","$U_{geo}$","$V_{geo}$"]

cints       = np.arange(272,306,1)
cmap        = 'cmo.balance'

fsz_title   = 32
fig,axs,_   = viz.init_orthomap(2,2,bboxplot,figsize=(20,14.5))


ii          = 0
for vv in range(2):
    for yy in range(2): # Loop for experinent
        
        # Select Axis
        ax  = axs[vv,yy]
        
        # Set Labels
        blb = viz.init_blabels()
        if yy !=0:
            blb['left']=False
        else:
            blb['left']=True
            #viz.add_ylabel(vnames[vv],ax=ax,rotation='horizontal',fontsize=fsz_title)
        if vv == 1:
            
            blb['lower'] =True
        
        # Include the several aspects
        
        ax.set_title(plotvnames[ii],fontsize=fsz_title)
        
        #ax.set_title("%s\n$\sigma^2$=%.5f $%s^2$"% (plotnames[ii],plotstds[ii],vunit),fontsize=fsz_title)
        ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=fsz_tick,blabels=blb,
                                        fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        plotvar      = plotvars[ii] * dtmon
        
        if ii < 2: # Do a scatter
            vmax    = 10
            pcm     = ax.scatter(tlon,tlat,c=plotvar,s=ssize,vmin=-vmax,vmax=vmax,transform=proj,cmap=cmap)
            
        
        else:
            vmax    = 1
            pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar * -1,vmin=-vmax,vmax=vmax,transform=proj,cmap=cmap)
        
        
        # Plot the Geostrophic Currents
        qint  = 2
        plotu = (ds_ugeo_mean.ug * mask).data #/ 100 * mask
        plotv = (ds_ugeo_mean.vg * mask).data #/ 100
        lonmesh,latmesh = np.meshgrid(ds_ugeo_mean.lon.data,ds_ugeo_mean.lat.data)
        ax.quiver(lonmesh[::qint,::qint],latmesh[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
                  color='magenta',transform=proj,alpha=1,scale=5)
        
        
        # Plot mean SST Contours
        plotvar = ds_sst.isel(ens=1).mean('mon') * mask
        cl = ax.contour(plotvar.lon,plotvar.lat,plotvar.SST,levels=cints,
                   linewidths=0.75,colors='k',transform=proj)
        ax.clabel(cl,fontsize=fsz_tick)
        
        
        
        
        cb = viz.hcbar(pcm,ax=ax)
        cb.ax.tick_params(labelsize=fsz_axis)
        
        
        ii += 1


savename = "%sMean_Total_v_Geostrophic_Transport_Sep_Terms.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

        
#%% Sum both and visualize...

utotal      = plotvars[0] + plotvars[1]
ugeototal   = plotvars[2] + plotvars[3]

invars      = [utotal,ugeototal]
intitles    = ["Total Transport","Geostrophic Transport"]


vmax  = 5


ssize = 66


fsz_axis        = 24
fsz_title       = 28
fsz_tick        = 18



titles = intitles
cb_lab = "$\degree$C per month"


fig,axs,_ = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))

ii = 0
for vv in range(2):
    
    ax      = axs[vv]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    ax.set_title(titles[vv],fontsize=fsz_title)
    
    plotvar = invars[vv] * dtmon
    
    if vv < 1: # Do a scatter
        pcm     = ax.scatter(tlon,tlat,c=plotvar,s=ssize,vmin=-vmax,vmax=vmax,transform=proj,cmap=cmap)
        
    else:
        
        print(None)
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar*-1,vmin=-vmax,vmax=vmax,transform=proj,cmap=cmap)
    
    
    #cb = viz.hcbar(pcm,ax=ax,fraction=0.05,pad=0.01)
    
    # Plot the Geostrophic Currents
    qint  = 2
    plotu = (ds_ugeo_mean.ug * mask).data #/ 100 * mask
    plotv = (ds_ugeo_mean.vg * mask).data #/ 100
    lonmesh,latmesh = np.meshgrid(ds_ugeo_mean.lon.data,ds_ugeo_mean.lat.data)
    ax.quiver(lonmesh[::qint,::qint],latmesh[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
              color='magenta',transform=proj,alpha=1,scale=5)
    
    
    # Plot mean SST Contours
    plotvar = ds_sst.isel(ens=1).mean('mon') * mask
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar.SST,levels=cints,
               linewidths=0.75,colors='k',transform=proj)
    ax.clabel(cl,fontsize=fsz_tick)
    
    viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,x=0.05)
    ii += 1

cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.05,pad=0.01)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label(cb_lab,fontsize=fsz_axis)

savename = "%sMean_Total_v_Geostrophic_Transport.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')



    
#%% Check at a single location

lonf,latf       = -70,40

locfn,loctitle = proc.make_locstring(lonf,latf)

ds_vnt_uet_pt   = proc.find_tlatlon(ds_vnt_uet,lonf+360,latf)
ds_ugeos_ens_pt = proc.selpt_ds(ds_ugeos_ens[0],lonf,latf)

savg     = lambda ds: ds.groupby('time.month').mean('time')
plotvars = [ds_vnt_uet_pt.UET,ds_vnt_uet_pt.VNT,
            ds_ugeos_ens_pt.UET,ds_ugeos_ens_pt.VNT]
plotvars = [savg(ds) for ds in plotvars]

plotcolors = ["red","midnightblue","hotpink","cornflowerblue"]
ls         = ["dashed",'dashed',"dotted",'dotted']

fig,ax     = viz.init_monplot(1,1,figsize=(8,4.5))
for ii in range(4):
    
    
    plotvar = plotvars[ii] * dtmon
    
    if ii > 1:
        plotvar = plotvar * -1
    
    # if ii < 2:
    #     plotvar = plotvar / 100
    
    ax.plot(mons3,plotvar,label=plotvnames[ii],lw=3.5,c=plotcolors[ii],ls=ls[ii])
    
# Now Plot the Net
utotal_pt = (plotvars[0] + plotvars[1])  *dtmon
ugeo_pt   = (plotvars[2] + plotvars[3])  *dtmon

ax.plot(mons3,utotal_pt,label="Total Transport",lw=3.5,c='k')
ax.plot(mons3,ugeo_pt*-1,label="Geostrophic Transport",lw=3.5,c='gray')

ax.set_title("Transport Terms @ %s" % (loctitle),fontsize=16)

ax.legend()


#%% Check Mean Geostrophic Advection

ugeo_mag = np.sqrt(ds_ugeo_mean.ug ** 2 + ds_ugeo_mean.vg **2 )
plt.pcolormesh(ugeo_mag,vmin=-.5,vmax=.5)
plt.colorbar()


#%% Draft 04 --> Visualize the Standard Deviation of Geostrophic Advection


ugeo_transports   = [ds.UET + ds.VNT for ds in ds_ugeos]
ugeo_stdev_monvar = [ds.groupby('time.month').std('time') for ds in ugeo_transports]
ugeo_stdev        = [ds.groupby.std('time') for ds in ugeo_transports]


#%%

#vmaxes       = [0.5,0.075]


plotcontour=True
plotadv    =True

vmaxes      = [0.75,0.1]
vcints      = [np.arange(0,1.2,0.2),np.arange(0,0.28,0.04)]
vcmaps      = [cm.lajolla_r,cm.acton_r]
vunits      = ["\degree C","psu"]
pmesh       = True


fsz_axis    = 24
fsz_title   = 28
fsz_tick    = 16

titles      = [r"$u_{geo} \cdot \nabla SST$",r"$u_{geo} \cdot \nabla SSS$"]


mean_contours = [ds_sst.mean('ens').mean('mon').SST,ds_sss.mean('ens').mean('mon').SSS]
cints_sst   = np.arange(250,310,2)
cints_sss   = np.arange(34,37.6,0.2)
cints_mean   = [cints_sst,cints_sss]

fig,axs,_   = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))
ii = 0
for vv in range(2):
    
    vmax    = vmaxes[vv]
    cmap    = vcmaps[vv]
    vunit   = vunits[vv]
    cints   = vcints[vv]
    
    ax      = axs[vv]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    ax.set_title(titles[vv],fontsize=fsz_title)
    
    plotvar = ugeo_stdev_monvar[vv].mean('ens').max('month') * dtmon * mask
    
    if pmesh:
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,vmin=0,vmax=vmax,cmap=cmap)
    else:
        pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints,
                                transform=proj,cmap=cmap,extend='both')
    
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,
                            transform=proj,colors="lightgray",lw=0.75)
    ax.clabel(cl,fontsize=fsz_tick)
    
    cb_lab = "[$%s$/mon]" % vunit
    
    cb = viz.hcbar(pcm,ax=ax,fraction=0.05,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label(cb_lab,fontsize=fsz_axis)
    
    # Add Other Features
    # Plot Gulf Stream Position
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,
            c='cornflowerblue',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=-1)
    
    
    nregs = len(ptnames)
    for ir in range(nregs):
        pxy   = ptcoords[ir]
        ax.plot(pxy[0],pxy[1],transform=proj,markersize=20,markeredgewidth=.5,c=ptcols[ir],
                marker='*',markeredgecolor='k')
    
    # Plot the Geostrophic Currents
    if plotadv:
        qint  = 2
        plotu = (ds_ugeo_mean.ug * mask).data #/ 100 * mask
        plotv = (ds_ugeo_mean.vg * mask).data #/ 100
        lonmesh,latmesh = np.meshgrid(ds_ugeo_mean.lon.data,ds_ugeo_mean.lat.data)
        ax.quiver(lonmesh[::qint,::qint],latmesh[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
                  color='cornflowerblue',transform=proj,alpha=1,scale=5)
    if plotcontour:
        # Plot mean Contours
        plotvar = mean_contours[vv] * mask
        cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    linewidths=1.5,colors="dimgray",levels=cints_mean[vv],linestyles='dashed')
        ax.clabel(cl,fontsize=fsz_tick)
        
    
    
    viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,x=0.05)
    ii += 1
    
    
savename = "%sUgeo_Contribution_Total.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Look at total mean transport (Ugeo, Ekman, )

tpin      = [ugeo_transport[0].isel(ens=1),
        uek_transport[0].isel(ens=1),
        total_transport]

plot_tp   = [ds.mean('time') for ds in tpin]

plotnames = ["Geostrophic","Ekman","Total"]

#%%

fsz_title   = 32
ssize       = 50

vmax_ugeo        = 1

fig,axs,_   = viz.init_orthomap(1,3,bboxplot,figsize=(24,10))

for vv in range(3):
    

    ax      = axs[vv]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    ax.set_title(plotnames[vv],fontsize=fsz_title)
    
    plotvar = plot_tp[vv] * dtmon #* mask
    
    if vv == 1:
        vmax = vmax_ugeo
        plotvar = plotvar *  100
    else:
        vmax = vmax_ugeo
    
    if vv == 2:
        pcm = ax.scatter(tlon,tlat,c=plotvar,s=ssize,transform=proj,
                         vmin=-vmax,vmax=vmax,cmap='cmo.balance')
    else:
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar*-1,transform=proj,
                            vmin=-vmax,vmax=vmax,cmap='cmo.balance')
    
    cb = viz.hcbar(pcm,ax=ax,fraction=0.025)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("$\degree$C/month",fontsize=fsz_axis)

#%%



plot_tp   = [ds.std('time') for ds in tpin]

plotnames = ["Geostrophic","Ekman","Total"]

#%

fsz_title   = 32
ssize       = 50

vmax_ugeo        = 1

fig,axs,_   = viz.init_orthomap(1,3,bboxplot,figsize=(24,10))

for vv in range(3):
    

    ax      = axs[vv]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    ax.set_title(plotnames[vv],fontsize=fsz_title)
    
    plotvar = plot_tp[vv] * dtmon #* mask
    
    if vv == 1:
        vmax = vmax_ugeo
        plotvar = plotvar *  100
    else:
        vmax = vmax_ugeo
    
    if vv == 2:
        pcm = ax.scatter(tlon,tlat,c=plotvar,s=ssize,transform=proj,
                         vmin=0,vmax=vmax,cmap='cmo.thermal')
    else:
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            vmin=0,vmax=vmax,cmap='cmo.thermal')
    
    cb = viz.hcbar(pcm,ax=ax,fraction=0.025)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("$\degree$C/month",fontsize=fsz_axis)

#%% Plot the total term and compare

#fsz_ticks     = 14

tpin_sum      = [proc.format_ds_dims(tpin[0]) + proc.format_ds_dims(tpin[1]) * 100, tpin[2]]

ylabs         = ["Mean","Mean Interannual Variability"]

plotnames_sum = ["Geostrophic + Ekman","Total Transport (UET + VNT)"]

fig,axs,_     = viz.init_orthomap(2,2,bboxplot,figsize=(18,14.5))

for mm in range(2):
    if mm == 0:
        infunc = lambda ds: ds.mean('time')
        cmap   = 'cmo.balance'
        vlms   = [-1,1]
        cints  = np.arange(-1,1.4,0.4)
    else:
        infunc = lambda ds: ds.groupby('time.month').std('time').mean('month')
        cmap   = cm.lajolla_r
        vlms   = [0,1]
        cints  = np.arange(0,1.2,0.2)
    
    for vv in range(2):
        
        ax      = axs[mm,vv]
        ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
        if mm == 0:
            ax.set_title(plotnames_sum[vv],fontsize=fsz_title)
        if vv == 0:
            viz.add_ylabel(ylabs[mm],fontsize=fsz_title,ax=ax)
        
        plotvar = infunc(tpin_sum[vv]) * dtmon
        
        if vv == 1:
            pcm = ax.scatter(tlon,tlat,c=plotvar,s=ssize,transform=proj,
                             vmin=vlms[0],vmax=vlms[1],cmap=cmap)
            
            cl  = ax.contour(tlon,tlat,plotvar,transform=proj,
                             levels=cints,colors="k",linewidths=0.75)
            ax.clabel(cl,fontsize=fsz_tick)
        else:
            if mm == 0:
                plotvar = plotvar * -1
            
            pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                                vmin=vlms[0],vmax=vlms[1],cmap=cmap)
            
            cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                             levels=cints,colors="k",linewidths=0.75)
            ax.clabel(cl,fontsize=fsz_tick)
        
        
        cb = viz.hcbar(pcm,ax=ax,fraction=0.025)
        cb.ax.tick_params(labelsize=fsz_tick)
        cb.set_label("$\degree$C/month",fontsize=fsz_axis)

savename = "%sSanity_Check_Heat_Transport_Ens02.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Examine the Differences (nvm cant because they are on different grids)

# fig,axs,_     = viz.init_orthomap(1,2,bboxplot,figsize=(18,10))

# for vv in range(2):
    
#     ax      = axs[vv]
#     ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    

#     if vv == 0: # Look at difference in the means
#         tpin_sum[1]

#%% Compare Monthly Interannual Variability, compared to stochastic model and CESM1


fig,axs = viz.init_monplot(1,3,constrained_layout=True,
                          figsize=(16,4.5))


for pt in range(3):
    ax        = axs[pt]
    lonf,latf = ptcoords[pt]
    
    #ax.set_title()
    
    ax.set_title(ptnames[pt],fontsize=fsz_title)
    
    for vv in range(3):
        
        if vv == 1:
            
            ptval = proc.find_tlatlon(ds_vnt_uet,lonf+360,latf)
            
            while np.all(np.isnan(ptval.UET)) or np.all(np.isnan(ptval.VNT)):
                ptval = proc.find_tlatlon(ds_vnt_uet,lonf+360-1,latf)
                
            ptval = ptval.UET + ptval.VNT
            
        elif vv == 0:
            
            ptval = proc.selpt_ds(tpin_sum[0],lonf,latf)
            
        else:
            
            ptval = proc.selpt_ds(tpin[0],lonf,latf)
            
            
        
        plotvar = ptval * dtmon
        plotvar = proc.xrdeseason(plotvar)
        plotvar = plotvar.groupby('time.month').var('time')
        
        if vv <2:
            lab = plotnames_sum[vv]
            ls  = "solid"
        else:
            lab = "Geostrophic Transport"
            ls  = 'dotted'
        
        ax.plot(mons3,plotvar,label=lab,lw=2.5,ls=ls)
        
    if pt == 0:
        ax.set_ylabel("Interannual Variance ($\degree$C/month)",fontsize=fsz_axis)
    ax.legend()

#%%

