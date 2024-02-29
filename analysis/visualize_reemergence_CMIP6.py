#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Re-emergence in CMIP6

Created on Fri Feb  2 13:51:13 2024

@author: gliu

"""



import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob

import matplotlib as mpl

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx


#%%

datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CMIP6/proc/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240215/"
proc.makedir(figpath)


#%% Load ACFs

e       = 0
varname = "SSS"
nc      = "%sCESM2_%s_ACF_1850to2014_lag00to60_ens%02i.nc" % (datpath,varname,e+1)

ds  = xr.open_dataset(nc).isel(thres=2)[varname]
acf = ds.values # [lon x lat x month x lag]
lon = ds.lon.values
lat = ds.lat.values

T2 = np.sum(acf**2,3)

# DOESN'T SEEM TO BE WORKING?
# remidx_all = []
# for  kmonth in range(12):
#     reidx = proc.calc_remidx_simple(acf,kmonth,monthdim=-2,lagdim=-1)
#     remidx_all.append(reidx)
# remidx_all = np.array(remidx_all)

#%%
kmonth = 1

for kmonth in range(12):
    mons3 = proc.get_monstr(nletters=3)
    levels = np.arange(0,21,1)
    
    mpl.rcParams['font.family'] = 'JetBrains Mono'
    bboxplot                    = [-80,0,0,65]
    fig,ax                      = viz.geosubplots(1,1,figsize=(10,6))
    ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
    
    pcm = ax.contourf(lon,lat,T2[:,:,kmonth].T,cmap='cmo.dense',levels=levels)
    cl = ax.contour(lon,lat,T2[:,:,kmonth].T,colors='k',linewidths=0.8,linestyles='dotted',levels=levels)
    ax.set_title("CESM2 Salinity %s $T^2$ (Ens %02i) " % (mons3[kmonth],e+1))
    fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)
    
    savename = "%sCESM2_%s_T2_mon%02i_ens%02i.png" % (figpath,varname,kmonth+1,e+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Try for CESM1

cesm1ncs  = ['HTR-FULL_SST_autocorrelation_thres0.nc','HTR-FULL_SSS_autocorrelation_thres0.nc']
cesm1path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
vnames = ["SST","SSS"]

ds_opn = [xr.open_dataset(cesm1path+nc) for nc in cesm1ncs] # [thres x ens x lag x mon x lat x lon]
ds_opn = [ds.isel(thres=2) for ds in ds_opn]

ds  = ds_opn[0]
lon = ds.lon.values
lat = ds.lat.values
lag = ds.lag.values
ens = ds.ens.values
mon = ds.mon.values 

#%% Plot for a variable

ds = ds_opn[0]
vv = 0
vname = vnames[vv]
acf      = ds_opn[vv][vname].values # [Ens Lag Mon Lat Lon]
T2_cesm1 = np.sum(acf**2,1) 

#%% Additional Plotting settings
bboxplot     = [-80,0,0,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3 = proc.get_monstr(nletters=3)

fsz_tick=18
fsz_axis=20
fsz_title=24
#%% Plot T2 for each month (default style)

e      = 0
kmonth = 0

for kmonth in range(12):
    
    levels = np.arange(0,21,1)
    
    #mpl.rcParams['font.family'] = 'JetBrains Mono'
    #bboxplot                    = [-80,0,0,65]
    fig,ax                      = viz.geosubplots(1,1,figsize=(10,6))
    ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
    plotvar = T2_cesm1[e,kmonth,:,:]
    pcm = ax.contourf(lon,lat,plotvar,cmap='cmo.dense',levels=levels)
    cl = ax.contour(lon,lat,plotvar,colors='k',linewidths=0.8,linestyles='dotted',levels=levels)
    ax.set_title("CESM1 %s %s $T^2$ (Ens %02i) " % (vname,mons3[kmonth],e+1))
    fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)
    
    savename = "%sCESM1_%s_T2_mon%02i_ens%02i.png" % (figpath,varname,kmonth+1,e+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Fancier Plot of above, for single month and ensemble member

kmonth = 0
e      = 0
varname=vname
lc = 'midnightblue'

levels = np.arange(0,21,1)

# Initialize Plot
fig,ax,mdict = viz.init_orthomap(1,1,bboxplot,figsize=(10,8.5),constrained_layout=True,)
ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")

# PLot it
plotvar = T2_cesm1[e,kmonth,:,:]

pcm = ax.contourf(lon,lat,plotvar,cmap='cmo.dense',levels=levels,transform=mdict['noProj'])
cl = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'])
ax.clabel(cl,levels[::2],fontsize=fsz_tick)
ax.set_title("CESM1 %s %s T$^2$ (Ens %02i) " % (vname,mons3[kmonth],e+1),fontsize=26)
cb = fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)#size=16)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Timescale (Month)",fontsize=fsz_axis)
savename = "%sCESM1_%s_T2_mon%02i_ens%02i.png" % (figpath,varname,kmonth+1,e+1)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#%% Compute the re-emmergence Index


rhocrit = proc.ttest_rho(0.05,1,86)

vv       = 1
vname    = vnames[vv]
acf      = ds_opn[vv][vname].values # [Ens Lag Mon Lat Lon]

norm_rem = False
sigmask  = False

plotbbox = True

if sigmask:
    rempeaks  = mmcorr[1,:,:,:,:].mean(1) # [year x lat x lon]
    sigmaskin = rempeaks > rhocrit

if vv == 0:
    cmapin='cmo.dense'
else:
    cmapin='cmo.deep'

remidxs = []
for kmonth in range(12):
    print(kmonth)
    
    mmcorr = proc.calc_remidx_simple(acf,kmonth,monthdim=2,lagdim=1) # [min/max,yr,mon,lat,lon]
    remidx = mmcorr[1,...] - mmcorr[0,...] #  (3, 42, 69, 65)
    if norm_rem:
        remidx = remidx/mmcorr[1,...]
    remidxs.append(remidx)
    
    
    
    #%Plot 3-year re-emergence, 1 ensemble Avg
    #e = 3
    bbplot2 = [-80,0,15,65]
    levels  = np.arange(0,0.55,0.05)
    plevels = np.arange(0,0.6,0.1)
    
    fig,axs,mdict = viz.init_orthomap(1,3,bbplot2,figsize=(16,8),constrained_layout=True,centlat=45)
    
    for yy in range(3):
        
        ax = axs[yy]
        blb = viz.init_blabels()
        if yy !=0:
            blb['left']=False
        else:
            blb['left']=True
        blb['lower']=True
        ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                        fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        plotvar = remidx[yy,:,:,:].mean(0)
        
        smsk    = sigmaskin[yy,:,:]
        
        pcm = ax.contourf(lon,lat,plotvar,cmap=cmapin,levels=levels,transform=mdict['noProj'],extend='both')
        cl = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'])
        
        if sigmask:
            mskk = viz.plot_mask(lon,lat,smsk.T,ax=ax,proj=mdict['noProj'],geoaxes=True,
                                 markersize=.75)
        
        #ax.clabel(cl,plevels,fontsize=fsz_tick-8)
        ax.set_title("Year %i" % (yy+1),fontsize=fsz_title)
        
        if plotbbox:
            if vv == 0:
                bbsel = [-45,-25,50,60]
                bcol  = "orange"
            elif vv == 1:
                bbsel = [-70,-55,30,40]
                bcol  = "midnightblue"
            if yy == 0:
                zz = viz.plot_box(bbsel,ax=ax,color=bcol,linewidth=2.5)
        
    
    cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.0105,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("%s Re-emergence Index" % vname,fontsize=fsz_axis)
    
    savename = "%sCESM1_%s_RemIdx_normrem%0i_mon%02i_EnsAvg.png" % (figpath,vname,norm_rem,kmonth+1)
    plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)

#%% Recompute Mean Wintertime T2


T2_cesm_all = []
for vv in range(2):
    vname    = vnames[vv]
    acf      = ds_opn[vv][vname].values # [Ens Lag Mon Lat Lon]
    T2       = proc.calc_T2(acf,axis=1)
    T2_cesm_all.append(T2)
    
T2_cesm_all = np.array(T2_cesm_all)
T2_cesm_all[T2_cesm_all == 1.] = np.nan

#%% Plot Ensemble Mean Wintertime T2


bbplot3 = [-80,0,15,65]
#djf_T2  = np.nanmean(T2_cesm1,(0,1))
levels  = np.arange(0,21,1)
plevels = levels[::3]

exlevels = np.arange(21,28,1)
vv = 1

vname    = vnames[vv]

fig,ax,mdict = viz.init_orthomap(1,1,bbplot3,figsize=(10,8.5),constrained_layout=True,centlat=45)

blb = viz.init_blabels()
if yy !=0:
    blb['left']=False
else:
    blb['left']=True
blb['lower']=True

ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")

plotvar = np.mean(T2_cesm_all[vv,:,[11,0,1],:,:],(0,1))

pcm = ax.contourf(lon,lat,plotvar,cmap='cmo.dense',levels=levels,transform=mdict['noProj'],extend='both')


cl = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'])
ax.clabel(cl,plevels,fontsize=fsz_tick)

# Add extended ticks
cl2 = ax.contour(lon,lat,plotvar,colors='w',linewidths=.5,linestyles='solid',levels=exlevels,transform=mdict['noProj'])
ax.clabel(cl2,exlevels[::2],fontsize=fsz_tick)

cb = fig.colorbar(pcm,ax=ax,fraction=0.045,pad=-0.1000,orientation='horizontal')
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("%s Persistence Timescale (T$^2$) [months]" % (vname),fontsize=fsz_axis)



savename = "%sCESM1_%s_T2_DJF_EnsAvg.png" % (figpath,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#%% Load ACFs from HadISST

hadpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
hadnpz  = np.load(hadpath+"HadISST_SST_autocorrelation_thres0.npz",allow_pickle=True)

hlon  = hadnpz['lon']
hlat  = hadnpz['lat']
hacs  = hadnpz['acs'][:,:,:,-1,:] # [80 x 65 x 12 x 3 (thresh?) x 37]
hlags = hadnpz['lags']


hT2 = proc.calc_T2(hacs,axis=-1)
hT2[hT2==1.] = np.nan

# Plot it

fig,ax,mdict = viz.init_orthomap(1,1,bbplot3,figsize=(10,8.5),constrained_layout=True,centlat=45)

blb = viz.init_blabels()
if yy !=0:
    blb['left']=False
else:
    blb['left']=True
blb['lower']=True

ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")

plotvar = hT2[:,:,[0,1,11]].mean(-1).T

pcm = ax.contourf(hlon,hlat,plotvar,cmap='cmo.dense',levels=levels,transform=mdict['noProj'],extend='both')


cl = ax.contour(hlon,hlat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'])
ax.clabel(cl,plevels,fontsize=fsz_tick)

# Add extended ticks
cl2 = ax.contour(hlon,hlat,plotvar,colors='w',linewidths=.5,linestyles='solid',levels=exlevels,transform=mdict['noProj'])
ax.clabel(cl2,exlevels[::2],fontsize=fsz_tick)

cb = fig.colorbar(pcm,ax=ax,fraction=0.045,pad=-0.1000,orientation='horizontal')
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("%s Persistence Timescale (T$^2$) [months]" % ("SST"),fontsize=fsz_axis)



savename = "%sHadiSST_%s_T2_DJF_EnsAvg.png" % (figpath,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#%% PUT INTO A NETCDF (HadISST)


hcoords = {
    
    'lon'  :hlon,
    'lat'  :hlat,
    'mon'  :np.arange(1,13,1),
    'thres': ["-","+","ALL"],
    'lag'  : ds_opn[0].lag.values,
    
    }
da_had = xr.DataArray(hadnpz['acs'],coords=hcoords,dims=hcoords,name="SST")
edict  = {'SST' : {'zlib':True}}
savename = "%sHadISST_SST_autocorrelation_thres0.nc" % hadpath
da_had.to_netcdf(savename,encoding=edict)

#%% Explore The ACF over different regions



#%% Examine Integrations of the entraining stochastic model



#%% First Select A region

kmonth = 1
remidx_plot = remidxs[kmonth][0,:,:,:].mean(0)




bbplot3 = [-80,0,15,65]
levels  = np.arange(0,0.55,0.05)
plevels = np.arange(0,0.6,0.1)

exlevels = np.arange(21,28,1)
vname    = vnames[0]

fig,ax,mdict = viz.init_orthomap(1,1,bbplot3,figsize=(10,8.5),constrained_layout=True,centlat=45)

blb = viz.init_blabels()
if yy !=0:
    blb['left']=False
else:
    blb['left']=True
blb['lower']=True


ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")

plotvar = remidx_plot

pcm = ax.contourf(lon,lat,plotvar,cmap='cmo.dense',levels=levels,transform=mdict['noProj'],extend='both')


cl = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'])
ax.clabel(cl,plevels,fontsize=fsz_tick)

# Add extended ticks
cl2 = ax.contour(lon,lat,plotvar,colors='w',linewidths=.5,linestyles='solid',levels=exlevels,transform=mdict['noProj'])
ax.clabel(cl2,exlevels[::2],fontsize=fsz_tick)

cb = fig.colorbar(pcm,ax=ax,fraction=0.045,pad=-0.1000,orientation='horizontal')
cb.ax.tick_params(labelsize=fsz_tick)
#cb.set_label("%s Persistence Timescale (T$^2$) [months]" % (vname),fontsize=fsz_axis)


selreg = [-45,-25,50,60]
viz.plot_box(selreg,ax=ax,proj=mdict['noProj'],linewidth=2.5,color="k",linestyle="solid")

savename = "%sCESM1_%s_REIDX_Box_mon%02i_EnsAvg.png" % (figpath,vname,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#%% Plot ACFs for each variable for the given Region
# It would be interesting to be able to control the line color such that it matches the contours

ds_in_cesm = [proc.sel_region_xr(ds.isel(mon=kmonth),selreg).mean('ens') for ds in ds_opn]
ds_in_had  = proc.sel_region_xr(da_had.isel(mon=kmonth,thres=-1,),selreg)



lonregc = ds_in_cesm[0].lon.values
latregc = ds_in_cesm[0].lat.values

lonregh = ds_in_had.lon.values
latregh = ds_in_had.lat.values


nlag,nlat,nlon= ds_in_cesm[0].SST.shape
nlonh,nlath,nlagh = ds_in_had.shape 

npts_cesm = nlat*nlon
npts_had = nlath*nlonh

#%% Get Colormap from cntour

# Get Colormap
cmap_rem  = pcm.get_cmap()
remcolors = cmap_rem(levels)

inarr   = levels
targval = inputval
idx = (np.abs(inarr - targval)).argmin()

#proc.find_nearest()



#%% 

# Make the Plot
xtksl   = np.arange(0,37,3)
lags    = np.arange(37)
fig,axs = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))
ax,_    = viz.init_acplot(kmonth,xtksl,lags)

# Plot CESM
acf_allpt_cesm = []
for o in tqdm(lonregc):
    #klon = list(lon).index(o)
    for a in latregc:
        
        # Plot SST for a point
        sst_in = ds_in_cesm[0].SST.sel(lon=o,lat=a,method='nearest')
        
        # Retrieve value (for SST)
        klon,klat = proc.find_latlon(o,a,lon,lat,verbose=False) # Note Lon lat is from other calculation
        inputval  = remidx_plot[klat,klon]
        
        # Now find the nearest contour level and corresponding color
        inarr   = levels
        targval = inputval
        idx     = (np.abs(inarr - targval)).argmin() + 2
        if idx > len(levels)-1:
            idx = len(levels)-1
        
        valcol  = remcolors[idx] # Value is 2 more than the current one, I guess counting for the extend? and values indicate upper bound?
        ax.plot(lags,sst_in,c=valcol,alpha=.2,lw=2,label="")
        
        acf_allpt_cesm.append(sst_in.values)

acf_allpt_cesm = np.array(acf_allpt_cesm)
ax.plot(lags,np.nanmean(acf_allpt_cesm,0),label="CESM (Region Average ACF)",color='midnightblue',lw=2,zorder=10)

acf_allpt_had = []
# Plot HadISST
for o in tqdm(lonregh):
    for a in latregh:
        # Plot SST for a point
        sst_in = ds_in_had.sel(lon=o,lat=a,method='nearest')
        
        # Retrieve value (for SST)
        #klon,klat = proc.find_latlon(o,a,lon,lat,verbose=False) # Note Lon lat is from other calculation
        #inputval  = remidx_plot[klat,klon]
        ax.plot(lags,sst_in,c="gray",alpha=.2,lw=2,label="")
        
        acf_allpt_had.append(sst_in.values)
        
acf_allpt_had = np.array(acf_allpt_had)

ax.plot(lags,acf_allpt_had.mean(0),label="HadISST (Region Average ACF)",color='k',lw=2,zorder=10)

#%%

# Debugging Purposes

ac        = acf
kmonth    = 11
monthdim  = 2
lagdim    = 1


minref   =6 
maxref   =12
tolerance=3
debug    =True

levels = np.arange(0,0.75,0.05)

# Initialize Plot
fig,ax,mdict = viz.init_orthomap(1,1,bboxplot,figsize=(10,8.5),constrained_layout=True,)
ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")

# PLot it
plotvar = T2_cesm1[e,kmonth,:,:]

pcm = ax.contourf(lon,lat,plotvar,cmap='cmo.dense',levels=levels,transform=mdict['noProj'])
cl = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'])
ax.clabel(cl,levels[::2],fontsize=fsz_tick)
ax.set_title("CESM1 %s %s T$^2$ (Ens %02i) " % (vname,mons3[kmonth],e+1),fontsize=26)
cb = fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)#size=16)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Persistence Timescale (T$^2$, months)",fontsize=fsz_axis)
savename = "%sCESM1_%s_T2_mon%02i_ens%02i.png" % (figpath,varname,kmonth+1,e+1)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)





#%% C


