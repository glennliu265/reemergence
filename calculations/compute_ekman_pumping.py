#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Using continuity (from Buckley et al. 2014 and 2015 papers), compute the vertical Ekman velocities

Created on Thu Mar 20 13:49:07 2025

@author: gliu
"""


import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
import xarray as xr
import time
from   tqdm import tqdm
import matplotlib as mpl

#%% Import modules

stormtrack    = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20250402"
   
    lipath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    rawpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    # Path of model input
    outpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
    outpathdat  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    
elif stormtrack == 1:
    #datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    #rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
    datpath     = rawpath
    outpathdat  = datpath + '/proc/'
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    figpath     = "/home/glliu/02_Figures/00_Scrap/"
    
    outpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/forcing/"

from amv import proc,viz
import scm
import tbx

proc.makedir(figpath)

#%% User Edits

# Load Ekman Advection Files
output_path_uek     = outpathdat 
savename_uek        = "%sCESM1LE_uek_NAtl_19200101_20050101_bilinear.nc" % (output_path_uek)
ds                  = xr.open_dataset(savename_uek).load()

vek                 = ds.v_ek
uek                 = ds.u_ek

# Wind Stress Information (for reference)
tauxnc              = "CESM1LE_TAUX_NAtl_19200101_20050101_bilinear.nc"
tauync              = "CESM1LE_TAUY_NAtl_19200101_20050101_bilinear.nc"
dstaux                = xr.open_dataset(output_path_uek + tauxnc).load() # (ensemble: 42, time: 1032, lat: 96, lon: 89)
dstauy                = xr.open_dataset(output_path_uek + tauync).load()

# Convert stress from stress on OCN on ATM --> ATM on OCN
taux                = dstaux.TAUX * -1
tauy                = dstauy.TAUY * -1

# Load Mixed-Layer gradients
vnames              = ["TEMP","SALT"]
path3d              = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
#salt_mld_nc         =  "CESM1_HTR_SALT_MLD_Gradient.nc"
mld_dz_nc           = path3d + "CESM1_HTR_%s_MLD_Gradient.nc"
ds_3d_all           = [xr.open_dataset(mld_dz_nc % vnames[vv]).load() for vv in range(2)]#xr.open_dataset(path3d + salt_mld_nc)


# Load in Mixed-;layer depths
ncmld2 = output_path_uek + "CESM1LE_HMXL_NAtl_19200101_20050101_bilinear.nc"
dsh_full = xr.open_dataset(ncmld2).HMXL.load()

hanom  = proc.xrdeseason(dsh_full) / 100 # Convert to meters

# Get Time Mean Values
uek_mean    = uek.mean('time')
vek_mean    = vek.mean('time')
taux_mean   = taux.mean('time')
tauy_mean   = tauy.mean('time')


# Load Mixed-Layer values

#%% Compute Vertical Pumping

wek         = uek + vek
wek_mean    = wek.mean('time')

#%% Plot Settings

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28
proj                        = ccrs.PlateCarree()


bbplot_nnat = [-80,0,20,65]



#%% Do a quick visualization just for reference

qint_ek  = 1
qint_tau = 2
iens     = 0
tauscale = 2
ekscale  = .005

bboxplot = [-80,0,10,70]
# Initialize Plot and Map
fig,ax = viz.init_regplot(bboxin=bboxplot)


plotvar = wek_mean.isel(ens=iens) * -1
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                    cmap='cmo.balance',vmin=-0.0002,vmax=0.0002,
                    transform=proj)
cb  = viz.hcbar(pcm,ax=ax,fraction=0.045,rotation=45)

# First, plot the Vek and Uek
qint    = qint_ek
plotu   = uek_mean.isel(ens=iens) * -1
plotv   = vek_mean.isel(ens=iens) * -1
umod    = np.sqrt(plotu**2 + plotv **2)
lon     = plotu.lon.data
lat     = plotu.lat.data
# qv      = ax.quiver(lon[::qint],lat[::qint],
#                     plotu.data[::qint,::qint],plotv.data[::qint,::qint],
#                     scale=ekscale,transform=proj,color='darkblue')
qv      = ax.streamplot(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                    color=umod.data[::qint,::qint],
                    density=2,
                    transform=proj)



# Colors are wek


#qk = ax.quiverkey(qv,.0,1,0.1,r"0.1 $\frac{m}{s}$",fontproperties=dict(size=10))


# # Plot Tau
qint    = qint_tau
plotu   = taux_mean.isel(ensemble=iens)
plotv   = tauy_mean.isel(ensemble=iens)
lon     = plotu.lon.data
lat     = plotu.lat.data
qv      = ax.quiver(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                    scale=tauscale,transform=proj,color='gray')

#qk = ax.quiverkey(qv,.0,1,0.1,r"0.1 $\frac{m}{s}$",fontproperties=dict(size=10))

#fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,6.5))
#x          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)


#%% Load in Detrain gradients (from compute_dz_ens)

vnames = ['TEMP','SALT']
path3d  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"


ds3d = []
for vv in range(2):
    vname   = vnames[vv]
    nc3d    = "%sCESM1_HTR_%s_MLD_Gradient.nc" % (path3d,vname)
    ds3d.append(xr.open_dataset(nc3d).load())
    

#%% Plot the maximum gradient at depth



# Plot SST
vv      = 0
fig,ax  = viz.init_regplot(bboxin=bbplot_nnat)

if vv == 0:
    vunit = "$\degree$C"
    cmap  = "cmo.thermal"
    vmax  = 0.02
    
elif vv == 1:
    vunit = "psu"
    cmap  = "cmo.haline"
    vmax  = 0.0020

# Plot the Variable
vname   = vnames[vv]
plotvar = np.abs(ds3d[vv][vname].mean('ens')).min('mon')
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                    transform=proj,cmap=cmap,
                    vmin=0,vmax=vmax)

cb      = viz.hcbar(pcm,fontsize=16)
cb.set_label("Max Absolute Gradient at Mixed-Layer Base [%s]" % vunit,
             fontsize=fsz_axis)


# Contour Ekman Pumping velocities
cints   = np.arange(-0.0002,0.00021,0.00002) * 10000
plotvar = wek_mean.mean('ens') * 10000
cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                     levels=cints,
                    transform=proj,linewidth=.75,colors="lightgray")
ax.clabel(cl,fontsize=fsz_tick)


#%% Examine Mean and Anomalous W_ek


cints_wek_std = np.arange(0,25)

fig,ax  = viz.init_regplot(bboxin=bboxplot)

plotvar = wek.std('time').mean('ens') * 1e5
pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints_wek_std,
                    transform=proj,cmap=cmap)

cb      = viz.hcbar(pcm,fontsize=16)
cb.set_label(r"$\sigma(w'_{ek})$ [$\frac{m}{s} \, \times \, 10^{-5}$]",
             fontsize=fsz_axis)

ax.set_title("Anomalous (Color) and Mean (Contours)\nVertical Ekman Velocities",fontsize=fsz_axis)


# Contour Ekman Pumping velocities
cints   = np.arange(-0.0002,0.00021,0.00002) * 10000
plotvar = wek_mean.mean('ens') * 10000
cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                     levels=cints,
                    transform=proj,linewidth=.55,colors="lightgray")
ax.clabel(cl,fontsize=fsz_tick)

#%% Calculate w_ek from anomalous wind stress

# Get rho and f
omega       = 7.2921e-5 # rad/sec
rho         = 1026      # kg/m3
lat         = dstaux.lat
lon         = dstaux.lon
xx,yy       = np.meshgrid(lon,lat) 
f           = 2*omega*np.sin(np.radians(yy))
dividef     = 1/f 
dividef[np.abs(yy)<=6] = np.nan # Remove large values around equator''
llcoords    =  {'lat':lat,'lon':lon,} 
da_dividef  = xr.DataArray(dividef,coords=llcoords,dims=llcoords)

# Calculate Anomalous Tau [N/m2]
tauxa   = proc.xrdeseason(taux)
tauya   = proc.xrdeseason(tauy)

# Compute the gradients
dx2,dy2 = proc.calc_dx_dy(lon.data,lat.data,centered=True)

# Compute the curl
dtaux_dy = tauxa.differentiate('lat') / dy2
dtauy_dx = tauya.differentiate('lon') / dx2

curltau  = dtauy_dx - dtaux_dy

# Calculate wek'
weka  = (curltau) / rho * da_dividef

#%% Debug Section (can delete later)

# # Check differencing quickly
# lonf        = -30
# latf        = 50
# testpt      = tauy.isel(lat=33,ensemble=0,time=0)#proc.selpt_ds(tauy,lonf,latf).isel(ensemble=0)
# testdiff    = testpt.differentiate('lon')
# testdiff    = testpt.diff('lon')
# print((testpt[2]-testpt[0])/2)
# print((testpt[2]-2*testpt[1]+testpt[0])/1)
# #testdiff    = testpt.differentiate('time')
# #testdiff2   = testpt.diff('time',n=1)

#%% -- <|S|> --- Save Anomalous Ekman Velocities

outpath_anom = rawpath
savename = "%sCESM1_HTR_FULL_Weka_curlcalc.nc" % outpath_anom
weka     = weka.rename('weka')
edict    = proc.make_encoding_dict(weka)
weka.to_netcdf(savename,encoding=edict)

#%% Compute dh'/dt

# Note, this differentiation automatically takes into account datetime unites
# and is converted to seconds.. [m/s]
dtmon     = 60*60*24*30
dhprime_dt = hanom.differentiate('time') #* dtmon

# # Check differencing quickly
# lonf     = -30
# latf     = 50
# testpt   = proc.selpt_ds(hanom,lonf,latf).isel(ensemble=0)
# testdiff = testpt.differentiate('time')
# testdiff2 = testpt.diff('time',n=1)

#%% --- <0> ---- Make a plot of Ekman Advection

itime     = 0
iens      = 0



vlm_weka  = 5e-6

cints_curltau = np.arange(-2,2.1,.1) * 1e-6

fsz_ticks = 16

fig,ax    = viz.init_regplot(bboxin=bbplot_nnat)

# Plot Wek
plotvar   = weka.isel(ensemble=iens,time=itime) #* dtmon
pcm       = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                        vmin=vlm_weka*-1,vmax=vlm_weka,
                        cmap='cmo.balance')
cb        = viz.hcbar(pcm,ax=ax,fontsize=fsz_ticks)
cb.set_label(r"$w_{ek}'$ $[\frac{m}{s}]$",fontsize=fsz_axis)

# Plot Wind Stress Vectors
qint     = 2
tauscale = 2
plotu   = tauxa.isel(ensemble=iens,time=itime)
plotv   = tauya.isel(ensemble=iens,time=itime)
lon     = plotu.lon.data
lat     = plotu.lat.data
qv      = ax.quiver(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                    scale=tauscale,transform=proj,color='gray',zorder=1)

# # # Contour Wind Stress Curl
# plotvar = curltau.isel(ensemble=iens,time=itime)# * dtmon
# cl      = ax.contour(lon,lat,plotvar,transform=proj,colors='k',levels=cints_curltau)
# ax.clabel(cl,fontsize=fsz_ticks)

title = "Ekman Vertical Velocities\nEns %02i, Time=%s" % (iens+1,str(plotvar.time.data)[:10])
ax.set_title(title,fontsize=fsz_title)

figname = "%s/Ekman_Vertical_Velocity_Ens%0i_t%04i.png" % (figpath,iens+1,itime)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% --- <0> ---- Make a plot of MLD variability

fig,ax    = viz.init_regplot(bboxin=bbplot_nnat)


vlm_dh    = 1e-4

# Plot dprime_dt
plotvar   = dhprime_dt.isel(ensemble=iens,time=itime)#/dtmon #* dtmon
pcm       = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                        vmin=vlm_dh*-1,vmax=vlm_dh,
                        cmap='cmo.balance')
cb        = viz.hcbar(pcm,ax=ax,fontsize=fsz_ticks,rotation=45)
cb.set_label(r"$\frac{\partial h'}{\partial t}$ $[\frac{m}{sec}]$",fontsize=fsz_axis)

# Plot Wind Stress Vectors
qint     = 2
tauscale = 2
plotu   = tauxa.isel(ensemble=iens,time=itime)
plotv   = tauya.isel(ensemble=iens,time=itime)
lon     = plotu.lon.data
lat     = plotu.lat.data
qv      = ax.quiver(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                    scale=tauscale,transform=proj,color='gray',zorder=1)

title = "Anomalous MLD Tendency\nEns %02i, Time=%s" % (iens+1,str(plotvar.time.data)[:10])
ax.set_title(title,fontsize=fsz_title)

figname = "%s/Anomalous_MLD_Tendency_Ens%0i_t%04i.png" % (figpath,iens+1,itime)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% --- <0> ---- Make a plot of Gradients of TEMP

imon      = dhprime_dt.isel(ensemble=iens,time=itime).month.data.item()-1

vlm_sst   = 0.1



fig,ax    = viz.init_regplot(bboxin=bbplot_nnat)

# Plot TEMP
plotvar   = ds3d[0].isel(mon=imon,ens=iens).TEMP #* -1
pcm       = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        vmin=vlm_sst*-1,vmax=vlm_sst,
                        cmap='cmo.balance')

# Plot SALT
# plotvar   = ds3d[1].isel(mon=imon,ens=iens).SALT
# cl       = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#                         levels=cints_sss,
#                         colors='k')
# ax.clabel(cl,fontsize=fsz_ticks)

cb        = viz.hcbar(pcm,ax=ax,fontsize=fsz_ticks,rotation=45)
cb.set_label(r"$\overline{T - T_b}$ [$\degree$C]",fontsize=fsz_axis)


title = "Gradients at Mixed-Layer Base\nEns %02i, Month=%s" % (iens+1,imon+1)
ax.set_title(title,fontsize=fsz_title)

figname = "%s/TEMP_SST_MLD_BASE_Ens%0i_t%04i.png" % (figpath,iens+1,itime)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% --- <0> ---- Make a plot of Gradients of SALT


cints_sss = np.arange(-0.05,0.055,0.005)
vlm_sss   = 0.025

fig,ax    = viz.init_regplot(bboxin=bbplot_nnat)

# Plot SALT
plotvar   = ds3d[1].isel(mon=imon,ens=iens).SALT
pcm       = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        vmin=vlm_sss*-1,vmax=vlm_sss,
                        cmap='cmo.balance')

# Plot SALT
# plotvar   = ds3d[1].isel(mon=imon,ens=iens).SALT
# cl       = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#                         levels=cints_sss,
#                         colors='k')
# ax.clabel(cl,fontsize=fsz_ticks)

cb        = viz.hcbar(pcm,ax=ax,fontsize=fsz_ticks,rotation=45)
cb.set_label(r"$\overline{S - S_b}$ [$\degree$C]",fontsize=fsz_axis)


title = "Gradients at Mixed-Layer Base\nEns %02i, Month=%s" % (iens+1,imon+1)
ax.set_title(title,fontsize=fsz_title)

figname = "%s/TEMP_SSS_MLD_BASE_Ens%0i_t%04i.png" % (figpath,iens+1,itime)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Combine the terms

wek_temp = weka.groupby('time.month') * ds3d[0].TEMP.rename(dict(mon='month',ens='ensemble'))
wek_salt = weka.groupby('time.month') * ds3d[1].SALT.rename(dict(mon='month',ens='ensemble'))

dh_temp  = dhprime_dt.groupby('time.month') * ds3d[0].TEMP.rename(dict(mon='month',ens='ensemble'))
dh_salt  = dhprime_dt.groupby('time.month') * ds3d[1].SALT.rename(dict(mon='month',ens='ensemble'))

#%% Plot the standard deviation of the terms

vv = 1
 
if vv == 0:
    temp_stdev = []
    vunit      = "\degree C"
    vname      = "TEMP"
    cmap       = 'cmo.thermal'
else:
    salt_stdev = []
    vunit      = "psu"
    vname      = "SALT"
    cmap       = 'cmo.haline'
for ii in range(2):
    fig,ax    = viz.init_regplot(bboxin=bbplot_nnat)
    vname     = "SALT"
    
    # Wek Term
    if ii == 0:
        if vv == 0:
            plotvar = wek_temp.std('time').mean('ensemble') * dtmon
            vmax    = 0.2
        elif vv == 1:
            plotvar = wek_salt.std('time').mean('ensemble') * dtmon
            vmax    = 0.05
        
        iiname  = "Wek"
        clab    = r"$w_{ek}' \, \overline{%s - %s_b}$ [$%s \, \frac{m}{month}$]" % (vname[0],vname[0],vunit)
        
    
    # dh'dt Term
    elif ii == 1:
        if vv == 0:
            plotvar = dh_temp.std('time').mean('ensemble') * dtmon
            vmax    = 1.0
        elif vv == 1:
            plotvar = dh_salt.std('time').mean('ensemble') * dtmon
            vmax    = 0.25
        
        
        iiname  = "dhdt"
        clab    = r"$\frac{\partial h'}{\partial t} \, \overline{%s - %s_b}$ [$%s \, \frac{m}{month}$]" % (vname[0],vname[0],vunit)
       
    
    if vv == 0:
        temp_stdev.append(plotvar)
    else:
        salt_stdev.append(plotvar)
    
    pcm       = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                            vmin=0,vmax=vmax,
                            cmap=cmap)
    
    cb        = viz.hcbar(pcm,ax=ax,fontsize=fsz_ticks,rotation=45)
    cb.set_label(clab,fontsize=fsz_axis)
    
    figname = "%s/%s_Forcing_%s_EnsAvg_Stdev.png" % (figpath,vname,iiname)
    plt.savefig(figname,dpi=150,bbox_inches='tight')   
        
#%% Plot the TEMP/SALT Stdev ratio


vv    = 1

if vv == 0:
    vunit      = "\degree C"
    vname      = "TEMP"
    cmap       = 'cmo.thermal'
    plot_stdev = temp_stdev
    vmax       = 3
else:
    
    vunit      = "psu"
    vname      = "SALT"
    cmap       = 'cmo.haline'
    plot_stdev = salt_stdev
    vmax       = 4
    
cints           = np.log(np.array([0.25, 0.5, 1, 2, 4,8,10]))
clabs           = ["0.25x","0.5x","1x", "2x", "4x","8x","10x"]


fig,ax    = viz.init_regplot(bboxin=bbplot_nnat)

plotvar   = np.log(plot_stdev[1]/plot_stdev[0]) # dh'/wek

pcm       = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        vmin=-vmax,vmax=vmax,
                        cmap='cmo.balance')


cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        levels=cints,colors="k",linewidths=0.75)
fmt= {}
for l, s in zip(cl.levels, clabs):
    fmt[l] = s
    
cl = ax.clabel(cl,fmt=fmt,fontsize=fsz_ticks)
viz.add_fontborder(cl)




clab      = r"Log($\frac{\partial_t h'}{w_{ek}'}$)"
cb        = viz.hcbar(pcm,ax=ax,fontsize=fsz_ticks)
cb.set_label(clab,fontsize=fsz_axis)

title     = "Stdev. Log Ratio, %s Forcing" % vname
ax.set_title(title,fontsize=fsz_title)
figname = "%s/Forcing_%s_LogRatios_EnsAvg_Stdev.png" % (figpath,vname)
plt.savefig(figname,dpi=150,bbox_inches='tight') 


