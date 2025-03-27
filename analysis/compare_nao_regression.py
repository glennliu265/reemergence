#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Qek formulations

(1) Load Wind stress anomalies associated with NAO, use to advect mean gradients
(2) Load Qek anomalies associated with NAO
(3) Compare the amplitude and patterns of forcing

copied stuff from [preprocessing/calc_ekman_advection_htr]

Created on Fri Mar 21 14:17:01 2025

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
    #figpath     = projpath + "02_Figures/20250402/"
    figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20250402/"
    
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
import amv.loaders as dl
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

# Convert to 

# Get Time Mean Values
uek_mean    = uek.mean('time')
vek_mean    = vek.mean('time')
taux_mean   = taux.mean('time')
tauy_mean   = tauy.mean('time')

# Load Mean SST/SSS
ds_sstmean = dl.load_monmean('SST').SST




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

#%% 

# Data Pathway
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
forcepath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
mldpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"


# NAO-related wind stress
tau_eof_nc = "CESM1_HTR_FULL_Monthly_TAU_NAO_nomasklag1_nroll0.nc"


# This one is in unites of W/m2
st             = time.time()
qek_tau_nao_nc = "CESM1_HTR_FULL_Qek_SST_NAO_nomasklag1_nroll0_NAtl.nc"
ds_qek_taunao  = xr.open_dataset(forcepath + qek_tau_nao_nc).load()
print("Loaded in %.2fs" % (time.time()-st))

# Note, this seemed to be the wrong one
# units, it seems to be in degC/sec
#qek_tau_nao_nc = "CESM1_HTR_FULL_Qek_SST_Monthly_TAU_NAO.nc"
#ds_qek_taunao  = xr.open_dataset(datpath + qek_tau_nao_nc).load() # (mode, ens, mon, lat, lon)
#


# Qek Term (Computed by regressing Qek Anomalies), Probably [degC/mon]
qek_nc_sst      = "CESM1_HTR_FULL_Qek_SST_NAO_DirReg_NAtl_corrected_EnsAvg.nc"
ds_qek_dirreg   = xr.open_dataset(forcepath + qek_nc_sst).load()

# Load Climatological MLD
ds_mld  = xr.open_dataset(mldpath + "CESM1_HTR_FULL_HMXL_NAtl.nc").h

# Load NAO stuff
naonc  = "CESM1_HTR_FULL_Monthly_TAU_NAO_nomasklag1_nroll0.nc"
ds_nao = xr.open_dataset(datpath + naonc).load()
taux_nao = ds_nao.TAUX * -1
tauy_nao = ds_nao.TAUY * -1

# Load the ice mask


# Load Mean SST gradients


#%% Plot how it looks like
# Constants for conversion
dtmon   = 3600*24*30
rho     = 1026
cp0     = 3996

cints_sst = np.arange(270,305,2)
#proj    = ccrs.PlateCarree()

#fsz_tick


qint    = 2
bbplot  = [-80,0,10,65]

imode   = 0
iens    = 0
imon    = 0


fig,axs,_  = viz.init_orthomap(1,2,bbplot,figsize=(18,10))

nao_pats = []
for ii in range(2):
    
    ax = axs[ii]
    ax        = viz.add_coast_grid(ax,bbox=bbplot,fill_color='lightgray',fontsize=fsz_tick)
    if ii == 0:
        # Convert to [degC]/mon
        plotvar           = ds_qek_taunao.Qek.isel(mode = imode, mon = imon).mean('ens') #* dtmon
        hmon              = ds_mld.isel(mon=imon).mean('ens')
        conversion_factor = dtmon / (rho*cp0*hmon)
        plotvar           = plotvar * conversion_factor
        
        # Additional conversion for MLD (used cm instead of m, for incorrect file)
        #plotvar = plotvar * 100
        
        title   = "Regress Wind Stress to NAO"
        vlms    = 0.2 * np.array([-1,1])
        outname = "RegrTau"
        
    else:
        plotvar = ds_qek_dirreg.Qek.isel(mode = imode, mon = imon) * dtmon
        title   = "Regress $Q_{ek}$ to NAO"
        vlms    = 0.2 * np.array([-1,1])
        outname = "RegrQek"
     
    nao_pats.append(plotvar)
    # Plot the taus
    qvlab = r"$ \tau'_{NAO}$, [N m$^{-2}$]"
    plotu = taux_nao.isel(mon=imon,mode=imode).mean('ens')
    plotv = tauy_nao.isel(mon=imon,mode=imode).mean('ens')
    lon     = plotu.lon.data
    lat     = plotu.lat.data
    qv      = ax.quiver(lon[::qint],lat[::qint],
                        plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                        transform=proj,zorder=2,color='gray')
    ax.quiverkey(qv,.9,0.95,0.1,qvlab,fontproperties={'size':fsz_tick})
    
    # Plot the Monthly Means
    plotsst = ds_sstmean.isel(mon=imon).mean('ens')
    cl = ax.contour(plotsst.lon,plotsst.lat,plotsst,levels=cints_sst,
                    colors="darkred",transform=proj,linewidths=0.55)
        
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        transform=proj,vmin=vlms[0],vmax=vlms[1],cmap='cmo.balance')
    cb      = viz.hcbar(pcm,ax=ax,fontsize=fsz_tick,rotation=0,fraction=0.025)
    
    ax.set_title(title,fontsize=26)
    cb.set_label(r"$Q_{ek}$ Forcing [$\frac{\degree C}{mon}$]"+"\n EOF: %0i, Month: %02i" % (imode+1,imon+1),fontsize=fsz_axis)
    
savename = "%sNAO_EOF_Comparison_mon%02i_eof%02i.png" % (figpath,imon+1,imode+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Examine the differences

fig,ax = viz.init_regplot(bboxin=bbplot)

# Plot Taus
qvlab = r"$ \tau'_{NAO}$, [N m$^{-2}$]"
plotu = taux_nao.isel(mon=imon,mode=imode).mean('ens')
plotv = tauy_nao.isel(mon=imon,mode=imode).mean('ens')
lon     = plotu.lon.data
lat     = plotu.lat.data
qv      = ax.quiver(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                    transform=proj,zorder=2,color='gray')
#ax.quiverkey(qv,.9,0.95,0.1,qvlab,fontproperties={'size':fsz_tick-4})


# Plot the Monthly Means
plotsst = ds_sstmean.isel(mon=imon).mean('ens')
cl = ax.contour(plotsst.lon,plotsst.lat,plotsst,levels=cints_sst,
                colors="darkred",transform=proj,zorder=-1)


# Compute Differences
vlms    = [-0.001,0.001]
plotvar = nao_pats[1] - nao_pats[0]
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                    transform=proj,vmin=vlms[0],vmax=vlms[1],cmap='cmo.balance')
cb      = viz.hcbar(pcm,ax=ax,fontsize=fsz_tick,rotation=0,fraction=0.025)
cb.set_label(r"$Q_{ek}$ Forcing Difference [$\frac{\degree C}{mon}$]"+"\n EOF: %0i, Month: %02i" % (imode+1,imon+1),fontsize=fsz_axis)

ax.set_title("Differences (RegrQek - RegrTau)",fontsize=fsz_title-4)


savename = "%sNAO_EOF_Comparison_Differences_mon%02i_eof%02i.png" % (figpath,imon+1,imode+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Compute total forcing amplitude

regrtau_total = np.sqrt((ds_qek_taunao.Qek**2).sum('mode')).mean('ens')
regrqek_total = np.sqrt((ds_qek_dirreg.Qek**2).sum('mode'))#.mean('ens')

#%% Viauzlie Total forcing amploitude

for imon in range(12):
    
    fig,axs,_  = viz.init_orthomap(1,3,bbplot,figsize=(18,14))
    
    total_pat = []
    for ii in range(3):
        ax = axs[ii]
        ax        = viz.add_coast_grid(ax,bbox=bbplot,fill_color='lightgray',fontsize=fsz_tick)
        
        if ii == 0:
            plotvar           = regrtau_total.isel(mon=imon)
            hmon              = ds_mld.isel(mon=imon).mean('ens')
            conversion_factor = dtmon / (rho*cp0*hmon)
            plotvar           = plotvar * conversion_factor
            
            title             = "(1) Regress Wind Stress to NAO"
            vlms              = [0,0.3]
            cmap              = 'cmo.amp'
        elif ii == 1:
            plotvar           = regrqek_total.isel(mon=imon)  * dtmon
            title             = "(2) Regress $Q_{ek}$ to NAO"
            vlms              = [0,0.3]
            
        else:
            plotvar           = total_pat[1] - total_pat[0]
            vlms              = [-.2,.2]
            cmap              = 'cmo.balance'
            title             = "Difference, (2) - (1)"
            
        total_pat.append(plotvar)
        
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                            transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
        cb      = viz.hcbar(pcm,ax=ax,fontsize=fsz_tick,rotation=0,fraction=0.025)
        
        ax.set_title(title,fontsize=26)
        cb.set_label(r"$Q_{ek}$ Forcing [$\frac{\degree C}{mon}$]"+"\n All Modes, Month: %02i" % (imon+1),fontsize=fsz_axis)
        
    savename = "%sNAO_EOF_Comparison_Differences_mon%02i_eofALL.png" % (figpath,imon+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')    



#%% Plot Side by Side


#%% Let's redo the calculation






