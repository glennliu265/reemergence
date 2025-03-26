#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Qek formulations

(1) Load Wind stress anomalies associated with NAO, use to advect mean gradients
(2) Load Qek anomalies associated with NAO
(3) Compare the amplitude and patterns of forcing


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
    figpath     = projpath + "02_Figures/20240207"
   
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



# Convert to 


# Get Time Mean Values
uek_mean    = uek.mean('time')
vek_mean    = vek.mean('time')
taux_mean   = taux.mean('time')
tauy_mean   = tauy.mean('time')


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

# Qek (SST), same for SSS
st             = time.time()
qek_tau_nao_nc = "CESM1_HTR_FULL_Qek_SST_Monthly_TAU_NAO.nc"
ds_qek_taunao  = xr.open_dataset(datpath + qek_tau_nao_nc).load() # (mode, ens, mon, lat, lon)
print("Loaded in %.2fs" % (time.time()-st))

# Qek Term (Computed by regressing Qek Anomalies), Probably [degC/mon]
qek_nc_sst      = "CESM1_HTR_FULL_Qek_SST_NAO_DirReg_NAtl_corrected_EnsAvg.nc"
ds_qek_dirreg   = xr.open_dataset(forcepath + qek_nc_sst).load()

# Load Climatological MLD
ds_mld  = xr.open_dataset(mldpath + "CESM1_HTR_FULL_HMXL_NAtl.nc").h



#%% Plot how it looks like

dtmon   = 3600*24*30
imode   = 0
iens    = 0
imon    = 0
proj    = ccrs.PlateCarree()
bbplot  = [-80,0,10,65]
fig,ax  = viz.init_regplot(bboxin=bbplot)

rho     = 1026
cp0     = 3996

plotvar = ds_qek_taunao.Qek.isel(mode = imode, mon = imon).mean('ens') * dtmon
#plotvar = ds_qek_dirreg.Qek.isel(mode = imode, mon = imon) * dtmon
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        transform=proj,vmin=-0.075,vmax=0.075,cmap='cmo.balance')
cb      = viz.hcbar(pcm)


# %%






