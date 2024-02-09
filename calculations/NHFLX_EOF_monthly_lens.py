#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute EOF Forcing (NAO, EAP) for Fprime for a CESM1 LENS simulation
Also computes Fprime given a selected Damping.
Currently written to run on Astraeus.

# On 2024.02.08
# Copy of NHFLX_EOF_monthly from stochmod/preprocessing
# Copied Fprime calculation step from preproc_sm_inputs_SSS

Procedure:

For each ensemble member
(1) Load in Fprime (or recompute it as in preproc_sm_inputs_SSS)
(2) Perform EOF Analysis (for each month)
(3) Get regression pattern and regress it to the full timeseries

Created on Thu Feb  8 16:43:44 2024
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

#%% Locate Target File

stormtrack = 0

# Path to variables processed by prep_data_byvariable_monthly, Output will be saved to rawpath1
if stormtrack:
    rawpath1 = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
    dpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/damping/"
    mldpath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/mld/"
else:
    rawpath1 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    mldpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
    dpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"
# /Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1

ncstr1   = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc"

#%% Indicate some settings

# Fprime calulation settings
dampstr = None # Damping String  (see "load damping of choice")

# EOF parameters
bboxeof = [-80,20,0,65]
N_mode  = 200 # Maxmum mode will be adjusted to number of years...

#%% Part (1): Load Inputs for Fprime Computation

# Load TS, flux
varnames =['SST','qnet']
ds_load  =[xr.open_dataset(rawpath1+ ncstr1 % vn).load() for vn in varnames]

# Anomalize
ds_anom  = [proc.xrdeseason(ds) for ds in ds_load]

# Detrend
ds_dt    = [ds-ds.mean('ensemble') for ds in ds_anom] # [ens x time x lat x lon]

# Transpose to [mon x ens x lat x lon]
ds_dt    = [ds.transpose('time','ensemble','lat','lon') for ds in ds_dt]

#% Load damping of choice
if dampstr == "Expfitlbda123":
    convert_wm2=True
    hff_nc   = "CESM1_HTR_FULL_Expfit_lbda_damping_lagsfit123.nc"#"CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"
elif dampstr == None:
    convert_wm2=False
    hff_nc = "CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"
elif dampstr == "ExpfitSST123":
    convert_wm2=True
    hff_nc   = "CESM1_HTR_FULL_Expfit_SST_damping_lagsfit123.nc"#"CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"
hff_path     = dpath
dshff    = xr.open_dataset(hff_path + hff_nc) # [mon x ens x lat x lon']

# Load h
mldnc  = "%sCESM1_HTR_FULL_HMXL_NAtl.nc" % mldpath
ds_mld = xr.open_dataset(mldnc)

# Check sizes
if dampstr is not None: # Not sure why, but it seems that this cuts regions oddly...
    ds_list = ds_dt + [dshff,ds_mld]
    ds_rsz  = proc.resize_ds(ds_list)
    ds_dt = ds_rsz[:2]
    dshff = ds_rsz[2]
    ds_mld = ds_rsz[3]

# Convert if need
if convert_wm2:
    dt  = 3600*24*30
    cp0 = 3996
    rho = 1026
    dshff = dshff.damping * (rho*cp0*ds_mld.h) / dt  *-1 #need to do a check for - value!!
else:
    dshff= dshff.damping

# Output to numpy
hff = dshff.values
sst = ds_dt[0].SST.values
qnet = ds_dt[1].qnet.values

# Check sizes
# Get dimension and tile
ntime,nens,nlat,nlon=qnet.shape
ntimeh,nensh,nlath,nlonh=hff.shape

# Tile
nyrs = int(ntime/12)
hfftile=np.tile(hff.transpose(1,2,3,0),nyrs)
hfftile= hfftile.transpose(3,0,1,2)
# Check plt.pcolormesh(hfftile[0,0,:,:]-hfftile[12,0,:,:]),plt.colorbar(),plt.show()


#% Calculate F'
nroll    = 0
rollstr  = "nroll%0i"  % nroll
Fprime   = qnet + hfftile*np.roll(sst,nroll)

#%% Save Output


coords   = dict(time=ds_dt[0].time.values,ens=dshff.ens.values,lat=dshff.lat.values,lon=dshff.lon.values)
daf      = xr.DataArray(Fprime,coords=coords,dims=coords,name="Fprime")
savename = "%sCESM1_HTR_FULL_Fprime_timeseries_%s_%s_NAtl.nc" % (rawpath1,dampstr,rollstr)

edict    = {"Fprime":{'zlib':True}}
daf.to_netcdf(savename,encoding=edict)


#%% Part (2): Perform EOF Analysis on Fprime (copy from NHFLX_EOF_monthly)

flxa   = daf # [Time x Ens x Lat x Lon] # Anomalize variabless

# Apply area weight
wgt    = np.sqrt(np.cos(np.radians(daf.lat.values))) # [Lat]
flxwgt = flxa * wgt[None,None,:,None]

# Select Region
flxreg = proc.sel_region_xr(flxwgt,bboxeof)



flxout     = flxreg.values
ntime,nens,nlatr,nlonr = flxout.shape
npts       = nlatr*nlonr
nyr        = int(ntime/12)

# Repeat for full variable
flxout_full= flxa.values
_,_,nlat,nlon=flxout_full.shape
npts_full  = nlat*nlon

# Check to see if N_mode exceeds nyrs
if N_mode > nyrs:
    print("Requested N_mode exists the maximum number of years, adjusting....")
    N_mode=nyrs

# Preallocate for EOF Analysis
eofall    = np.zeros((N_mode,12,nens,nlat*nlon)) * np.nan
pcall     = np.zeros((N_mode,12,nens,nyr)) * np.nan
varexpall = np.zeros((N_mode,12,nens)) * np.nan
    
# Loop for ensemble memmber
for e in tqdm(range(nens)):
    
    # Remove NaN Points
    flxens            = flxout[:,e,:,:].reshape(ntime,npts) #  Time x Space
    okdata,knan,okpts = proc.find_nan(flxens,0)
    _,npts_valid = okdata.shape
    
    # Repeat for full data
    flxens_full       = flxout_full[:,e,:,:].reshape(ntime,npts_full)
    okdataf,knanf,okptsf = proc.find_nan(flxens_full,0)
    _,npts_validf = okdataf.shape
    
    # Reshape to [yr x mon x pts]
    okdatar  = okdata.reshape(nyr,12,npts_valid)
    okdatarf = okdataf.reshape(nyr,12,npts_validf)
    
    # Calculate EOF by month
    for im in range(12):
        
        # Compute EOF
        datain          = okdatar[:,im,:].T # --> [space x time]
        eofs,pcs,varexp = proc.eof_simple(datain,N_mode,1)
        
        # Standardize PCs
        pcstd = pcs / pcs.std(0)[None,:]
        
        # Regress back to dataset
        datainf = okdatarf[:,im,:].T
        eof,b = proc.regress_2d(pcstd.T,datainf.T) # [time x pts]
        
        
        # Save the data
        eofall[:,im,e,okptsf] = eof.copy()
        pcall[:,im,e,:] = pcs.T.copy()
        varexpall[:,im,e] = varexp.copy()

# Reshape the variable
eofall = eofall.reshape(N_mode,12,nens,nlat,nlon) # (86, 12, 42, 96, 89)

#%% Flip sign to match NAO+ (negative heat flux out of ocean/ -SLP over SPG)

spgbox     = [-60,20,40,80]
eapbox     = [-60,20,40,60] # Shift Box west for EAP

N_modeplot = 5

for N in tqdm(range(N_modeplot)):
    if N == 1:
        chkbox = eapbox # Shift coordinates west
    else:
        chkbox = spgbox
    for e in range(nens):
        for m in range(12):
            
            
            sumflx = proc.sel_region(eofall[N,[m],e,:,:].transpose(2,1,0),flxa.lon.values,flxa.lat.values,chkbox,reg_avg=True)
            #sumslp = proc.sel_region(eofslp[:,:,[m],N],lon,lat,chkbox,reg_avg=True)
            
            if sumflx > 0:
                print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
                eofall[N,m,e,:,:]*=-1
                pcall[N,m,e,:] *= -1


#%% Convert to Data Array

coordseof = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,43,1),lat=flxa.lat,lon=flxa.lon)
daeof     = xr.DataArray(eofall,coords=coordseof,dims=coordseof,name="eofs")

coordspc  = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,43,1),yr=np.arange(1920,2005+1))
dapcs     = xr.DataArray(pcall,coords=coordspc,dims=coordspc,name="pcs")

coordsvar = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,43,1))
davarexp  = xr.DataArray(varexpall,coords=coordsvar,dims=coordsvar,name="varexp")


ds_eof    = xr.merge([daeof,dapcs,davarexp])
edict_eof = proc.make_encoding_dict(ds_eof)

savename  = "%sEOF_Monthly_NAO_EAP_Fprime_%s_%s_NAtl.nc" % (rawpath1,dampstr,rollstr)

ds_eof.to_netcdf(savename,encoding=edict_eof)


#%% Visualize to check

bboxchk = [-40,10,40,65]
e       = 1
N       = 0
im      = 11

fig,ax,mdict = viz.init_orthomap(1,1,bboxeof)
ax = viz.add_coast_grid(ax,bbox=bboxeof)

pcm = ax.pcolormesh(flxa.lon,flxa.lat,eofall[N,im,e,:,:],transform=mdict['noProj'])
fig.colorbar(pcm,ax=ax)
ax.set_title("EOF %02i Month %02i Ens %02i" % (N+1,im+1,e+1))

ax = viz.plot_box(bboxchk,ax=ax)

 # [Time x Ens x Lat x Lon]

#%% Make innto data array

#%% Part (2): Perform EOF Analysis on Fprime (copy from NHFLX_EOF_monthly)

#% (**) Apply Area Weight (to region) ----------------------------------------------
# ~1m5s

wgt = np.sqrt(np.cos(np.radians(lat)))

#plt.plot(wgt)

flxwgt = flxa * wgt[None,:,None]
#slpwgt = slpa * wgt[None,:,None] # Don't apply area-weight to regressed variable

# Select region --------------------------------------------------------------
flxreg,lonr,latr = proc.sel_region(flxwgt.transpose(2,1,0),lon,lat,bboxeof)
nlonr,nlatr,_ = flxreg.shape
flxreg = flxreg.transpose(2,1,0) # Back to time x lat x lon

# Remove NaN Points [time x npts] --------------------------------------------
flxwgt = flxa.reshape((ntime,nlat*nlon)) # Dont use weighted variable
okdata,knan,okpts = proc.find_nan(flxwgt,0)
npts = okdata.shape[1]

flxreg = flxreg.reshape((ntime,nlatr*nlonr)) # Use lat weights for EOF region
okdatar,knanr,okptsr = proc.find_nan(flxreg,0)
nptsr = okdatar.shape[1]

nptsall = nlat*nlon
#slpwgt = slpwgt.reshape(ntime,nptsall) # Repeat for slp 
slpwgt = slpa.reshape(ntime,nptsall) # Repeat for slp 
okslp  = slpwgt#[:,okpts]

# Calculate Monthly Anomalies, change to [yr x mon x npts] -------------------
okdata = okdata.reshape((nyr,12,npts))
okdata = okdata - okdata.mean(0)[None,:,:]
okdatar = okdatar.reshape((nyr,12,nptsr)) # Repeat for region
okdatar = okdatar - okdatar.mean(0)[None,:,:]
okslp = okslp.reshape((nyr,12,nptsall))

# Prepare for eof anaylsis ---------------------------------------------------
eofall    = np.zeros((N_mode,12,nlat*nlon)) * np.nan
eofslp    = eofall.copy()
pcall     = np.zeros((N_mode,12,nyr)) * np.nan
varexpall = np.zeros((N_mode,12)) * np.nan
# Looping for each month
for m in tqdm(range(12)):
    
    # Calculate EOF
    datain = okdatar[:,m,:].T # [space x time]
    regrin = okdata[:,m,:].T
    slpin  = okslp[:,m,:].T
    
    eofs,pcs,varexp = proc.eof_simple(datain,N_mode,1)
    
    # Standardize PCs
    pcstd = pcs / pcs.std(0)[None,:]
    
    # Regress back to dataset
    eof,b = proc.regress_2d(pcstd.T,regrin.T)
    eof_s,_ = proc.regress_2d(pcstd.T,slpin.T)
    
    # if debug:
    #     # Check to make sure both regress_2d methods are the same
    #     # (comparing looping by PC, and using A= [P x N])
    #     eof1 = np.zeros((N_mode,npts))
    #     b1  = np.zeros(eof1.shape)
    #     # Regress back to the dataset
    #     for n in range(N_mode):
    #         eof1[n,:],b1[n,:] = proc.regress_2d(pcstd[:,n],regrin)
    #     print("max diff for eof (matrix vs loop) is %f"%(np.nanmax(np.abs(eof-eof1))))
    #     print("max diff for b (matrix vs loop) is %f"%(np.nanmax(np.abs(b-b1))))
        
    # Save the data
    eofall[:,m,okpts] = eof
    eofslp[:,m,:] = eof_s
    pcall[:,m,:] = pcs.T
    varexpall[:,m] = varexp

# Flip longitude ------------------------------------------------------------
eofall = eofall.reshape(N_mode,12,nlat,nlon)
eofall = eofall.transpose(3,2,1,0) # [lon x lat x mon x N]
lon180,eofall = proc.lon360to180(lon,eofall.reshape(nlon,nlat,N_mode*12))
eofall = eofall.reshape(nlon,nlat,12,N_mode)
# Repeat for SLP eofs
eofslp = eofslp.reshape(N_mode,12,nlat,nlon)
eofslp = eofslp.transpose(3,2,1,0) # [lon x lat x mon x N]
lon180,eofslp = proc.lon360to180(lon,eofslp.reshape(nlon,nlat,N_mode*12))
eofslp = eofslp.reshape(nlon,nlat,12,N_mode)

#%% F
#%% (**) Save the results
bboxtext = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
bboxstr  = "Lon %i to %i, Lat %i to %i" % (bbox[0],bbox[1],bbox[2],bbox[3])
savename = "%sNHFLX_%s_%iEOFsPCs_%s.npz" % (datpath,mcname,N_mode,bboxtext)
if correction:
    savename = proc.addstrtoext(savename,correction_str)

np.savez(savename,**{
    "eofall":eofall,
    "eofslp":eofslp,
    "pcall":pcall,
    "varexpall":varexpall,
    'lon':lon180,
    'lat':lat},allow_pickle=True)



#dsmerge = xr.concat(ds_all,dim='ens',join='left')




