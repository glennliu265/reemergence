#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Salinity-Heat Flux Feedback

Takes regridded SSS preprocessed by [prep_MLD_PIC]
Takes ENSO Index calculated by      [calc_enso_general]

Created on Tue Aug 29 13:52:56 2023

@author: gliu
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob
import sys
from tqdm import tqdm
import time

#%% Import custom modules

stormtrack = 1
if stormtrack:
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
else:
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
   # sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")
    
from amv import proc,viz
import scm
#import amv_dataloader as adl
#%% Further Edits

# Data that has been processed by [preproc_CESM1_LENS]
outpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_HTR/"

# Location ofsalinity data regridded to atmospheric grid using [prep_mld_PIC]
# Name structure: SSS_FULL_HTR_bilinear_num##.nc
ssspath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/SSS/"


# Processed CESM1-HTR data 
dataset_name = "htr" 
datpath_htr  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_HTR/" # Where ensavg is stored

# Intermediate Outputs
datpath      =  "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_HTR/01_PREPROC/"


# ENSO information

# Name of the variables
vnames    = ("SSS","TS") # Variables
fluxnames = ("FSNS","FLNS","LHFLX","SHFLX") # Fluxes
dimnames  = ("lat","lon","time")
vnames_in = ("SSS","FSNS","FLNS","LHFLX","SHFLX")
vnames_in_long = (
    "Sea Surface Salinity",
    "Shortwave",
    "Longwave",
    "Latent Heat Flux",
    "Sensible Heat Flux"
    )

# Additional Variables copied from calc_enso_general --------
# Select time crop (prior to preprocessing)
croptime          = True # Cut the time prior to detrending, EOF, etc
tstart            =  '1920-01-01' # "2006-01-01" # 
tend              =  '2006-01-01' #"2101-01-01" # 

# Select time crop (for the estimate)
croptime_estimate = True # Cut time right before estimating the heat flux feedback
tcrop_start       = "1970-01-01"#'1920-01-01' '2070-01-01'#
tcrop_end         = "1999-12-31"#'1970-01-01' '2099-12-31'#
tcrop_fname       = ""
if croptime_estimate:
    tcrop_fname      = "_%sto%s" % (tcrop_start.replace('-',''),tcrop_end.replace('-',''))
    
# ENSO Parameters
pcrem    = 3                   # PCs to calculate
bbox     = [120, 290, -20, 20] # ENSO Bounding Box

# Part 3 (ENSO Removal) -------------------------------------------
ensolag  = 1    # Lag between ENSO month and response month in NATL
reduceyr = True # Drop years due to ENSO lag
monwin   = 3    # Window of months to consider


# Part 4 (HFF Calculations) -------------------------------------------
ensorem  = True

# Set coordinate names for dataset
lonname  = 'lon'
latname  = 'lat'
tname    = 'time'

overwrite = False
detrend   = True # Set this as default for large ensemble
lensflag  = True
# ----------


#%% Get SSS and combine ensemble average

# Get list of regridded SSS
nclist = glob.glob(ssspath+"SSS_FULL_HTR_bilinear_num*.nc")
nclist.sort()
nens   = len(nclist)
    
# First, check if ensemble average has already been calculated
savename    = "%sCESM1_htr_SSS_ensAVG.nc" % outpath
query       = glob.glob(savename)
if len(query) < 1:
    print("Calculating SSS Ens. Avg")
    
    # Compute Ensemble Average for SSS, load SSS datasets
    ds_all_sss  = []
    for e in tqdm(range(nens)):
        
        ds = xr.open_dataset(nclist[e])
        ds_all_sss.append(ds)
        if e == 0:
            ds          = ds.sel(time=slice("1920-01-01","2005-12-31"))
            ens_avg_sss = ds.SSS.values
        else:
            ens_avg_sss = ens_avg_sss + ds.SSS.values
    ens_avg_sss = ens_avg_sss / nens
    
    # Save ensemble average
    
    coordsdict  = {'time':ds.time,
                   'z_t' :ds.z_t,
                   'lat' :ds.lat,
                   'lon' :ds.lon}
    sss_ensavg  = xr.DataArray(ens_avg_sss,coords=coordsdict,dims=coordsdict,name="SSS")
    sss_ensavg.to_netcdf(savename,encoding={"SSS": {'zlib': True}})
else:
    print("SSS Ens. Avg Found!")
    
#%% Get nclists for the other variables

nclists_all = [nclist,]
for v in vnames_in:
    if v == "SSS": # Skip SSS
        continue
    else:
        ncl = glob.glob("%sCESM1_%s_%s_ens*.nc" % (datpath,dataset_name,v)) # Note that this grabs ensAVG file...
        ncl = [nc for nc in ncl if "AVG" not in nc]
    nclists_all.append(ncl)
    print("Found %i files for %s" % (len(ncl),v))

#%% The section below is adapted from calc_enso_general with additional steps for preprocessing SSS

allstart=time.time()

for ensnum in range(nens):
    
    # 1. Preprocess Variable (currently just works for historical)
    # -------------------------------------------------------------o
    for iv,v in enumerate(vnames_in):
        #vname = vnames_in[v]
        # Open dataset and slice to time period of itnerest
        da = xr.open_dataset(nclists_all[iv][ensnum])
        
        # Crop time if option is set
        if e == 0:
            da = da.sel(time=slice("1920-02-01","2006-01-01"))
        # else:
        #     da = da.sel(time=slice("1920-01-01","2006-01-10"))
        
        # Check time, and skip file if it already exists
        # ----------------------------------------------
        times   = da[tname].values
        timesyr = times.astype('datetime64[Y]').astype(int) +1970
        timestr = "%ito%i" % (timesyr[0],timesyr[-1])
        
        # Set Save Name
        savename = "%s%s_%s_manom_detrend1_%s_ens%02i.nc" % (datpath,dataset_name,v,timestr,ensnum+1)
        query = glob.glob(savename)
        
        if (len(query) < 1) or (overwrite == True):
            
            # Read out the other variables # [time x lat x lon]
            # -------------------------------------------------
            st    = time.time()
            invar = da[v].values
            lon   = da[lonname].values
            lat   = da[latname].values
            print("Data loaded in %.2fs"%(time.time()-st))
            
            # For LENs case, remove ensavg
            # ----------------------------
            if dataset_name in ["rcp85", "htr"]:
                eavg_fname  = "%sCESM1_%s_%s_ensAVG.nc" % (datpath,dataset_name,v)
            else:
                eavg_fname  = "%s%s_%s_ensAVG.nc" % (datpath,dataset_name,v)
            ensavg      = xr.open_dataset(eavg_fname)
            ensavg      = ensavg.sel(time=slice(tstart,tend),drop=True)
            ensavg      = ensavg[v].values
            invar       = invar - ensavg
            
            
            # Remove monthly anomalies
            # ------------------------
            invar          = invar.squeeze() # for cases where there is an additional dimension
            nmon,nlat,nlon = invar.shape
            manom,invar = proc.calc_clim(invar,0,returnts=1) # Calculate clim with time in axis 0
            vanom = invar - manom[None,:,:,:]
            vanom = vanom.reshape(nmon,nlat,nlon) # Reshape back to [time x lat x lon]
        
            # Flip latitude
            if lat[0] > lat[-1]: # If latitude is decreasing...
                lat   = np.flip(lat)
                vanom = np.flip(vanom,axis=1)
                
            # Flip longitude (for SSS) back to 0 to 360
            if np.any(lon<0):
                lon,vanom=proc.lon180to360(lon,vanom.transpose(2,1,0))
                vanom = vanom.transpose(2,1,0)
            
            # Detrend the variable (taken from calc_amv_hadisst.py)
            # ----------------------------------------------------
            data_dt = vanom
                
            
            # Save detrended data if option is set
            # ------------------------------------
            da = proc.numpy_to_da(data_dt,times,lat,lon,v,savenetcdf=savename)
        else:
            print("Skipping. Found existing file: %s" % (str(query)))
    
    
    # 2. Load ENSO Indices (computed already)
    # -------------------------------------------------------------o
    savename = "%senso/%s_ENSO_detrend%i_pcs%i_%s_ens%02i.npz" % (datpath,dataset_name,detrend,pcrem,timestr,ensnum+1)
    # Stupid dummy fix (need to be more consistent with naming...)
    if dataset_name == "htr":
        savename = savename.replace('2005','2006')
    query = glob.glob(savename)
    if len(query) < 1:
        print("ENSO file not found at %s. Please recalculate..." % savename)
    else:
        print("ENSO file found.")
        ld = np.load(savename,allow_pickle=True)
        ensoid = ld['pcs'] # [year x  month x pc]
        
        
        for v in vnames_in:
            
            # Load Target variable
            savename = "%s%s_%s_manom_detrend%i_%s_ens%02i.nc" % (datpath,dataset_name,v,detrend,timestr,ensnum+1)
            da       = xr.open_dataset(savename)
            
            # Check if (ENSO index file) already exists, and skip if so.
            savename = "%senso/%s_%s_detrend%i_ENSOrem_lag%i_pcs%i_monwin%i_%s_ens%02i.nc" % (datpath,dataset_name,v,detrend,ensolag,pcrem,monwin,timestr,ensnum+1)
            
            query    = glob.glob(savename)
            if (len(query) < 1) or (overwrite == True):
                # Read out the variables # [time x lat x lon]
                st        = time.time()
                invar     = da[v].values
                lon       = da[lonname].values
                lat       = da[latname].values
                times     = da[tname].values
                
                # # Check if a particular time is all nan:
                for t in range(invar.shape[0]):
                    if np.all(np.isnan(invar[t,:,:])):
                        print("All NaN Points at time=%s. Setting to zero." % (da.time.isel(time=t).values))
                        invar[t,:,:] = 0 # Set times there to zero
                    
                
                # Remove ENSO
                vout,ensopattern,times = scm.remove_enso(invar,ensoid,ensolag,monwin,reduceyr=reduceyr,times=times)
                
                da = proc.numpy_to_da(vout,times,lat,lon,v,savenetcdf=savename)
                # Save ENSO component
                savename = "%senso/%s_%s_detrend%i_ENSOcmp_lag%i_pcs%i_monwin%i_%s.npz" % (datpath,dataset_name,v,detrend,ensolag,pcrem,monwin,timestr)
                if lensflag:
                    savename = proc.addstrtoext(savename,"_ens%02i"%(ensnum+1),adjust=0)
                np.savez(savename,**{
                    'ensopattern':ensopattern,
                    'lon':lon,
                    'lat':lat}
                         )
                
                print("Completed variable %s (t=%.2fs)" % (v,time.time()-allstart))
                # End Skip
            else:
                print("Skipping. Found existing file: %s" % (str(query)))
            # <End Variable Loop>
        # <End ENSO Removal Loop>
    
    # --------------------------------
    #%% Compute the heat flux feedback
    # --------------------------------
    
    # Load inputs with variables removed
    invars = []
    for v in vnames_in:
        
        if ensorem:
            savename = "%senso/%s_%s_detrend%i_ENSOrem_lag%i_pcs%i_monwin%i_%s.nc" % (datpath,dataset_name,v,detrend,ensolag,pcrem,monwin,timestr)
        else:
            savename = "%s%s_%s_manom_detrend%i_%s.nc" % (datpath,dataset_name,v,detrend,timestr)
        if lensflag:
            savename = proc.addstrtoext(savename,"_ens%02i"%(ensnum+1),adjust=-1)
        ds       = xr.open_dataset(savename)
        
        lat = ds.lat.values
        lon = ds.lon.values
        
        # Crop time if option is set
        if croptime_estimate:
            ds = ds.sel(time=slice(tcrop_start,tcrop_end),drop=True)
        
        loadvar = ds[v].values
        ntime,nlat,nlon = loadvar.shape
        loadvar = loadvar.reshape(int(ntime/12),12,nlat,nlon)
        
        invars.append(loadvar)
    
    #% Calculate heat flux
    for f in range(4):
        
        sst = invars[0]
        flx = invars[f+1]
        fname = vnames_in[f+1]
        
        damping,autocorr,crosscorr,autocov,cov = scm.calc_HF(sst,flx,[1,2,3],3,verbose=True,posatm=True,return_cov=True)
        
        # Save heat flux (from hfdamping_mat2nc.py)
        # ----------------------------------------
        outvars  = [damping,crosscorr,autocorr,cov,autocov]
        datpath_out = "%s%s_damping/" % (datpath,vnames_in[0])
        proc.makedir(datpath_out)
        
        savename = "%s%s_%s_hfdamping_ensorem%i_detrend%i_%s_%scrop.nc" % (datpath_out,dataset_name,fname,ensorem,detrend,timestr,tcrop_fname)
        if lensflag:
            savename = proc.addstrtoext(savename,"_ens%02i"%(ensnum+1),adjust=-1)
        dims     = {'month':np.arange(1,13,1),
                      "lag"  :np.arange(1,4,1),
                      "lat"  :lat,
                      "lon"  :lon}
        
        # Set some attributes
        varnames = ("%s_damping"        % fname,
                    "%s_%s_crosscorr"   % (vnames_in[0],fname),
                    "%s_autocorr"       % vnames_in[0],
                    "cov",
                    "autocov")
        varlnames = ("%s Damping"                       % vnames_in_long[f+1],
                     "%s-Heat Flux Cross Correlation"   % vnames_in[0],
                     "%s Autocorrelation"               % vnames_in[0],
                     "%s-Heat Flux Covariance"          % vnames_in[0],
                     "%s Autocovariance"                % vnames_in[0])
        if vnames_in[0] == "SST":
            vunit = "degC"
        elif vnames_in[0] == "SSS":
            vunit = "psu"
        units     = ("W/m2/%s" % vunit,
                     "Correlation",
                     "Correlation",
                     "W/m2*%s" % vunit,
                     "%s2" % vunit)
        
        das = []
        for v,name in enumerate(varnames):
        
            attr_dict = {'long_name':varlnames[v],
                         'units':units[v]}
            da = xr.DataArray(outvars[v],
                        dims=dims,
                        coords=dims,
                        name = name,
                        attrs=attr_dict
                        )
            if v == 0:
                ds = da.to_dataset() # Convert to dataset
            else:
                ds = ds.merge(da) # Merge other datasets
                
            # Append to list if I want to save separate dataarrays
            das.append(ds)
        
        #% Save as netCDF
        # ---------------
        st = time.time()
        encoding_dict = {name : {'zlib': True} for name in varnames} 
        print("Saving as " + savename)
        ds.to_netcdf(savename,
                 encoding=encoding_dict)
        print("Saved in %.2fs" % (time.time()-st))

# #%% Save 
# if debug: # Plot seasonal cycle
#     il = 0
#     proj = ccrs.PlateCarree()
#     fig,axs = plt.subplots(4,3,subplot_kw={'projection':proj},
#                            figsize=(12,12),constrained_layout=True)
    
    
#     plotmon = np.roll(np.arange(0,12),1)
    
#     for im in range(12):
        
#         monid = plotmon[im]
        
#         ax = axs.flatten()[im]
#         plotvar = damping[monid,il,:,:]
#         lon1,plotvar1 = proc.lon360to180(lon,(plotvar.T)[...,None])
        
#         blabel=[0,0,0,0]
#         if im%3 == 0:
#             blabel[0] = 1
#         if im>8:
#             blabel[-1] = 1
        
#         ax = viz.add_coast_grid(ax,bbox=[-80,0,-10,62],fill_color='gray',
#                                 blabels=blabel,ignore_error=True)
#         pcm = ax.contourf(lon1,lat,plotvar1.squeeze().T*-1,levels = np.arange(-50,55,5),extend='both',
#                             cmap='cmo.balance')
        
#         viz.label_sp(monid+1,usenumber=True,alpha=0.7,ax=ax,labelstyle="mon%s")
    
#     cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.035,pad=0.05)
#     cb.set_label("$\lambda_a$ : $W m^{2} \lambda_a$ ($\degree C ^{-1}$)")
#     plt.suptitle("Heat Flux Damping For %s \n Enso Removed: %s | Lag: %i" % (dataset_name,ensorem,il+1))
#     plt.savefig("%sNHFLX_damping_lag%i_%s_detrend%i_%s.png" % (figpath,il+1,dataset_name,detrend,timestr),dpi=150)
# print("Script Ran to Completion in %.2fs"%(time.time()-st_script))
# #%%

# # 3. Calculate HFF (computed already)
# # -------------------------------------------------------------o



# #%% Rewrite this part as appropriate

# # >> list of netcdfs to process for each variable

# #%% Looping for each ensemble member

# #%% Step 1 (Formatting for Intake)

# #%% Step 2 (Preprocessing)

# #%% Step 3 (ENSO Removal)

# #%% Step 4 (HFF Calculation)

