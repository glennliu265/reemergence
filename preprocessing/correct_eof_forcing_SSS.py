#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Correct EOF Forcing for SSS (Evaporation, Precipitation)
Also Do this for Qek? (SST and SSS)

Copied correct_eof_forcing on 2024.03.01
(started) Adding Qek Corrections 2024.08.26

Perform EOF filtering based on a variance threshold.
Compute the Required Variance needed to correct back to 100% (monthly std(E') or std(P')) at each month.

Currently works on Astraeus but need to make this machine adaptable

Inputs:
------------------------
    varname             : dims                              - units                 - Full Name
    LHFLX               : (month,ens,lat,lon)
    PRECTOT             : (mon,ens,lat,lon)
#    Qek_SST             : (mon,ens,lat,lon)                 [degC / sec]
#    Qek_SSS             : (mon,ens,lat,lon)                 [degC/sec]
    

Outputs: 
------------------------

    varname             : dims                              - units                 - Full Name
    correction_factor   : (mon, lat, lon)                   [W/m2]
    eofs                : (mode, mon, lat, lon)             [W/m2 per std(pc)]
    
    
    

What does this script do?
------------------------
(1) Load in EOF Output and Fprime and take ensemble mean of variance explained, patterns, and std(F')
(2) Apply EOF Filtering, retaining only modes explaining up to N %
(3) Take difference of std(F') and std(EOF_filtered) to get pointwise variance correction factor
                                                                                                 
                                                                                                 

Note that correct ion is performed on the ENSEMBLE MEAN forcing and Fstd!!


Created on Tue Feb 13 20:21:00 2024

@author: gliu
"""


import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import tqdm

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "stormtrack"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
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
procpath    = pathdict['procpath']


#%% Indicate Paths

#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240411/"
datpath   = pathdict['raw_path']
outpath   = pathdict['input_path']+"forcing/"
rawpath   = pathdict['raw_path']


#%% User Edits


# Indicate Filtering OPtions
eof_thres = 0.90
bbox_crop = [-90,0,0,90]
# Indicate dataset name
dataset = "CESM1_HTR"
concat_ens=True


if dataset == "CESM1_HTR":
    
    datpath   = outpath
    # Indicate Forcing Options
    dampstr   = "nomasklag1"
    rollstr   = "nroll0"

    # Load EOF results
    nceof     = rawpath + "CESM1_HTR_EOF_Monthly_NAO_EAP_Fprime_%s_%s_NAtl_concatEns.nc" % (dampstr,rollstr)
    
    # Load Fprime
    ncfprime  = rawpath + "CESM1_HTR_FULL_Fprime_timeseries_%s_%s_NAtl.nc" % (dampstr,rollstr)
    
    # Load Evap, Precip
    ncevap    = rawpath + "CESM1_HTR_FULL_Eprime_timeseries_LHFLXnomasklag1_nroll0_NAtl.nc"#"CESM1_HTR_FULL_Eprime_nroll0_NAtl.nc"
    ncprec    = rawpath + "PRECTOT_HTR_FULL.nc"#"CESM1_HTR_FULL_PRECTOT_NAtl.nc"

    # Load EOF Regression output
    ncprec_eof = outpath + "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_concatEns.nc"
    #ncevap_eof = "CESM1_HTR_FULL_LHFLX_EOF_nomasklag1_nroll0_NAtl.nc"
    ncevap_eof = outpath + "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_concatEns.nc"
    ncfprime_eof = outpath + "CESM1_HTR_FULL_Fprime_EOF_nomasklag1_nroll0_NAtl_concatEns.nc"
    
elif dataset == "cesm1le_htr_5degbilinear":
    
    datpath         = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/proc/"
    
    # Indicate Forcing Options
    dampstr         = "cesm1le5degLHFLX"
    dampstr_qnet    = "cesm1le5degqnet"  
    rollstr         = "nroll0"
    regstr          = "Global"
    
    # Load EOF Results (NHFLX_EOF_Monthly)
    nceof     = datpath + "%s_EOF_Monthly_NAO_EAP_Fprime_%s_%s_%s.nc" % (dataset,dampstr_qnet,rollstr,regstr)
    
    # Load Fprime (Qnet)
    ncfprime  = "%s_Fprime_timeseries_%s_%s_%s.nc" %  (dataset,dampstr_qnet,rollstr,regstr)
    
    # Load Evap, Precip (to compute Monthly Standard Deviations)
    # Evap: hfcalc/Main/calc_Fprime
    # Prec: reemergence/preprocesing/combine_precip)
    ncevap    = datpath + "%s_Eprime_timeseries_%s_%s_%s.nc" % (dataset,dampstr,rollstr,regstr)
    ncprec    = datpath + "cesm1_htr_5degbilinear_PRECTOT_%s_1920to2005.nc" % (regstr)
    
    # Load Evap, Precip Regressions computed in /regress_EOF_forcing/
    ncevap_eof    = outpath + "cesm1le_htr_5degbilinear_Eprime_EOF_cesm1le5degLHFLX_nroll0_Global.nc"#
    ncprec_eof    = outpath + "cesm1le_htr_5degbilinear_PRECTOT_EOF_cesm1le5degLHFLX_nroll0_Global.nc" #
    

    
    
    
    
    
    
    

# Load Ekman Forcing

#fp1  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
#ncek = "CESM1_HTR_FULL_Qek_SSS_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc"
#dsek = xr.open_dataset(fp1+ncek)

# Other things
crop_sm     = True # Save Cropped Version
regstr_crop = "NAtl"
bbox_crop   = [-90,0,0,90]
bbox        = [-80,0,0,65]
debug       = True


def anomalize(ds):
    ds = ds - ds.mean('ens')
    ds = proc.xrdeseason(ds)
    return ds

def reshape_concat_ens(ds):
    # IMPT: To concatenate properly, Ensemble must be first...
    ds = ds.transpose('ens','time','lat','lon')
    nens,ntime,nlat,nlon=ds.shape
    dsout     = ds.data.reshape(ntime*nens,1,nlat,nlon)
    timefake  = proc.get_xryear('0000',nmon=ntime*nens)
    coordsout = dict(time=timefake,ens=np.arange(1,2),lat=ds.lat,lon=ds.lon)
    dsout     = xr.DataArray(dsout,coords=coordsout,dims=coordsout)
    return dsout
    

def stdstqsumpt(ds,axis,lonf,latf):
    return np.sqrt(np.nansum(proc.selpt_ds(ds,lonf,latf)**2,axis))

#%% Procedure

# (1) Load EOF Results, compute variance explained
dseof    = xr.open_dataset(nceof).load()
eofs     = dseof.eofs.mean('ens')   # (mode: 86, mon: 12, lat: 96, lon: 89)
varexp   = dseof.varexp.mean('ens') # (mode: 86, mon: 12)

# # (2) Load Fprime, compute std(F') at each point
dsfp     = xr.open_dataset(ncfprime).load().Fprime
dsfp    = proc.fix_febstart(dsfp)
dsfp    = anomalize(dsfp)
#%%

'''
Trouble shooting
'''
dsfp_funcrs = reshape_concat_ens(dsfp)

#%%
if concat_ens:
    dsfp      = reshape_concat_ens(dsfp)
monvarF   = dsfp.groupby('time.month').std('time')#.mean('ens') # (month: 12, lat: 96, lon: 89)
monvarF   = monvarF.rename({'month':'mon'})

# (2) Load  Evap, Precip to compute Monthly Stdev 
dsevap    = xr.open_dataset(ncevap).load().LHFLX # (time, ens, lat, lon)
dsevap    = proc.fix_febstart(dsevap)
dsevap    = anomalize(dsevap)
if concat_ens:
    dsevap      = reshape_concat_ens(dsevap)
monvarE   = dsevap.groupby('time.month').std('time')
monvarE   = monvarE.rename({'month':'mon'})

#dsevap    = dsevap.rename({'month':'mon'})
dsprec    = xr.open_dataset(ncprec).load().PRECTOT
dsprec    = proc.fix_febstart(dsprec)
dsprec    = anomalize(dsprec)
if concat_ens:
    dsprec    = reshape_concat_ens(dsprec)
monvarP   = dsprec.groupby('time.month').std('time')
monvarP   = monvarP.rename({'month':'mon'})

# (3) Load EOF regressions of Evap, Precip
dsevap_eof      = xr.open_dataset(ncevap_eof).load().LHFLX # (mode, ens, mon, lat, lon)
dsprec_eof      = xr.open_dataset(ncprec_eof).load().PRECTOT # (mode, ens, mon, lat, lon)
dsfprime_eof    = dseof.eofs.transpose('mode','mon','ens','lat','lon')

# Make dsure variables are the same
dsevap_eof,monvarE = proc.resize_ds([dsevap_eof,monvarE])
dsprec_eof,monvarP = proc.resize_ds([dsprec_eof,monvarP])
dsfprime_eof,monvarF = proc.resize_ds([dsfprime_eof,monvarF])

#%% 3. Perform EOF filtering (retain enough modes to explain [eof_thres]% of variance for each month)

# Inputs
#eofs_std   = dseof.eofs
varexp_in  = varexp.values           # Variance explained (for Fprime Analysis) [mode, mon]
vnames     = ["Fprime","LHFLX","PRECTOT"]     # names of variables
ds_eof_raw = [dsfprime_eof,dsevap_eof,dsprec_eof] # EOF regressions    (mode, ens, mon, lat, lon)
ds_std     = [monvarF,monvarE,monvarP]         # Monthly standard deviation (ens, mon , lat, lon)
ncnames    = [ncfprime_eof,ncevap_eof,ncprec_eof]
nvars      = len(vnames)
ds_varfull = [dsfp,dsevap,dsprec]

ensavg_first = True





# By Default, use the longitude coordinates of the regressed variables
# CAUTION: This script DOES NOT check if longitude is same across all variables
# Need to write a check for this..
lon_out    = dsprec_eof.lon
lat_out    = dsprec_eof.lat

check_eof  = []
check_corr = []
for v in range(nvars): # Loop by Variable
    

    if ensavg_first:
        # Index variables (take ensemble average)
        eofvar_in   = np.nanmean(ds_eof_raw[v].transpose('mode','ens','mon','lat','lon').values,1)
        monvarfp    = np.nanmean(ds_std[v].transpose('ens','mon','lat','lon').data,0)
        varexp_eavg = varexp_in#np.nanmean(varexp_in,1) # 
        
        # Perform Filtering
        eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt=proc.eof_filter(eofvar_in,varexp_eavg,
                                                            eof_thres,axis=0,return_all=True)
        
        
        # Compute Stdev of EOFs
        eofs_std = np.sqrt(np.sum(eofs_filtered**2,0)) # [Mon x Lat x Lon]
        
        
        lon_out    = ds_eof_raw[v].lon
        lat_out    = ds_eof_raw[v].lat
        
        # Compute pointwise correction
        correction_diff = monvarfp - eofs_std
        
        
        # Prepare to Save -------------------------
        
        corcoords     = dict(mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        eofcoords     = dict(mode=ds_eof_raw[0].mode,mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        
        da_correction = xr.DataArray(correction_diff,coords=corcoords,dims=corcoords,name="correction_factor")
        da_eofs_filt  = xr.DataArray(eofs_filtered,coords=eofcoords,dims=eofcoords  ,name=vnames[v])
    
        ds_out        = xr.merge([da_correction,da_eofs_filt])
        edict         = proc.make_encoding_dict(ds_out)
        
        
        check_corr.append(da_correction.copy())
        check_eof.append(da_eofs_filt.copy())
        
        # Save for all ensemble members
        savename       = proc.addstrtoext(ncnames[v],"_corrected",adjust=-1)
        savename       = proc.addstrtoext(savename,"_EnsAvgFirst",adjust=-1)
        ds_out.to_netcdf(savename,encoding=edict)
        print("Saved output to %s" % savename)
        
    else:
            
            
                
        # Index variables
        eofvar_in = ds_eof_raw[v].transpose('mode','ens','mon','lat','lon').values
        monvarfp  = ds_std[v].transpose('ens','mon','lat','lon').values
            
        # Perform Filtering
        eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt=proc.eof_filter(eofvar_in,varexp_in,
                                                           eof_thres,axis=0,return_all=True)
        
        # Compute Stdev of EOFs
        eofs_std = np.sqrt(np.sum(eofs_filtered**2,0)) # [Ens x Mon x Lat x Lon]
        
        
        lon_out    = ds_eof_raw[v].lon
        lat_out    = ds_eof_raw[v].lat
        
        
    
            
            
        # Compute pointwise correction
        correction_diff = monvarfp - eofs_std
        
        # Prepare for output -----
        corcoords     = dict(ens=ds_std[0].ens,mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        eofcoords     = dict(mode=dseof.mode,ens=ds_std[0].ens,mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        
        da_correction = xr.DataArray(correction_diff,coords=corcoords,dims=corcoords,name="correction_factor")
        da_eofs_filt  = xr.DataArray(eofs_filtered,coords=eofcoords,dims=eofcoords  ,name=vnames[v])
        
        
        ds_out        = xr.merge([da_correction,da_eofs_filt])
        edict         = proc.make_encoding_dict(ds_out)
        
        # Save for all ensemble members
        savename       = proc.addstrtoext(ncnames[v],"_corrected",adjust=-1)
        ds_out.to_netcdf(savename,encoding=edict)
        
        savename_emean = proc.addstrtoext(savename,"_EnsAvg",adjust=-1)
        ds_out_ensavg  = ds_out.mean('ens')
        ds_out_ensavg.to_netcdf(savename_emean,encoding=edict)
        
        # if crop_sm:
        #     print("Cropping to region %s" % (regstr_crop))
        #     ds_out = proc.lon360to180_xr(ds_out)
            
        #     ds_out_reg = proc.sel_region_xr(ds_out,bbox_crop)
        #     savename_reg = proc.addstrtoext(ncnames[v],"_corrected",adjust=-1).replace(regstr,regstr_crop)
        #     ds_out_reg.to_netcdf(savename_reg,encoding=edict)
            
        #     savename_emean = proc.addstrtoext(savename_reg,"_EnsAvg",adjust=-1)
        #     ds_out_reg_ensavg  = ds_out_reg.mean('ens')
        #     ds_out_reg_ensavg.to_netcdf(savename_emean,encoding=edict)
        #     print("Saved Ens Avg. Cropped Output to %s" % savename_emean)


#%% Part II: Load Portions and Compute the Correlations of the Residuals....
# Note: Currently works only for concatenated case...

#%% Reload the EOFs


eofs_byvar = []
for v in tqdm.tqdm(range(nvars)):
    
    # Save for all ensemble members
    savename       = proc.addstrtoext(ncnames[v],"_corrected",adjust=-1)
    savename       = proc.addstrtoext(savename,"_EnsAvgFirst",adjust=-1)
    
    ds = xr.open_dataset(savename).load()
    eofs_byvar.append(ds)

#%% Project the EOF (by month

v         = 0
vsel = [1,2]

for v in vsel:
    eofs_filt = eofs_byvar[v][vnames[v]] # (200, 12, 96, 89)
    pcs       = dseof.pcs # (200, 12, 1, 3612), They are already stanfardized
    
    
    nmode,nmon,nlat,nlon = eofs_filt.shape
    nmode,nmon,nens,nyr  = pcs.shape
    
    # Project pattern onto eof
    # eofs_in  = eofs_filt.data[...,None]
    # pcs_in   = pcs.data[:,:,None,:]
    
    eof_proj = np.zeros((nyr,nmon,nlat,nlon))
    for im in range(12):
        
    
        
        for n in tqdm.tqdm(range(nmode)):
            eofs_in = eofs_filt.isel(mon=im,mode=n).data     # {Lat x Lon}
            pcs_in  = pcs.isel(mon=im,mode=n).squeeze().data # {Year}
            
            eof_proj[:,im,:,:] += eofs_in[None,:,:] * pcs_in[:,None,None]
    
    
    
    coords  = dict(yr=np.arange(nyr),mon=np.arange(1,13,1),lat=eofs_filt.lat,lon=eofs_filt.lon)
    daout   = xr.DataArray(eof_proj,coords=coords,dims=coords,name=vnames[v])
    edict   = proc.make_encoding_dict(daout)
    
    outname = "%sCESM1_%s_EOF_Projection_concatEns.nc" % (rawpath,vnames[v],)
    daout.to_netcdf(outname,encoding=edict)
    
#%% Reload the filtered variables
eofs_filt_sum = []
for v in range(3):
    outname = "%sCESM1_%s_EOF_Projection_concatEns.nc" % (rawpath,vnames[v],)
    ds = xr.open_dataset(outname).load()
    eofs_filt_sum.append(ds)
    
eofs_filt_sum = [eofs_filt_sum[vv][vnames[vv]] for vv in range(3)]



#%% Compute the residual and check the correlation

# Do it for 1 point
testeof  = eofs_filt_sum[0].sel(lon=-30,lat=50,method='nearest').data.reshape(3612*12)
testfull = ds_varfull[0].sel(lon=-30,lat=50,method='nearest').squeeze()

fig,ax = plt.subplots(1,1)
ax.plot(testeof[0:100],label="EOF")
ax.plot(testfull[0:100],label="FULL")
ax.legend()
plt.show()

#%% Now compute residual for all points

vsel = [1,2]
for v in vsel:
    eofsum  = eofs_filt_sum[v]
    varfull = ds_varfull[v]
    _,_,nlat,nlon = eofsum.shape
    diff   = varfull.data.squeeze() - eofsum.data.reshape(3612*12,nlat,nlon)
    
    coords = dict(time=np.arange(varfull.shape[0]),lat=eofsum.lat,lon=eofsum.lon)
    da_res = xr.DataArray(diff,coords=coords,dims=coords,name=vnames[v])
    
    edict   = proc.make_encoding_dict(da_res)
    outname = "%sCESM1_%s_EOF_Residual_concatEns.nc" % (rawpath,vnames[v],)
    da_res.to_netcdf(outname,encoding=edict)
    
#%% Now load the computed residuals...

da_res_all = []
for v in range(3):
    outname = "%sCESM1_%s_EOF_Residual_concatEns.nc" % (rawpath,vnames[v],)
    ds = xr.open_dataset(outname)[vnames[v]].load()
    da_res_all.append(ds)



def pointcorr(ts1,ts2):
    if np.all(ts1==0) or np.all(ts2==0):
        return np.nan
    if np.any(np.isnan(ts1)) or  np.any(np.isnan(ts2)):
        return np.nan
    return np.corrcoef(ts1,ts2)[0,1]

bbsim      = [-80,0,20,60]
da_res_all = [proc.sel_region_xr(ds,bbsim) for ds in da_res_all]
for ii in range(3):
    da_res_all[ii]['lat'] = da_res_all[0].lat
    da_res_all[ii]['lon'] = da_res_all[0].lon
    
    
#da_res_all = [ da['lat'] = da_res_all[0].lat for da in da_res_all]
#da_res_all = proc.resize_ds(da_res_all)
#%% Now Compute pairwise point correlations using xr.ufunc


def loop_pointcorr(ds1,ds2):
    ntime,nlat,nlon = ds1.shape
    lon,lat = ds1.lon,ds1.lat
    ds1 = ds1.data
    ds2 = ds2.data
    outcorr = np.zeros((nlat,nlon)) * np.nan
    for o in range(nlon):
        for a in range(nlat):
            outcorr[a,o] = pointcorr(ds1[:,a,o],ds2[:,a,o])
    return xr.DataArray(outcorr,coords=dict(lat=lat,lon=lon))
            
            
st = time.time()
outcorr = loop_pointcorr(da_res_all[0],da_res_all[2])
print("Computed in %.2fs" % (time.time() - st))

st = time.time()

#calc_leadlag    = lambda x,y: proc.leadlag_corr(x,y,lags,corr_only=True)
crosscorrs = xr.apply_ufunc(
    pointcorr,
    da_res_all[0], # Fprime
    da_res_all[2], # Precip
    input_core_dims=[['time'],['time']],
    output_core_dims=[[]],
    vectorize=True,
    )
print("Computed in %.2fs" % (time.time() - st))



leadlags     = np.concatenate([np.flip(-1*lags)[:-1],lags],) 
crosscorrs['lags'] = leadlags

#%%
#%%
#eof_proj = eofs_in * pcs_in

#%%
    
    
#%%



#%%


#%% Older Debugging sections (check and remove later)

# Check values
if debug:
    mons3   = proc.get_monstr()
    fig,ax  = viz.init_monplot(1,1)
    ax.bar(mons3,nmodes_needed,alpha=0.5,color='darkred')
    ax.set_xlim([-1,12])
    ax.set_title("Number of Modes Needed to Explain %.2f" % (eof_thres*100) + "% of Variance")
    ax.set_ylabel("Number of Modes")
    savename = "%sNAO_EAP_Fprime_Forcing_NumModes_thres%03i.png" % (figpath,eof_thres*100)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    fig,ax  = viz.init_monplot(1,1)
    ax.plot(np.sum(varexp,0),label="Raw")
    ax.plot(np.sum(varexps_filt,0),label="Post-Filtering")
    ax.set_ylabel("Total Variance Explained")
    ax.legend()

#%% Check values

lonf = -36
latf = 50

monvar_pt = [proc.selpt_ds(ds,lonf,latf).squeeze() for ds in ds_std]
corr_pt   = [np.sqrt(proc.selpt_ds(ds,lonf,latf).squeeze()**2) for ds in check_corr]
eofs_pt   = [np.sqrt(np.nansum(proc.selpt_ds(ds,lonf,latf)**2,0)) for ds in check_eof]


fig,axs = plt.subplots(1,3,)

for a,ax in enumerate(axs):
    ax.set_xticks(np.arange(1,13,1))
    
    ax.plot(monvar_pt[a],color="k",label="ori")
    ax.plot(corr_pt[a],color="red",label="correction")
    ax.plot(eofs_pt[a],color='blue',label="eof")
    ax.plot(corr_pt[a]+eofs_pt[a],color='cyan',ls='dashed',label="final")
    ax.legend()
    
plt.show()

#%% Do some debugging

cf       = ds_out.correction_factor.isel(mon=0)
totalvar = monvarF.isel(mon=0)
rat      = cf/totalvar