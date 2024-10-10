#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the residual component, post EOF filtering
Subtract it from the full variable.

Copied upper section of correct_eof_forcing_SSS


Created on Thu Sep 19 22:45:49 2024

@author: gliu
"""



import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt

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


    # Load Qek
    nc_qek_sst = rawpath + "CESM1LE_Qek_SST_NAtl_19200101_20050101_bilinear.nc"
    nc_qek_sss = rawpath + "CESM1LE_Qek_SSS_NAtl_19200101_20050101_bilinear.nc"
    
    ncqek_sst_eof = outpath + "CESM1_HTR_FULL_Qek_SST_NAO_DirReg_NAtl_concatEns_corrected_EnsAvgFirst.nc"
    ncqek_sss_eof = outpath + "CESM1_HTR_FULL_Qek_SSS_NAO_DirReg_NAtl_concatEns_corrected_EnsAvgFirst.nc"
    
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
    
#%% Load the files


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

# # # (2) Load Fprime, compute std(F') at each point
# dsfp     = xr.open_dataset(ncfprime).load().Fprime
# dsfp    = proc.fix_febstart(dsfp)
# dsfp    = anomalize(dsfp)


# Load Qek SST and SSS
qek_ncs = [nc_qek_sst, nc_qek_sss]
ds_qeks = []
for ii in range(2):
    ds = xr.open_dataset(qek_ncs[ii])['Qek'].load()
    ds = proc.fix_febstart(ds)
    ds = ds.rename(dict(ensemble='ens'))#proc.format_dims(ds)
    dsanom = anomalize(ds)
    ds_qeks.append(dsanom)

ds_qeks = [reshape_concat_ens(ds) for ds in ds_qeks]

qek_eofs    = [ncqek_sst_eof,ncqek_sss_eof]
ds_qek_eofs = []
# Load Filtered EOF Regressions
ds_qeks_eof = []
for ii in range(2):
    ds = xr.open_dataset(qek_eofs[ii])['Qek'].load()
    ds_qeks_eof.append(ds)
    
#%%
vnames     = ["Qek_SST","Qek_SSS"]#"PRECTOT"]     # names of variables
v          = 0


for v in range(2):
    eofs_filt = ds_qeks_eof[v] # (200, 12, 96, 89)
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
for v in range(2):
    outname = "%sCESM1_%s_EOF_Projection_concatEns.nc" % (rawpath,vnames[v],)
    ds = xr.open_dataset(outname).load()
    eofs_filt_sum.append(ds)
    
eofs_filt_sum = [eofs_filt_sum[vv][vnames[vv]] for vv in range(len(vsel))]



# #%% Compute the residual and check the correlation

# # Do it for 1 point
# testeof  = eofs_filt_sum[0].sel(lon=-30,lat=50,method='nearest').data.reshape(3612*12)
# testfull = ds_varfull[0].sel(lon=-30,lat=50,method='nearest').squeeze()

# fig,ax = plt.subplots(1,1)
# ax.plot(testeof[0:100],label="EOF")
# ax.plot(testfull[0:100],label="FULL")
# ax.legend()
# plt.show()

#%% Now compute residual for all points

vsel = [0,1]
for v in vsel:
    eofsum  = eofs_filt_sum[v]
    varfull = ds_qeks[v]
    _,_,nlat,nlon = eofsum.shape
    diff   = varfull.data.squeeze() - eofsum.data.reshape(3612*12,nlat,nlon)
    
    coords = dict(time=np.arange(varfull.shape[0]),lat=eofsum.lat,lon=eofsum.lon)
    da_res = xr.DataArray(diff,coords=coords,dims=coords,name=vnames[v])
    
    edict   = proc.make_encoding_dict(da_res)
    outname = "%sCESM1_%s_EOF_Residual_concatEns.nc" % (rawpath,vnames[v],)
    da_res.to_netcdf(outname,encoding=edict)

#%%
#%% Add intermediate loading scripts ----



#%%----


ds_eof_raw = #[dsfprime_eof,dsevap_eof,dsprec_eof] # EOF regressions    (mode, ens, mon, lat, lon)
ds_std     = [monvarF,monvarE,monvarP]         # Monthly standard deviation (ens, mon , lat, lon)
#ncnames    = #[ncfprime_eof,ncevap_eof,ncprec_eof]
nvars      = len(vnames)
ds_varfull = ds_qeks #[dsfp,dsevap,dsprec]
    
#%% Load the 

    
#%%

'''
Trouble shooting
'''
dsfp_funcrs = reshape_concat_ens(dsfp)



    