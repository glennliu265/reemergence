#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Run a (single) stochastic SSS model experiment

# Entrainment Only
# Works with output from preproc_sm_inputs_SSS

# General Steps

(1) Initialize Directory

(2) Load Inputs

(3) Unit Conversions + Prep Parameters

(4) Integrate Model

(5) Analyze Output

(6) Save


Created on Thu Feb  1 17:10:51 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

#%% Import Custom Modules

# Import AMV Calculation
amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
sys.path.append(amvpath)
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
import scm

# Import Hf Calc params
hfpath  = "/stormtrack/home/glliu/00_Scripts/01_Projects/01_AMV/01_hfdamping/hfcalc/" # hfcalc module 
sys.path.append(hfpath)
import hfcalc_params as hp

#%% Input Options

# Paths and Experiment
input_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/"
output_path= "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"

expname    = "Test_Td0.1_SPG"

expparams   = {
    'bbox_sim'      : [-65,0,45,65],
    'nyrs'          : 1000,
    'runids'        : ["test01",],
    'PRECTOT'       : "CESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc",
    'LHFLX'         : "CESM1_HTR_FULL_QL_NAtl_EnsAvg.nc",
    'h'             : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'         : 0.10,
    'Sbar'          : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
    'beta'          : None, # If None, just compute entrainment damping
    'kprev'         : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'         : None, # NEEDS TO BE ALREADY CONVERTED TO 1/Mon !!!
    }

# Constants
dt  = 3600*24*30 # Timestep [s]
cp  = 3850       # 
rho = 1026       # Density [kg/m3]
B   = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L   = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

debug = True

#%% Load Params

# Parameters to check for []
#chk_params = ["mld","evap_forcing","precip_forcing","Sbar","lbd_d","beta","kprev","lbd_a"]


# ds_mld   = xr.open_dataset(input_path + "mld/" + expparams['mld'])['h']#.h.values
# ds_e     = xr.open_dataset(input_path + "forcing/" + expparams['evap_forcing'])['LHFLX']
# ds_p     = xr.open_dataset(input_path + "forcing/" + expparams['precip_forcing'])['PRECTOT']
# ds_sbar  = xr.open_dataset(input_path + "forcing/" + expparams['Sbar'])['Sbar']

# if type(expparams['lbd_d'])==str:
#     ds_lbdd   = xr.open_dataset(input_path + "damping/" + expparams['lbd_d'])['lbd_d']#.h.values
# else: # Assuming Td is a single value...
#     ds_lbdd = xr.ones_like(ds_p).rename("lbd_d") * expparams['lbd_d']
    
    
#%% Check and Load Params

chk_params = ["h","LHFLX","PRECTOT","Sbar","lbd_d","beta","kprev","lbd_a"]
param_type = ["mld","forcing","forcing","forcing","damping","mld","mld","damping"]

ninputs       = len(chk_params)
inputs_ds     = {}
inputs        = {}
missing_input = []
for nn in range(ninputs):
    pname = chk_params[nn]
    ptype = param_type[nn]
    if type(expparams[pname])==str: # If String, Load from input folder
        
        # Load DS
        ds = xr.open_dataset(input_path + ptype + "/" + expparams[pname])[pname]
        
        # Crop to region
        # Load dataarrays for debugging
        dsreg    = proc.sel_region_xr(ds,expparams['bbox_sim']).load()
        inputs_ds[pname] = dsreg.copy() 
        
        # Load to numpy arrays 
        varout   = dsreg.values
        inputs[pname] = dsreg.values.copy()
        
    else:
        missing_input.append(pname)
    # elif type(expparams[pname])==float:
    #     # Make empty data_array, multiplied by the given value
    #     print("For <%s> Making Empty DataArray with the repeated value %f" % (pname,expparams[pname]))
    #     ds = xr.ones_like(inputs[nn-1]).rename(pname) * expparams[pname]
    # else:
    #     missing_input.append(pname)
# Crop to Region

#varcrop     = [proc.sel_region_xr(ds,expparams['bbox_sim']).load().values for ds in inputs] 

#%% Process Missing Inputs

_,nlat,nlon=inputs['h'].shape

for pname in missing_input:
    if type(expparams[pname]) == float:
        print("Float detected for <%s>. Making array with the repeated value %f" % (pname,expparams[pname]))
        inputs[pname] = np.ones((12,nlat,nlon)) * expparams[pname]
    else:
        print("No value found for <%s>. Setting to zero." % pname)
        inputs[pname] = np.zeros((12,nlat,nlon))
        
        
        



# #%% Crop to Region

# inputs      = [ds_mld,ds_e,ds_p,ds_sbar,ds_lbdd]
# varcrop     = [proc.sel_region_xr(ds,expparams['bbox_sim']).load().values for ds in inputs] 
# [print(v.shape) for v in varcrop]
# h,e,p,sbar,lbd_d = varcrop

#%% For Debugging
dsreg =inputs_ds['h']
latr = dsreg.lat.values
lonr = dsreg.lon.values
klon,klat=proc.find_latlon(-30,50,lonr,latr)

#%% Initialize An Experiment folder for output

expdir = output_path + expname + "/"
proc.makedir(expdir + "Input")
proc.makedir(expdir + "Output")
proc.makedir(expdir + "Metrics")
proc.makedir(expdir + "Figures")

#%% Prepare White Noise timeseries


# Add Loop here for run ids
noisefile = "%sInput/whitenoise_%s_%s.npy" % (expdir,expname,expparams['runids'][0])
if len(glob.glob(noisefile)) > 0:
    print("White Noise file has been found! Loading...")
    wn = np.load(noisefile)
else:
    print("Generating new white noise file: %s" % noisefile)
    wn = np.random.normal(0,1,expparams['nyrs']*12)
    np.save(noisefile,wn)
    
#%% Do Conversions for Model Run



# Do Unit Conversions
Econvert = inputs['LHFLX'].copy() / (rho*L*inputs['h'])*dt*inputs['Sbar'] # [Mon x Lat x Lon]
Pconvert = inputs['PRECTOT']*dt


# Create Forcing (Up to here, check this)
EP_forcing = np.tile((Econvert + Pconvert).transpose(1,2,0),expparams['nyrs']).transpose(2,0,1) # [Time x Lat x Lon]
#EP_Forcing = wn[:,None,None] * 

# Calculate beta and kprev
beta       = scm.calc_beta(inputs['h'].transpose(2,1,0))
if expparams['kprev'] is None: # Compute Kprev
    print("Recalculating Kprev")
    kprev = np.zeros((12,nlat,nlon))
    for o in range(nlon):
        for a in range(nlat):
            kprevpt,_=scm.find_kprev(inputs['h'][:,a,o])
            kprev[:,a,o] = kprevpt.copy()
    inputs['kprev'] = kprev


# Set parameters, and transpose to [lon x lat x mon] for old script
smconfig = {}


smconfig['h']       = inputs['h'].transpose(2,1,0)           # Mixed Layer Depth in Meters [Lon x Lat x Mon]
smconfig['forcing'] = EP_forcing.transpose(2,1,0) # Forcing in psu/mon [Lon x Lat x Mon]
smconfig['lbd_a']   = inputs['lbd_a'].transpose(2,1,0) # 
smconfig['beta']    = beta # Entrainment Damping [1/mon]
smconfig['kprev']   = inputs['kprev'].transpose(2,1,0)
smconfig['lbd_d']   = inputs['lbd_d'].transpose(2,1,0)

if debug: #Just run at a point
    ivnames = list(smconfig.keys())
    [print(smconfig[iv].shape) for iv in ivnames]
    
    for iv in ivnames:
        smconfig[iv] = smconfig[iv][klon,klat,:].squeeze()[None,None,:]
    
    [print(smconfig[iv].shape) for iv in ivnames]

#%% Integrate the model

outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],
                                return_dict=True,old_index=True)


#%% Save the output

SSS_out  = outdict['T']
timedim  = xr.cftime_range(start="0001",periods=SSS_out.shape[-1],freq="MS",calendar="noleap")


cdict    = {
    "time" : timedim,
    "lat" : latr,
    "lon" : lonr,
    }

da       = xr.DataArray(SSS_out.transpose(2,1,0),coords=cdict,dims=cdict,name="SSS")
edict    = {"SSS":{"zlib":True}}
savename = "%sOutput/SSS_runid%s.nc" % (expdir,expparams['runids'][0])
da.to_netcdf(savename,encoding=edict)






# #%%  Debugging for above


# plt.plot(e[:,klat,klon]),plt.show()

# plt.plot(Econv[:,klat,klon]),plt.show()
# plt.plot(EP_forcing[:,klat,klon]),plt.show()


# #%% Crop to region


# xr.cftime_range(start="0001",periods=SSS_out.shape[-1],freq="MS",calendar="noleap")


# #%% Load Inputs



# #%%





# exp_params = {

#     }

#%% PCompare Point Output -----------
mons3 = proc.get_monstr(nletters=3)
locfn,loctitle=proc.make_locstring(330,50)

fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(6,8))


# Plot MLD
ax = axs.flatten()[0]
ax = viz.viz_kprev(inputs['h'][:,klat,klon],inputs['kprev'][:,klat,klon],ax=ax,usetitle=False,lw=2.5)
#ax.plot(mons3,inputs['h'][:,klat,klon],label="MLD",marker="o",lw=3.5)
ax.set_ylabel("MLD (meters)")
ax.legend()

# Plot Beta
ax = axs.flatten()[1]
ax.plot(mons3,beta[klon,klat,:],label="beta",marker="o",lw=3.5,color="darkblue")
ax.plot(mons3,inputs['lbd_d'][:,klat,klon],label="$\lambda^d$",marker="d",lw=3.5,color="limegreen")
ax.plot(mons3,inputs['lbd_a'][:,klat,klon],label="$\lambda^a$",marker="d",lw=3.5,color="violet")
ax.set_ylabel("Damping (1/mon)")
ax.legend()

ax = axs.flatten()[2]
ax.plot(mons3,Econvert[:,klat,klon],label="E'",marker="o",lw=3.5,color="orange")
ax.plot(mons3,Pconvert[:,klat,klon],label="P'",marker="o",lw=3.5,color="b")
ax.set_ylabel("Forcing (psu/mon)")
ax.legend()

plt.suptitle("SSS Input Parameters @ %s" % loctitle)
for ax in axs:
    ax.grid(True,ls='dotted')

plt.show()

# 
#%% Examine/compare for point output

# Load in CESM Output
ssspath= "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
sssnc  = "CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc"
dsc    = xr.open_dataset(ssspath+sssnc)
sss_cesm = dsc.sel(lon=-30,lat=50,method='nearest').SSS.load()#.values

# Deseason, Detrend
sss_anom = proc.xrdeseason(sss_cesm)
sss_dt   = sss_anom - sss_anom.mean('ensemble')
sss_cesm = sss_dt.values

sss_cesm[np.isnan(sss_cesm)] = 0

#%% Compute Metrics

sss_pt = SSS_out.squeeze()
tsm    = scm.compute_sm_metrics([sss_pt,sss_cesm.flatten()],)


#%% PLot ACF
kmonth = 1
lags = np.arange(37)
xtks = np.arange(0,38,3)

fig,ax = plt.subplots(1,1,constrained_layout=True)

ax,_=viz.init_acplot(kmonth,xtks,lags)
ax.plot(lags,tsm['acfs'][kmonth][0],label="SM")
ax.plot(lags,tsm['acfs'][kmonth][1],label="CESM")
ax.legend()
plt.show()

#%% Plot Monvar

fig,ax = viz.init_monplot(1,1)
ax.plot(mons3,tsm['monvars'][0],label="SM")
ax.plot(mons3,tsm['monvars'][1],label="CESM")
ax.set_title("Monthly Variance")
ax.set_ylabel("SST Variance ($\degree C^2$)")
ax.legend()
plt.show()

#%% Plot Timseries

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,4))
ax =axs[0]
ax.plot(sss_pt,label="SM")
ax.set_title("SM")
ax = axs[1]

for e in range(42):
    ax.plot(sss_cesm[e,:],label="",color='orange',alpha=0.2)
ax.set_title("CESM")
ax.legend()
plt.show()



#%%

plt.plot(tsm['acfs'][1][0]),plt.show()
plt.plot(tsm['monvars'][0]),plt.show()


#%% Briefly examine output

sss_pt = SSS_out[klon,klat]
tsm = scm.compute_sm_metrics([sss_pt,],)


plt.plot(SSS_out.squeeze()),plt.show()

plt.plot(Econvert[:,klat,klon]),plt.show()
plt.plot(Pconvert[:,klat,klon]),plt.show()



plt.plot(inputs['h'][:,klat,klon]),plt.show()
plt.plot(inputs['lbd_d'][:,klat,klon]),plt.show()
plt.plot(beta[klon,klat,:]),plt.show()

