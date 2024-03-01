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

#%% 

"""
SSS_EOF_Qek_Pilot

Note: The original run (2/14) had the incorrect Ekman Forcing and used ens01 detrainment damping with linear detrend
I reran this after fixing these issues (2/29)

"""

# Paths and Experiment
input_path  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/"
output_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"

expname     = "SSS_EOF_Qek_pilot_corrected"

expparams   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,5,1)],
    'runid_path'        : None,#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'LHFLX'             : "CESM1_HTR_FULL_LHFLX_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_SSS_Expfit_lbdd_monvar_detrendensmean_lagmax3_Ens01.nc",
    'Sbar'              : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : None, # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : "CESM1_HTR_FULL_Qek_SSS_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc", # Must be in W/m2
    'convert_Fprime'    : False,
    'convert_lbd_a'     : False,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    "entrain"           : True,
    "eof_forcing"       : True,
    }

# expname     = "SST_EOF_Qek_pilot"

# expparams   = {
#     'varname'           : "SST",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,5,1)],
#     'runid_path'        : None, # If not None, load a runid from another directory
#     'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_corrected_nomasklag1_nroll0_perc090_NAtl_EnsAvg.nc",
#     'PRECTOT'           : None,
#     'LHFLX'             : None,
#     'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'             : "CESM1_HTR_FULL_SST_Expfit_lbdd_monvar_detrendensmean_lagmax3_Ens01.nc",
#     'Sbar'              : None,
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
#     'Qek'               : "CESM1_HTR_FULL_Qek_SST_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc", # Must be in W/m2
#     'convert_Fprime'    : True,
#     'convert_lbd_a'     : True, # ALERT!! Need to rerun with this set to true....
#     'convert_PRECTOT'   : False,
#     'convert_LHFLX'     : False,
#     'froll'             : 0,
#     'mroll'             : 0,
#     'droll'             : 0,
#     'halfmode'          : False,
#     "entrain"           : True,
#     "eof_forcing"       : True,
#     }

# Constants
dt    = 3600*24*30 # Timestep [s]
cp    = 3850       # 
rho   = 1026       # Density [kg/m3]
B     = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L     = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

debug = False

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

# First, Check if there is EOF-based forcing (remove this if I eventually redo it)
if expparams['eof_forcing']:
    print("EOF Forcing Detected.")
    eof_flag = True
else:
    eof_flag = False

# Indicate the Parameter Names (sorry, it's all hard coded...)
if expparams['varname']== "SSS": # Check for LHFLX, PRECTOT, Sbar
    chk_params = ["h","LHFLX","PRECTOT","Sbar","lbd_d","beta","kprev","lbd_a","Qek"]
    param_type = ["mld","forcing","forcing","forcing","damping","mld","mld","damping",'forcing']
elif expparams['varname'] == "SST": # Check for Fprime
    chk_params = ["h","Fprime","lbd_d","beta","kprev","lbd_a","Qek"]
    param_type = ["mld","forcing","damping","mld","mld","damping",'forcing']

# Check the params
ninputs       = len(chk_params)
inputs_ds     = {}
inputs        = {}
inputs_type   = {}
missing_input = []
for nn in range(ninputs):
    # Get Parameter Name and Type
    pname = chk_params[nn]
    ptype = param_type[nn]
    
    # Check for Exceptions (Can Fix this in the preprocessing stage)
    if pname == 'lbd_a':
        da_varname = 'damping'
    else:
        da_varname = pname
    
    print(pname)
    if type(expparams[pname])==str: # If String, Load from input folder
        
        # Load ds
        ds = xr.open_dataset(input_path + ptype + "/" + expparams[pname])[da_varname]
        

        # Crop to region
        
        # Load dataarrays for debugging
        dsreg            = proc.sel_region_xr(ds,expparams['bbox_sim']).load()
        inputs_ds[pname] = dsreg.copy() 
        
        # Load to numpy arrays 
        varout           = dsreg.values
        inputs[pname]    = dsreg.values.copy()
        
        if ((da_varname == "Fprime") and (eof_flag)) or ("corrected" in expparams[pname]):
            print("Loading correction factor for EOF forcing...")
            ds_corr                          = xr.open_dataset(input_path + ptype + "/" + expparams[pname])['correction_factor']
            ds_corr_reg                      = proc.sel_region_xr(ds_corr,expparams['bbox_sim']).load()
            
            # set key based on variable type
            if da_varname == "Fprime":
                keyname = "correction_factor"
            elif da_varname == "LHFLX":
                keyname = "correction_factor_evap"
            elif da_varname == "PRECTOT":
                keyname = "correction_factor_prec"
                
            inputs_ds[keyname]   = ds_corr_reg.copy()
            inputs[keyname]      = ds_corr_reg.values.copy()
            inputs_type[keyname] = "forcing"
        
    else:
        missing_input.append(pname)
    inputs_type[pname] = ptype

    # elif type(expparams[pname])==float:
    #     # Make empty data_array, multiplied by the given value
    #     print("For <%s> Making Empty DataArray with the repeated value %f" % (pname,expparams[pname]))
    #     ds = xr.ones_like(inputs[nn-1]).rename(pname) * expparams[pname]
    # else:
    #     missing_input.append(pname)
# Crop to Region
#varcrop     = [proc.sel_region_xr(ds,expparams['bbox_sim']).load().values for ds in inputs] 

#%% Detect and Process Missing Inputs

_,nlat,nlon=inputs['h'].shape

for pname in missing_input:
    if type(expparams[pname]) == float:
        print("Float detected for <%s>. Making array with the repeated value %f" % (pname,expparams[pname]))
        inputs[pname] = np.ones((12,nlat,nlon)) * expparams[pname]
    else:
        print("No value found for <%s>. Setting to zero." % pname)
        inputs[pname] = np.zeros((12,nlat,nlon))

# Get number of modes
if eof_flag:
    if expparams['varname'] == "SST":
        nmode = inputs['Fprime'].shape[0]
    elif expparams['varname'] == "SSS":
        nmode = inputs['LHFLX'].shape[0]
    
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

# Save the parameter file
savename = "%sexpparams.npz" % (expdir+"Input/")
np.savez(savename,**expparams)

# Load out some parameters
runids = expparams['runids']
nruns  = len(runids)

froll = expparams['froll']
droll = expparams['droll']
mroll = expparams['mroll']

# Make a function to simplify rolling
def roll_input(invar,rollback,halfmode=False,axis=0):
    rollvar = np.roll(invar,rollback,axis=axis)
    if halfmode:
        rollvar = (rollvar + invar)/2
    return rollvar
    
for nr in range(nruns):
    
    #%% Prepare White Noise timeseries ----------------------------------------
    runid = runids[nr]
    
    # Check if specific path was indicated, and set filename accordingly
    if expparams['runid_path'] is None:
        noisefile = "%sInput/whitenoise_%s_%s.npy" % (expdir,expname,runid)
    else:
        expname_runid = expparams['runid_path'] # Name of experiment to take runid from
        print("Searching for runid path in specified experiment folder: %s" % expname_runid)
        noisefile     = "%sInput/whitenoise_%s_%s.npy" % (output_path + expname_runid + "/",expname_runid,runid)
    
    # Generate or reload white noise
    if len(glob.glob(noisefile)) > 0:
        print("White Noise file has been found! Loading...")
        wn = np.load(noisefile)
    else:
        print("Generating new white noise file: %s" % noisefile)
        noise_size = [expparams['nyrs'],12,]
        if eof_flag: # Generate white noise for each mode
            nmodes_plus1 = nmode + 1 
            print("Detected EOF Forcing. Generating %i white noise timeseries" % (nmodes_plus1))
            noise_size   = noise_size + [nmodes_plus1]
        
        wn = np.random.normal(0,1,noise_size) # [Yr x Mon x Mode]
        np.save(noisefile,wn)
        
    #%% Do Conversions for Model Run ------------------------------------------
    if nr == 0: # Only perform this once
        
        # Apply roll/shift to seasonal cycle
        ninputs = len(inputs)
        for ni in range(ninputs):
            
            pname = list(inputs.keys())[ni]
            ptype = inputs_type[pname]
            
            if ptype == "mld":
                rollback = mroll
            elif ptype == "forcing":
                rollback = froll
            elif ptype == "damping":
                rollback = droll
            else:
                print("Warning, Parameter Type not Identified. No roll performed.")
                rollback = 0
            
            if rollback != 0:
                print("Rolling %s back by %i" % (pname,rollback))
                
                if eof_flag and len(inputs[pname].shape) > 3:
                    rollaxis=1 # Switch to 1st dim to avoid mode dimension
                else:
                    rollaxis=0
                inputs[pname] = roll_input(inputs[pname],rollback,axis=rollaxis,halfmode=expparams['halfmode'])
        
        # Do Unit Conversions ---
        if expparams["varname"] == "SSS": # Convert to psu/mon
            
            # Evap Forcing
            if expparams['convert_LHFLX']: 
                if eof_flag: # Check for EOFs
                    conversion_factor = ( dt*inputs['Sbar'] / (rho*L*inputs['h']))[None,...]
                    Econvert          = inputs['LHFLX'].copy() * conversion_factor # [Mon x Lat x Lon]
                    
                    # Add Correction Factor, if it exists
                    if 'correction_factor_evap' in list(inputs.keys()):
                        print("Processing LHFLX/Evaporation Correction factor")
                        QfactorE      = inputs['correction_factor_evap'].copy() * conversion_factor.squeeze()
                    else:
                        QfactorE      = np.zeros((inputs['LHFLX'].shape[1:])) # Same Shape minus the mode dimension
                else:
                    Econvert          = inputs['LHFLX'].copy() / (rho*L*inputs['h'])*dt*inputs['Sbar'] # [Mon x Lat x Lon]
            else:
                Econvert   = inputs['LHFLX'].copy()
            
            # Precip Forcing
            if expparams['convert_PRECTOT']:
                
                Pconvert   = inputs['PRECTOT']*dt
                
                if (eof_flag) and ('correction_factor_prec' in list(inputs.keys())):
                    print("Processing Precip Correction factor")
                    QfactorP   = inputs['correction_factor_prec'] * dt
                else:
                    QfactorP   = np.zeros((inputs['PRECTOT'].shape[1:])) # Same Shape minus the mode dimension
            else:
                Pconvert   = inputs['PRECTOT'].copy()
            
            # Atmospheric Damping
            if expparams['convert_lbd_a']:
                print("WARNING: lbd_a unit conversion for SSS currently not supported")
                Dconvert = inputs['lbd_a'].copy()
            else:
                Dconvert = inputs['lbd_a'].copy()
                if np.nansum(Dconvert) < 0:
                    print("Flipping Sign")
                    Dconvert *= -1
            
            # Add Ekman Forcing, if it Exists (should be zero otherwise)
            Qekconvert = inputs['Qek'].copy()  * dt #  [(mode) x Mon x Lat x Lon]
            
            # Corrrection Factor
            if eof_flag:
                Qfactor    = QfactorE + QfactorP # Combine Evap and Precip correction factor
            
            # Combine Evap and Precip (and Ekman Forcing)
            alpha         = Econvert + Pconvert + Qekconvert
            
        elif expparams['varname'] == "SST": # Convert to degC/mon
            
            # Convert Stochastic Heat Flux Forcing
            if expparams['convert_Fprime']:
                if eof_flag:
                    Fconvert   = inputs['Fprime'].copy()           / (rho*cp*inputs['h'])[None,:,:,:] * dt # Broadcast to mode x mon x lat x lon
                    # Also convert correction factor
                    Qfactor    = inputs['correction_factor'].copy()/ (rho*cp*inputs['h'])[:,:,:] * dt
                    
                else:
                    Fconvert   = inputs['Fprime'].copy() / (rho*cp*inputs['h']) * dt
            else:
                Fconvert   = inputs['Fprime'].copy()
            
            # Convert Atmospheric Damping
            if expparams['convert_lbd_a']:
                
                Dconvert   = inputs['lbd_a'].copy() / (rho*cp*inputs['h']) * dt
            else:
                Dconvert   = inputs['lbd_a'].copy()
                if np.nansum(Dconvert) < 0:
                    print("Flipping Sign")
                    Dconvert *= -1
            
            # Add Ekman Forcing, if it Exists (should be zero otherwise)
            if eof_flag:
                Qekconvert = inputs['Qek'].copy() / (rho*cp*inputs['h'])[None,:,:,:] * dt
            else:
                Qekconvert = inputs['Qek'].copy() / (rho*cp*inputs['h']) * dt
            
            # Compute forcing amplitude
            alpha = Fconvert + Qekconvert
            # <End Variable Conversion Check>
        
        # Tile Forcing (need to move time dimension to the back)
        if eof_flag: # Append Qfactor as an extra mode
            # alpha_tile=[]
            # for ii in tqdm.tqdm(range(nmode)): # Tile for each mode
            #     inalphatile = np.tile(alpha[ii,:,:,:].transpose(1,2,0),expparams['nyrs']).transpose(2,0,1) 
                
            #     alpha_tile.append(inalphatile)
            # corr_tile  = np.tile(Qfactor.transpose(1,2,0),expparams['nyrs']).transpose(2,0,1) 
            # alpha_tile = np.array(alpha_tile) # Note this takes forever....
            # # Combine them This also takes forver
            # alpha_tile = np.concatenate([alpha_tile,corr_tile[None,...]],axis=0)
            alpha = np.concatenate([alpha,Qfactor[None,...]],axis=0)
        #else:
            #alpha_tile = np.tile((alpha).transpose(1,2,0),expparams['nyrs']).transpose(2,0,1) # [Time x Lat x Lon]
        
        # Calculate beta and kprev
        beta       = scm.calc_beta(inputs['h'].transpose(2,1,0)) # {lon x lat x time}
        if expparams['kprev'] is None: # Compute Kprev if it is not supplied
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
        smconfig['lbd_a']   = Dconvert.transpose(2,1,0) # 
        smconfig['beta']    = beta # Entrainment Damping [1/mon]
        smconfig['kprev']   = inputs['kprev'].transpose(2,1,0)
        smconfig['lbd_d']   = inputs['lbd_d'].transpose(2,1,0)
        
    # Use different white noise for each runid
    #wn_tile = wn.reshape()
    if eof_flag:
        forcing_in = (wn.transpose(2,0,1)[:,:,:,None,None] * alpha[:,None,:,:,:]) # [mode x yr x mon x lat x lon]
        forcing_in = np.nansum(forcing_in,0) # Sum over modes
        #forcing_in = np.nansum(wn.T[:,:,None,None] * alpha_tile,0) # Multiple then sum the tiles
    else:
        forcing_in  = wn.T[:,:,None,None] * alpha[None,:,:,:]
        #forcing_in = wn[:,None,None] * alpha_tile
    nyr,_,nlat,nlon = forcing_in.shape
    forcing_in      = forcing_in.reshape(nyr*12,nlat,nlon)
    smconfig['forcing'] = forcing_in.transpose(2,1,0) # Forcing in psu/mon [Lon x Lat x Mon]
    
    if debug: #Just run at a point
        ivnames = list(smconfig.keys())
        [print(smconfig[iv].shape) for iv in ivnames]
        
        for iv in ivnames:
            smconfig[iv] = smconfig[iv][klon,klat,:].squeeze()[None,None,:]
        
        [print(smconfig[iv].shape) for iv in ivnames]
    
    #%% Integrate the model
    if expparams['entrain'] is True:
        outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                        Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],
                                        return_dict=True,old_index=True)
    else:
        outdict = scm.integrate_noentrain(smconfig['lbd_a'],smconfig['forcing'],T0=0,multFAC=True,debug=True,old_index=True,return_dict=True)
        
    #%% Save the output
    if debug:
        ts = outdict['T'].squeeze()
        plt.plot(ts),plt.show()
    else:
        var_out  = outdict['T']
        timedim  = xr.cftime_range(start="0001",periods=var_out.shape[-1],freq="MS",calendar="noleap")
        cdict    = {
            "time" : timedim,
            "lat" : latr,
            "lon" : lonr,
            }
    
        da       = xr.DataArray(var_out.transpose(2,1,0),coords=cdict,dims=cdict,name=expparams['varname'])
        edict    = {expparams['varname']:{"zlib":True}}
        savename = "%sOutput/%s_runid%s.nc" % (expdir,expparams['varname'],runid)
        da.to_netcdf(savename,encoding=edict)
    



#%%


# def ssr(v,dim):
#     return np.sqrt(np.sum(v**2,dim))



# Mean Stdev (in time) of all points
# Qekconvert: 0.008140762034669971
# Econvert  : 0.004727825910126515
# Pconvert  : 0.009327264991185927
# alpha     : 0.016826414463990486
# forcing_in : 1.0124996310252043


# # #%%  Debugging for above


# # plt.plot(e[:,klat,klon]),plt.show()

# # plt.plot(Econv[:,klat,klon]),plt.show()
# # plt.plot(EP_forcing[:,klat,klon]),plt.show()


# # #%% Crop to region


# # xr.cftime_range(start="0001",periods=SSS_out.shape[-1],freq="MS",calendar="noleap")


# # #%% Load Inputs



# # #%%





# # exp_params = {

# #     }

# #%% PCompare Point Output -----------
# mons3 = proc.get_monstr(nletters=3)
# locfn,loctitle=proc.make_locstring(330,50)

# fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(6,8))


# # Plot MLD
# ax = axs.flatten()[0]
# ax = viz.viz_kprev(inputs['h'][:,klat,klon],inputs['kprev'][:,klat,klon],ax=ax,usetitle=False,lw=2.5)
# #ax.plot(mons3,inputs['h'][:,klat,klon],label="MLD",marker="o",lw=3.5)
# ax.set_ylabel("MLD (meters)")
# ax.legend()

# # Plot Beta
# ax = axs.flatten()[1]
# ax.plot(mons3,beta[klon,klat,:],label="beta",marker="o",lw=3.5,color="darkblue")
# ax.plot(mons3,inputs['lbd_d'][:,klat,klon],label="$\lambda^d$",marker="d",lw=3.5,color="limegreen")
# ax.plot(mons3,inputs['lbd_a'][:,klat,klon],label="$\lambda^a$",marker="d",lw=3.5,color="violet")
# ax.set_ylabel("Damping (1/mon)")
# ax.legend()

# ax = axs.flatten()[2]
# ax.plot(mons3,Econvert[:,klat,klon],label="E'",marker="o",lw=3.5,color="orange")
# ax.plot(mons3,Pconvert[:,klat,klon],label="P'",marker="o",lw=3.5,color="b")
# ax.set_ylabel("Forcing (psu/mon)")
# ax.legend()

# plt.suptitle("SSS Input Parameters @ %s" % loctitle)
# for ax in axs:
#     ax.grid(True,ls='dotted')

# plt.show()

# # 
# #%% Examine/compare for point output

# # Load in CESM Output
# ssspath= "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
# sssnc  = "CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc"
# dsc    = xr.open_dataset(ssspath+sssnc)
# sss_cesm = dsc.sel(lon=-30,lat=50,method='nearest').SSS.load()#.values

# # Deseason, Detrend
# sss_anom = proc.xrdeseason(sss_cesm)
# sss_dt   = sss_anom - sss_anom.mean('ensemble')
# sss_cesm = sss_dt.values

# sss_cesm[np.isnan(sss_cesm)] = 0

# #%% Compute Metrics

# sss_pt = SSS_out.squeeze()
# tsm    = scm.compute_sm_metrics([sss_pt,sss_cesm.flatten()],)


# #%% PLot ACF
# kmonth = 1
# lags = np.arange(37)
# xtks = np.arange(0,38,3)

# fig,ax = plt.subplots(1,1,constrained_layout=True)

# ax,_=viz.init_acplot(kmonth,xtks,lags)
# ax.plot(lags,tsm['acfs'][kmonth][0],label="SM")
# ax.plot(lags,tsm['acfs'][kmonth][1],label="CESM")
# ax.legend()
# plt.show()

# #%% Plot Monvar

# fig,ax = viz.init_monplot(1,1)
# ax.plot(mons3,tsm['monvars'][0],label="SM")
# ax.plot(mons3,tsm['monvars'][1],label="CESM")
# ax.set_title("Monthly Variance")
# ax.set_ylabel("SST Variance ($\degree C^2$)")
# ax.legend()
# plt.show()

# #%% Plot Timseries

# fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,4))
# ax =axs[0]
# ax.plot(sss_pt[1800:],label="SM")
# ax.set_title("SM")
# ax = axs[1]
# ax.set_ylim([1800,1824])

# for e in range(42):
#     ax.plot(sss_cesm[e,:],label="",color='orange',alpha=0.2)
# ax.set_title("CESM")
# ax.legend()
# plt.show()

# #%%p

# plt.plot(EP_forcing[:,klat,klon])
# ax.set_ylim([1800,1824])
# plt.show()


# #%%

# plt.plot(tsm['acfs'][1][0]),plt.show()
# plt.plot(tsm['monvars'][0]),plt.show()


# #%% Briefly examine output

# sss_pt = SSS_out[klon,klat]
# tsm = scm.compute_sm_metrics([sss_pt,],)


# plt.plot(SSS_out.squeeze()),plt.show()

# plt.plot(Econvert[:,klat,klon]),plt.show()
# plt.plot(Pconvert[:,klat,klon]),plt.show()



# plt.plot(inputs['h'][:,klat,klon]),plt.show()
# plt.plot(inputs['lbd_d'][:,klat,klon]),plt.show()
# plt.plot(beta[klon,klat,:]),plt.show()

