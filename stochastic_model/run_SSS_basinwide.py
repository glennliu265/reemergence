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

expname     = "SSS_EOF_LbddCorr_Rerun_lbdE"

expparams   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SST_EOF_LbddCorr_Rerun",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'LHFLX'             : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
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
    "Td_corr"           : True,
    "lbd_e"             : "CESM1LE_HTR_FULL_lbde_Bcorr3_lbda_qnet_damping_nomasklag1_EnsAvg.nc",
    "Tforce"            : "SST_EOF_LbddCorr_Rerun"
    }

# expname     = "SST_EOF_LbddCorr_Rerun"

# expparams   = {
#     'varname'           : "SST",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : None, # If not None, load a runid from another directory
#     'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_corrected_nomasklag1_nroll0_perc090_NAtl_EnsAvg.nc",
#     'PRECTOT'           : None,
#     'LHFLX'             : None,
#     'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'             : "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
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
#     "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
#     }

# expname     = "SSS_EOF_NoLbdd"

# expparams   = {
#     'varname'           : "SSS",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(5,10,1)],
#     'runid_path'        : "SST_EOF_LbddEnsMean",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
#     'Fprime'            : None,
#     'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
#     'LHFLX'             : "CESM1_HTR_FULL_LHFLX_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
#     'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'             : None,
#     'Sbar'              : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'             : None, # NEEDS TO BE CONVERTED TO 1/Mon !!!
#     'Qek'               : "CESM1_HTR_FULL_Qek_SSS_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc", # Must be in W/m2
#     'convert_Fprime'    : False,
#     'convert_lbd_a'     : False,
#     'convert_PRECTOT'   : True,
#     'convert_LHFLX'     : True,
#     'froll'             : 0,
#     'mroll'             : 0,
#     'droll'             : 0,
#     'halfmode'          : False,
#     "entrain"           : True,
#     "eof_forcing"       : True,
#     }

# expname     = "SST_EOF_LbddEnsMean"

# expparams   = {
#     'varname'           : "SST",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : None, # If not None, load a runid from another directory
#     'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_corrected_nomasklag1_nroll0_perc090_NAtl_EnsAvg.nc",
#     'PRECTOT'           : None,
#     'LHFLX'             : None,
#     'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'             : "CESM1_HTR_FULL_SST_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAvg.nc",
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

#%% Check and Load Params

print("Loading inputs for %s" % expname)

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
    
    #print(pname)
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
            print("Loading %s correction factor for EOF forcing..." % pname)
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
        print("Did not find %s" % pname)
        missing_input.append(pname)
    inputs_type[pname] = ptype

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
chk = proc.checkfile(savename)
if chk is False:
    print("Saving Parameter Dictionary...")
    np.savez(savename,**expparams,allow_pickle=True)

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
                
                conversion_factor = ( dt*inputs['Sbar'] / inputs['h'] )
                
                if eof_flag:
                    Pconvert =  inputs['PRECTOT'] * conversion_factor[None,...]
                else:
                    Pconvert =  inputs['PRECTOT'] * conversion_factor
                
                if (eof_flag) and ('correction_factor_prec' in list(inputs.keys())):
                    print("Processing Precip Correction factor")
                    QfactorP   = inputs['correction_factor_prec'] * conversion_factor
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

            alpha = np.concatenate([alpha,Qfactor[None,...]],axis=0)
            
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
        smconfig['Td_corr'] = expparams['Td_corr']
        
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
    
    # New Section: Check for SST-Evaporation Feedback ------------------------
    smconfig['add_F'] = None
    if 'lbd_e' in expparams.keys() and expparams['varname'] == "SSS":
        if expparams['lbd_e'] is not None: 
            print("Adding SST-Evaporation Forcing on SSS!")
            # Load lbd_e
            lbd_e = xr.open_dataset(input_path + "forcing/" + expparams['lbd_e']).lbd_e.load() # [mon x lat x lon]
            lbd_e = proc.sel_region_xr(lbd_e,bbox=expparams['bbox_sim'])
            
            # Convert [sec --> mon]
            lbd_emon = lbd_e * dt
            lbd_emon = lbd_emon.transpose('lon','lat','mon').values
            
            # Load temperature timeseries
            assert expparams['Tforce'] is not None,"Experiment for SST timeseries [Tforce] must be specified"
            sst_nc = "%s%s/Output/SST_runid%s.nc" % (output_path,expparams['Tforce'],runid)
            sst_in = xr.open_dataset(sst_nc).SST.load()
            sst_in = sst_in.transpose('lon','lat','time').values
            
            # Tile and combine
            lbd_emon_tile     = np.tile(lbd_emon,nyr) #
            lbdeT             = lbd_emon_tile * sst_in
            smconfig['add_F'] = lbdeT
    
    if debug: #Just run at a point
        ivnames = list(smconfig.keys())
        [print(smconfig[iv].shape) for iv in ivnames]
        
        for iv in ivnames:
            smconfig[iv] = smconfig[iv][klon,klat,:].squeeze()[None,None,:]
        
        [print(smconfig[iv].shape) for iv in ivnames]
    #  ------------------------------------------------------------------------
    
    #%% Integrate the model
    if expparams['entrain'] is True:
        outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                        Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],add_F=smconfig['add_F'],
                                        return_dict=True,old_index=True,Td_corr=smconfig['Td_corr'])
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


#%% Information ----


