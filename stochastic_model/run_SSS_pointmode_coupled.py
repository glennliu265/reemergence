#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Run a (single) stochastic model for both SST and SSS

SSS model experiment

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

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+"/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']


#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm

#%% 

"""
LHFLX Run (SST_SSS  Coupled, from early may prior to 2024.05.07)
"""

# Paths and Experiment
expname     = "SST_SSS_TdcorrFalse" # Borrowed from "SST_EOF_LbddCorr_Rerun"
expparams_sst   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SST_SSS_LHFLX", # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_LHFLX_damping_nomasklag1_EnsAvg.nc", # Only latent heat flux # "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc",#
    'Qek'               : None, # No Qekman #Must be in W/m2
    'convert_Fprime'    : True,
    'convert_lbd_a'     : True, # ALERT!! Need to rerun with this set to true....
    'convert_PRECTOT'   : False,
    'convert_LHFLX'     : False,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    "entrain"           : True,
    "eof_forcing"       : True,
    "Td_corr"           : False, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None,
    "Tforce"            : None,
    }

expparams_sss   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SST_SSS_LHFLX",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : None, # No Precip
    'LHFLX'             : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_SALT_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
    'Sbar'              : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : None, # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : None, # Must be in W/m2
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
    "Td_corr"           : False, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : "CESM1LE_HTR_FULL_lbde_Bcorr3_lbda_qnet_damping_nomasklag1_EnsAvg.nc",
    "Tforce"            : "SST_SSS_LHFLX",
    }

# """
# Another Version, but without lbd_d

# """


# expname     = "SST_SSS_LHFLX_NoLbdd" # Borrowed from "SST_EOF_LbddCorr_Rerun"
# expparams_sst   = {
#     'varname'           : "SST",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : None, # If not None, load a runid from another directory
#     'Fprime'            : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
#     'PRECTOT'           : None,
#     'LHFLX'             : None,
#     'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'             : False,
#     'Sbar'              : None,
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'             : "CESM1_HTR_FULL_LHFLX_damping_nomasklag1_EnsAvg.nc", # Only latent heat flux
#     'Qek'               : None, # No Qekman #Must be in W/m2
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
#     "Td_corr"           : False, # Set to True if lbd_d is provided as a correlation, rather than 1/months
#     "lbd_e"             : None,
#     "Tforce"            : None,
#     }

# expparams_sss   = {
#     'varname'           : "SSS",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : "SST_SSS_LHFLX",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
#     'Fprime'            : None,
#     'PRECTOT'           : None, # No Precip
#     'LHFLX'             : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
#     'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'             : False,
#     'Sbar'              : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'             : None, # NEEDS TO BE CONVERTED TO 1/Mon !!!
#     'Qek'               : None, # Must be in W/m2
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
#     "Td_corr"           : False, # Set to True if lbd_d is provided as a correlation, rather than 1/months
#     "lbd_e"             : "CESM1LE_HTR_FULL_lbde_Bcorr3_lbda_qnet_damping_nomasklag1_EnsAvg.nc",
#     "Tforce"            : "SST_SSS_LHFLX",
#     }



# Constants
dt    = 3600*24*30 # Timestep [s]
cp    = 3850       # 
rho   = 1026       # Density [kg/m3]
B     = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L     = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

debug = False

lonf  = -30
latf  = 50

#%%



# -----------------------
#%% Check and Load Params
# -----------------------
print("Loading inputs for %s" % expname)

exparams_in   = [expparams_sst,expparams_sss]
inputs_all    = []
inputs_types  = []
inputs_ds_all = []
for vv in range(2):
    
    expparams = exparams_in[vv]
    
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
        
        # Load DataArray
        if type(expparams[pname])==str: # If String, Load from input folder
            
            # Load ds
            if (expparams['varname'] == "SST") and (pname =="Fprime") and "Fprime" not in expname:
                
                da_varname   = "LHFLX" # Swap to LHFLX for now
                varname_swap = True # True so "Fprime" is input as da_varname later
                swapname     = "Fprime" 
                
            else:
                varname_swap = False
                
                #inputs_type['Fprime'] = 'forcing' # Add extra Fprime variable
            ds = xr.open_dataset(input_path + ptype + "/" + expparams[pname])[da_varname]
            
            
            # Crop to region
            
            # Load dataarrays for debugging
            dsreg            = proc.selpt_ds(ds,lonf,latf)#proc.sel_region_xr(ds,expparams['bbox_sim']).load()
            inputs_ds[pname] = dsreg.copy() 
            
            
            # Load to numpy arrays 
            varout           = dsreg.values
            varout           = varout[...,None,None]
            if debug:
                print(pname) # Name of variable
                print("\t%s" % str(ds.shape)) # Original Shape
                print("\t%s" % str(varout.shape)) # Point Shape
            inputs[pname]    = varout.copy()
            
            if ((da_varname == "Fprime") and (eof_flag)) or ("corrected" in expparams[pname]):
                print("Loading %s correction factor for EOF forcing..." % pname)
                ds_corr                          = xr.open_dataset(input_path + ptype + "/" + expparams[pname])['correction_factor']
                ds_corr_reg                      = proc.selpt_ds(ds_corr,lonf,latf).load()
                
                
                if varname_swap == True:
                    da_varname = pname # Swap from LHFLX back to Fprime for SST Integration
                    
                # set key based on variable type
                if da_varname == "Fprime":
                    keyname = "correction_factor"
                elif da_varname == "LHFLX":
                    keyname = "correction_factor_evap"
                elif da_varname == "PRECTOT":
                    keyname = "correction_factor_prec"
                
                inputs_ds[keyname]   = ds_corr_reg.copy()
                inputs[keyname]      = ds_corr_reg.values.copy()[...,None,None]
                if debug:
                    print(da_varname + " Corr") # Variable Name
                    print("\t%s" % str(ds_corr.shape))
                    print("\t%s" % str(inputs[keyname].shape)) # Corrected Shape
                inputs_type[keyname] = "forcing"
            
        else:
            print("Did not find %s" % pname)
            missing_input.append(pname)
        inputs_type[pname] = ptype
    
    inputs_types.append(inputs_type)
    inputs_all.append(inputs)
    inputs_ds_all.append(inputs_ds)

    #% Detect and Process Missing Inputs
    
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
    
    # ninputs = len(inputs)
    # lons    = []
    # lats    = []
    # for ni in range(ninputs):
        
    #     pname        = list(inputs.keys())[ni]
    #     #inputs_basin = inputs_[ni]
        
    #     lons.append(inputs_ds[pname].lon.values)
    #     lats.append(inputs_ds[pname].lat.values)
        
        #inputs_basin = inputs_all[vv][ni]
        
    # Narrow Down to Specific Point... (copied from run_stochmod_regionalavg)
    # #
    
    # #invar    = inputs_ds[pname] * mask # Load and apply a mask
    # #invar    = invar.transpose('lon','lat','mon')#.values

    # inputreg = []
    # for rr in range(nregs):
        
    #     bbsel  = bboxes[rr]
    #     regsel = proc.sel_region_xr(invar,bbsel)#.values
    #     ravg   = proc.area_avg_cosweight(regsel).values
    #     inputreg.append(ravg)
    # inputreg = np.array(inputreg)
    # if len(inputreg.shape) > 2:
    #     inputreg = inputreg.transpose(1,2,0)[...,None] # [mode x mon x reg x 1 ]
    # else:    
    #     inputreg = inputreg.T[...,None] # [mon x reg x 1]
    
    # print(pname)
    # print(inputreg.shape)
    # inputs_reg[pname] = inputreg.copy()

    
    
    

#%% For Debugging
dsreg =inputs_ds['h']
#latr = dsreg.lat.values
#lonr = dsreg.lon.values
#klon,klat=proc.find_latlon(-30,50,lonr,latr)
debug=True


#plt.plot((inputs_all[0]['Fprime']**2).sum(0).squeeze())
#plt.plot(inputs_all[0]['lbd_a'].squeeze())
#%%

#%% Initialize An Experiment folder for output

expdir = output_path + expname + "/"
proc.makedir(expdir + "Input")
proc.makedir(expdir + "Output")
proc.makedir(expdir + "Metrics")
proc.makedir(expdir + "Figures")

# Save the parameter file
for vv in range(2):
    savename = "%sexpparams_%s.npz" % (expdir+"Input/",exparams_in[vv]['varname'])
    np.savez(savename,**expparams)
    
    
    
# Make a function to simplify rolling
def roll_input(invar,rollback,halfmode=False,axis=0):
    rollvar = np.roll(invar,rollback,axis=axis)
    if halfmode:
        rollvar = (rollvar + invar)/2
    return rollvar

#%% First, Run for SST


for vv in range(2):
    inputs      = inputs_all[vv]
    inputs_type = inputs_types[vv]
    expparams   = exparams_in[vv]
    

    # Load out some parameters
    runids = expparams['runids']
    nruns  = len(runids)

    froll  = expparams['froll']
    droll  = expparams['droll']
    mroll  = expparams['mroll']
    
    for nr in range(nruns):
        
        #% Prepare White Noise timeseries ----------------------------------------
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
        
        #% Do Conversions for Model Run ------------------------------------------
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
                            QfactorE      = inputs['correction_factor_evap'].copy() * conversion_factor#.squeeze()
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
                if len(Qfactor.shape) < 4:    
                    Qfactor = Qfactor[None,...]
                alpha = np.concatenate([alpha,Qfactor],axis=0)
                
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
        
        # if debug: #Just run at a point
        #     ivnames = list(smconfig.keys())
        #     #[print(smconfig[iv].shape) for iv in ivnames]
            
        #     for iv in ivnames:
        #         smconfig[iv] = smconfig[iv][klon,klat,:].squeeze()[None,None,:]
            
            #[print(smconfig[iv].shape) for iv in ivnames]
        
        #  ------------------------------------------------------------------------
        
        #% Integrate the model
        if expparams['entrain'] is True:
            outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                            Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],
                                            return_dict=True,old_index=True,add_F=smconfig['add_F'])
        else:
            outdict = scm.integrate_noentrain(smconfig['lbd_a'],smconfig['forcing'],T0=0,multFAC=True,debug=True,old_index=True,return_dict=True)
            
        
        
        #% Save the output
        var_out  = outdict['T']
        timedim  = xr.cftime_range(start="0001",periods=var_out.shape[-1],freq="MS",calendar="noleap")
        cdict    = {
            "time" : timedim,
            "lat" : [latf,],
            "lon" : [lonf,],
            }
    
        da       = xr.DataArray(var_out.transpose(2,1,0),coords=cdict,dims=cdict,name=expparams['varname'])
        edict    = {expparams['varname']:{"zlib":True}}
        savename = "%sOutput/%s_runid%s.nc" % (expdir,expparams['varname'],runid)
        da.to_netcdf(savename,encoding=edict)


#%% Information ----


