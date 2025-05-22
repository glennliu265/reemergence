#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Reconstruct/Recalculate Qnet from a stochastic model simulation


Procedure
    1. Load in Parameters (F', Lambda^a)
    2. Load in White Noise Timeseries
    3. Looping for each white noise segment/run
        a. Load in the Temperature
        b. Replicate/tile each parameter to simulation length
        c. Lag the temperature anomalies
        d. Combine and save...

Copied upper section of 

Created on Wed May 21 11:58:35 2025

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
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "stormtrack"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import amv.loaders as dl
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
# procpath    = pathdict['procpath']
# figpath     = pathdict['figpath']
# proc.makedir(figpath)

#%% 

"""
Paste Experiment Parameters Below (see basinwide_experiment_params.py)

"""

expname     = "SST_Obs_Pilot_00_Tdcorr0_qnet_noPositive"
expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : 'SST_Obs_Pilot_00_Tdcorr0_qnet', # If not None, load a runid from another directory
    'Fprime'            : "ERA5_Fprime_QNET_std_pilot.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "MIMOC_regridERA5_h_pilot.nc",
    'lbd_d'             : None,
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "MIMOC_regridERA5_kprev_pilot.nc",
    'lbd_a'             : "ERA5_qnet_damping_noPositive.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : None, # Now in degC/sec
    'convert_Fprime'    : True,
    'convert_lbd_a'     : True, 
    'convert_PRECTOT'   : False,
    'convert_LHFLX'     : False,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    "entrain"           : True,
    "eof_forcing"       : False, # CHECK THIS
    "Td_corr"           : False, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None, # Relevant for SSS
    "Tforce"            : None, # Relevant for SSS
    "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    }


# expname     = "SST_Obs_Pilot_00_Tdcorr1_qnet_noPositive_SPGNE"
# expparams   = {
#     'varname'           : "SST",
#     'bbox_sim'          : [-40,-15,52,62],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : 'SST_Obs_Pilot_00_Tdcorr0_qnet', # If not None, load a runid from another directory
#     'Fprime'            : "ERA5_Fprime_QNET_std_pilot.nc",
#     'PRECTOT'           : None,
#     'LHFLX'             : None,
#     'h'                 : "MIMOC_regridERA5_h_pilot.nc",
#     'lbd_d'             : None,
#     'Sbar'              : None,
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "MIMOC_regridERA5_kprev_pilot.nc",
#     'lbd_a'             : "ERA5_qnet_damping_noPositive.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
#     'Qek'               : None, # Now in degC/sec
#     'convert_Fprime'    : True,
#     'convert_lbd_a'     : True, 
#     'convert_PRECTOT'   : False,
#     'convert_LHFLX'     : False,
#     'froll'             : 0,
#     'mroll'             : 0,
#     'droll'             : 0,
#     'halfmode'          : False,
#     "entrain"           : True,
#     "eof_forcing"       : False, # CHECK THIS
#     "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
#     "lbd_e"             : None, # Relevant for SSS
#     "Tforce"            : None, # Relevant for SSS
#     "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
#     "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
#     }



# -----------------------------------------------------------------------------
# Begin copy of run_SSS_basinwide

#%% Other Constants

# Constants
dt    = 3600*24*30 # Timestep [s]
cp    = 3850       # 
rho   = 1026       # Density [kg/m3]
B     = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L     = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

debug = False

print("==========================")
print("Now Running Experiment: %s" % expname)
print("==========================")

#%% Check and Load Params

print("\tLoading inputs for %s" % expname)

# Apply patch to expdict
expparams = scm.patch_expparams(expparams)

# First, Check if there is EOF-based forcing (remove this if I eventually redo it)
if expparams['eof_forcing']:
    print("\t\tEOF Forcing Detected.")
    eof_flag = True
else:
    print("\t\tEOF Forcing will not be used.")
    eof_flag = False

inputs,inputs_ds,inputs_type,params_vv = scm.load_params(expparams,input_path)

#%% Detect and Process Missing Inputs

_,nlat,nlon=inputs['h'].shape

# Get number of modes
if eof_flag:
    if expparams['varname'] == "SST":
        nmode = inputs['Fprime'].shape[0]
    elif expparams['varname'] == "SSS":
        nmode = inputs['LHFLX'].shape[0]

#%% For Debugging

lonf = -50
latf = 40
dsreg =inputs_ds['h']
latr = dsreg.lat.values
lonr = dsreg.lon.values
klon,klat=proc.find_latlon(lonf,latf,lonr,latr)

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


def qfactor_noisemaker(expparams,expdir,expname,runid,share_noise=False):
    # Checks for separate wn timeseries for each correction factor
    # and loads the dictionary
    # If share noise is true, the same wn timeseries is used for:
    #     - Fprime and Evaporation
    #     - Qek forcing
    #     - Precipe
    
    # Makes (and checks for) additional white noise timeseries for the following (6) correction factors
    forcing_names = ("correction_factor",           # Fprime
                     "correction_factor_Qek_SST",   # Qek_SST
                     "correction_factor_evap",      # Evaporation
                     "correction_factor_prec",      # Precip
                     "correction_factor_Qek_SSS",   # Qek_SSS
                     )
    nforcings     = len(forcing_names)
    
    # Check for correction file
    noisefile_corr = "%sInput/whitenoise_%s_%s_corrections.npz" % (expdir,expname,runid)
    
    # Generate or reload white noise
    if len(glob.glob(noisefile_corr)) > 0:
        
        print("\t\tWhite Noise correction factor file has been found! Loading...")
        wn_corr = np.load(noisefile_corr)
        
        if share_noise:
            print("Checking for shared noise...")
            if wn_corr['correction_factor'] != wn_corr['correction_factor_evap']:
                print("\tSetting F' and E' white noise to be the same")
                wn_corr['correction_factor_evap'] = wn_corr['correction_factor']
            if wn_corr['correction_factor_Qek_SSS'] != wn_corr['correction_factor_Qek_SST']:
                print("\tSetting Qek white noise to be the same")
                wn_corr['correction_factor_Qek_SSS'] = wn_corr['correction_factor_Qek_SST']
                
        
    else:
        
        print("\t\tGenerating %i new white noise timeseries: %s" % (nforcings,noisefile_corr))
        noise_size  = [expparams['nyrs'],12,]
        
        wn_corr_out = {}
        for nn in range(nforcings):
            
            if (forcing_names[nn] == "correction_factor_evap") and share_noise:
                print("Copying same white noise timeseries as F' for E'")
                wn_corr_out[forcing_names[nn]] = wn_corr_out['correction_factor']
            elif (forcing_names[nn] == "correction_factor_Qek_SSS") and share_noise:
                print("Copying same white noise timeseries as Qek SST for Qek SSS")
                wn_corr_out[forcing_names[nn]] = wn_corr_out['correction_factor_Qek_SST']
            else: # Make a new noise timeseries
                wn_corr_out[forcing_names[nn]] = np.random.normal(0,1,noise_size) # [Yr x Mon x Mode]
        
        np.savez(noisefile_corr,**wn_corr_out,allow_pickle=True)
        wn_corr = wn_corr_out.copy()
        
    return wn_corr

#%%  
for nr in range(nruns):
    
    st = time.time()
    
    #%% Prepare White Noise timeseries ----------------------------------------
    runid = runids[nr]
    print("\tPreparing forcing...")
    
    # Check if specific path was indicated, and set filename accordingly
    if expparams['runid_path'] is None:
        noisefile = "%sInput/whitenoise_%s_%s.npy" % (expdir,expname,runid)
    else:
        expname_runid = expparams['runid_path'] # Name of experiment to take runid from
        print("\t\tSearching for runid path in specified experiment folder: %s" % expname_runid)
        noisefile     = "%sInput/whitenoise_%s_%s.npy" % (output_path + expname_runid + "/",expname_runid,runid)
    
    # Generate or reload white noise
    if len(glob.glob(noisefile)) > 0:
        print("\t\tWhite file has been found! Loading...")
        wn = np.load(noisefile)
    else:
        print("\t\tGenerating new white noise file: %s" % noisefile)
        noise_size = [expparams['nyrs'],12,]
        if eof_flag: # Generate white noise for each mode
            if expparams['qfactor_sep'] is False:
                nmodes_plus1 = nmode + 1 # Directly include white noise timeseries
            else:
                nmodes_plus1 = nmode # Just use separate timeseries for each correction factor loaded through the helper
            print("\t\tDetected EOF Forcing. Generating %i white noise timeseries" % (nmodes_plus1))
            noise_size   = noise_size + [nmodes_plus1]
        
        wn = np.random.normal(0,1,noise_size) # [Yr x Mon x Mode]
        np.save(noisefile,wn)
    
    # Check if separate white noise timeseries should be loaded using the helper function
    if expparams['qfactor_sep']:
        wn_corr = qfactor_noisemaker(expparams,expdir,expname,runid,share_noise=expparams['share_noise'])
    
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
        
        # # Do Unit Conversions ---
        inputs_convert  = scm.convert_inputs(expparams,inputs,dt=dt,rho=rho,L=L,cp=cp,return_sep=True)
        alpha    = inputs_convert['alpha'] # Amplitude of the forcing
        Dconvert = inputs_convert['lbd_a'] # Converted Damping
        # # End Unit Conversion ---
        
        # Tile Forcing (need to move time dimension to the back)
        if eof_flag and expparams['qfactor_sep'] is False: # Append Qfactor as an extra mode (old approach)
            Qfactor = inputs_convert['Qfactor']
            alpha   = np.concatenate([alpha,Qfactor[None,...]],axis=0)
        
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
    # wn_tile = wn.reshape()
    stfrc = time.time()
    if eof_flag:
        if expparams['qfactor_sep']:
            nmode_final = alpha.shape[0]
            if wn.shape[2] != nmode_final:
                print("Dropping last mode, using separate correction timeseries...")
                wn = wn[:,:,:nmode_final]
            
            # Transpose and make the eof forcing
            forcing_eof = (wn.transpose(2,0,1)[:,:,:,None,None] * alpha[:,None,:,:,:]) # [mode x yr x mon x lat x lon]
            
            # Prepare the correction
            qfactors = [qfz for qfz in list(inputs_convert.keys()) if "correction_factor" in qfz]
            
            qftotal  = []
            for qfz in range(len(qfactors)): # Make timseries for each white noise correction
                qfname      = qfactors[qfz]
                qfactor     = inputs_convert[qfname] # [Mon x Lat x Lon]
                if "Qek" in qfname:
                    qfname = "%s_%s" % (qfname,expparams['varname'])
                wn_qf       = wn_corr[qfname] # [Year x Mon]
                
                qf_combine  = wn_qf[:,:,None,None] * qfactor[None,:,:,:] # [Year x Mon x Lat x Lon]
                
                qftotal.append(qf_combine.copy())
            qftotal = np.array(qftotal) # [Mode x Year x Mon x Lat x Lon]
            
            print(forcing_eof.shape)
            print(qftotal.shape)
            forcing_in = np.concatenate([forcing_eof,qftotal],axis=0)
            
        else:
            forcing_in = (wn.transpose(2,0,1)[:,:,:,None,None] * alpha[:,None,:,:,:]) # [mode x yr x mon x lat x lon]
        forcing_in = np.nansum(forcing_in,0) # Sum over modes
        
    else:
        forcing_in  = wn[:,:,None,None] * alpha[None,:,:,:] # [Year x Mon]
    nyr,_,nlat,nlon = forcing_in.shape
    forcing_in      = forcing_in.reshape(nyr*12,nlat,nlon)
    smconfig['forcing'] = forcing_in.transpose(2,1,0) # Forcing in psu/mon [Lon x Lat x Mon]
    print("\tPrepared forcing in %.2fs" % (time.time()-stfrc))
    
    # End Copy of run_SSS_basinwide
    #  ------------------------------------------------------------------------
    
    # Part (1) Calculate in [degC/mon]
    fprime    = smconfig['forcing'] # Lon x Lat x Time
    lbd_a     = smconfig['lbd_a']
    h         = smconfig['h']
    
    # Load SST
    sst       = dl.load_smoutput(expname,output_path,runids=[nr,])[expparams['varname']]
    sst_arr   = sst.transpose('lon','lat','time').data # Time x Lat x Lon
     
   
    
    sst_roll        = np.roll(sst_arr,axis=2,shift=1)
    zero_mask       = np.where(np.isnan(sst_roll[:,:,1]),np.nan,0)
    sst_roll[:,:,0] = zero_mask
    
    nyrs            = int(sst_roll.shape[-1]/12)
    lbda_T          = sst_roll * np.tile(lbd_a,nyrs)
    Qnet            = fprime - lbda_T
    
    # Convert to W/m2
    htile           = np.tile(h,nyrs)
    Qnet_Wm2        = Qnet * (rho*cp*htile) / (dt)
    
    # To save other components, you can just uncomment and edit the following...
    coords    = dict(lon=sst.lon,lat=sst.lat,time=sst.time)
    #da_fprime = xr.DataArray(fprime,coords=coords,dims=coords,name="Fprime")
    #da_lbdaT  = xr.DataArray(lbda_T,coords=coords,dims=coords,name="Lbd_a_T")
    da_qnet  = xr.DataArray(Qnet_Wm2,coords=coords,dims=coords,name="Qnet")
    
    # Below just saves qnet, but can adjust this to save other components
    da_qnet  = da_qnet.transpose('time','lat','lon')
    edict    = proc.make_encoding_dict(da_qnet)
    outname  = "%s/Metrics/Qnet_runid%s.nc" % (expdir,runids[nr])
    da_qnet.to_netcdf(outname)
    print("Saved output in %.2fs" % (time.time()-st))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    