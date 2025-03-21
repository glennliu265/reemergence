#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Debugging Run SSS_basinwide, testing for the addition of qfactor sep


Copied run_SSS_basinwide on Wed Oct  2 14:02:31 2024

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
sys.path.append("../")
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
SSS_EOF_Qek_Pilot

Note: The original run (2/14) had the incorrect Ekman Forcing and used ens01 detrainment damping with linear detrend
I reran this after fixing these issues (2/29)

"""

# expname     = "SSS_Draft01_Rerun_QekCorr_NoLbde"

# expparams   = {
#     'varname'           : "SSS",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : "SST_EOF_LbddCorr_Rerun",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
#     'Fprime'            : None,
#     'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
#     'LHFLX'             : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
#     'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'             : "CESM1_HTR_FULL_corr_d_SALT_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
#     'Sbar'              : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'             : None, # NEEDS TO BE CONVERTED TO 1/Mon !!!
#     'Qek'               : "CESM1_HTR_FULL_Qek_SSS_NAO_DirReg_NAtl_corrected_EnsAvgFirst.nc", # Must be in W/m2
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
#     "Td_corr"           : True,
#     "lbd_e"             : None,
#     "Tforce"            : None,
#     "correct_Qek"       : True, # Set to True if correction factor to Qek was calculated
#     "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
#     }

# expname = "SST_Draft01_Rerun_QekCorr_NoLbdd"

# expparams   = {
#     'varname'           : "SST",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : "SST_EOF_LbddCorr_Rerun", # If not None, load a runid from another directory
#     'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_corrected_nomasklag1_nroll0_perc090_NAtl_EnsAvg.nc",
#     'PRECTOT'           : None,
#     'LHFLX'             : None,
#     'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'             : None,
#     'Sbar'              : None,
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
#     'Qek'               : "CESM1_HTR_FULL_Qek_SSS_NAO_DirReg_NAtl_corrected_EnsAvgFirst.nc", # Now in degC/sec
#     'convert_Fprime'    : True,
#     'convert_lbd_a'     : True, 
#     'convert_PRECTOT'   : False,
#     'convert_LHFLX'     : False,
#     'froll'             : 0,
#     'mroll'             : 0,
#     'droll'             : 0,
#     'halfmode'          : False,
#     "entrain"           : True,
#     "eof_forcing"       : True,
#     "Td_corr"           : False, # Set to True if lbd_d is provided as a correlation, rather than 1/months
#     "lbd_e"             : None, # Relevant for SSS
#     "Tforce"            : None, # Relevant for SSS
#     "correct_Qek"       : True, # Set to True if correction factor to Qek was calculated
#     "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
#     }


# expname     = "SSS_Draft01_Rerun_QekCorr_NoLbde_NoLbdd"

# expparams   = {
#     'varname'           : "SSS",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : "SST_EOF_LbddCorr_Rerun",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
#     'Fprime'            : None,
#     'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
#     'LHFLX'             : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
#     'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'             : None,
#     'Sbar'              : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'             : None, # NEEDS TO BE CONVERTED TO 1/Mon !!!
#     'Qek'               : "CESM1_HTR_FULL_Qek_SSS_NAO_DirReg_NAtl_corrected_EnsAvgFirst.nc", # Must be in W/m2
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
#     "Td_corr"           : False,
#     "lbd_e"             : None,
#     "Tforce"            : None,
#     "correct_Qek"       : True, # Set to True if correction factor to Qek was calculated
#     "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
#     }

# expname = "SST_Draft02_Rerun_QekCorr"

# expparams   = {
#     'varname'           : "SST",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : "SST_EOF_LbddCorr_Rerun", # If not None, load a runid from another directory
#     'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_corrected_nomasklag1_nroll0_perc090_NAtl_EnsAvg.nc",
#     'PRECTOT'           : None,
#     'LHFLX'             : None,
#     'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'             : "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
#     'Sbar'              : None,
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
#     'Qek'               : "CESM1_HTR_FULL_Qek_SST_NAO_DirReg_NAtl_corrected_EnsAvgFirst.nc", # Now in degC/sec
#     'convert_Fprime'    : True,
#     'convert_lbd_a'     : True, 
#     'convert_PRECTOT'   : False,
#     'convert_LHFLX'     : False,
#     'froll'             : 0,
#     'mroll'             : 0,
#     'droll'             : 0,
#     'halfmode'          : False,
#     "entrain"           : True,
#     "eof_forcing"       : True,
#     "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
#     "lbd_e"             : None, # Relevant for SSS
#     "Tforce"            : None, # Relevant for SSS
#     "correct_Qek"       : True, # Set to True if correction factor to Qek was calculated
#     "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
#     }


# expname = "SST_Draft02_Rerun_QekCorr_NoLbdd"

# expparams   = {
#     'varname'           : "SST",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : "SST_EOF_LbddCorr_Rerun", # If not None, load a runid from another directory
#     'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_corrected_nomasklag1_nroll0_perc090_NAtl_EnsAvg.nc",
#     'PRECTOT'           : None,
#     'LHFLX'             : None,
#     'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'             : None,
#     'Sbar'              : None,
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
#     'Qek'               : "CESM1_HTR_FULL_Qek_SST_NAO_DirReg_NAtl_corrected_EnsAvgFirst.nc", # Now in degC/sec
#     'convert_Fprime'    : True,
#     'convert_lbd_a'     : True, 
#     'convert_PRECTOT'   : False,
#     'convert_LHFLX'     : False,
#     'froll'             : 0,
#     'mroll'             : 0,
#     'droll'             : 0,
#     'halfmode'          : False,
#     "entrain"           : True,
#     "eof_forcing"       : True,
#     "Td_corr"           : False, # Set to True if lbd_d is provided as a correlation, rather than 1/months
#     "lbd_e"             : None, # Relevant for SSS
#     "Tforce"            : None, # Relevant for SSS
#     "correct_Qek"       : True, # Set to True if correction factor to Qek was calculated
#     "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
#     }

expname     = "SSS_Draft03_Rerun_QekCorr_Debug_QfactorSep"

expparams   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SST_Draft03_Rerun_QekCorr",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_concatEns_corrected_EnsAvgFirst.nc",
    'LHFLX'             : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_concatEns_corrected_EnsAvgFirst.nc",
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_SALT_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
    'Sbar'              : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : None, # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : "CESM1_HTR_FULL_Qek_SSS_NAO_DirReg_NAtl_concatEns_corrected_EnsAvgFirst.nc", # Must be in W/m2
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
    "lbd_e"             : "CESM1LE_HTR_FULL_lbde_Bcorr3_lbda_LHFLX_damping_nomasklag1_EnsAvg_noBowen.nc",
    "Tforce"            : "SST_Draft03_Rerun_QekCorr",
    "correct_Qek"       : True, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    "qfactor_sep"       : True,
    }


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
    eof_flag = False

inputs,inputs_ds,inputs_type,params_vv = scm.load_params(expparams,input_path)

#%% Detect and Process Missing Inputs

_,nlat,nlon=inputs['h'].shape

# for pname in missing_input:
#     if type(expparams[pname]) == float:
#         print("\t\tFloat detected for <%s>. Making array with the repeated value %f" % (pname,expparams[pname]))
#         inputs[pname] = np.ones((12,nlat,nlon)) * expparams[pname]
#     else:
#         print("\tNo value found for <%s>. Setting to zero." % pname)
#         inputs[pname] = np.zeros((12,nlat,nlon))

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


def qfactor_noisemaker(expparams,expdir,expname,runid):
    # Checks for separate wn timeseries for each correction factor
    # and loads the dictionary
    
    # Makes (and checks for) additional white noise timeseries for the following (6) correction factors
    forcing_names = ("correction_factor",           # Fprime
                     "correction_factor_Qek_SST",   # Qek_SST
                     "correction_factor_evap",      # Evaporation
                     "correction_factor_prec",      # Precip
                     "correction_factor_Qek_SSS",   # Qek_SSS
                     )
    nforcings = len(forcing_names)
    
    # Check for correction file
    noisefile_corr = "%sInput/whitenoise_%s_%s_corrections.npz" % (expdir,expname,runid)
    
    # Generate or reload white noise
    if len(glob.glob(noisefile_corr)) > 0:
        
        print("\t\tWhite Noise corretion factor file has been found! Loading...")
        wn_corr = np.load(noisefile_corr)
    else:
        
        print("\t\tGenerating %i new white noise timeseries: %s" % (nforcings,noisefile_corr))
        noise_size  = [expparams['nyrs'],12,]
        
        wn_corr_out = {}
        for nn in range(nforcings):
            wn_corr_out[forcing_names[nn]] = np.random.normal(0,1,noise_size) # [Yr x Mon x Mode]
        
        np.savez(noisefile_corr,**wn_corr_out,allow_pickle=True)
    return wn_corr

    
for nr in range(nruns):
    
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
        wn_corr = qfactor_noisemaker(expparams,expdir,expname,runid)
    
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
        # if expparams["varname"] == "SSS": # Convert to psu/mon ---------------
            
        #     # Evap Forcing
        #     if expparams['convert_LHFLX']: #~~
        #         if eof_flag: # Check for EOFs
        #             conversion_factor = ( dt*inputs['Sbar'] / (rho*L*inputs['h']))[None,...]
        #             Econvert          = inputs['LHFLX'].copy() * conversion_factor # [Mon x Lat x Lon]
                    
        #             # Add Correction Factor, if it exists
        #             if 'correction_factor_evap' in list(inputs.keys()):
        #                 print("Processing LHFLX/Evaporation Correction factor")
        #                 QfactorE      = inputs['correction_factor_evap'].copy() * conversion_factor.squeeze()
        #             else:
        #                 QfactorE      = np.zeros((inputs['LHFLX'].shape[1:])) # Same Shape minus the mode dimension
        #         else:
        #             Econvert          = inputs['LHFLX'].copy() / (rho*L*inputs['h'])*dt*inputs['Sbar'] # [Mon x Lat x Lon]
                
        #         # Multiply both by -1 (since Q'_LH = -LE')
        #         Econvert = Econvert * -1
        #         QfactorE = QfactorE * -1
                
        #     else:
        #         Econvert   = inputs['LHFLX'].copy()
            
        #     # Precip Forcing ~~
        #     if expparams['convert_PRECTOT']:
                
        #         conversion_factor = ( dt*inputs['Sbar'] / inputs['h'] )
                
        #         if eof_flag:
        #             Pconvert =  inputs['PRECTOT'] * conversion_factor[None,...]
        #         else:
        #             Pconvert =  inputs['PRECTOT'] * conversion_factor
                
        #         if (eof_flag) and ('correction_factor_prec' in list(inputs.keys())):
        #             print("Processing Precip Correction factor")
        #             QfactorP   = inputs['correction_factor_prec'] * conversion_factor
        #         else:
        #             QfactorP   = np.zeros((inputs['PRECTOT'].shape[1:])) # Same Shape minus the mode dimension
        #     else:
        #         Pconvert   = inputs['PRECTOT'].copy()
            
        #     # Atmospheric Damping ~~
        #     if expparams['convert_lbd_a']:
        #         print("WARNING: lbd_a unit conversion for SSS currently not supported")
        #         Dconvert = inputs['lbd_a'].copy()
        #     else:
        #         Dconvert = inputs['lbd_a'].copy()
        #         if np.nansum(Dconvert) < 0:
        #             print("Flipping Sign")
        #             Dconvert *= -1
            
        #     # Add Ekman Forcing, if it Exists (should be zero otherwise) ~~
        #     Qekconvert = inputs['Qek'].copy()  * dt #  [(mode) x Mon x Lat x Lon]
            
        #     # Corrrection Factor
        #     if eof_flag:
        #         Qfactor    = QfactorE + QfactorP # Combine Evap and Precip correction factor
            
        #     # Combine Evap and Precip (and Ekman Forcing)
        #     alpha         = Econvert + Pconvert + Qekconvert
        # # End SSS Conversion --------------------------------------------------
        # elif expparams['varname'] == "SST": # Convert to degC/mon
            
        #     # Convert Stochastic Heat Flux Forcing ~~
        #     if expparams['convert_Fprime']:
        #         if eof_flag:
        #             Fconvert   = inputs['Fprime'].copy()           / (rho*cp*inputs['h'])[None,:,:,:] * dt # Broadcast to mode x mon x lat x lon
        #             # Also convert correction factor
        #             Qfactor    = inputs['correction_factor'].copy()/ (rho*cp*inputs['h'])[:,:,:] * dt
                    
        #         else:
        #             Fconvert   = inputs['Fprime'].copy() / (rho*cp*inputs['h']) * dt
        #     else:
        #         Fconvert   = inputs['Fprime'].copy()
            
        #     # Convert Atmospheric Damping ~~
        #     if expparams['convert_lbd_a']:
                
        #         Dconvert   = inputs['lbd_a'].copy() / (rho*cp*inputs['h']) * dt
        #     else:
        #         Dconvert   = inputs['lbd_a'].copy()
        #         if np.nansum(Dconvert) < 0:
        #             print("Flipping Sign")
        #             Dconvert *= -1
            
        #     # Add Ekman Forcing, if it exists (should be zero otherwise) ~~
        #     if eof_flag:
        #         Qekconvert = inputs['Qek'].copy() / (rho*cp*inputs['h'])[None,:,:,:] * dt
        #     else:
        #         Qekconvert = inputs['Qek'].copy() / (rho*cp*inputs['h']) * dt
            
        #     # Compute forcing amplitude
        #     alpha = Fconvert + Qekconvert
        #     # <End Variable Conversion Check>
        
        # # End SST Conversion --------------------------------------------------
        
        #alpha,Dconvert,Qfactor = scm.convert_inputs(expparams,inputs,dt=dt,rho=rho,L=L,cp=cp,return_sep=False)
        
        outdict  = scm.convert_inputs(expparams,inputs,dt=dt,rho=rho,L=L,cp=cp,return_sep=True)
        alpha    = outdict['alpha'] # Amplitude of the forcing
        Dconvert = outdict['lbd_a'] # Converted Damping
        
        # Tile Forcing (need to move time dimension to the back)
        if eof_flag and expparams['qfactor_sep'] is False: # Append Qfactor as an extra mode (old approach)
            Qfactor = outdict['Qfactor']
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
    if eof_flag:
        stfrc = time.time()
        if expparams['qfactor_sep']:
            nmode_final = alpha.shape[0]
            if wn.shape[2] != nmode_final:
                print("Dropping last mode, using separate correction timeseries...")
                wn = wn[:,:,:nmode_final]
            
            # Transpose and make the eof forcing
            forcing_eof = (wn.transpose(2,0,1)[:,:,:,None,None] * alpha[:,None,:,:,:]) # [mode x yr x mon x lat x lon]
            
            # Prepare the correction
            qfactors = [qfz for qfz in list(outdict.keys()) if "correction_factor" in qfz]
            
            qftotal  = []
            for qfz in range(len(qfactors)): # Make timseries for each white noise correction
                qfname  = qfactors[qfz]
                qfactor = outdict[qfname] # [Mon x Lat x Lon]
                if "Qek" in qfname:
                    qfname = "%s_%s" % (qfname,expparams['varname'])
                wn_qf   = wn_corr[qfname] # [Year x Mon]
                
                qf_combine = wn_qf[:,:,None,None] * qfactor[None,:,:,:] # [Year x Mon x Lat x Lon]
                
                qftotal.append(qf_combine.copy())
            qftotal = np.array(qftotal) # [Mode x Year x Mon x Lat x Lon]
            
            forcing_in = np.concatenate([forcing_eof,qftotal],axis=0)
            
        else:
            
            
            forcing_in = (wn.transpose(2,0,1)[:,:,:,None,None] * alpha[:,None,:,:,:]) # [mode x yr x mon x lat x lon]
        forcing_in = np.nansum(forcing_in,0) # Sum over modes
    
    else:
        forcing_in  = wn.T[:,:,None,None] * alpha[None,:,:,:]
    nyr,_,nlat,nlon = forcing_in.shape
    forcing_in      = forcing_in.reshape(nyr*12,nlat,nlon)
    smconfig['forcing'] = forcing_in.transpose(2,1,0) # Forcing in psu/mon [Lon x Lat x Mon]
    print("\tPrepared forcing in %.2fs" % (stfrc - time.time()))
    
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
            
            sst_in = sst_in.drop_duplicates('lon')
            
            sst_in = sst_in.transpose('lon','lat','time').values
            
            # Tile and combine
            lbd_emon_tile     = np.tile(lbd_emon,nyr) #
            lbdeT             = lbd_emon_tile * sst_in
            smconfig['add_F'] = lbdeT
    
    if debug: #Just run at a point
        ivnames = list(smconfig.keys())
        #[print(smconfig[iv].shape) for iv in ivnames]
        
        
        for iv in ivnames:
            vtype = type(smconfig[iv])
            if vtype == np.ndarray:
                smconfig[iv] = smconfig[iv][klon,klat,:].squeeze()[None,None,:]
                
        #[print(smconfig[iv].shape) for iv in ivnames]
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
