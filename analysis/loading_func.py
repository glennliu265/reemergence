#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:08:56 2024

@author: gliu
"""

# Copied from run_SSS_basinwide =============================================== >> >> >>
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



# Paths and Experiment
input_path  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/"
output_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"

# Paths and Experiment
input_path  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/"
output_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"


# expname     = "SSS_CESM1_5deg_lbddcoarsen"

# expparams   = {
#     'varname'           : "SSS",
#     'bbox_sim'          : [-80,0,20,65],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : "SST_CESM1_5deg_lbddcoarsen",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
#     'Fprime'            : None,
#     'PRECTOT'           : "cesm1le_htr_5degbilinear_PRECTOT_EOF_cesm1le5degLHFLX_nroll0_NAtl_corrected_EnsAvg.nc",#"CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
#     'LHFLX'             : "cesm1le_htr_5degbilinear_Eprime_EOF_cesm1le5degLHFLX_nroll0_NAtl_corrected_EnsAvg.nc",#"CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
#     'h'                 : "cesm1_htr_5degbilinear_HMXL_NAtl_1920to2005_EnsAvg.nc",
#     'lbd_d'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_coarsen5deg.nc",
#     'Sbar'              : "cesm1_htr_5degbilinear_Sbar_Global_1920to2005_EnsAvg.nc",#"CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "cesm1_htr_5degbilinear_kprev_NAtl_1920to2005_EnsAvg.nc",
#     'lbd_a'             : None, # NEEDS TO BE CONVERTED TO 1/Mon !!!
#     'Qek'               : "cesm1_htr_5degbilinear_Qek_SALT_NAO_cesm1le5degqnet_nroll0_NAtl_EnsAvg.nc", # Must be in W/m2
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
#     "lbd_e"             : "cesm1le_htr_5degbilinear_lbde_Bcorr3_lbda_cesm1le5degqnetDamp_EnsAvg.nc",##"CESM1LE_HTR_FULL_lbde_Bcorr3_lbda_qnet_damping_nomasklag1_EnsAvg.nc",
#     "Tforce"            : "SST_CESM1_5deg_lbddcoarsen_rerun",#"SST_EOF_LbddCorr_Rerun"
#     }

expname     = "SSS_EOF_LbddCorr_Rerun_lbdE_neg"

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
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_SALT_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
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

#%%


# # Constants
# dt    = 3600*24*30 # Timestep [s]
# cp    = 3850       # 
# rho   = 1026       # Density [kg/m3]
# B     = 0.2        # Bowen Ratio, from Frankignoul et al 1998
# L     = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

# debug = False

#%%  ======================================================= Copied Section End >> >> >> >>


#%%


def load_params(expparams,debug=False):
    # Copied run_SSS_pointmode_coupled
    # Loads and checks for all necessary inputs of the stochastic model
    
    # To Do: Return eof_flag...
    
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
            if (expparams['varname'] == "SST") and (pname =="Fprime") and "Fprime" not in expparams[pname]:
                
                da_varname   = "LHFLX" # Swap to LHFLX for now
                varname_swap = True # True so "Fprime" is input as da_varname later
                swapname     = "Fprime" 
            
            else:
                varname_swap = False
                
                #inputs_type['Fprime'] = 'forcing' # Add extra Fprime variable
            
            # Load ds
            ds = xr.open_dataset(input_path + ptype + "/" + expparams[pname])[da_varname]
            
            
            # Crop to region
            
            # Load dataarrays for debugging
            dsreg            = proc.sel_region_xr(ds,expparams['bbox_sim']).load()
            dsreg            = dsreg.drop_duplicates('lon')# Drop duplicate Lon (Hard coded fix, remove this)
            inputs_ds[pname] = dsreg.copy() 
            
            # Load to numpy arrays 
            varout           = dsreg.values
            # varout           = varout[...,None,None]
            # if debug:
            #     print(pname) # Name of variable
            #     print("\t%s" % str(ds.shape)) # Original Shape
            #     print("\t%s" % str(varout.shape)) # Point Shape
            inputs[pname]    = varout.copy()
            
            if ((da_varname == "Fprime") and (eof_flag)) or ("corrected" in expparams[pname]):
                print("Loading %s correction factor for EOF forcing..." % pname)
                ds_corr                          = xr.open_dataset(input_path + ptype + "/" + expparams[pname])['correction_factor']
                ds_corr_reg                      = proc.sel_region_xr(ds_corr,expparams['bbox_sim']).load()
                ds_corr_reg                      = ds_corr_reg.drop_duplicates('lon')
                
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
                inputs[keyname]      = ds_corr_reg.values.copy()#[...,None,None]
                if debug:
                    print(da_varname + " Corr") # Variable Name
                    print("\t%s" % str(ds_corr.shape))
                    print("\t%s" % str(inputs[keyname].shape)) # Corrected Shape
                inputs_type[keyname] = "forcing"
            
        else:
            print("Did not find %s" % pname)
            missing_input.append(pname)
        
        inputs_type[pname] = ptype
    
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
            
    # Unpack things from dictionary (added after for debugging)
    ninputs     = len(inputs_ds)
    param_names = list(inputs_ds.keys())
    params_vv   = [] # Unpack from dictionary
    for ni in range(ninputs):
        pname = param_names[ni]
        dsin  = inputs_ds[pname]
        params_vv.append(dsin.copy())
    return inputs,inputs_ds,inputs_type,params_vv

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#%% Do a manual load copied from run_SSS_basinwide.py


print("Loading inputs for %s" % expname)

# First, Check if there is EOF-based forcing (remove this if I eventually redo it)
if expparams['eof_forcing']:
    print("\tEOF Forcing Detected.")
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
        
    
        # Handle Loading of Eprime/LHFLX as Fprime (copied from [run_SSS_pointmode_coupled.py])
        # Load ds
        if (expparams['varname'] == "SST") and (pname =="Fprime") and "Fprime" not in expparams[pname]:
            
            da_varname   = "LHFLX" # Swap to LHFLX for now
            varname_swap = True    # True so "Fprime" is input as da_varname later
            swapname     = "Fprime" 
            
        else:
            varname_swap = False
        
        # Load ds
        ds = xr.open_dataset(input_path + ptype + "/" + expparams[pname])[da_varname]
        
        # Crop to region
        
        # Load dataarrays for debugging
        dsreg            = proc.sel_region_xr(ds,expparams['bbox_sim']).load()
        dsreg            = dsreg.drop_duplicates('lon')# Drop duplicate Lon (Hard coded fix, remove this)
        inputs_ds[pname] = dsreg.copy() 
        
        
        
        # Load to numpy arrays 
        varout           = dsreg.values
        inputs[pname]    = dsreg.values.copy()
        
        if ((da_varname == "Fprime") and (eof_flag)) or ("corrected" in expparams[pname]):
            print("\tLoading %s correction factor for EOF forcing..." % pname)
            ds_corr                          = xr.open_dataset(input_path + ptype + "/" + expparams[pname])['correction_factor']
            ds_corr_reg                      = proc.sel_region_xr(ds_corr,expparams['bbox_sim']).load()
            ds_corr_reg                      = ds_corr_reg.drop_duplicates('lon')
            if varname_swap:
                da_varname = pname # Swap from LHFLX back to Fprime for SST Integration
            
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
        #print("\tDid not find: %s" % pname)
        missing_input.append(pname)
    inputs_type[pname] = ptype

#% Detect and Process Missing Inputs

_,nlat,nlon=inputs['h'].shape

for pname in missing_input:
    if type(expparams[pname]) == float:
        print("\t\tFloat detected for <%s>. Making array with the repeated value %f" % (pname,expparams[pname]))
        inputs[pname] = np.ones((12,nlat,nlon)) * expparams[pname]
    else:
        print("\tNo value found for <%s>. Setting to zero." % pname)
        inputs[pname] = np.zeros((12,nlat,nlon))

# Get number of modes
if eof_flag:
    if expparams['varname'] == "SST":
        nmode = inputs['Fprime'].shape[0]
    elif expparams['varname'] == "SSS":
        nmode = inputs['LHFLX'].shape[0]


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#%% Run the Function 

finputs,finputs_ds,finputs_type,fparams_vv = load_params(expparams)

#%% Check to see if they are identical

varkeys = list(finputs.keys())
nkeys   = len(varkeys)

for nk in range(nkeys):
    
    inkey = varkeys[nk]
    
    a = inputs[inkey]
    b = finputs[inkey]
    chk   = ((a == b) | (np.isnan(a) & np.isnan(b))).all()#.all()
    print("[inputs] is ok for %s" % inkey)
    
    
varkeys = list(finputs_ds.keys())
nkeys   = len(varkeys)
for nk in range(nkeys):
    inkey = varkeys[nk]
    
    a = inputs_ds[inkey].data
    b = finputs_ds[inkey].data
    chk   = ((a == b) | (np.isnan(a) & np.isnan(b))).all()#.all()
    print("[inputs_ds] is ok for %s" % inkey)
    
    a = inputs_type[inkey]
    b = finputs_type[inkey]
    chk   = a == b #| (np.isnan(a) & np.isnan(b))).all()#.all()
    print("[inputs_type] is ok for %s" % inkey)
    

# for nk in range(len(params_vv)):
#     a = params_vv[nk]
#     b = fparams_vv[nk]
#     chk   = ((a == b) | (np.isnan(a) & np.isnan(b))).all()#.all()
#     print("[params_vv] is ok for %s" % inkey)

#%% First, run the section where you roll everything. This should operate on the inputs variable

def convert_inputs(expparams,inputs,dt=3600*24*30,rho=1026,L=2.5e6,cp=3850):
    
    if expparams['eof_forcing']:
        eof_flag=True
    else:
        eof_flag=False
    
    # Do Unit Conversions --- O --- o --- o --- 0 --- o --- o --- 0 --- o --- o
    if expparams["varname"] == "SSS": # Convert all to psu/mon ----------------
        
        # Evaporation <Forcing> ~~ ********************************************
        if expparams['convert_LHFLX']: #~~
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
            
            # Multiply both by -1 (since Q'_LH = -LE')
            Econvert = Econvert * -1
            QfactorE = QfactorE * -1
            
        else:
            Econvert   = inputs['LHFLX'].copy()
        
        # Precip <Forcing> ~~ *************************************************
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
        
        # Atmospheric <Damping> ~~ ********************************************
        if expparams['convert_lbd_a']:
            print("WARNING: lbd_a unit conversion for SSS currently not supported")
            Dconvert = inputs['lbd_a'].copy()
        else:
            Dconvert = inputs['lbd_a'].copy()
            if np.nansum(Dconvert) < 0:
                print("Flipping Sign")
                Dconvert *= -1
        
        # Add Ekman Forcing, if it Exists (should be zero otherwise) ~~::::::::
        Qekconvert = inputs['Qek'].copy()  * dt #  [(mode) x Mon x Lat x Lon]
        
        # Corrrection Factor **************************************************
        if eof_flag:
            Qfactor    = QfactorE + QfactorP # Combine Evap and Precip correction factor
        
        # Combine Evap and Precip (and Ekman Forcing)
        alpha         = Econvert + Pconvert + Qekconvert
    # Main Output: (alpha - all the forcings), (Dconvert - atm dampings)
    # End SSS Conversion --------------------------------------------------
    elif expparams['varname'] == "SST": # Convert to degC/mon
        
        # Convert Stochastic Heat Flux Forcing ~~
        if expparams['convert_Fprime']:
            if eof_flag:
                
                Fconvert   = inputs['Fprime'].copy()           / (rho*cp*inputs['h'])[None,:,:,:] * dt # Broadcast to mode x mon x lat x lon
                # Also convert correction factor
                Qfactor    = inputs['correction_factor'].copy()/ (rho*cp*inputs['h'])[:,:,:] * dt
                
            else:
                Fconvert   = inputs['Fprime'].copy() / (rho*cp*inputs['h']) * dt
        else:
            Fconvert   = inputs['Fprime'].copy()
        
        # Convert Atmospheric Damping ~~
        if expparams['convert_lbd_a']:
            
            Dconvert   = inputs['lbd_a'].copy() / (rho*cp*inputs['h']) * dt
        else:
            
            Dconvert   = inputs['lbd_a'].copy()
            if np.nansum(Dconvert) < 0:
                print("Flipping Sign")
                Dconvert *= -1
        
        # Add Ekman Forcing, if it exists (should be zero otherwise) ~~
        if eof_flag:
            Qekconvert = inputs['Qek'].copy() / (rho*cp*inputs['h'])[None,:,:,:] * dt
        else:
            Qekconvert = inputs['Qek'].copy() / (rho*cp*inputs['h']) * dt
        
        # Compute forcing amplitude
        alpha = Fconvert + Qekconvert
        # <End Variable Conversion Check>
    # End SST Conversion --------------------------------------------------\
    
    return alpha,Dconvert,Qfactor

falpha,fDconvert,fQfactor = convert_inputs(expparams,inputs,dt=dt,rho=rho,L=L,cp=cp)
#%% 

def check_equal_nan(a,b):
    chk   = ((a == b) | (np.isnan(a) & np.isnan(b))).all()
    return chk

check_equal_nan(falpha,alpha)
check_equal_nan(fDconvert,Dconvert)

check_equal_nan(Qfactor,fQfactor)

#%% Check differences between the two runs

fds = xr.open_dataset("/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/SSS_CESM1_5deg_lbddcoarsen_testfunc/Output/SSS_runidrun00.nc").load()
ds  = xr.open_dataset("/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/SSS_CESM1_5deg_lbddcoarsen/Output/SSS_runidrun00.nc").load()


