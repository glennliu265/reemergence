#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test HPD (misnamed as lbde)

Copied stochmod_point_test_lbdd
Runs on Astraeus

Created on Wed Mar 20 17:08:43 2024

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob
import time

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

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

#%% User Edits

# Location
lonf           = -30
latf           = 50
locfn,loctitle = proc.make_locstring(lonf,latf)

# Get experiment parameters for base case
expname_base   = "SST_EOF_LbddCorr_Rerun" # "SST_EOF_LbddEnsMean"
dictpath       = output_path + expname_base + "/Input/expparams.npz"
expparams_raw  = np.load(dictpath,allow_pickle=True)

# Set experiment parameters for test case
nyrs           = 10000
expname_test   = "hpdtest"
expdir         = output_path + expname_test + "/"
scm.gen_expdir(expdir)

# Fix parameter dictionary (they are all 0-d arrays)
expparams      = scm.repair_expparams(expparams_raw)
expparams      = scm.patch_expparams(expparams)

# Constants
dt    = 3600*24*30 # Timestep [s]
cp    = 3850       # 
rho   = 1026       # Density [kg/m3]
B     = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L     = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

debug = False

# -----------------------------------------------------------------------------

# -----------------------
#%% Check and Load Params
# -----------------------
print("Loading inputs for %s" % expname_base)

# First, Check if there is EOF-based forcing (remove this if I eventually redo it)
if expparams['eof_forcing']:
    print("EOF Forcing Detected.")
    eof_flag = True
else:
    eof_flag = False

# Indicate the Parameter Names (sorry, it's all hard coded...)
if expparams['varname'] == "SSS": # Check for LHFLX, PRECTOT, Sbar
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
        

        # Crop to point ----------- Updated Section BEGIN
        
        # Load dataarrays for debugging
        dsreg            = ds.sel(lon=lonf,lat=latf,method='nearest').load()#proc.sel_region_xr(ds,expparams['bbox_sim']).load()
        # Crop to point ----------- Updated Section END
        inputs_ds[pname] = dsreg.copy() 
        
        # Load to numpy arrays 
        varout           = dsreg.values
        inputs[pname]    = dsreg.values[...,None,None].copy()
        
        if ((da_varname == "Fprime") and (eof_flag)) or ("corrected" in expparams[pname]):
            print("Loading %s correction factor for EOF forcing..." % pname)
            ds_corr                          = xr.open_dataset(input_path + ptype + "/" + expparams[pname])['correction_factor']
            ds_corr_reg                      = ds_corr.sel(lon=lonf,lat=latf,method='nearest').load()
            
            # set key based on variable type
            if da_varname == "Fprime":
                keyname = "correction_factor"
            elif da_varname == "LHFLX":
                keyname = "correction_factor_evap"
            elif da_varname == "PRECTOT":
                keyname = "correction_factor_prec"
                
            inputs_ds[keyname]   = ds_corr_reg.copy()
            inputs[keyname]      = ds_corr_reg.values[...,None,None].copy()
            inputs_type[keyname] = "forcing"
        
    else:
        print("Did not find %s" % pname)
        missing_input.append(pname)
    inputs_type[pname] = ptype
    
#%% Detect and Process Missing Inputs (this has been edited to run at a point)


nlat,nlon=1,1

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
        
#%% Save original here!
inputs_original = inputs.copy()


#%%

froll = expparams['froll']
droll = expparams['droll']
mroll = expparams['mroll']

# Make a function to simplify rolling
def roll_input(invar,rollback,halfmode=False,axis=0):
    rollvar = np.roll(invar,rollback,axis=axis)
    if halfmode:
        rollvar = (rollvar + invar)/2
    return rollvar


#%% Prepare White Noise timeseries ----------------------------------------

runid     = "test01"
noisefile = "%sInput/whitenoise_%s_%s.npy" % (expdir,expname_test,runid)

#nr     = 0

# runid = runids[nr]

# # Check if specific path was indicated, and set filename accordingly
# if expparams['runid_path'] is None:
#     noisefile = "%sInput/whitenoise_%s_%s.npy" % (expdir,expname,runid)
# else:
#     expname_runid = expparams['runid_path'] # Name of experiment to take runid from
#     print("Searching for runid path in specified experiment folder: %s" % expname_runid)
#     noisefile     = "%sInput/whitenoise_%s_%s.npy" % (output_path + expname_runid + "/",expname_runid,runid)

# # Generate or reload white noise
if len(glob.glob(noisefile)) > 0:
    print("White Noise file has been found! Loading...")
    wn = np.load(noisefile)
else:
    print("Generating new white noise file: %s" % noisefile)
    noise_size = [nyrs,12,]
    if eof_flag: # Generate white noise for each mode
        nmodes_plus1 = nmode + 1 
        print("Detected EOF Forcing. Generating %i white noise timeseries" % (nmodes_plus1))
        noise_size   = noise_size + [nmodes_plus1]

    wn = np.random.normal(0,1,noise_size) # [Yr x Mon x Mode]
    np.save(noisefile,wn)
    

#%% Do Conversions for Model Run ------------------------------------------

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
        print("Check ")
        Dconvert = inputs['lbd_a'].copy()
        #break
    else:
        print("WARNING: lbd_a unit conversion for SSS currently not supported")
        Dconvert = inputs['lbd_a'].copy()
        if np.nansum(Dconvert) < 0:
            print("Flipping Sign")
            Dconvert *= -1
        #break
    
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
    if len(Qfactor.shape) != 4:
        Qfactor=Qfactor[None,...] # Add extra mode dimension
    alpha = np.concatenate([alpha,Qfactor],axis=0)
    
# Calculate beta and kprev
beta       = scm.calc_beta(inputs['h'].transpose(2,1,0)) # {lon x lat x time}
if expparams['kprev'] is None: # Compute Kprev if it is not supplied
    print("Recalculating Kprev")
    kprev = np.zeros((12,1,1))
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

#%% Do White Noise Generation Stuff

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

# -------------------------------
#%% Load & set up experimental parameters
# -------------------------------

"""
Experiment 1: HPD vs HMXL, Use HPD rather than HMXL
"hpdvhmxl"

Experiment 2: Correlation estimates using shifting and interpolation
"corrmethodsSST"

Experiment 3: CSame as above but for SSS rather than SST
"corrmethodsSSS"

"""



expname ="hpdvhmxl"
if expname == "hpdvhmxl":
    
    # Load HPD
    #nc_hpd  = input_path + "mld/" + "CESM1_HTR_FULL_HPD_NAtl_Ens01.nc"
    #ds_hpd  = xr.open_dataset(nc_hpd).h.sel(lon=lonf,lat=latf,method='nearest')
    #hpd_pt  = ds_hpd.values[:,None,None]
    
    # Get HPD using last month version
    
    hpd_pt = np.array([ 6631.57894737,  8143.13701375, 10198.86568354,  9823.02873585,
            8149.74594774,  4621.23675185,  3644.29669896,  3251.31720729,
            3181.9254739 ,  3901.99004975,            np.nan,            np.nan])/100
    
    
    
    # Load HMXL
    hmxl_pt  = inputs_original['h']#.transpose(2,1,0)
    
    for im in range(12):
        if (im < 11):
            
            if im == np.argmax(hmxl_pt): # deepest month
                hpd_pt[im] = hmxl_pt[im]
            
            # If MLD is NOT shoaling (increasing h), or is NaN, replace with HMXL
            # or Hpd is shallower than HMXL
            if (hmxl_pt[im+1] - hmxl_pt[im] > 0) or np.isnan(hpd_pt[im]) or (hpd_pt[im] <  hmxl_pt[im]):
                #if :
                
                hpd_pt[im] = hmxl_pt[im]
            
        else:
            hpd_pt[im] = hmxl_pt[im]
    
    hpd_pt = hpd_pt[:,None,None]
    #fig,ax=plt.subplots()
    
    # # Load Surface Lbdd
    # nclbdd      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon330_lat50/Lbdd_estimate_surface_TEMP.nc"
    # ds_lbdd     = xr.open_dataset(nclbdd)
    # lbdd_surf   = ds_lbdd.mean('ens').lbd_d.values[None,None,:]
    
    # # Load Deep Lbdd
    # nclbdd      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon330_lat50/Lbdd_estimate_deep_TEMP.nc"
    # ds_lbdd     = xr.open_dataset(nclbdd)
    # lbdd_deep   = ds_lbdd.mean('ens').lbd_d.values[None,None,:]
    
    # # Load Correlation (Surface Lbdd)
    # nclbdd     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon330_lat50/Lbdd_estimate_surface1_TEMP.nc"
    # ds_lbdd     = xr.open_dataset(nclbdd)
    # lbdd_surf_corr   = ds_lbdd.mean('ens').corr_d.values[None,None,:]
    
    # # Load Correlation (Surface Lbdd)
    # nclbdd     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon330_lat50/Lbdd_estimate_surface0_TEMP.nc"
    # ds_lbdd     = xr.open_dataset(nclbdd)
    # lbdd_deep_corr   = ds_lbdd.mean('ens').corr_d.values[None,None,:]
    
    # Experiment 1: Loading in Deep vs. Surface Damping and Correlation
    vname      = "TEMP"
    testparam  = "h"
    testvalues = [hpd_pt,hmxl_pt]
    testnames  = ["Heat Penetration Depth","Mixed-Layer Depth",]
    testcols   = ["cornflowerblue"           ,"orange"]
    testls     = ["solid","dashed"]
    
    
    fig,ax = viz.init_monplot(1,1)
    ax.plot(hpd_pt.squeeze(),label="hpd")
    ax.plot(hmxl_pt.squeeze(),label="hmxl",ls='dashed')
    ax.legend()
    ax.invert_yaxis()
    ax.set_ylabel("Depth (meters)")


    





#%% Now Integrate the Model


ntest      = len(testvalues)

output = []
checkparam = []
for tt in range(ntest+1):
    
    
    
    
    if tt < ntest:
        inputs = inputs_original.copy()
        print("Replacing testparam %s FOR %s" % (testparam,testnames[tt]))
        inputs[testparam] = testvalues[tt]
        #smconfig_in[testparam]     = testvalues[tt]
        
    #% Copy from Above ========== (need to do reconversion)
        
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
            print("Check ")
            Dconvert = inputs['lbd_a'].copy()
            #break
        else:
            print("WARNING: lbd_a unit conversion for SSS currently not supported")
            Dconvert = inputs['lbd_a'].copy()
            if np.nansum(Dconvert) < 0:
                print("Flipping Sign")
                Dconvert *= -1
            #break
        
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
        if len(Qfactor.shape) != 4:
            Qfactor=Qfactor[None,...] # Add extra mode dimension
        alpha = np.concatenate([alpha,Qfactor],axis=0)
        
    # Calculate beta and kprev
    beta       = scm.calc_beta(inputs_original['h'].transpose(2,1,0)) # {lon x lat x time}
    if expparams['kprev'] is None: # Compute Kprev if it is not supplied
        print("Recalculating Kprev")
        kprev = np.zeros((12,1,1))
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
        
    #% Do White Noise Generation Stuff

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

    
    #% ================= Copied from above
    smconfig_in                    = smconfig.copy()
    
    if tt < ntest:
        inputs = inputs_original.copy()
        #print("Replacing testparam %s FOR %s" % (testparam,testnames[tt]))
        smconfig_in[testparam] = testvalues[tt].transpose(2,1,0)
        #smconfig_in[testparam]     = testvalues[tt]
    
    if expparams['entrain'] is True:
        outdict = scm.integrate_entrain(smconfig_in['h'],smconfig_in['kprev'],smconfig_in['lbd_a'],smconfig_in['forcing'],
                                        Tdexp=smconfig_in['lbd_d'],beta=smconfig_in['beta'],
                                        return_dict=True,old_index=True,Td_corr=smconfig_in['Td_corr'])
    else:
        outdict = scm.integrate_noentrain(smconfig_in['lbd_a'],smconfig_in['forcing'],T0=0,multFAC=True,debug=True,old_index=True,return_dict=True)
    
    
    output.append(outdict)
    checkparam.append(smconfig_in[testparam])
    
    
testnames.append("Default")

#%%  Compute Metrics

tsout   = [op['T'].squeeze() for op in output]

metdict = scm.compute_sm_metrics(tsout)

#plt.plot(metdict['acfs'][0][1])

#plt.plot(outdict['T'].squeeze())
#%% Load Metrics from CESM1

if vname == "SALT":
    vname_surf = "SSS"
elif vname == "TEMP":
    vname_surf = "SST"
else:
    print("vname must be SALT or TEMP")

acfpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/%s_CESM/Metrics/" % vname_surf
acfnc      = "Pointwise_Autocorrelation_thresALL_lag00to60.nc"
dscesmsst  = xr.open_dataset(acfpath+acfnc)[vname_surf].sel(lon=lonf,lat=latf,method='nearest').load()


# # Load ACF with no deep damping
# acfpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/%s_EOF_NoLbdd/Metrics/" % vname_surf
# acfnc      = "Pointwise_Autocorrelation_thresALL_lag00to60.nc"
# dssmsst    = xr.open_dataset(acfpath+acfnc)[vname_surf].sel(lon=lonf,lat=latf,method='nearest').load()


#%% Plot the ACF

kmonth = 1
nens  = 42

lags = np.arange(37)
xtks = np.arange(0,38,2)

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(10,4.5))
ax,_   = viz.init_acplot(kmonth,xtks,lags,title="")

for tt in range(ntest):
    ax.plot(metdict['acfs'][kmonth][tt],label=testnames[tt],c=testcols[tt],ls=testls[tt])
    
# # Plot the default run
# ax.plot(lags,metdict['acfs'][kmonth][ntest],label="Base (%s)" % expname_base,
#         c="gray",ls=testls[tt])

# plot CESM
dsc = dscesmsst.isel(mons=kmonth).mean('ens')
ax.plot(dsc.lags,dsc,color="k",label="CESM Ens. Mean")

# Plot CESM Ensemble Members
for e in range(nens):
    dsc = dscesmsst.isel(mons=kmonth,ens=e)
    ax.plot(dsc.lags,dsc,color="gray",label="",alpha=0.2,zorder=-1)

# dsc = dssmsst.isel(mons=kmonth,thres=0)
# ax.plot(dsc.lags,dsc,color="limegreen",label="Stochastic Model (No Detrainment Damping)")

    
ax.legend(ncol=3,fontsize=8)
savename = "%sACF_Lbdd_Experiment_%s_base_%s_mon%02i.png" % (figpath,expname,expname_base,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#savename = "%s" % ()

#%%

mons3  = proc.get_monstr()

fig,ax = viz.init_monplot(1,1)
for tt in range(ntest+1):
    ax.plot(mons3,checkparam[tt].squeeze(),label=testnames[tt])
    
# Plot CESM
ax.plot()

ax.legend()
#%%

#%%