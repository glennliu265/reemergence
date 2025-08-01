#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Set up experiment folder for stochatic model

Copied Upper Section of run SSS basinwide

Created on Wed Jul  2 08:47:37 2025

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
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
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


expname     = "SST_Obs_Pilot_00_Tdcorr1_qnet_noPositive_SPGNE"
expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-40,-15,52,62],
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None, # Relevant for SSS
    "Tforce"            : None, # Relevant for SSS
    "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    }



expname     = "SST_Obs_Pilot_00_Tdcorr0_qnet_AConly_SPGNE"
expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-40,-15,52,62],
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
    'lbd_a'             : "ERA5_qnet_damping_AConly.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
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


expname     = "SST_Obs_Pilot_00_Tdcorr0_qnet_p10_SPGNE"
expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-40,-15,52,62],
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
    'lbd_a'             : "ERA5_qnet_damping_p10.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
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


#%

expname     = "SST_Obs_Pilot_00_Tdcorr0_qnet_p20_SPGNE"
expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-40,-15,52,62],
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
    'lbd_a'             : "ERA5_qnet_damping_p20.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
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

#%% SST Full Run (RegTau)
expname     = "SST_Revision_Qek_TauReg_NoQek"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : None, # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_nomasklag1_nroll0_NAtl_concatEns_corrected_EnsAvgFirst.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
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
    "eof_forcing"       : True,
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None, # Relevant for SSS
    "Tforce"            : None, # Relevant for SSS
    "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    }


expname     = "SSS_Revision_Qek_TauReg_NoQek"

expparams   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SST_Revision_Qek_TauReg_NoQek",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_concatEns_corrected_EnsAvgFirst.nc",
    'LHFLX'             : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_concatEns_corrected_EnsAvgFirst.nc",
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_SALT_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
    'Sbar'              : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : None, # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : None, # See convert_Qek
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
    "Tforce"            : "SST_Revision_Qek_TauReg_NoQek",
    "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    }


expname     = "SST_Obs_Pilot_00_qnet_AConly_SPGNE_addlbdd"
expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-40,-15,52,62],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : 'SST_Obs_Pilot_00_Tdcorr0_qnet', # If not None, load a runid from another directory
    'Fprime'            : "ERA5_Fprime_QNET_std_pilot.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "MIMOC_regridERA5_h_pilot.nc",
    'lbd_d'             : "EN4_MIMOC_corr_d_TEMP_detrendbilinear_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2021_regridERA5.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "MIMOC_regridERA5_kprev_pilot.nc",
    'lbd_a'             : "ERA5_qnet_damping_AConly.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None, # Relevant for SSS
    "Tforce"            : None, # Relevant for SSS
    "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    }

expname     = "SST_Obs_Pilot_00_qnet_AConly_SPG_addlbdd"
expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-60,-0,45,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : 'SST_Obs_Pilot_00_Tdcorr0_qnet', # If not None, load a runid from another directory
    'Fprime'            : "ERA5_Fprime_QNETpilotObsAConly_std_pilot.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "MIMOC_regridERA5_h_pilot.nc",
    'lbd_d'             : "EN4_MIMOC_corr_d_TEMP_detrendbilinear_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2021_regridERA5.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "MIMOC_regridERA5_kprev_pilot.nc",
    'lbd_a'             : "ERA5_qnet_damping_AConly.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None, # Relevant for SSS
    "Tforce"            : None, # Relevant for SSS
    "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    }




expname     = "SST_Obs_Pilot_00_qnet_AConly_SPGNE_noREM"
expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-40,-15,50,62],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : 'SST_Obs_Pilot_00_Tdcorr0_qnet', # If not None, load a runid from another directory
    'Fprime'            : "ERA5_Fprime_QNETpilotObsAConly_std_pilot.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "MIMOC_regridERA5_h_pilot.nc",
    'lbd_d'             : None,
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "MIMOC_regridERA5_kprev_pilot.nc",
    'lbd_a'             : "ERA5_qnet_damping_AConly.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None, # Relevant for SSS
    "Tforce"            : None, # Relevant for SSS
    "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    }


# expname     = "SST_Obs_Pilot_00_qnet_AConly_SPGNE_ORAS5"
# expparams   = {
#     'varname'           : "SST",
#     'bbox_sim'          : [-40,-15,52,62],
#     'nyrs'              : 1000,
#     'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
#     'runid_path'        : 'SST_Obs_Pilot_00_Tdcorr0_qnet', # If not None, load a runid from another directory
#     'Fprime'            : "ERA5_Fprime_QNETpilotObsAConly_std_pilot.nc",
#     'PRECTOT'           : None,
#     'LHFLX'             : None,
#     'h'                 : "MIMOC_regridERA5_h_pilot.nc",
#     'lbd_d'             : "ORAS5_MIMOC_corr_d_TEMP_detrendRAW_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2018_regridERA5.nc",
#     'Sbar'              : None,
#     'beta'              : None, # If None, just compute entrainment damping
#     'kprev'             : "MIMOC_regridERA5_kprev_pilot.nc",
#     'lbd_a'             : "ERA5_qnet_damping_AConly.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
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



expname     = "SST_ORAS5_avg"
expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-40,-15,52,62],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : 'SST_Obs_Pilot_00_Tdcorr0_qnet', # If not None, load a runid from another directory
    'Fprime'            : "ERA5_Fprime_QNETpilotObsAConly_std_pilot.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "MIMOC_regridERA5_h_pilot.nc",
    'lbd_d'             : "ORAS5_avg_MIMOC_corr_d_TEMP_detrendRAW_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2024_regridERA5.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "MIMOC_regridERA5_kprev_pilot.nc",
    'lbd_a'             : "ERA5_qnet_damping_AConly.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None, # Relevant for SSS
    "Tforce"            : None, # Relevant for SSS
    "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    }

expname     = "SST_ORAS5_avg_mld003"
expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-40,-15,52,62],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : 'SST_Obs_Pilot_00_Tdcorr0_qnet', # If not None, load a runid from another directory
    'Fprime'            : "ERA5_Fprime_QNETpilotObsAConly_std_pilot.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "ORAS5_CDS_regridERA5_h.nc",
    'lbd_d'             : "ORAS5_avg_mld003_ORAS5mld003_corr_d_TEMP_detrendRAW_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2024_regridERA5.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "ORAS5_CDS_regridERA5_kprev.nc",
    'lbd_a'             : "ERA5_qnet_damping_AConly.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None, # Relevant for SSS
    "Tforce"            : None, # Relevant for SSS
    "correct_Qek"       : False, # Set to True if correction factor to Qek was calculated
    "convert_Qek"       : False, # Set to True if Qek is in W/m2 (True for old SST forcing...) False if in psu/sec or degC/sec (for new scripts)
    }


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
