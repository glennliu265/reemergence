#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basinwide expeirment parameters for run_SSS_basinwide
Created on Sun Feb  4 19:06:41 2024

@author: gliu
"""




#%%


"""
SST_EOF_Lbddcorr Rerun, but with LHFLX Forcing/damping only!
"""

expname     = "SST_EOF_LHFLX"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SST_EOF_LbddCorr_Rerun", # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc", # Can use Fprime or Eprime (stochastic LHFLX, NOT sign converted..)
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_LHFLX_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : None, # Must be in W/m2
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    }




#%%

"""
SST EOF Lbdd Update (Corrected Fprime, Correlation based Lbdd taken at the detrainment
                     depth)

"""

expname     = "SST_EOF_LbddCorr_Rerun"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : None, # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_corrected_nomasklag1_nroll0_perc090_NAtl_EnsAvg.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : "CESM1_HTR_FULL_Qek_SST_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc", # Must be in W/m2
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None,
    "Tforce"            : None,
    }

#%%

"""
Same as above, but for SSS

"""
expname     = "SSS_EOF_LbddCorr_Rerun"

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
    "Td_corr"           : False, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None,
    "Tforce"            : None,
    }

#%%
"""

SST EOF Lbdd Update (Same as Above, but no detrainment damping lbd_d)

"""

expname     = "SST_EOF_LbddCorr_Rerun_NoLbdd"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SST_EOF_LbddCorr_Rerun", # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_corrected_nomasklag1_nroll0_perc090_NAtl_EnsAvg.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : None,
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : "CESM1_HTR_FULL_Qek_SST_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc", # Must be in W/m2
    'convert_Fprime'    : True,
    'convert_lbd_a'     : True, # ALERT!! Need to rerun with this set to true....
    'convert_PRECTOT'   : False,
    'convert_LHFLX'     : False,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    "entrain"           : True ,
    "eof_forcing"       : True ,
    "Td_corr"           : False, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None ,
    "Tforce"            : None ,
    }

#%%

"""

Same as above, but for SSS

"""
expname     = "SSS_EOF_LbddCorr_Rerun_NoLbdd"

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
    'lbd_d'             : None,
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
    "Td_corr"           : False,
    "lbd_e"             : None,
    "Tforce"            : None,
    }
#%%

"""

SSS With SST-Evaporation Feedback (same as SSS experiment above, but add SST-Evaporation)

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
"""
Same as above, but rerun after the script was corrected to flip Eprime's sign.
"""
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



#%% 2024.05.07 SST-SSS Coupled Runs (copied from run_SSS_pointmode_coupled.)

"""
LHFLX Run (SST_SSS  Coupled, from early may prior to 2024.05.07)
"""

# Paths and Experiment
expname     = "SST_SSS_LHFLX" # Borrowed from "SST_EOF_LbddCorr_Rerun"
expparams_sst   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : None, # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_LHFLX_damping_nomasklag1_EnsAvg.nc", # Only latent heat flux
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
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
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : "CESM1LE_HTR_FULL_lbde_Bcorr3_lbda_qnet_damping_nomasklag1_EnsAvg.nc",
    "Tforce"            : "SST_SSS_LHFLX",
    }


"""
Another Version, but without lbd_d

"""

expname     = "SST_SSS_LHFLX_NoLbdd" # Borrowed from "SST_EOF_LbddCorr_Rerun"
expparams_sst   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : None, # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : False,
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_LHFLX_damping_nomasklag1_EnsAvg.nc", # Only latent heat flux
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
    'lbd_d'             : False,
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




#%% ------------------------------------------------------------------------------------


"""
SST EOF  LbddEnsMean

Same as Pilot run, but more simulations, and using ens-averaged detrainment damping

"""

expname     = "SST_EOF_LbddEnsMean"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : None, # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_corrected_nomasklag1_nroll0_perc090_NAtl_EnsAvg.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_SST_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAvg.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : "CESM1_HTR_FULL_Qek_SST_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc", # Must be in W/m2
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
    }

#%%
"""
Save as Above, but for SSS
"""


expname     = "SSS_EOF_Qek_LbddEnsMean"

expparams   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(5,10,1)],
    'runid_path'        : "SST_EOF_LbddEnsMean",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'LHFLX'             : "CESM1_HTR_FULL_LHFLX_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_SSS_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAvg.nc",
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

#%%


"""
SST EOF  NoLbdd

Same as SST Run above, but with no detrainment damping.


"""

expname     = "SST_EOF_NoLbdd"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : None, # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_corrected_nomasklag1_nroll0_perc090_NAtl_EnsAvg.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : None,
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : "CESM1_HTR_FULL_Qek_SST_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc", # Must be in W/m2
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
    }

#%%
"""
Save as Above for SSS, no detrainment damping
"""


expname     = "SSS_EOF_NoLbdd"

expparams   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(5,10,1)],
    'runid_path'        : "SST_EOF_LbddEnsMean",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'LHFLX'             : "CESM1_HTR_FULL_LHFLX_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : None,
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


#%%

#%%


"""
SSS_EOF_Qek_Pilot

Note: The original run (2/14) had the incorrect Ekman Forcing and used ens01 detrainment damping with linear detrend
I reran this after fixing these issues (2/29)

"""


expname     = "SSS_EOF_Qek_pilot"

expparams   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,5,1)],
    'runid_path'        : None,#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_EnsAvg.nc",
    'LHFLX'             : "CESM1_HTR_FULL_LHFLX_EOF_nomasklag1_nroll0_NAtl_EnsAvg.nc",
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


#%%

"""
Initial Run for SST with EOF Forcing, Qek, Lbd_d

Same as above: 
The original run (2/14) had the incorrect Ekman Forcing and used ens01 detrainment damping with linear detrend
I reran this after fixing these issues (2/29)


"""
expname     = "SST_EOF_Qek_pilot"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,5,1)],
    'runid_path'        : None, # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_EOF_corrected_nomasklag1_nroll0_perc090_NAtl_EnsAvg.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_SST_Expfit_lbdd_monvar_detrendensmean_lagmax3_Ens01.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : "CESM1_HTR_FULL_Qek_SST_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc", # Must be in W/m2
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
    }

#%%

"""
Same as above but for SSS

"""

expname     = "SSS_EOF_Qek_pilot"

expparams   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,5,1)],
    'runid_path'        : "SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_EnsAvg.nc",
    'LHFLX'             : "CESM1_HTR_FULL_LHFLX_EOF_nomasklag1_nroll0_NAtl_EnsAvg.nc",
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_SSS_Expfit_lbdd_monvar_detrendlinear_lagmax3_Ens01.nc",
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


# ---------- o ---------- o ---------- o ---------- o ---------- o ---------- o 
#%% ---------------------------

"""

Run for OSM but for SSS

"""

expname     = "SSS_OSM_Tddamp"

expparams   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,0,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(1,11,1)],
    'runid_path'        : "SST_OSM_Tddamp", # If true, load a runid from another directory
    'Fprime'            : None, 
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc",
    'LHFLX'             : "CESM1_HTR_FULL_Eprime_nroll0_NAtl_EnsAvg.nc",
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_SSS_Expfit_lbdd_maxhclim_lagsfit123_Ens01.nc",
    'Sbar'              : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : None, # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'convert_Fprime'    : False,
    'convert_lbd_a'     : True,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    "entrain"           : True,
    }

#%% ---------------------------
"""
Run for OSM, with default qnet damping and Fprime forcing
No Ekman Advection,  No Shift

Tddamp estimated from ensemble 01

"""

expname     = "SST_OSM_Tddamp"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,0,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(1,11,1)],
    'runid_path'        : "SST_OSM_Tddamp", # If true, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_nroll0_NAtl_EnsAvg.nc",       
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_SST_Expfit_lbdd_maxhclim_lagsfit123_Ens01.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'convert_Fprime'    : True,
    'convert_lbd_a'     : True,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    "entrain"           : True,
    }

#%% ---------------------------

"""
Same as SSS Run above, but with no entrainment
"""
expname     = "SSS_OSM_Tddamp_noentrain"
expparams   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,0,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(1,11,1)],
    'runid_path'        : "SST_OSM_Tddamp", # If true, load a runid from another directory
    'Fprime'            : None, 
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc",
    'LHFLX'             : "CESM1_HTR_FULL_Eprime_nroll0_NAtl_EnsAvg.nc",
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_SSS_Expfit_lbdd_maxhclim_lagsfit123_Ens01.nc",
    'Sbar'              : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : None, # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'convert_Fprime'    : False,
    'convert_lbd_a'     : True,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    "entrain"           : False, # Set to True to use entraining model
    }


#%% 

"""
same as above for for SST
"""

expname     = "SST_OSM_Tddamp_noentrain"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,0,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(1,11,1)],
    'runid_path'        : "SST_OSM_Tddamp", # If true, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_nroll0_NAtl_EnsAvg.nc",       
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_SST_Expfit_lbdd_maxhclim_lagsfit123_Ens01.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'convert_Fprime'    : True,
    'convert_lbd_a'     : True,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    "entrain"           : False,
    }


#%%

"""

Entraining SST Run, but with Qek

"""

expname     = "SST_OSM_Tddamp_Qek_monvar"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,0,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(1,11,1)],
    'runid_path'        : "SST_OSM_Tddamp", # If true, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_nroll0_NAtl_EnsAvg.nc",       
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_SST_Expfit_lbdd_maxhclim_lagsfit123_Ens01.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'Qek'               : "CESM1_HTR_FULL_Qek_monstd_NAtl_EnsAvg.nc", # Must be in W/m2
    'convert_Fprime'    : True,
    'convert_lbd_a'     : True,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    "entrain"           : True,
    }


#%% Next Section, Testing Different Shifts

# ---------- o ---------- o ---------- o ---------- o ---------- o ---------- o 
# ---------- o ---------- o ---------- o ---------- o ---------- o ---------- o 

"""
Run of half-shifted damping/forcing/mld over the Atlantic Basin

"""
expname     = "Test_Td0.1_SPG_allroll1_halfmode"

expparams   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-65,0,45,65],
    'nyrs'              : 1000,
    'runids'            : ["test%02i" % i for i in np.arange(1,6,1)],
    'runid_path'        : None, # If true, load a runid from another directory
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc",
    'LHFLX'             : "CESM1_HTR_FULL_Eprime_nroll0_NAtl_EnsAvg.nc",
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : 0.10,
    'Sbar'              : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : None, # NEEDS TO BE ALREADY CONVERTED TO 1/Mon !!!
    'convert_Fprime'    : True,
    'convert_lbd_a'     : True,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'             : 1,
    'mroll'             : 1,
    'droll'             : 1,
    'halfmode'          : True,
    }

"""


"""




"""
NAT Extratropics run with Expfit Lbda
"""


expname     = "SST_expfit_damping_20to65"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(1,6,1)],
    'runid_path'        : "SST_covariance_damping_20to65", # If true, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_Expfitlbda123_nroll0_NAtl_EnsAvg.nc",       
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : None,
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_Expfit_lbda_damping_lagsfit123_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'convert_Fprime'    : True,
    'convert_lbd_a'     : False,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    }

"""
Same as above but with SST ACF fit (full rather than just lbd_a estimate)
"""

expname     = "SST_expfit_SST_damping_20to65"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(1,6,1)],
    'runid_path'        : "SST_covariance_damping_20to65", # If true, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_ExpfitSST123_nroll0_NAtl_EnsAvg.nc",       
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : None,
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_Expfit_SST_damping_lagsfit123_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'convert_Fprime'    : True,
    'convert_lbd_a'     : False,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    }

#%%