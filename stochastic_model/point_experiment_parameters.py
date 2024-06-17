#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:43:40 2024

@author: gliu
"""



"""
"""

# =============================================================================
#%% Testing Effect of Adding Precipitation, Qek, and other forcings
# =============================================================================

"""
Experiment Set: Adding to LHFLX Base Run

- Base: "SSS_EOF_LbddCorr_Rerun_lbdE_neg"
- Add Precip: "SSS_LHFLX_addP"
- Add Qek Only:   "SSS_LHFLX_addQek"
- Add Qek: "SSS_LHFLX_addP_addQek"


"""


#%% Add Qek Only

"""

"SSS_LHFLX_addP_addQek"

# Add Qek and P

# Rerun "SSS_EOF_LbddCorr_Rerun_lbdE_neg" but with precip

"""

expname         = "SSS_LHFLX_addQek"#_DiffWn" # Borrowed from "SST_EOF_LbddCorr_Rerun"
expparams_sst   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SSS_EOF_LbddCorr_Rerun_lbdE_neg",#expname, # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_LHFLX_damping_nomasklag1_EnsAvg.nc", # Only latent heat flux # "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc",#
    'Qek'               : "CESM1_HTR_FULL_Qek_SST_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc", # No Qekman #Must be in W/m2
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
    'runid_path'        : "SSS_EOF_LbddCorr_Rerun_lbdE_neg",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : None, # No Precip
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : "CESM1LE_HTR_FULL_lbde_Bcorr3_lbda_qnet_damping_nomasklag1_EnsAvg.nc",
    "Tforce"            : expname,#"SST_SSS_LHFLX_2_neg",#None,#expname,
    }




#%% Add P and Qek

"""

"SSS_LHFLX_addP_addQek"

# Add Qek and P

# Rerun "SSS_EOF_LbddCorr_Rerun_lbdE_neg" but with precip

"""

expname         = "SSS_LHFLX_addP_addQek"#_DiffWn" # Borrowed from "SST_EOF_LbddCorr_Rerun"
expparams_sst   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SSS_EOF_LbddCorr_Rerun_lbdE_neg",#expname, # If not None, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc",
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_LHFLX_damping_nomasklag1_EnsAvg.nc", # Only latent heat flux # "CESM1_HTR_FULL_qnet_damping_nomasklag1_EnsAvg.nc",#
    'Qek'               : "CESM1_HTR_FULL_Qek_SST_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc", # No Qekman #Must be in W/m2
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
    'runid_path'        : "SSS_EOF_LbddCorr_Rerun_lbdE_neg",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc", # No Precip
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : "CESM1LE_HTR_FULL_lbde_Bcorr3_lbda_qnet_damping_nomasklag1_EnsAvg.nc",
    "Tforce"            : expname,#"SST_SSS_LHFLX_2_neg",#None,#expname,
    }





#%% Add P
"""

"SSS_LHFLX_addP"

# Rerun "SSS_EOF_LbddCorr_Rerun_lbdE_neg" but with precip

"""
expname         = "SSS_LHFLX_addP"#_DiffWn" # Borrowed from "SST_EOF_LbddCorr_Rerun"
expparams_sst   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SSS_EOF_LbddCorr_Rerun_lbdE_neg",#expname, # If not None, load a runid from another directory
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None,
    "Tforce"            : None,
    }

expparams_sss   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SSS_EOF_LbddCorr_Rerun_lbdE_neg",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
    'Fprime'            : None,
    'PRECTOT'           : "CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc", # No Precip
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : "CESM1LE_HTR_FULL_lbde_Bcorr3_lbda_qnet_damping_nomasklag1_EnsAvg.nc",
    "Tforce"            : expname,#"SST_SSS_LHFLX_2_neg",#None,#expname,
    }



# =============================================================================
#%% Stuff Below this was from the LHFLX Runs (Testing SST-Lbd_e Coupling)
# =============================================================================


"""
LHFLX Run (SST_SSS  Coupled, from early may prior to 2024.05.07)
"""

# Paths and Experiment
expname         = "SST_SSS_LHFLX_2_noLbdE"#_DiffWn" # Borrowed from "SST_EOF_LbddCorr_Rerun"
expparams_sst   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SST_SSS_LHFLX_2",#expname, # If not None, load a runid from another directory
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None,
    "Tforce"            : None,
    }

expparams_sss   = {
    'varname'           : "SSS",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(0,10,1)],
    'runid_path'        : "SST_SSS_LHFLX_2",#"SST_EOF_Qek_pilot", # If not None, load a runid from another directory
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
    "Td_corr"           : True, # Set to True if lbd_d is provided as a correlation, rather than 1/months
    "lbd_e"             : None,#"CESM1LE_HTR_FULL_lbde_Bcorr3_lbda_qnet_damping_nomasklag1_EnsAvg.nc",
    "Tforce"            : None,#"SST_SSS_LHFLX_2_neg",#None,#expname,
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