#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

re-emergence project parameters

Created on Mon Mar  4 13:15:55 2024

@author: gliu

"""

# Main Variables
#

outdate = "20240308" # Date of the next weekly meeting

# -----------------------------------------------------------------------
#%% Module and (Raw) Data Paths
# -----------------------------------------------------------------------
# Add Relative paths here
# Copied from predict_amv_params

# Template below:
"""
mdict0 = {
    "machine"           : None,     # Name of the machine
    "raw_path"          : 0,        # Path to post-processed CESM1 Output
    "input_path"        : 0,        # Path to stochastic model input (forcing, damping, mld)
    "output_path"       : 0,        # Path to stochastic model output
    "amvpath"           : 0,        # Path to analysis tools module...
    "scmpath"           : 0,        # Path to stochastic model module
    "figpath"           : 0,        # Path to figure output
    }
"""

# Astraeus Local ---
mdict0 = {
    "machine"           : "Astraeus",     # Name of the machine
    "raw_path"          : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/", # Path to pre-processed CESM1 Output, cropped to NATL
    "input_path"        : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/",        # Path to stochastic model input (forcing, damping, mld)
    "output_path"       : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/",        # Path to stochastic model output
    "amvpath"           : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/",        # Path to analysis tools module...
    "scmpath"           : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/",        # Path to stochastic model module
    "figpath"           : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/%s/"  % outdate,# Path to figure output
    }

# Stormtrack Server ---
mdict1 = {
    "machine"           : "stormtrack", # Name of the machine
    "raw_path"          : "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/",    # Path to post-processed CESM1 Output
    "input_path"        : "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/",        # Path to stochastic model input (forcing, damping, mld)
    "output_path"       : "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/",          # Path to stochastic model output
    "amvpath"           : "/home/glliu/00_Scripts/01_Projects/00_Commons/",                                         # Path to analysis tools module...
    "scmpath"           : "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/",                  # Path to stochastic model module
    "figpath"           : "/home/glliu/02_Figures/01_WeeklyMeetings/%s/"  % outdate,                                # Path to figure output
    }

# Make the [machine_paths] dictionary
machine_path_dicts  = (mdict0,mdict1,)
machine_names       = [d["machine"] for d in machine_path_dicts]
machine_paths       = dict(zip(machine_names,machine_path_dicts))


#%%

