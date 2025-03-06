#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

re-emergence project parameters

Created on Mon Mar  4 13:15:55 2024

@author: gliu

"""

#
# Main Variables
#

import numpy as np
import cartopy.crs as ccrs
outdate = "20250307" # Date of the next weekly meeting

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
    "procpath"          : 0,        # Path to analysis output/processed data
    "lipath"            : 0,        # Path to Land Ice Mask
    "amvpath"           : 0,        # Path to analysis tools module...
    "scmpath"           : 0,        # Path to stochastic model module
    "figpath"           : 0,        # Path to figure output
    }
"""

# Astraeus Local ---
mdict0 = {
    "machine"           : "Astraeus",     # Name of the machine
    "raw_path"          : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/",      # Path to pre-processed CESM1 Output, cropped to NATL
    "input_path"        : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/",          # Path to stochastic model input (forcing, damping, mld)
    "output_path"       : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/",            # Path to stochastic model output
    "procpath"          : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/",                      # Path to analysis output/processed data
    "lipath"            : "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Masks/",               # Path to Land Ice Mask
    "amvpath"           : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/",                            # Path to analysis tools module...
    "scmpath"           : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/",            # Path to stochastic model module
    "figpath"           : "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/%s/"  % outdate,          # Path to figure output
    }

# Stormtrack Server ---
mdict1 = {
    "machine"           : "stormtrack", # Name of the machine
    "raw_path"          : "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/",    # Path to post-processed CESM1 Output
    "input_path"        : "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/",        # Path to stochastic model input (forcing, damping, mld)
    "output_path"       : "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/",          # Path to stochastic model output
    "procpath"          : "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/",                    # Path to analysis output/processed data
    "lipath"            : "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/masks/",  # Path to Land Ice Mask    
    "amvpath"           : "/home/glliu/00_Scripts/01_Projects/00_Commons/",                                         # Path to analysis tools module...
    "scmpath"           : "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/",                  # Path to stochastic model module
    "figpath"           : "/home/glliu/02_Figures/01_WeeklyMeetings/%s/"  % outdate,                                # Path to figure output
    }

# Make the [machine_paths] dictionary
machine_path_dicts  = (mdict0,mdict1,)
machine_names       = [d["machine"] for d in machine_path_dicts]
machine_paths       = dict(zip(machine_names,machine_path_dicts))

# --------------------------------------
#%% Regional Analysis Subsets
# -------------------------------------
rsubset1 = dict(
    selname      = "SMPaper",
    bboxes       = [[-60, -15,  40,  60], [-80,   0,  10,  60]      , [-40, -10,  20,  40]      , [-80, -40,  20,  40]],
    regions      = ['SPG'               , 'NNAT'                    , 'STGe'                    , 'STGw'],
    regions_long = ("Subpolar Gyre"     , "Northern North Atlantic" , "Subtropical Gyre East"   , "Subtropical Gyre West"),
    rcols        = ("blue"              , "black"                   , "magenta"                 , "r"),
    rsty         = ("solid"             , "dotted"                  , "dashed"                  ,"dotted")
    )

rsubset2 = dict(
    selname      = "OSM24",
    bboxes       = [[-70,-55,30,40] , [-45,-25,50,60]],
    regions      = ["STG"           , "SPG"],
    regions_long = ("Sargasso Sea"  , "SE Greenland"),
    rcols        = ("limegreen"     , "cornflowerblue"),
    rsty         = ("solid"         , "solid"),
    )

rsubset3 = dict(
    selname     = "TCMPi24",
    bboxes      = [[-45,-25,50,60], [-65,-40,40,47]     , [-50,-20,20,30]   , [-20,-10,30,50]  ],
    regions      = ["SPG"           , "TZ"              , "AZO"             , "STGe"],
    regions_long = ("SE Greenland"  , "Transition Zone" , "Azores High"     , "Eastern Subtropics"),
    rcols        = ("navy"          , "firebrick"       , "limegreen"       , "magenta"      ),
    rsty         = ("solid"         , "solid"           , "dashed"          , "dotted"      ),
        
    )

# Regions based on SSS Re-emergence Maxima
rsubset4 = dict(
    selname     = "SSSCSU",
    bboxes      = [[-70,-55,35,40]        , [-40,-30,40,50]     , [-45,-40,20,25]   , [-40,-25,50,60]  ],
    regions      = ["SAR"                 , "NAC"               , "AZO"             , "STGe"], # Lastone is a misnomer
    regions_long = ("Sargasso Sea"        , "N. Atl. Current"   , "Azores High"     , "Irminger Sea"),
    rcols        = ("navy"                , "firebrick"         , "limegreen"       , "magenta"      ),
    rsty         = ("solid"               , "solid"             , "dashed"          , "dotted"      ),
    )

# Make the [region_sets] dictionary
region_dicts  = (rsubset1,rsubset2,rsubset3,rsubset4)
region_names  = (r['selname'] for r in region_dicts)
region_sets   = dict(zip(region_names,region_dicts))


# --------------------------------------
#%% Point Subsets
# -------------------------------------

# Regions based on SSS Re-emergence Maxima
psubset1 = dict(
    selname     = "PaperDraft02",
    bboxes      =  [[-65,36]              , [-39,44]            , [-35,53]  ],
    regions      = ["SAR"                 , "NAC"               , "IRM"], 
    regions_long = ("Sargasso Sea"        , "N. Atl. Current"   , "Irminger Sea"),
    rcols        = ("navy"                , "firebrick"         , "magenta"      ),
    rsty         = ("solid"               , "solid"             , "dotted"      ),
    )

# Make the [region_sets] dictionary
point_dicts  = (psubset1,)
point_names  = (r['selname'] for r in point_dicts)
point_sets   = dict(zip(point_names,point_dicts))

#%% Bounding Box For regional plots

# rdict = dict(
#     name        = "Irminger Sea",
#     name_short  = "IRM",
#     bbox        = [-40,-25,50,60],
#     bboxplot    = [-70,-10,55,70],
#     centlon     = -
    
#     )
# bbox_irm


#%% Input Variable Names and Information

v0 = dict(
    name="",        # Variable Name (used in keys for dictionary, load_params, etcs)
    longname='',    # Variable Longname
    vname='',        # Which equation (SSS or SST)
    vunit='',       # Unconverted UNit
    vunit_conv='',   # Converted Unit
    c='',           # Color
    ls='',          # Linestyle
    mk='',          # Marker Style
    )

v1 = dict(
    name        ="Qek",        # Variable Name (used in keys for dictionary, load_params, etcs)
    longname    ='Ekman Forcing',    # Variable Longname
    vname       ='SST',        # Which equation (SSS or SST)
    vunit       ='W/m2',       # Unconverted UNit
    vunit_conv  ='',   # Converted Unit
    c='',           # Color
    ls='',          # Linestyle
    mk='',          # Marker Style
    )

#%% Plotting Parameters (SST and SSS)

vnames          = ["SST","SSS"]                 # Abbrv.
vnames_long     = ["Sea Surface Temperature","Sea Surface Salinity"] # Full Name
vnames_3d       = ["TEMP","SALT"]               # 3-D variable name in CESM1

vunits          = ["\degree C","psu"]           # Units
vmarkers        = ["o",'d']                     # Markers
vcolors         = ["hotpink","navy"]            # Individual Colors for Lines, etc
vcmaps          = ["cmo.thermal",'cmo.haline']  # Colormaps


#%% Plotting Params (Maps)

proj            = ccrs.PlateCarree()





