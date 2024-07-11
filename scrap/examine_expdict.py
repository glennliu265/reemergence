#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Examine the experiment parameters for selected experiments

Copied upper section of compare_regional_metrics


Created on Wed Jul  3 09:49:58 2024

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
import cartopy.crs as ccrs
import os

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
sys.path.append(pathdict['scmpath'] + "../")
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx
import stochmod_params as sparams

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']

# Make Needed Paths
proc.makedir(figpath)

#%% Indicate experiments to load

#  Same as comparing lbd_e effect, but with Evaporation forcing corrections
regionset       = "SSSCSU"
comparename     = "SSS_Paper_Draft01"
expnames        = ["SSS_EOF_LbddCorr_Rerun_lbdE_neg","SSS_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_NoLbdd","SSS_CESM"]
expnames_long   = ["Stochastic Model (sign corrected + $\lambda^e$)","Stochastic Model (with $\lambda^e$)","Stochastic Model","CESM1"]
expnames_short  = ["SM_lbde_neg","SM_lbde","SM","CESM"]
ecols           = ["magenta","forestgreen","goldenrod","k"]
els             = ['dotted',"solid",'dashed','solid']
emarkers        = ['+',"d","x","o"]


#%%

def load_expdict(expname,output_path):
    expdir  = "%s%s/Input/" % (output_path,expname)
    expdict = np.load(expdir + "expparams.npz",allow_pickle=True)
    
    # Convert all inputs into string, if needed
    keys = expdict.files
    expdict_out = {}
    for key in keys:
        if type(expdict[key]) == np.ndarray and (len(expdict[key].shape) < 1): #  Check if size is zero, andnumpy array is present
            expdict_out[key] = expdict[key].item()
        else:
            expdict_out[key] = expdict[key]
    
    return expdict_out

#%%

ex          = 2
expname     = expnames[ex]
exd         = load_expdict(expname,output_path)
exd.keys()

nexps = len(expnames)
for ex in range(nexps):
    expname     = expnames[ex]
    exd         = load_expdict(expname,output_path)
    print("%s : %s" % (expname,exd['lbd_d']))
    

