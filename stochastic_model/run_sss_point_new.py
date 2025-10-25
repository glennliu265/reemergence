
import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
import matplotlib.patheffects as PathEffects
from cmcrameri import cm

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine     = "Astraeus"

# First Load the Parameter File
cwd         = os.getcwd()
sys.path.append(cwd+ "/..")
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
machine     = "Astraeus"
pathdict    = rparams.machine_paths[machine]

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
rawpath     = pathdict['raw_path']


#%%


Fprime = np.ones(12)