#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Revision Figures

Figures for revision of the SSS Paper

Created on Thu Mar  6 08:59:37 2025

@author: gliu

"""


import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import tqdm
import os

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
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']


#%% Indicate Paths

#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240411/"
datpath   = pathdict['raw_path']
outpath   = pathdict['input_path']+"forcing/"
rawpath   = pathdict['raw_path']

revpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/revision_data/"

# ===================================
#%% Figure R1: Number of Modes Needed
# ===================================
# Data Source: /preprocessing/correct_EOF_forcing_SSS.py

fname           = "EOF_Number_Modes_Needed_90pct_Fprime.npy"
nmodes_needed   = np.load(revpath + fname)
eof_thres       = 0.90
mons3           = proc.get_monstr()

fig,ax          = viz.init_monplot(1,1,figsize=(6,4.5))

ax.bar(mons3,nmodes_needed,alpha=0.5,color='darkred',edgecolor="k")
ax.set_xlim([-1,12])
ax.set_title("Number of Modes \n Needed to Explain %.2f" % (eof_thres*100) + "% of Variance",fontsize=16)
ax.set_ylabel("Number of Modes",fontsize=14)
ax.set_xlabel("Month",fontsize=14)
ax.bar_label(ax.containers[0],label_type='center',c='k')

savename        = "%sEOF_Nmodes_Needed.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')
