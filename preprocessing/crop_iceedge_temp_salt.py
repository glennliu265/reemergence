#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Crop TEMP and SALT data preprocessed by preprocess_bylevel_ens ()

For a single ensemble member, take a centerpoint and points N/S/E/W of it with range 3


Created on Fri May 10 11:09:53 2024

@author: gliu
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys


# Load Information
datpath       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/ocn_var_3d/"
outpath       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/ptdata/profile_analysis/"
fns           = ["SALT_NATL_ens01.nc","TEMP_NATL_ens01.nc"]
centerpoints  = ([-37,62],[-55,59],) # Taken from viz_icefrac

#%%
# stormtrack
amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%%

def make_locstring(cp):
    outstr = "lon%03i_lat%02i" % (cp[0],cp[1])
    return outstr

#%%

dirnames = ["Center","N","S","E","W"]
vnames   = ["SALT","TEMP"]
cpnames  = ["IrmingerEns01","LabradorEns01"]

cc = 0
#ff = 0

rng = 3


ds_byvar = []


for cc in range(2):
    
    ds_byvar = []
    for vv in range(2):
        dspt_grid = [] # Initialize List
        
        vname     = vnames[vv]
        
        # Looping for each point
        cp    = centerpoints[cc]
        ds    = xr.open_dataset(datpath + fns[vv])
        
        # First read in the center point
        dspt  = proc.find_tlatlon(ds,cp[0]+360,cp[1])
        dspt_grid.append(dspt)
        
        # Next do each direction
        for di in range(4):
            if di == 0: # North
                cp_in = [cp[0],cp[1]+rng]
                dspt  = proc.find_tlatlon(ds,cp_in[0]+360,cp_in[1])
            elif di == 1: # South
                cp_in = [cp[0],cp[1]-rng]
                dspt  = proc.find_tlatlon(ds,cp_in[0]+360,cp_in[1])
            elif di == 2: # East
                cp_in = [cp[0]+rng,cp[1]]
                dspt  = proc.find_tlatlon(ds,cp_in[0]+360,cp_in[1])
            elif di == 3: # West
                cp_in = [cp[0]-rng,cp[1]]
                dspt  = proc.find_tlatlon(ds,cp_in[0]+360,cp_in[1])
            dspt_grid.append(dspt)
            
        ds_grid  = xr.concat(dspt_grid,dim='dir')
        ds_grid  = ds_grid.assign({'dir':dirnames})
        
        ds_byvar.append(ds_grid)

        # End Variable Loop
    
    
    # End Grid Type Loop
    
    ds_byvar = xr.merge(ds_byvar)
    edict    = proc.make_encoding_dict(ds_byvar)
    savename = "%s%s_CrossPoints.nc" % (outpath,cpnames[cc])
    ds_byvar.to_netcdf(savename,encoding=edict)   
    
    # End Cross Point Loop
    
            
#%%


#edict    = proc.make_encoding_dict(dspt)
#locfn,_  = proc.make_locstring(cp[0]+360,cp[1])
savename = ""

#%%











