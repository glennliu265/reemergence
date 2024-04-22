#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copy ocn variable for CESM1 LENS, looping by ensemble
Trying to write this to work on Casper
Copied from [process_bylevel_ens]

Created on Tue Apr 16 23:25:45 2024

@author: gliu
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


#%% Set the paths

outpath = "/glade/u/home/glennliu/scratch/CESM1_NATL_crop/"
datpath = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/ocn/proc/tseries/monthly/"
vname   = "PD"
keepvars = [vname,"TLONG","TLAT","z_t","time"]
mnums   = np.hstack([np.arange(1,36),np.arange(101,108)])

# b.e11.B20TRC5CNBDRD.f09_g16.001.pop.h.PD.185001-200512.nc  
ncstr   = "b.e11.B20TRC5CNBDRD.f09_g16.%03i.pop.h.%s.*.nc" # % (mnum,vname)


# Set Crop Region
bbox    = [-80,0,20,65]


#%% Functions

def sel_region_xr_cv(ds2,bbox,vname,debug=False):
    
    # Get mesh
    tlat = ds2.TLAT.values
    tlon = ds2.TLONG.values
    
    # Make Bool Mask
    latmask = (tlat >= bbox[2]) * (tlat <= bbox[3])
    
    # Three Cases
    # Case 1. Both are degrees west
    # Case 2. Crossing prime meridian (0,360)
    # Case 3. Crossing international date line (180,-180)
    # Case 4. Both are degrees east
    if np.any(np.array(bbox)[:2] < 0):
        print("Degrees West Detected")
        
        if np.all(np.array(bbox[:2])) < 0: # Case 1 Both are degrees west
            print("Both are degrees west")
            lonmask = (tlon >= bbox[0]+360) * (tlon <= bbox[1]+360)
            
        elif (bbox[0] < 0) and (bbox[1] >= 0): # Case 2 (crossing prime meridian)
            print("Crossing Prime Meridian")
            lonmaskE = (tlon >= bbox[0]+360) * (tlon <= 360) # [lonW to 360]
            if bbox[1] ==0:
                lonmaskW = 1
            else:
                lonmaskW = (tlon >= 0)           * (tlon <= bbox[1])       # [0 to lonE]
            
            lonmask = lonmaskE * lonmaskW
        elif (bbox[0] > 0) and (bbox[1] < 0): # Case 3 (crossing dateline)
            print("Crossing Dateline")
            lonmaskE = (tlon >= bbox[0]) * (tlon <= 180) # [lonW to 180]
            lonmaskW = (tlon >= 180)     * (tlon <= bbox[1]+360) # [lonW to 180]
            lonmask = lonmaskE * lonmaskW
    else:
        print("Everything is degrees east")
        lonmask = (tlon >= bbox[0]) * (tlon <= bbox[1])


    regmask = lonmask*latmask

    # Select the box
    if debug:
        plt.pcolormesh(lonmask*latmask),plt.colorbar(),plt.show()
    
    # Make a mask
    ds2 = ds2[vname]#.isel(z_t=1)
    
    ds2.coords['mask'] = (('nlat', 'nlon'), regmask)
    
    st = time.time()
    ds2 = ds2.where(ds2.mask,drop=True)
    print("Loaded in %.2fs" % (time.time()-st))
    return ds2

# Copied from amv.proc on 2024.04.17
def ds_dropvars(ds,keepvars):
    '''Drop variables in ds whose name is not in the list [keepvars]'''
    # Drop unwanted dimension
    dsvars = list(ds.variables)
    remvar = [i for i in dsvars if i not in keepvars]
    ds = ds.drop(remvar)
    return ds

def make_encoding_dict(ds,encoding_type='zlib'):
    keys   = list(ds.keys())
    values = ({encoding_type:True},) * len(keys)
    encoding_dict = { k:v for (k,v) in zip(keys,values)}
    return encoding_dict


# Up to here =========================================


#%% Debug Segment, given the file

#datpath = "/Users/gliu/Globus_File_Transfer/CESM1_LE/Historical/PD/"
#ncname  = "b.e11.B20TRC5CNBDRD.f09_g16.002.pop.h.PD.192001-200512.nc"
#outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"



ie   = 1

for ie in range(42):
    if ie == 0:
        continue
    
    mnum   = mnums[ie]
    if ie == 0:
        timestr = "185001-200512"
    else:
        timestr = "192001-200512"
    ncname = "%s//b.e11.B20TRC5CNBDRD.f09_g16.%03i.pop.h.%s.%s.nc"  % (vname,mnum,vname,timestr)
    
    
    # Get necessary variables
    ds      = xr.open_dataset(datpath+ncname)#[vname]
    tlat    = ds.TLAT
    tlon    = ds.TLONG
    zt      = ds.z_t
    dsvar   = ds.PD
    
    st      = time.time()
    ds2     = ds_dropvars(ds,keepvars)
    dsreg   = sel_region_xr_cv(ds2,bbox,vname,debug=False)
    print("Cropped data in %.2fs" % (time.time()-st)) # 1484.40
    
    
    #%% Save Data
    st      = time.time()
    edict    = {vname:dict(zlib=True)}
    savename = "%sCESM1_HTR_FULL_%s_NAtl_Crop.nc" % (outpath,vname)
    dsreg.to_netcdf(savename,encoding=edict)
    print("Saved data in %.2fs" % (time.time()-st)) # 16.84s, 425 MB
 

