#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get MLD Max

Grab the deepest MLD (basinwide, or pointwise)
for detrainment damping calculations

Created on Tue Feb  6 16:03:14 2024

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Get the deepest mixed layer depth

fp = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
fn = "CESM1_HTR_FULL_HMXL_NAtl.nc"


ds = xr.open_dataset(fp+fn)

#  Get maximum MLD by ensemble member [lat x lon]
hmax_byens       = ds.h.max(('lat','lon','mon'))  # Mean by Ens
hmax_bypoint     = ds.h.max(('mon','ens'))        # Ens Mean


# Plot it
hmax_bypoint.plot(cmap='cmo.dense'),plt.show()

# Get overall hmax
hmax_tot = hmax_byens.max()

# Check max and min by ensemble (to see if it is worth doing this)



# This is the value: 1578.39808548 meters


#%% Lets Load z_t


nc1= "CESM1_FULL_PIC_SALT.nc"
ncp = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon330_lat50/"
dspt = xr.open_dataset(ncp+nc1)

z_t = dspt.z_t.values
(z_t < (hmax_tot.values*100)).sum()


#%% Try loading file ons tormtrack
import time
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

hthres =  1578.3980854802585*100

st     = time.time()
fn1    = "b.e11.B20TRC5CNBDRD.f09_g16.002.pop.h.SALT.192001-200512.nc"
fp1    = "/stormtrack/data4/glliu/01_Data/CESM1_LE/SALT/"

ds2    = xr.open_dataset(fp1+fn1)


# Get mesh
tlat = ds2.TLAT.values
tlon = ds2.TLONG.values

# Bounding box
bbox = [-80,0,0,65]

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
plt.pcolormesh(lonmask*latmask),plt.colorbar(),plt.show()


# Make a mask
ds2 = ds2.SALT.isel(z_t=1)

ds2.coords['mask'] = (('nlat', 'nlon'), regmask)

st = time.time()
ds2 = ds2.where(ds2.mask,drop=True)
print("Loaded in %.2fs" % (time.time()-st))
#%%


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
    ds2 = ds2[vname].isel(z_t=1)
    
    ds2.coords['mask'] = (('nlat', 'nlon'), regmask)

    st = time.time()
    ds2 = ds2.where(ds2.mask,drop=True)
    print("Loaded in %.2fs" % (time.time()-st))
    return ds2
        



