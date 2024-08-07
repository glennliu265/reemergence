#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regrid POP data, but average across the mixed-layer
Copied  code as [regrid_POP_1level]
Currently works with UVEL/VVEL Output

Created on Aug 7 13:31:33 2024

@author: gliu
"""

import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import scipy as sp
import cartopy.crs as ccrs

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "stormtrack"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict    = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
rawpath     = pathdict['raw_path']

mldpath     = input_path + "mld/"
outpath     = rawpath + "ocn_var_3d/"

vnames      = ["UVEL","VVEL"]
nens        = 42
loopens     = np.arange(1,43,1)#[32,]

idz         = 0

#%% Indicate which files to process

fns_all = []
for v in range(len(vnames)):
    vname = vnames[v]
    fns = []
    for e in loopens:
        fn = "%socn_var_3d/%s/%s_NATL_ens%02i.nc" % (rawpath,vname,vname,e)
        
        fns.append(fn)
    fns_all.append(fns)

# bounding box for final output
bbox    = [-80,0,20,65]

#%% Retrieve TLAT/TLON from a file in outpath
# Note this part should change as I modified preproc_detrainment_data to include tlat and tlon as a coord
fnlat   = "SALT_NATL_ens01.nc" # Name of file to take tlat/tlon information from

dstl    = xr.open_dataset(outpath+fnlat)
tlat    = dstl.TLAT.values
tlon    = dstl.TLONG.values

#%% Retrieve dimensions of CESM1 from another file

mldnc   = "CESM1_HTR_FULL_HMXL_NAtl.nc"
dsh     = xr.open_dataset(mldpath+mldnc).h

dshreg  = dsh.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))

outlat  = dshreg.lat.values
outlon  = dshreg.lon.values

#%% Seems the easier way might just be to do this loopwise (silly but I guess...)

nvars   = len(vnames)

for v in range(nvars):   
    vname = vnames[v] 
    for e in tqdm(range(nens)):
        
        st = time.time()
        fn = fns_all[v][e]
        # Open the Dataset
        ds          = xr.open_dataset(fn).isel(z_t=idz).load()#.load()
        dsl         = ds.assign(lon=(['nlat','nlon'],tlon),lat=(['nlat','nlon'],tlat))
        nlon,nlat   = len(outlon),len(outlat)
        
        # Get Dimensions and preallocate
        ntime,ntlat,ntlon = ds[vname].shape
        ocn_var_avg = np.zeros((ntime,nlat,nlon))
        
        # Looping for each point
        for a in range(nlat):
            
            latf   = outlat[a]
            
            for o in tqdm(range(nlon)): # Took (1h 11 min if you don't load, 2 sec if you load, T-T)
                
                # Longitudes
                lonf = outlon[o]
                if lonf < 0:
                    lonf += 360
                
                # Get the nearest point
                outids = proc.get_pt_nearest(dsl,lonf,latf,tlon_name="lon",tlat_name="lat",returnid=True,debug=False)
                dspt   = dsl.isel(nlat=outids[0],nlon=outids[1])
                
                ocn_var_avg[:,a,o] = dspt[vname].data
                # End Longitude Loop ---
                
            # End Latitude Loop ---
                
        #% Apply mask based on h
        mask                    = np.sum(dshreg.values,(0,1))
        mask[~np.isnan(mask)]   = 1
        da_mask                 = xr.DataArray(mask,coords=dict(lat=outlat,lon=outlon))
        
        # Save Regridded Data
        coords                  = dict(time=ds.time,lat=dshreg.lat,lon=dshreg.lon,)
        da_out                  = xr.DataArray(ocn_var_avg * mask[None,:,:],coords=coords,dims=coords,name=vname)
        
        
        savename = proc.addstrtoext(fn,"_regridNN",adjust=-1)
        edict    = proc.make_encoding_dict(da_out)
        da_out.to_netcdf(savename,encoding=edict)
        print("Saved output in %.2fs" % (time.time()-st))
        # End Ensemble Loop
    # End Variable Loop
    
#%% Open all files and save

for v in range(len(vnames)):
    
    # Variables 
    vname = vnames[v]
    fns   = []
    for e in loopens:
        fn = "%socn_var_3d/%s/%s_NATL_ens%02i_regridNN.nc" % (rawpath,vname,vname,e)
        fns.append(fn)
    
    # Indicate which datasets
    ds_all  = xr.open_mfdataset(fns,concat_dim='ens',combine='nested').load()
    
    # Save the ouput as a bulk file
    edict   = proc.make_encoding_dict(ds_all)
    fn_out  = "%socn_var_3d/%s/%s_NATL_AllEns_regridNN.nc" % (rawpath,vname,vname)
    ds_all.to_netcdf(fn_out,encoding=edict)




    
    
    
    
    


