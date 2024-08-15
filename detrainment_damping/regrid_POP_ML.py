
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regrid POP data, but average across the mixed-layer
Copied  code as [regrid_POP_1level]
Currently works with UVEL/VVEL Output

Optionally, remain in pop grid and work with HMXL processed in
[process_bylevel_ens]

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

# Additional Option
keep_POPgrid = True # Set to True to keep the POP grid
tol          = 1e-7 # Tolerance for Lat.Lon Differences


#%% Indicate which files to process

fns_all     = []
fns_hmxl    = []
for v in range(len(vnames)):
    vname = vnames[v]
    fns = []
    for e in loopens:
        # Get filename for variable
        fn = "%socn_var_3d/%s/%s_NATL_ens%02i.nc" % (rawpath,vname,vname,e)
        fns.append(fn)
        # Get filename for hxml
        if keep_POPgrid:
            fnh = "%socn_var_3d/%s_NATL_ens%02i.nc" % (rawpath,"HMXL",e)
            fns_hmxl.append(fnh)
        
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

if keep_POPgrid:
    # First, compute the mean seasonal MLD
    nlat,nlon = tlon.shape
    
    # Get TLAT and TLONG
    dsh     = xr.open_dataset(fns_hmxl[0]).load()
    tlat = dsh.TLAT.data
    tlon = dsh.TLONG.data
    
    tdim     = dict(nlat=dsh.nlat,nlon=dsh.nlon)
    tlong_ds = xr.DataArray(dsh.TLONG.data,coords=tdim,name='TLONG')
    tlat_ds  = xr.DataArray(dsh.TLAT.data,coords=tdim,name='TLAT')
    
    
    # Directly Go into Loop and process pointwise
    nvars   = len(vnames)
    for e in tqdm(range(nens)): # Loop by ensemble first
        
        # Load Mixed Layer Depth, Compute mean seasonal Cycle
        dsh     = xr.open_dataset(fns_hmxl[e]).load()
        hcycle  = dsh.HMXL.groupby('time.month').mean('time')
        
        
        for v in range(nvars):
            vname = vnames[v] 
            
            # Load the Variable
            st = time.time()
            fn = fns_all[v][e]
            
            # Open the Dataset
            st1         = time.time()
            ds          = xr.open_dataset(fn).load()
            #dsl         = ds.assign(lon=(['nlat','nlon'],tlon),lat=(['nlat','nlon'],tlat))
            #nlon,nlat   = len(outlon),len(outlat)
            print("Loaded Dataset in %.2fs" % (time.time()-st1))
            
            # Check Region Extent and TLONG/TLAT
            if not (ds.TLONG.shape == dsh.TLONG.shape):
                print("Warning, shapes are not equal. Please make sure they are cropped to the same region...")
                break
            if not np.all(np.abs(ds.TLONG.data - dsh.TLONG.data) < tol):
                print("Warning, not all longitudes are equal/within tolerance %.2e" % tol)
                break
            if not np.all(np.abs(ds.TLAT.data - dsh.TLAT.data) < tol):
                print("Warning, not all latitudes are equal/within tolerance %.2e" % tol)
                break
            
            ntime,nz,nlat,nlon   = ds[vname].shape
            mld_avg_var          = np.zeros((ntime,nlat,nlon)) * np.nan
            
            for o in tqdm(range(nlon)):
                
                for a in range(nlat):
                    
                    # Get Lat and Lon
                    lonf = tlon[a,o]
                    latf = tlat[a,o]
                    
                    # Locate the depth
                    hpt  = hcycle.isel(nlon=o,nlat=a)#(mon=im,ens=e,lon=o,lat=a).item()
                    if np.any(np.isnan(hpt)).item():
                        continue
                    
                    # Get the point
                    dspt = ds[vname].isel(nlon=o,nlat=a)
                    
                    ptval_bymon = []
                    for im in range(12):
                        # Select points for that month
                        dspt_mon = dspt.sel(time=ds.time.dt.month.isin([im+1]))
                        
                        # Restrict to depth
                        dspt_mon = dspt_mon.sel(z_t=slice(0,hpt.isel(month=im).data.item()))
                        
                        # Take mean over mixed-layer depth
                        if len(dspt_mon.z_t) < 1: # Exit if no MLD was found...
                            print("Warning, values no found within the mixed-layer depth range at Lon=%i Lat=%i (month %i..." % (o,a,im+1))
                            continue
                        else:
                            dspt_mon = dspt_mon.mean('z_t')
                            ptval_bymon.append(dspt_mon)
                    
                    if len(ptval_bymon) < 12: # Exit if not all months had a MLD value...
                        print("Warning, less than 12 months were found at Lon=%i Lat=%i..." % (o,a))
                    else:
                        ptval_bymon = xr.concat(ptval_bymon,dim='time')
                        
                        mld_avg_var[:,a,o] = ptval_bymon.data
            
            coords   = dict(time=ds.time,nlat=ds.nlat,nlon=ds.nlon)
            ds_out   = xr.DataArray(mld_avg_var,coords=coords,dims=coords,name=vname)
            ds_out   = xr.merge([ds_out,tlong_ds,tlat_ds])
            edict    = proc.make_encoding_dict(ds_out)
            
            savename = proc.addstrtoext(fn,"_MLDAVG_POP",adjust=-1)
            ds_out.to_netcdf(savename,encoding=edict)
            print("Saved %s" % savename)
            
            # End Variable Loop
        # End Ensemble Loop
    
    # Reopen and Merge all the files
    
    for v in range(len(vnames)):
        
        # Variables 
        vname = vnames[v]
        fns   = []
        for e in loopens:
            fn = "%socn_var_3d/%s/%s_NATL_ens%02i_MLDAVG_POP.nc" % (rawpath,vname,vname,e)
            fns.append(fn)
        
        # Indicate which datasets
        ds_all  = xr.open_mfdataset(fns,concat_dim='ens',combine='nested').load()
        
        # Save the ouput as a bulk file
        edict   = proc.make_encoding_dict(ds_all)
        fn_out  = "%socn_var_3d/%s/%s_NATL_AllEns_MLDAVG_POP.nc" % (rawpath,vname,vname)
        ds_all.to_netcdf(fn_out,encoding=edict)
                
    
    
else:
    # Otherwise... Load HMXL and do stuff
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
            st1 = time.time()
            ds          = xr.open_dataset(fn).load()
            dsl         = ds.assign(lon=(['nlat','nlon'],tlon),lat=(['nlat','nlon'],tlat))
            nlon,nlat   = len(outlon),len(outlat)
            print("Loaded Dataset in %.2fs" % (time.time()-st1))
            
            # Get Dimensions and preallocate
            ntime,nz,ntlat,ntlon = ds[vname].shape
            ocn_var_avg = np.zeros((ntime,nlat,nlon))
            
            # Looping for each point
            for a in range(nlat):
                
                latf   = outlat[a]
                
                for o in tqdm(range(nlon)): # Took (1h 11 min if you don't load, 2 sec if you load, T-T)
                    
                    
                    # Longitudes
                    lonf = outlon[o]
                    if lonf < 0:
                        lonf += 360
                        
                    for t in range(ntime):
                        # Get the month
                        im   = ds.time.isel(time=t).time.item().month - 1 # Month Index
                        
                        # Locate the depth (and convert to cm)
                        hpt  = dshreg.isel(mon=im,ens=e,lon=o,lat=a).item() * 100
                        
                        # Get the nearest point
                        outids = proc.get_pt_nearest(dsl,lonf,latf,tlon_name="lon",tlat_name="lat",returnid=True,debug=False)
                        dspt   = dsl.isel(nlat=outids[0],nlon=outids[1],time=t)
                        
                        # Slice to depth and average
                        dspt_mldavg                = dspt.sel(z_t = slice(0,hpt)).mean('z_t')
                        ocn_var_avg[t,a,o]         = dspt_mldavg[vname].data
                        
                    # End Longitude Loop ---
                    
                # End Latitude Loop ---
                    
            #% Apply mask based on h
            mask                    = np.sum(dshreg.values,(0,1))
            mask[~np.isnan(mask)]   = 1
            da_mask                 = xr.DataArray(mask,coords=dict(lat=outlat,lon=outlon))
            
            # Save Regridded Data
            coords                  = dict(time=ds.time,lat=dshreg.lat,lon=dshreg.lon,)
            da_out                  = xr.DataArray(ocn_var_avg * mask[None,:,:],coords=coords,dims=coords,name=vname)
            
            
            savename = proc.addstrtoext(fn,"_regridNN_MLavg",adjust=-1)
            edict    = proc.make_encoding_dict(da_out)
            da_out.to_netcdf(savename,encoding=edict)
            print("Saved output in %.2fs" % (time.time()-st))
            # End Ensemble Loop
        # End Variable Loop
    
#%% Open all files and save

# for v in range(len(vnames)):
    
#     # Variables 
#     vname = vnames[v]
#     fns   = []
#     for e in loopens:
#         fn = "%socn_var_3d/%s/%s_NATL_ens%02i_regridNN_MLavg.nc" % (rawpath,vname,vname,e)
#         fns.append(fn)
    
#     # Indicate which datasets
#     ds_all  = xr.open_mfdataset(fns,concat_dim='ens',combine='nested').load()
    
#     # Save the ouput as a bulk file
#     edict   = proc.make_encoding_dict(ds_all)
#     fn_out  = "%socn_var_3d/%s/%s_NATL_AllEns_regridNN_MLavg.nc" % (rawpath,vname,vname)
#     ds_all.to_netcdf(fn_out,encoding=edict)




    
    
    
    
    


