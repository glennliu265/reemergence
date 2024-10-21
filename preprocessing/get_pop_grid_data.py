#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get POP Data

Created on Wed Sep 25 10:01:57 2024

@author: gliu

"""

import xarray as xr
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import xgcm
import pop_tools

#%%
amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl


#%%

vname  = "WTT"
ncname = "b.e11.B20TRC5CNBDRD.f09_g16.002.pop.h.%s.192001-200512.nc" % vname
ncpath = "/stormtrack/data4/glliu/01_Data/CESM1_LE/%s/" % vname
ds = xr.open_dataset(ncpath+ncname)


#%% Get Necessary coordinates (based on notebook from Who...)



# Assign thickness of cells, in centimeters
ds["DZT"] = xr.DataArray(ds.dz.values[:,None,None]*np.ones((len(ds.dz),len(ds.nlat),len(ds.nlon)))
                , dims=['z_t','nlat','nlon'], coords={'z_t':ds.z_t,'nlat':ds.nlat,'nlon':ds.nlon})
ds["DZU"] = xr.DataArray(ds.dz.values[:,None,None]*np.ones((len(ds.dz),len(ds.nlat),len(ds.nlon)))
                , dims=['z_t','nlat','nlon'], coords={'z_t':ds.z_t,'nlat':ds.nlat,'nlon':ds.nlon})


ds.DZT.attrs["long_name"] = "Thickness of T cells"
ds.DZT.attrs["units"] = "centimeter"
ds.DZT.attrs["grid_loc"] = "3111"
ds.DZU.attrs["long_name"] = "Thickness of U cells"
ds.DZU.attrs["units"] = "centimeter"
ds.DZU.attrs["grid_loc"] = "3221"

# make sure we have the cell volumne for calculations
VOL = (ds.DZT * ds.DXT * ds.DYT).compute()
KMT = ds.KMT.compute()

for j in tqdm(range(len(KMT.nlat))):
    for i in range(len(KMT.nlon)):
        k = KMT.values[j, i].astype(int)
        VOL.values[k:, j, i] = 0.0

ds["VOL"] = VOL

ds.VOL.attrs["long_name"] = "volume of T cells"
ds.VOL.attrs["units"] = "centimeter^3"

ds.VOL.attrs["grid_loc"] = "3111"

#%% Keep selected variables relevant for xgcm

keepvars = ["KMT","VOL",
            "TLAT","TLONG",
            "ULAT","ULONG",
            "z_t","dz","z_w",
            "TAREA","UAREA",
            "DXT","DYT","DZT",
            "DXU","DYU","DZU"]
dscoords = proc.ds_dropvars(ds,keepvars)
edict    = proc.make_encoding_dict(dscoords)
outpath  = "/home/glliu/01_Data/"
savename = "%sCESM1_POP_Coords.nc" % (outpath)

dscoords.to_netcdf(savename,encoding=edict)

#%% Alright, Budget calculation begins here I guess

# Load the transport terms
vnames = ("WTT",)#"UET","VNT"]
nvars = len(vnames)
ds_all = []
for vv in range(nvars):
    ncname = "b.e11.B20TRC5CNBDRD.f09_g16.002.pop.h.%s.192001-200512.nc" % vnames[vv]
    ncpath = "/stormtrack/data4/glliu/01_Data/CESM1_LE/%s/" % vnames[vv]
    ds = xr.open_dataset(ncpath+ncname,chunks={"time": 12})
    ds_all.append(ds)

# Load the coordinates from above
outpath_coords  = "/home/glliu/01_Data/"
savename_coords = "%sCESM1_POP_Coords.nc" % (outpath_coords)
dscoords        = xr.open_dataset(savename_coords)

# Load in the values
for ii in range(nvars):
    dscoords[vnames[ii]] = ds_all[ii][vnames[ii]]
#dscoords['UET'] = ds_all[0].UET
#dscoords['VNT'] = ds_all[1].VNT

#%% Set grid and dataset for xgcm (from Who's notebook)

metrics = {
    ("X",): ["DXU", "DXT"],  # X distances
    ("Y",): ["DYU", "DYT"],  # Y distances
    ("Z",): ["DZU", "DZT"],  # Z distances
    ("X", "Y"): ["UAREA", "TAREA"],
}

# here we get the xgcm compatible dataset
gridxgcm, dsxgcm = pop_tools.to_xgcm_grid_dataset(
    dscoords,
    periodic=False,
    metrics=metrics,
    boundary={"X": "extend", "Y": "extend", "Z": "extend"},
)

for coord in ["nlat", "nlon"]:
    if coord in dsxgcm.coords:
        dsxgcm = dsxgcm.drop_vars(coord)
        
        
#dsxgcm.sel(lat=slice(0,90))
#dsxgcm = dsxgcm.isel()
#%% Set up budget comutation %Start computing the budget
#st = time.time()
budget = xr.Dataset()
budget["WTT"] = (
    gridxgcm.diff(dsxgcm.WTT.fillna(0) * (dsxgcm.dz * dsxgcm.DXT * dsxgcm.DYT).values, axis="Z")
    / dsxgcm.VOL
)
budget["UET"] = -(gridxgcm.diff(dsxgcm.UET * dsxgcm.VOL.values, axis="X") / dsxgcm.VOL)
budget["VNT"] = -(gridxgcm.diff(dsxgcm.VNT * dsxgcm.VOL.values, axis="Y") / dsxgcm.VOL)
#print("Computed UET and VNT in %.2fs" % (time.time()-st))

#%% Save this component
outpath = "/stormtrack/data4/glliu/01_Data/CESM1_LE/proc/"
edict   = proc.make_encoding_dict(budget)
if vname == "WTT":
    savename = outpath + "CESM1_WTT_Budget.nc"
else:
    savename = outpath + "CESM1_VNT_UET_Budget.nc"
budget.to_netcdf(savename,encoding=edict)

#%% Reload component and just grab data for 1 point
outpath = "/stormtrack/data4/glliu/01_Data/CESM1_LE/proc/"
savename = outpath + "CESM1_VNT_UET_Budget.nc"
ds = xr.open_dataset(savename)
ds   = ds.rename(dict(nlat_t='nlat',nlon_t='nlon'))

locname = "NAC"
lonf = -39 + 360
latf = 44

locname = "SAR"
lonf = -65 + 360
latf = 36

locname = "IRM"
lonf = -35 + 360
latf = 53

dspt = proc.find_tlatlon(ds,lonf,latf)


locfn,loctitle= proc.make_locstring(lonf,latf)

savename = outpath + "CESM1_VNT_UET_Budget_%s_%s.nc" % (locname,locfn)
dspt.to_netcdf(savename)
print(savename)



