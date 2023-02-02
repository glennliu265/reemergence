#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate and process data for ORAS5 (re-emergence analysis)

Created on Thu May  5 11:05:39 2022

@author: gliu
"""


import glob
import xarray as xr


#%%
datpath = "/mnt/CMIP6/data/ocean_reanalysis/ORAS5/oras5/monthly/global-reanalysis-phy-001-031/"
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/ORAS5/"

#%%


# List of variables to get
varnames_raw = ("mlotst_oras","thetao_oras","so_oras","zos_oras",)
varnames_ud  = ("mld"        ,"sst"        ,"sss"    ,"ssh")


# Get list of NCFiles
searchstr = "%sglobal-reanalysis-phy-001-031-grepv2-monthly_*.nc" % datpath
nclist    = glob.glob(searchstr)
nclist.sort()
print(nclist)


# At this point, I found out they were cropped to just the Western North Atlantic
# Need to re-unzip, and find out the meanings of the folder opa0 to opa5 