#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:28:44 2024

@author: gliu
"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os

from tqdm import tqdm

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
machine    = "Astraeus"
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
rawpath     = pathdict['raw_path']

# ----------------------------------
#%% User Edits
# ----------------------------------

# # Indicate the experiment
# expname_sss         = "SSS_Draft01_Rerun_QekCorr"#SSS_EOF_LbddCorr_Rerun_lbdE_neg" #"SSS_EOF_Qek_LbddEnsMean"#"SSS_EOF_Qek_LbddEnsMean"
# expname_sst         = "SST_Draft01_Rerun_QekCorr"#"SST_EOF_LbddCorr_Rerun"


expname_uet = "UET_NATL_ens02.nc"
expname_vnt = "VNT_NATL_ens02.nc"
#expname_wtt = "WTT"

# Constants
dt          = 3600*24*30 # Timestep [s]
cp          = 3850       # 
rho         = 1026       # Density [kg/m3]
B           = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L           = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

fsz_tick    = 18
fsz_title   = 24
fsz_axis    = 22


debug       = False

#%% Load the point coordinates
# Get Point Info
pointset    = "PaperDraft02"
ptdict      = rparams.point_sets[pointset]
ptcoords    = ptdict['bboxes']
ptnames     = ptdict['regions']
ptnames_long = ptdict['regions_long']
ptcols      = ptdict['rcols']
ptsty       = ptdict['rsty']

#%% Load Mean Mixed layer Depths
# Load Mixed-Layer Depth
mldpath = input_path + "mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc"
ds_mld  = xr.open_dataset(mldpath+mldnc).h.load()

#%% Load the POP coordinates (as computed by get_pop_grid_data)

popnc  = "CESM1_POP_Coords.nc"
ds_pop = xr.open_dataset(rawpath + popnc).load()


#%% Get tlon/tlat

dtmon   = 3600*24*30

lonf    = 330
latf    = 50

ds_vnt  = xr.open_dataset(rawpath + expname_vnt)#.load()
ds_uet  = xr.open_dataset(rawpath + expname_uet)#.load()

tlon    = ds_vnt.TLONG.data
tlat    = ds_vnt.TLAT.data

#%% Load the data

ds_vnt = ds_vnt.load()
ds_uet = ds_uet.load()

vnt_pt  = proc.find_tlatlon(ds_vnt,lonf,latf).VNT
uet_pt  = proc.find_tlatlon(ds_uet,lonf,latf).UET

#%% Compute what is coming into the box

lonf = 330
latf = 50


# Get Eastward/Northward transport at the point
klon,klat = proc.find_tlatlon(ds_vnt,lonf,latf,return_index=True)


vnt_upper = ds_vnt.isel(nlon=klon,nlat=klat).VNT
uet_right = ds_uet.isel(nlon=klon,nlat=klat).UET
vnt_lower = ds_vnt.isel(nlon=klon,nlat=klat-1).VNT
uet_left  = ds_uet.isel(nlon=klon-1,nlat=klat).UET


# Get net transport into point (degC/sec) across all levels
net_v  = vnt_lower - vnt_upper
net_u  = uet_left - uet_right 



#%% Try to roll all points at once (debugging)

# Do this for all points at once (minus boundary points)
vnt_roll_lower = ds_vnt.roll(dict(nlat=1))
uet_roll_left  = ds_uet.roll(dict(nlon=1))

# Check point values, comparing with above
vnt_roll_pt = proc.find_tlatlon(vnt_roll_lower,lonf,latf).VNT
print(vnt_roll_pt - vnt_lower)
uet_roll_pt = proc.find_tlatlon(uet_roll_left,lonf,latf).UET
print(uet_roll_pt - uet_left)
print(np.all(uet_roll_pt == uet_left))

#
net_v_roll =  vnt_roll_lower.VNT - ds_vnt.VNT
print(np.all(proc.find_tlatlon(net_v_roll,lonf,latf)==net_v))
net_u_roll = uet_roll_left.UET - ds_uet.UET
print(np.all(proc.find_tlatlon(net_u_roll,lonf,latf)==net_u))


#%%

def get_net_transport(ds_uet,ds_vnt):
    # Do this for all points at once (minus boundary points)
    vnt_roll_lower = ds_vnt.roll(dict(nlat=1))
    uet_roll_left  = ds_uet.roll(dict(nlon=1))
    
    net_v_roll =  vnt_roll_lower.VNT - ds_vnt.VNT
    net_u_roll = uet_roll_left.UET - ds_uet.UET
    return net_u_roll, net_v_roll


net_uT,net_vT = get_net_transport(ds_uet,ds_vnt)

#plt.scatter(tlat,tlon,net_uT.isel(time=0))

plt.scatter(tlon,tlat,c=net_uT.isel(time=0,z_t=0),vmin=-1e-6,vmax=1e-6)


#%% Load VNT and UET in 

process_wtt = False

if process_wtt:
    ncname_wtt      = "CESM1_WTT_Budget_NAtl.nc"
    ds_budget       = xr.open_dataset(rawpath + ncname_wtt).load()
else:
    ncname_vntuet   = "CESM1_VNT_UET_Budget_NAtl.nc"
    ds_budget       = xr.open_dataset(rawpath + ncname_vntuet).load()
    
#%% Sum across the mixed-layer 

iz      = 25
uet_z   = ds_budget.UET.isel(z_t=iz).std("time")
vnt_z   = ds_budget.VNT.isel(z_t=iz).std("time")

dtmon   = 3600 * 24 * 30
bbox    = [-80,0,20,60]

z_t     = ds_budget.z_t.data

vmin    = 0
vmax    = 15
s       = 22

fig,axs  = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},
                        constrained_layout=True,figsize=(12,4))

for ax in axs:
    ax.coastlines()
    ax.set_extent(bbox)

ax = axs[0]
plotvar = uet_z * dtmon

sc = ax.scatter(tlon,tlat,c=plotvar,s=s,vmin=vmin,vmax=vmax)
cb = viz.hcbar(sc,ax=ax)
cb.set_label("Eastward Trasport (degC/mon)")

ax = axs[1]
plotvar = vnt_z * dtmon

sc = ax.scatter(tlon,tlat,c=plotvar,s=s,vmin=vmin,vmax=vmax)
cb = viz.hcbar(sc,ax=ax)
cb.set_label("Northward Trasport (degC/mon)")

plt.suptitle("Transport at z=%i cm" % (z_t[iz]))
savename = "%sMeanTransport_iz%0i.png" % (figpath,iz)
print(savename)

plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Select a point

ii        = 1
lonf,latf = ptcoords[ii][0],ptcoords[ii][1]
if lonf < 0:
    lonf += 360

# Get Deepest Mixed Layer depth and convert to cm
pt_hmax   = proc.selpt_ds(ds_mld,lonf-360,latf).max('mon').item() * 100
pt_budget = proc.find_tlatlon(ds_budget,lonf,latf)


# Get the corresponding coordinates for the cell thickness
pop_grid_pt = proc.find_tlatlon(ds_pop,lonf,latf).sel(z_t=slice(0,pt_hmax)).dz
dz          = pop_grid_pt / pop_grid_pt.sum('z_t')


# Restrict Budget Terms to the specified depth
pt_budget_ml = pt_budget.sel(z_t=slice(0,pt_hmax))
nz           = len(pt_budget_ml.z_t)

# Multiply by dz to get the weighted average
ml_sum_budget = (pt_budget_ml * dz).sum('z_t')

# We expect VNT, UET to be strong over certain points


# Make the above into a function

def sum_ml(ds_in,ds_mld,ds_pop,lonf,latf):
    # Given a DS, sum over the maximum mixed layer depth and take the area weighted mean...
    
    
    if lonf < 0:
        lonf += 360
    
    # Get Deepest Mixed Layer depth and convert to cm
    pt_hmax   = proc.selpt_ds(ds_mld,lonf-360,latf).max('mon').item() * 100
    pt_budget = proc.find_tlatlon(ds_budget,lonf,latf,verbose=False)
    
    # Get the corresponding coordinates for the cell thickness
    pop_grid_pt = proc.find_tlatlon(ds_pop,lonf,latf,verbose=False).sel(z_t=slice(0,pt_hmax)).dz
    dz          = pop_grid_pt / pop_grid_pt.sum('z_t')
    
    # Restrict Budget Terms to the specified depth
    pt_budget_ml = pt_budget.sel(z_t=slice(0,pt_hmax))
    nz           = len(pt_budget_ml.z_t)

    # Multiply by dz to get the weighted average
    ml_sum_budget = (pt_budget_ml * dz).sum('z_t')
    
    return ml_sum_budget

    
testout = sum_ml(ds_budget,ds_mld,ds_pop,lonf,latf)
    

#%% Now Loop over all points

if process_wtt:
    ntime,_,nlat,nlon = ds_budget.WTT.shape
    budget_sum        = np.zeros((1,ntime,nlat,nlon)) * np.nan
else:
    ntime,_,nlat,nlon = ds_budget.UET.shape
    budget_sum        = np.zeros((2,ntime,nlat,nlon)) * np.nan

for o in tqdm(range(nlon)):
    
    
    for a in range(nlat):
        
        lonf        = tlon[a,o]
        latf        = tlat[a,o]
        
        ml_pt_sum   = sum_ml(ds_budget,ds_mld,ds_pop,lonf,latf)
        
        
        if process_wtt:
            if np.all(ml_pt_sum.WTT.data == 0.0):
                continue
            else:
                budget_sum[0,:,a,o] = ml_pt_sum.WTT.data
        else:
            
            if np.all(ml_pt_sum.UET.data == 0.0) or np.all(ml_pt_sum.VNT.data == 0.0):
                continue
            else:
                budget_sum[0,:,a,o] = ml_pt_sum.UET.data
                budget_sum[1,:,a,o] = ml_pt_sum.VNT.data
        
#%% Place into DataArray

coords      = dict(time=ds_budget.time,nlat=ds_budget.nlat,nlon=ds_budget.nlon)

coords_xy   = dict(nlat=ds_budget.nlat,nlon=ds_budget.nlon)
da_tlat     = xr.DataArray(ds_budget.TLAT.data,coords=coords_xy,dims=coords_xy,name="TLAT")
da_tlong    = xr.DataArray(ds_budget.TLONG.data,coords=coords_xy,dims=coords_xy,name="TLONG")


if process_wtt:
    da_wtt = xr.DataArray(budget_sum[0,...],coords=coords,dims=coords,name="WTT")
    
    ds_ml_out = xr.merge([da_wtt,da_tlat,da_tlong])#,ds_budget.TLONG,ds_budget.TLAT])
    savename = "%sCESM1_WTT_MLSum_Budget_NAtl.nc" % rawpath
else:
    da_uet      = xr.DataArray(budget_sum[0,...],coords=coords,dims=coords,name="UET")
    da_vnt      = xr.DataArray(budget_sum[1,...],coords=coords,dims=coords,name="VNT")

    ds_ml_out = xr.merge([da_uet,da_vnt,da_tlat,da_tlong])#,ds_budget.TLONG,ds_budget.TLAT])
    
    savename = "%sCESM1_VNT_UET_MLSum_Budget_NAtl.nc" % rawpath
    
edict     = proc.make_encoding_dict(ds_ml_out)

ds_ml_out.to_netcdf(savename,encoding=edict)

#%% Repeat for WTT

#%% plot some stuff

#%% Visualize Monthly Variance Contribution 

pt_ts = []
for ii in range(3):
    
    lonf,latf = ptcoords[ii][0],ptcoords[ii][1]
    if lonf < 0:
        lonf += 360
        
    ds_pt = proc.find_tlatlon(ds_ml_out,lonf,latf)
    
    pt_ts.append(ds_pt.copy())
    

uet_monvar = [proc.xrdeseason(ds.UET).groupby('time.month').var('time') for ds in pt_ts]
vnt_monvar = [proc.xrdeseason(ds.VNT).groupby('time.month').var('time') for ds in pt_ts]



#%% Plot Monthly Variance

mons3=proc.get_monstr()

fig,axs = viz.init_monplot(1,3,figsize=(16,4.5))
for ii in range(3):
    ax = axs[ii]
    ax.plot(mons3,uet_monvar[ii].data*dtmon**2,label="UET",lw=2.5)
    ax.plot(mons3,vnt_monvar[ii].data*dtmon**2,label="VNT",lw=2.5)
    ax.set_title(ptnames[ii])
    ax.legend()


#%% Plot the field (variance contributions)
bboxplot = bbox
vnames   = ["UET","VNT"]
proj     = ccrs.PlateCarree()


tlon

# Initialize Plot
fig,axs,_    = viz.init_orthomap(1,2,bboxplot,figsize=(26,12))

for ax in axs:
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)


for ii in range(2):
    ax = axs[ii]
    vname = vnames[ii]
    ax.set_title(vname,fontsize=fsz_title)
    plotvar = ds_ml_out[vname].groupby('time.month').var('time').mean('month') * dtmon**2
    sc = ax.scatter(tlon,tlat,c=plotvar,s=250,transform=proj,vmin=0,vmax=55)
    cb = viz.hcbar(sc,ax=ax)    
    cb.ax.tick_params(labelsize=16)
    
    for iip in range(3):
        ax.plot(ptcoords[iip][0],ptcoords[iip][1],transform=proj,marker="x",markersize=30,c='lightgray')


#%% Save several simpler crops 

#%% Just the surface value
#
z_t             = ds_budget.z_t
ds_budget_surf  = ds_budget.isel(z_t=0)

# Take weighted average over first 100 meters (first 10 levels)
crop_depth = 100 * 100 # Convert from m --> cm
z_t_crop   = z_t.sel(z_t=slice(0,crop_depth))

ds_budget_100m = ds_budget.sel(z_t=slice(0,crop_depth))
weights        = z_t_crop / z_t_crop.sum('z_t')
ds_budget_100m = (ds_budget_100m * weights).sum('z_t')

#%% Load ML sum to compare


savename = "%sCESM1_VNT_UET_MLSum_Budget_NAtl.nc" % rawpath
ds_mlsum = xr.open_dataset(savename).load()


#%% Select monthy variability at a point

def selpt_monvar(ds,lonf,latf):
    dspt        = proc.find_tlatlon(ds,lonf,latf,)
    dsmonvar    = dspt.groupby('time.month').var('time')
    return dsmonvar
    
lonf    = -35 + 360
latf    = 53
dsmonvar_surf  = selpt_monvar(ds_budget_surf,lonf,latf)
dsmonvar_100m  = selpt_monvar(ds_budget_100m,lonf,latf)
dsmonvar_mlsum = selpt_monvar(ds_mlsum,lonf,latf)



#%% Compare the three quantities
dtmon   = 3600*24*30
mons3   = proc.get_monstr()
in_ds   = [dsmonvar_mlsum,dsmonvar_surf,dsmonvar_100m]
innames = ["Mixed Layer Average","Surface","100 meter"]
incols  = ["k","cornflowerblue","limegreen"]

plot_titles = ["Eastward Transport (UET)","Northward Transport (VNT)","Total Transport (UET + VNT)"]

fig,axs = viz.init_monplot(1,3,figsize=(16,4.5))


for vv in range(3):
    
    ax = axs[vv]
    
    for ee in range(3):
        
        if vv == 0:
            plotvar = in_ds[ee].UET
        elif vv == 1:
            plotvar = in_ds[ee].VNT
        else:
            plotvar = in_ds[ee].VNT + in_ds[ee].UET
        plotvar = plotvar * dtmon**2
        ax.plot(mons3,plotvar,label=innames[ee],c=incols[ee],lw=2.5)
    ax.legend()
    ax.set_title(plot_titles[vv])
    
    
    if vv == 0:
        ax.set_ylabel("Interannual Variance [degC/month]$^2$")
#%% What if we just compute the net before everything

in_all          = [ds_mlsum,ds_budget_surf,ds_budget_100m]
in_all          = [proc.find_tlatlon(ds,lonf,latf) for ds in in_all]
transport_sum   = [ds.UET + ds.VNT for ds in in_all]
sum_monvar      = [ds.groupby('time.month').var('time') for ds in transport_sum]


plot_comparison = False
#for ii in range(3):
    
fig,ax = viz.init_monplot(1,1,figsize=(8,4.5))

for vv in range(3):
    if plot_comparison:
        # Plot takking the stdev first, then summing
        plotvar = (in_ds[vv].VNT + in_ds[ee].UET) * dtmon**2
        ax.plot(mons3,plotvar,label=innames[vv] + " (Sum After)",c=incols[vv],lw=2.5)
        
    # Plot taking the sum first, then stdev
    plotvar = sum_monvar[vv] * dtmon**2
    ax.plot(mons3,plotvar,label=innames[vv] + " (Sum Before)",c=incols[vv],lw=2.5,ls='dotted')
    
ax.legend()
ax.set_ylim([0,0.2])
ax.set_ylabel("Interannual Variance [degC/month]$^2$")



#transport_surf = proc.selpt_ds(ds_budget_surf.UET,lonf,latf) + ds_budget_surf.VNT

#%%




