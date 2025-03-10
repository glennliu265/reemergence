#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis Plots

Copied Sections from compare_AMV_HadISST

Created on Fri Nov  8 09:27:51 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import matplotlib.path as mpath

import matplotlib.pyplot as plt

from scipy.signal import butter,filtfilt,detrend
import sys
import glob
import os

import tqdm
import time

#%% Import Custom Modules

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

#%% User Edits

datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/"#"../../CESM_data/"
outpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/02_Figures/Thesis/"

# Set File Names
vnames      = ['sst','sss','psl']
vnamelong   = ["Sea Surface Temperature (degC)","Sea Surface Salinity (psu)","Sea Level Pressure (hPa)"]
fn1         = "CESM1LE_sst_NAtl_19200101_20051201_Regridded2deg.nc"
fn2         = "CESM1LE_sss_NAtl_19200101_20051201_Regridded2deg.nc"
fn3         = "CESM1LE_psl_NAtl_19200101_20051201_Regridded2deg.nc"
fns         = [fn1,fn2,fn3]

# Plotting Box
bbox        = [-80,0,0,65] # North Atlantic [lonW, lonE, latS, latN]

#%% Dark mode Settings

darkmode = False
if darkmode:
    dfcol = "w"
    bgcol = np.array([15,15,15])/256
    transparent = True
    plt.style.use('dark_background')
    mpl.rcParams['font.family']     = 'Avenir'
else:
    dfcol = "k"
    bgcol = "w"
    transparent = False
    plt.style.use('default')

#%% Old Functions


def calc_AMV_index(region,invar,lat,lon,lp=False,order=5,cutofftime=120,dtr=False):
    """
    Select bounding box for a given AMV region for an input variable
        "SPG" - Subpolar Gyre
        "STG" - Subtropical Gyre
        "TRO" - Tropics
        "NAT" - North Atlantic
    
    Parameters
    ----------
    region : STR
        One of following the 3-letter combinations indicating selected region
        ("SPG","STG","TRO","NAT")
        
    var : ARRAY [Ensemble x time x lat x lon]
        Input Array to select from
    lat : ARRAY
        Latitude values
    lon : ARRAY
        Longitude values    

    Returns
    -------
    amv_index [ensemble x time]
        AMV Index for a given region/variable

    """
    
    # Select AMV Index region
    bbox_SP = [-60,-15,40,65]
    bbox_ST = [-80,-10,20,40]
    bbox_TR = [-75,-15,0,20]
    bbox_NA = [-80,0 ,0,65]
    regions = ("SPG","STG","TRO","NAT")        # Region Names
    bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA) # Bounding Boxes
    
    # Get bounding box
    bbox = bboxes[regions.index(region)]
    
    # Select Region
    selvar = invar.copy()
    klon = np.where((lon>=bbox[0]) & (lon<=bbox[1]))[0]
    klat = np.where((lat>=bbox[2]) & (lat<=bbox[3]))[0]
    selvar = selvar[:,:,klat[:,None],klon[None,:]]
    
    # Take mean ove region
    amv_index = np.nanmean(selvar,(2,3))
    
    # If detrend
    if dtr:
        for i in range(amv_index.shape[0]):
            amv_index[i,:] = detrend(amv_index[i,:])
    
    if lp:
        
        for i in range(amv_index.shape[0]):
            amv_index[i,:]=lp_butter(amv_index[i,:],cutofftime,order)
        
    
    return amv_index

def regress_2d(A,B,nanwarn=1):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    # Determine if A or B is 2D and find anomalies
    
    # Compute using nan functions (slower)
    if np.any(np.isnan(A)) or np.any(np.isnan(B)):
        if nanwarn == 1:
            print("NaN Values Detected...")
    
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.nanmean(A,axis=a_axis)[:,None]
            Banom = B - np.nanmean(B,axis=b_axis)
            
        
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis)
            Banom = B - np.nanmean(B,axis=b_axis)[None,:]
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.nansum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.nansum(B,axis=b_axis) - beta * np.nansum(A,axis=a_axis))/A.shape[a_axis]
    else:
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)
            
        
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)
            Banom = B - np.mean(B,axis=b_axis)[None,:]
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.sum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.sum(B,axis=b_axis) - beta * np.sum(A,axis=a_axis))/A.shape[a_axis]
    
    
    return beta,b


def lp_butter(varmon,cutofftime,order):
    # Input variable is assumed to be monthy with the following dimensions:
    flag1d=False
    if len(varmon.shape) > 2:
        nmon,nlat,nlon = varmon.shape
    else:
        
        flag1d = True
        nmon = varmon.shape[0]
    
    # Design Butterworth Lowpass Filter
    filtfreq = nmon/cutofftime
    nyquist  = nmon/2
    cutoff = filtfreq/nyquist
    b,a    = butter(order,cutoff,btype="lowpass")
    
    # Reshape input
    if flag1d is False: # For 3d inputs, loop thru each point
        varmon = varmon.reshape(nmon,nlat*nlon)
        # Loop
        varfilt = np.zeros((nmon,nlat*nlon)) * np.nan
        for i in tqdm(range(nlon*nlat)):
            varfilt[:,i] = filtfilt(b,a,varmon[:,i])
        
        varfilt=varfilt.reshape(nmon,nlat,nlon)
    else: # 1d input
        varfilt = filtfilt(b,a,varmon)
    return varfilt


#%% Load and compute AMV


dtr      = True

#  Read out values
fn       = "hadisst.1870-01-01_2018-12-01.nc"
dsh      = xr.open_dataset(datpath+fn)
ssth     = dsh.sst.values # [time x lat x lon]
lath     = dsh.lat.values
lonh     = dsh.lon.values
times    = dsh.time.values

# Get Strings for Plotting later
timesmon = np.datetime_as_string(times,unit="M")
timesyr  = np.datetime_as_string(times,unit="Y")[:]

# Calculate Monthly Anomalies
ssts = ssth.transpose(2,1,0)
nlon,nlat,nmon = ssts.shape
ssts = ssts.reshape(nlon,nlat,int(nmon/12),12)
ssta = ssts - ssts.mean(2)[:,:,None,:]
ssta = ssta.reshape(nlon,nlat,nmon)

# Transpose [time lat lon] -- > [lon x lat x time]
ssta                = ssta.transpose(2,1,0)

amvid,amvpattern    = proc.calc_AMVquick(ssta,lonh,lath,bbox,runmean=False,)

#plt.pcolormesh(amvpattern.T,vmin=-.4,vmax=.4,cmap='cmo.balance')
#plt.contourf(amvpattern.T,levels=np.arange(-.4,.44,0.04),cmap='cmo.balance')

#%% Let's try again
nmon

# # Deseason Things
# dsha            = dsh.copy()
# dsha['time']    = xr.cftime_range(start="1870",periods=nmon,freq="MS")
# dsha            = proc.xrdeseason(dsha.sst)

# ssta            = dsha.transpose('lon','lat','time').data
# ssta_ann        = dsha.groupby('time.year').mean('time').transpose('lon','lat','year').data

# ssta_dt,b,c,d      = proc.detrend_dim(ssta,2,)

# amvid,amvpattern = proc.calc_AMVquick(ssta_dt,lonh,lath,bbox,)



# amvid_ann,amvpattern_ann   = proc.calc_AMVquick(ssta_dt,lonh,lath,bbox,anndata=True)

# plt.contourf(amvpattern_ann.T,levels=np.arange(-.4,.44,0.04),cmap='cmo.balance')

#%% Detrend by remove the global mean signal ****

detrend_method = "global"#"global"

#sst        = dsh.sst.transpose('lon','lat','time')
#sst        = (sst - sst.mean('time')).data

ssta_in     = ssta.transpose(2,1,0)

# Get Global Mean SST
if detrend_method == "global":
    gm_sst     = proc.area_avg(ssta_in,[-180,180,-90,90],lonh,lath,1)
    
    # Regress to get pattern
    gm_pattern = proc.regress2ts(ssta_in,gm_sst,normalizeall=0,nanwarn=0,verbose=True)
    plt.pcolormesh(lonh,lath,gm_pattern.T,vmin=-2,vmax=2,cmap='cmo.balance')
    plt.show()
    
    # Remove that from the sst
    glob_dt     = (gm_sst[:,None,None] * gm_pattern.T[None,:,:])
    ssta_dt_gm =  ssta - glob_dt#(gm_sst[:,None,None] * gm_pattern.T[None,:,:])
    ssta_dt_gm = ssta_dt_gm.transpose(2,1,0) # lon x lat x time
else:
    ssta_dt_gm,b,c,d = proc.detrend_dim(ssta_in,2)

# Note this is the pattern you probably want to use
amvid_dt_gm,amvpattern_dt_gm,nasst_dt_gm = proc.calc_AMVquick(ssta_dt_gm,lonh,lath,bbox,
                                                              return_unsmoothed=True)
plt.contourf(amvpattern_dt_gm.T,levels=np.arange(-.4,.44,0.04),cmap='cmo.balance')


# Also get the monthly values
natl_ssta   = proc.sel_region(ssta_dt_gm,lonh,lath,bbox,reg_avg=1,awgt='cos')
_,lonr,latr = proc.sel_region(ssta_dt_gm,lonh,lath,bbox,awgt='cos')

#nasst_monthly 


#%% Oarametric way to compute sign of corr coef


from scipy import stats

# Take annual average of SST
ssta_ann_avg = proc.ann_avg(ssta_dt_gm,2)

# Inputs (to function)
in_var = ssta_ann_avg
in_ts  = amvid_dt_gm / amvid_dt_gm.std(0)

mciter  = 1000
p       = 0.05
tails   = 2
nparams = 2

outdict_sigttest = proc.regress_ttest(in_var,in_ts,)
sigmask = outdict_sigttest['sigmask']

# #%% Started nonparametric way but gave up (for now)


# # Step (3), Compute R1
# calc_r1     = lambda ts: np.corrcoef(ts[:-1],ts[1:])[0,1]
# r1_map      = np.apply_along_axis(calc_r1,1,invar_rs)

# # Step (4), Make Montecarlo Simulations
# npts       = invar_rs.shape[0]
# thresholds = np.zeros((npts,2)) # (Pt, [Lower,Upper])
# for nn in range(invar_rs):
    
#     iter_slopes = []
#     for mc in range(mciter):
#         ar1_mc = proc.
        
    

# # Check R1 Map
# remap_r1    = np.zeros((nlat*nlon))
# remap_r1[nandict['ok_indices']] = r1_map
# remap_r1    = remap_r1.reshape(nlon,nlat)

#%% Compute Significance of regression coefficient

# Is the coefficient significantly different from zero?

# Need to get a t-critical value
ts_in  = amvid_dt_gm
p      = 0.05
tails  = 2
ptilde = p/tails
dof    = len(ts_in)

critval = stats.t.ppf(1-ptilde,dof)

# https://www.geo.fu-berlin.de/en/v/soga-r/Basics-of-statistics/Hypothesis-Tests/Inferential-Methods-in-Regression-and-Correlation/Inferences-About-the-Slope/index.html







#%%
# Ordinary Least Squares Regression
tper    = np.arange(0,tdim)
m,b     = regress_2d(tper,varok)

# Squeeze things? (temp fix, need to check)
m       = m.squeeze()
b       = b.squeeze()

# Detrend [Space,1][1,Time]
ymod = (m[:,None] * tper + b[:,None]).T




#%% Make A nicer plot for the thesis

fsz_tick  = 18
fsz_axis  = 22
fsz_title = 36

plot_sig  = True


mpl.rcParams['font.family'] = 'Avenir'


plot_index_top = True # set to True to set the index plot as panel A (on top)


plot_bbox = [-90,10,-10,70]

cints     = np.arange(-0.60,0.65,0.05)
cintsl    = np.arange(-0.6,0.7,0.1)

fig       = plt.figure(figsize=(22,22),constrained_layout=True)
gs        = gridspec.GridSpec(7,5)

# ------------------------
# Plot The AMV regression pattern
# ------------------------

# Set up orth0map ~~~~~~~~~~~~
centlon     = -40
centlat     = 35
precision   = 40
dx          = 10
dy          = 5
frame_lw    = 2
frame_col   = dfcol
noProj      = ccrs.PlateCarree()
proj        = noProj

# Set Orthographic Projection
myProj            = ccrs.Orthographic(central_longitude=centlon, central_latitude=centlat)
myProj._threshold = myProj._threshold/precision  #for higher precision plot


if plot_index_top:
    ax1               = fig.add_subplot(gs[2:,:],projection=myProj)
    subplotii         = 1
else:
    ax1               = fig.add_subplot(gs[:5,:],projection=myProj)
    subplotii         = 0

# Get Line Coordinates
xp,yp  = viz.get_box_coords(plot_bbox,dx=dx,dy=dy)

ax     = ax1
ndaxis = False

[ax_hdl] = ax.plot(xp,yp,
    color=frame_col, linewidth=frame_lw,
    transform=noProj)

# Make a polygon and crop
tx_path                = ax_hdl._get_transformed_path()
path_in_data_coords, _ = tx_path.get_transformed_path_and_affine()
polygon1s              = mpath.Path( path_in_data_coords.vertices)
ax.set_boundary(polygon1s) # masks-out unwanted part of the plot
        

# End Orthomap setup ~~~~~~~~~~~~

# Set up coastlines
ax = viz.add_coast_grid(ax,bbox=plot_bbox,proj=proj,
                        fontsize=fsz_tick,fill_color="dimgray")#[.12,.12,.12])

# Plot the variable
plotvar = amvpattern_dt_gm.T
pcm     = ax.contourf(lonh,lath,plotvar,transform=proj,
                      levels=cints,
                      cmap='cmo.balance',zorder=-1)


cl = ax.contour(lonh,lath,plotvar,transform=proj,
                levels=cints,colors="w",linewidths=0.75)
cl = ax.clabel(cl,fontsize=fsz_tick)

# Plot Significance if option is set
if plot_sig:
    viz.plot_mask(lonh, lath, sigmask, ax=ax, markersize=.5,color='gray',
                  reverse=True,geoaxes=True,proj=proj)
    

# Plot the bounding Box
viz.plot_box(bbox,linestyle='dashed',color='k',ax=ax,proj=proj,linewidth=4.5)

# Add Colorbar
cb      = viz.hcbar(pcm,ax=ax,pad=.05,fraction=0.025)
cb.ax.tick_params(labelsize=fsz_tick,rotation=45)
cb.set_label("AMV Pattern ($\degree$C per $1\sigma_{AMV}$)",fontsize=fsz_axis)



# Add Subplot Label
viz.label_sp(subplotii,alpha=0.75,ax=ax,fontsize=fsz_title,x=-.02)
        
# ------------------------
# Plot Timeseries
# ------------------------
if plot_index_top:
    ax2               = fig.add_subplot(gs[:2,1:4])
    #ax1               = fig.add_subplot(gs[:5,:],projection=myProj)
    subplotii         = 0
    leg_loc           = (0.4, .72, 0.5, 0.5)
    splaby            = 1.15
else:
    ax2               = fig.add_subplot(gs[6:,1:4])
    subplotii         = 1
    leg_loc           = (0.4, .96, 0.5, 0.5)
    splaby            = 1.4 # y position of subplot label
    
#ax2 = fig.add_subplot(gs[6:,1:4])
ax  = ax2

#ax.plot(nasst_dt_gm,label="NASST Index")
#ax.plot(amvid_dt_gm,color=dfcol,lw=2.5,label="AMV Index")


plotid   = natl_ssta.squeeze()
timeplot = np.arange(0,len(plotid),1)
maskneg  = plotid < 0
maskpos  = plotid >=0
ax.bar(timeplot[maskneg],plotid[maskneg],label="NASST-",color='cornflowerblue',width=1,alpha=1)
ax.bar(timeplot[maskpos],plotid[maskpos],label="NASST+",color='tomato',width=1,alpha=1)

# Plot the smoothed INdex
ax.plot(timeplot[::12],amvid_dt_gm,color=dfcol,lw=2.5,label="AMV Index")



ax2.set_ylim([-.75,.75])
ax2.set_xlim([timeplot[0],timeplot[-1]])

ax2.axhline([0,],lw=0.75,c=dfcol,ls='dotted')

ax2.set_ylabel("SST ($\degree C$)",fontsize=fsz_axis)

ax2.tick_params(labelsize=fsz_tick)
ax2.legend(fontsize=fsz_axis,ncol=3,bbox_to_anchor=leg_loc)

#ax2.grid(True,which='major',ls='dotted')

ax2.set_yticks(np.arange(-.5,0.75,0.25))
ax2.set_xticks(timeplot[::240],labels=(timeplot[::240]/12+1870).astype(int))
viz.add_ticks(ax=ax2,fontsize=fsz_tick,add_grid=False,
              facecolor=bgcol,spinecolor=dfcol,tickcolor=dfcol,ticklabelcolor="k")

ax.set_facecolor(bgcol)
ax2.set_facecolor(bgcol)

#plt.show()
viz.label_sp(subplotii,alpha=0.75,ax=ax2,fontsize=fsz_title,y=splaby,x=-.156)

savename = "%sAMV_Pattern_Thesis_Figure.png" % (outpath)

if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
    
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)

#%% Troubleshoot it


var_in  = ssta
lon     = lonh
lat     = lath
bbox    = bbox

order = 5
cutofftime=10
anndata=False
runmean=False
dropedge=0
monid=None
nmon=12
mask=None
return_unsmoothed=False
verbose = True

# Resample to monthly data 
if anndata == False:
    sst      = np.copy(var_in)
    annsst   = ann_avg(sst,2,monid=monid,nmon=nmon)
    
else:
    annsst   = var_in.copy()

# Calculate Index
if mask is not None: 
    if verbose:
        print("Masking SST for Index Calculation!")
    amvidx,aasst   = calc_AMV(lon,lat,annsst*mask[:,:,None],bbox,order,cutofftime,1,runmean=runmean)
else:
    amvidx,aasst   = calc_AMV(lon,lat,annsst,bbox,order,cutofftime,1,runmean=runmean)



annsst         = proc.ann_avg(sst,2,monid=monid,nmon=nmon)
amvidx,aasst   = proc.calc_AMV(lon,lat,annsst,bbox,order,cutofftime,1,runmean=runmean)
amvpattern     = proc.regress2ts(annsst,idxnorm,nanwarn=0,verbose=verbose)


aavg_sst       = proc.area_avg(sst,bbox,lonh,lath,1)

apat_nosmooth     = proc.regress2ts(sst,aavg_sst,nanwarn=0,verbose=verbose)




#%%

#plt.pcolormesh(amvpattern.T,vmin=-.4,vmax=.4,cmap='cmo.balance')
#plt.contourf(amvpattern.T,levels=np.arange(-.4,.44,0.04),cmap='cmo.balance')
# Detrend, if option is set



# Calculate AMV and AMV Pattern
#amvid,amvpattern=proc.calc_AMVquick(ssta,lonh,lath,bbox,runmean=False,)

dtr = 1

amvid       = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=True,dtr=dtr)
amvidstd    = amvid/amvid.std(1)[:,None] # Standardize

amvid       = amvid.squeeze()
amvidraw    = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=False,dtr=dtr)
amvidraw    = amvidraw.squeeze()

# Calculate undetrended version
amvid_undtr     = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=True,dtr=False)
amvidraw_undtr  = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=False,dtr=False)

# Regress back to sstanomalies to obtain AMV pattern
#ssta   = ssta.transpose(1,0,2,3) # [time x ens x lon x lat]
sstar         = ssta.reshape(nmon,nlat*nlon) 
beta,_        = regress_2d(amvidstd.squeeze(),sstar)
beta_undtr,_  = regress_2d((amvid_undtr/amvid_undtr.std(1)[:,None]).squeeze(),sstar)
amvpath       = beta
amvpath       = amvpath.reshape(nlat,nlon)

#%% Make a Robinson plot for the AMV pattern/impacts plot

plot_sig = True

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(16,12),
                      subplot_kw={'projection':ccrs.Robinson()})


# # # # Plot Significance if option is set
# if plot_sig:
#     viz.plot_mask(lonh, lath, sigmask, ax=ax, markersize=1,color='gray',
#                   reverse=True,geoaxes=True,proj=proj)
    
    
    
ax = viz.add_coast_grid(ax,bbox=[-179,179,-75,75],fill_color="dimgray",
                        fontsize=fsz_tick)

# Plot the variable
plotvar = amvpattern_dt_gm.T
pcm     = ax.contourf(lonh,lath,plotvar,transform=proj,
                      levels=cints,
                      cmap='cmo.balance',zorder=-1)


cl = ax.contour(lonh,lath,plotvar,transform=proj,
                levels=cints,colors="k",linewidths=0.55)
#cl = ax.clabel(cl,fontsize=fsz_tick)

if plot_sig:
    htch = ax.contourf(lonh,lath,sigmask.T,transform=proj,
                          levels=[0,1],
                          cmap='gray',extend='both',alpha=0.00,
                          hatches=['/',""],zorder=-1)
    #htch.set_alpha(0.02)
    #for i,bar in enumerate()

# Add Colorbar
cb      = viz.hcbar(pcm,ax=ax,pad=.02,fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick,rotation=0)
cb.set_label("AMV Pattern ($\degree$C per $1\sigma_{AMV}$)",fontsize=fsz_axis)

savename = "%sAMV_Impacts_Map_Thesis_Figure.png" % (outpath)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
    
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)



