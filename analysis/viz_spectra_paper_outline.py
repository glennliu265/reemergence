#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied upper section from [viz_T2_paper_outline.py]

Created on Wed May  1 11:17:36 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']

#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm


#%% Load other Plotting things ------------------------------------------------


#%% Load BSF and Ice Mask (copied from compare_detrainment_damping)

bsf      = dl.load_bsf()

# Load Land Ice Mask
icemask  = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# Resize
bsf,icemask    = proc.resize_ds([bsf,icemask])
bsf_savg = proc.calc_savg_mon(bsf)

#
mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0


# Load Gulf Stream
ds_gs   = dl.load_gs()
ds_gs   = ds_gs.sel(lon=slice(-90,-50))
ds_gs2  = dl.load_gs(load_u2=True)

# Load velocities
ds_uvel,ds_vvel = dl.load_current()
tlon  = ds_uvel.TLONG.mean('ens').data
tlat  = ds_uvel.TLAT.mean('ens').data

# Load Region Information
regionset       = "SSSCSU"
regiondicts     = rparams.region_sets[regionset]
bboxes          = regiondicts['bboxes']
regions_long    = regiondicts['regions_long']
rcols           = regiondicts['rcols']
rsty            = regiondicts['rsty']
regplot         = [0,1,3]
nregs           = len(regplot)

# -----------------------------------------------------------------------------


#%% Compute the variance

# Indicate files containing ACFs
sst_expname = "SST_EOF_LbddCorr_Rerun"##"SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
sss_expname = "SSS_EOF_LbddCorr_Rerun_lbdE_neg"#"SM_SSS_EOF_LbddCorr_Rerun_lbdE_SSS_autocorrelation_thresALL_lag00to60.nc"

#%% 

# Indicate Experiment Name and Variable
expname = sss_expname
vname   = "SSS"


# Indicate smoothing options
nsmooth = 100
pct     = 0.10
dt      = 3600*24*30

# Locate files
st      = time.time()
expdir  = "%s%s/Output/" % (output_path,expname,)
nclist  = glob.glob(expdir + "*runid*.nc")
nclist.sort()
print("Found %i files!" % len(nclist))

# Load files
ds_all  = xr.open_mfdataset(nclist,combine='nested',concat_dim='ens')[vname].load()
print("Loaded data in %.2fs" % (time.time()-st))

# Deseason
dsa     = proc.xrdeseason(ds_all)

# Now what...
#%% Compute spectra using numpy vectorize/xarray unfuncs
# Taken from visualize_atmospheric persistence
calc_spectra = lambda x: scm.point_spectra(x,nsmooth=nsmooth,pct=pct,dt=dt)
st = time.time()
daspec = xr.apply_ufunc(
    calc_spectra,  # Pass the function
    dsa,  # The inputs in order that is expected
    # Which dimensions to operate over for each argument...
    input_core_dims=[['time'],],
    output_core_dims=[['freq'],],  # Output Dimension
    exclude_dims=set(("freq",)),
    vectorize=True,  # True to loop over non-core dims
)
print("Finished pointwise spectra calculation in %.2fs" % (time.time()-st))
ts1  = dsa.isel(ens=0,lon=0,lat=0).values
freq = scm.get_freqdim(ts1)
daspec['freq'] = freq

# Save the output
st = time.time()
daspec   = daspec.rename('spec')
edict    = dict(spec=dict(zlib=True))
savename = "%s../Metrics/Pointwise_Spectra_nsmooth%03i_pct%02i.nc" % (expdir,nsmooth,pct*100)
daspec.to_netcdf(savename,encoding=edict)
print("Saved spectra to %s.\nCompleted in %.2fs" % (savename,time.time()-st))


# #%% Load BSF and Ice Mask (copied from compare_detrainment_damping)

# bsf      = dl.load_bsf()

# # Load Land Ice Mask
# icemask  = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# # Resize
# #bsf,icemask,_    = proc.resize_ds([bsf,icemask,acfs_in_rsz[0]])
# bsf_savg = proc.calc_savg_mon(bsf)

# #
# mask = icemask.MASK.squeeze()
# mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

# mask_apply = icemask.MASK.squeeze().values
# #mask_plot[np.isnan(mask)] = 0

#%% Check the spectra (debug)

ts_test  = dsa.sel(lon=-30,lat=50,method='nearest').isel(ens=2)
specout  = scm.quick_spectrum([ts_test.values,],nsmooth,pct,dt=dt,return_dict=True)

spec_man = specout['specs'][0]
spec_xr  = daspec.sel(lon=-30,lat=50,method='nearest').isel(ens=2)

plt.plot(spec_man-spec_xr)

#%% Repeat Above for CESM1 (calculations)

vname  = "SST"

# Load the file, set the output
ncname       = "%sCESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % (rawpath,vname)
cesm_outpath = "%s%s_CESM/Metrics/" % (output_path,vname,)


# Indicate smoothing options
nsmooth = 10
pct     = 0.10
dt      = 3600*24*30


st     = time.time()
ds_all = xr.open_dataset(ncname)[vname].load()
ds_all = ds_all.rename({'ensemble':'ens'})
dsa    = ds_all - ds_all.mean('ens')
dsa    = proc.xrdeseason(dsa)
print("Loaded and proccesed data in %.2fs" % (time.time()-st))


# Compute spectra using numpy vectorize/xarray unfuncs
# Taken from visualize_atmospheric persistence
calc_spectra = lambda x: scm.point_spectra(x,nsmooth=nsmooth,pct=pct,dt=dt)
st = time.time()
daspec = xr.apply_ufunc(
    calc_spectra,  # Pass the function
    dsa,  # The inputs in order that is expected
    # Which dimensions to operate over for each argument...
    input_core_dims=[['time'],],
    output_core_dims=[['freq'],],  # Output Dimension
    exclude_dims=set(("freq",)),
    vectorize=True,  # True to loop over non-core dims
)
print("Finished pointwise spectra calculation in %.2fs" % (time.time()-st))
ts1  = dsa.isel(ens=0,lon=0,lat=0).values
freq = scm.get_freqdim(ts1)
daspec['freq'] = freq

#%
# Save the output
st = time.time()
daspec   = daspec.rename('spec')
edict    = dict(spec=dict(zlib=True))
savename = "%sPointwise_Spectra_nsmooth%03i_pct%02i.nc" % (cesm_outpath,nsmooth,pct*100)
daspec.to_netcdf(savename,encoding=edict)
print("Saved spectra to %s.\nCompleted in %.2fs" % (savename,time.time()-st))


#%% Load the calculated spectra and visualize (maybe move this to a separate script)

"""
sst_expname = "SST_EOF_LbddCorr_Rerun"##"SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
sss_expname = "SSS_EOF_LbddCorr_Rerun_lbdE"#"SM_SSS_EOF_LbddCorr_Rerun_lbdE_SSS_autocorrelation_thresALL_lag00to60.nc"


"""

expnames      = ["SST_CESM","SST_EOF_LbddCorr_Rerun","SSS_CESM","SSS_EOF_LbddCorr_Rerun_lbdE_neg",]
nsmooths      = [10,100,10,100]

expnames_long = ["SST (CESM)","SST (SM)","SSS (CESM)","SSS (SM)"]



# Smoothing Settings
#nsmooth  = 10
pct      = 0.10
dt       = 3600*24*30

# Load the data
nexps    = len(expnames)
ds_all   = []
for ex in range(nexps):
    
    expname=expnames[ex]
    
    ncname = "%s%s/Metrics/Pointwise_Spectra_nsmooth%03i_pct%0i.nc" % (output_path,expname,nsmooths[ex],pct*100)
    ds     = xr.open_dataset(ncname).load()
    ds_all.append(ds)


#%% Compute the specvar

thresvals = [1,10,100] # intervals
sumname   = "OutlineInt" # Name of thresvals selected
nthres    = len(thresvals) + 1
dtplot    = 3600*24*365 # Unit thresvals is expressed in (years in this case)
recompute  = False
ex        = 0 # Loop for ds


specsum_exp = []
if recompute:
    for ex in range(nexps):
        ds_in     = ds_all[ex]
        freq      = ds_in.freq.values
        specs     = ds_in.spec
        expname   = expnames[ex]
    
        labels = []
        spec_bythres = []
        for th in range(nthres):# Loop by Threshold
            #print(th)
            
        
            if th == 0: # First value
                thres_in       = 1/thresvals[th]
                upperthres     = thres_in
                lowerthres     = 0
                label          = "<%i" % (1/thres_in)
            elif th == (nthres-1): # Intermediate value
                thres_in       = 1/thresvals[th-2]
                lowerthres     = thres_in
                upperthres     = freq[-1] * dtplot
                label          = ">%i" % (1/thres_in)
            else:
                lowerthres     = 1/thresvals[th]
                upperthres     = 1/thresvals[th-1]
                label          = "%i<x<%i" % (1/upperthres,1/lowerthres)
            
            labels.append(label)
            
            # Compute Spectra sum under selected frequencies
            specsum     = proc.calc_specvar(freq,specs,upperthres,dtplot,lowerthres=lowerthres)
        
            spec_bythres.append(specsum.copy())
        
        coords      = dict(thres=labels,ens=ds_in.ens,lat=ds_in.lat,lon=ds_in.lon)
        daspecsum   = xr.DataArray(spec_bythres,coords=coords,dims=coords)
        edict       = dict(spec=dict(zlib=True))
        daspecsum   = daspecsum.rename('spec')
        savename    = "%s%s/Metrics/Pointwise_Spectra_nsmooth%03i_pct%0i_specsum_%s.nc" % (output_path,expname,nsmooths[ex],pct*100,sumname)
        daspecsum.to_netcdf(savename,encoding=edict)
        
        print("Saved Output to %s" % savename)
        specsum_exp.append(daspecsum.copy())
else:
    
    specsum_exp = []
    for ex in range(nexps):
        expname     = expnames[ex]
        
        savename    = "%s%s/Metrics/Pointwise_Spectra_nsmooth%03i_pct%0i_specsum_%s.nc" % (output_path,expname,nsmooths[ex],pct*100,sumname)
        ds          = xr.open_dataset(savename).load()
        specsum_exp.append(ds.spec)
        
    daspecsum = specsum_exp[0]
    


#%% Load the Summed Specs


#%% Check it (Debug for spec sum)
lonf   = -30
latf   = 50
e      = 0
ptspec = ds_in.sel(lon=lonf,lat=latf,method='nearest').isel(ens=e)
    

func_val = daspecsum.sel(lon=lonf,lat=latf,method='nearest').isel(ens=e)

fig,ax   = plt.subplots(1,1)

freq_sel = (freq*dtplot < 1/10)

freqrange = (freq*dtplot)[freq_sel]
specrange = (ptspec.spec/dtplot)[freq_sel]

ax.plot(freq*dtplot,ptspec.spec/dtplot,label='Full Spectra')
ax.plot(freqrange,specrange,marker="x",c="orange",label="Selected Section")

# Seems relatively approximate (dropping first power?)
df      = (freqrange[1:] - freqrange[:-1]).mean()
man_val = ((freqrange[1:] - freqrange[:-1]) * specrange[1:]).sum() #+ (freqrange[1]-freqrange[0]) * specrange[0]


ax.axvline([1/(10)],color="k")
ax.axvline([1/(20)],color="g")
ax.axvline([1/(100)],color="g")

ax.set_xlim([0,1])
ax.set_title("For Freq < 1/10 years,\nFunc Val: %.6f vs. Manual Val: %.6f" % (func_val,man_val))



#%% Plotting Params
mpl.rcParams['font.family'] = 'Avenir'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
lon                         = daspecsum.lon.values
lat                         = daspecsum.lat.values
mons3                       = proc.get_monstr()

#%% Visualize some differences

thres     = 2

use_contour = True
imsk        = icemask.MASK.squeeze()

# Visualize Interannual Variability
fig,axs,_ = viz.init_orthomap(2,2,bboxplot,figsize=(12,10))


for a in range(4):
    
    ax      = axs.flatten()[a]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
    ax.set_title(expnames_long[a])
    
    plotvar = specsum_exp[a].isel(thres=thres,).mean('ens') * imsk
    
    if thres == 2:
        if a < 2:
            vmax   = .1
            cints  = np.arange(0,0.11,0.01) 
        else:
            vmax   = .01
            cints  = np.arange(0,0.011,0.001)
    elif thres == 1:
        if a < 2:
            vmax   = .1
            cints  = np.arange(0,0.5,0.05) 
        else:
            vmax   = .01
            cints  = np.arange(0,0.02,0.002)
        
    
    if use_contour:
        pcm = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cints,cmap='cmo.deep_r',extend='both')
        cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cints,linewidths=0.75,
                         colors='lightgray',extend='both')
        ax.clabel(cl)
    else:
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmin=0,vmax=vmax,cmap='cmo.deep_r')
    cb  = viz.hcbar(pcm,ax=ax,fraction=0.055)
plt.suptitle("Variance between years %s" % (labels[thres]))


figname = "%sVariance_Specsum_thres%0i.png" % (figpath,thres)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Examine Log Ratios

fsz_axis        = 18
fsz_title       = 24
fsz_tick        = 14

imsk = icemask.MASK.squeeze()
thres_sel       = [1,2]
thresnames      = ["Interannual (1-10 years)","Multidecadal (10-100 years)"]
vnames          = ["SST","SSS"]
plotcurrent     = False

cints           = np.log(np.array([.1,.5,2,10]))
cints           = np.sort(np.append(cints,0))
cints_lab       = [1/10,1/5,1/2,0,2,5,10]

# Visualize Interannual Variability
ii = 0
fig,axs,_       = viz.init_orthomap(2,2,bboxplot,figsize=(13.5,10))

for vv in range(2):
    for th in range(2):
        thres = thres_sel[th]

        ax      = axs[vv,th]
        ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        #ax.set_title(expnames_long[ex])
        
        #ax.set_title("%s (%s)" % (vnames[vv],thresnames[th]))
        
        if vv == 0:
            ax.set_title(thresnames[th],fontsize=fsz_axis)
        if th == 0:
            viz.add_ylabel(vnames[vv],ax=ax,rotation='horizontal',fontsize=fsz_axis)
        
        if vv == 0: # Log Ratio (SSTs)
            plotvar = np.log(specsum_exp[1].isel(thres=thres).mean('ens')/specsum_exp[0].isel(thres=thres).mean('ens'))
        else:
            plotvar = np.log(specsum_exp[3].isel(thres=thres).mean('ens')/specsum_exp[2].isel(thres=thres).mean('ens'))
        
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar*imsk,transform=proj,vmin=-2.5,vmax=2.5,cmap="cmo.balance",zorder=-1)
        cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar*imsk,transform=proj,levels=cints,colors="dimgray",zorder=2,linewidths=1.5)
        ax.clabel(cl,fmt="%.2f",fontsize=fsz_tick)
        
        # Plot Gulf Stream Position
        ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
        
        # Plot Ice Edge
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                   transform=proj,levels=[0,1],zorder=-1)
        
        # Plot Currents
        if plotcurrent:
            qint  = 2
            plotu = ds_uvel.UVEL.mean('ens').mean('month').values
            plotv = ds_vvel.VVEL.mean('ens').mean('month').values
            ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
                      color=[.9,.9,.9],transform=proj,alpha=.67,zorder=1)#scale=1e3)
    
        # Plot Regions
        for ir in range(nregs):
            rr   = regplot[ir]
            rbbx = bboxes[rr]
            
            ls_in = rsty[rr]
            if ir == 2:
                ls_in = 'dashed'
            
            viz.plot_box(rbbx,ax=ax,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=1.5,return_line=True)

        
        #cb  = viz.hcbar(pcm,ax=ax)
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
        ii+=1
        
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
cb.set_label("Log(Stochastic Model / CESM1)",fontsize=fsz_axis)
cb.ax.tick_params(labelsize=fsz_tick)
#plt.suptitle("Log Ratio (SM/CESM)")
    
#figname = "%sVariance_Specsum_LogRatio.png" % (figpath)
figname = "%sLogratio_Spectra.png" % (figpath)

if plotcurrent:
    figname = proc.addstrtoext(figname,"_withcurrent",)
plt.savefig(figname,dpi=150,bbox_inches='tight')
    
#%% Examine the power spectra over each region

vv    = 0
thres = 2

if vv == 0: # Log Ratio (SSTs)
    plotvar = np.log(specsum_exp[1].isel(thres=thres).mean('ens')/specsum_exp[0].isel(thres=thres).mean('ens'))
else:
    plotvar = np.log(specsum_exp[3].isel(thres=thres).mean('ens')/specsum_exp[2].isel(thres=thres).mean('ens'))

plt.pcolormesh(plotvar < np.log(0.5))

# -----------------------------------------------------------------------------
#%% The other approach is to do a lowpass filter and examine the results
# (CESM)

# Indicate Experiment Name and Variable
expname = sss_expname
vname   = "SSS"



# Locate files
st      = time.time()
expdir  = "%s%s/Output/" % (output_path,expname,)
nclist  = glob.glob(expdir + "*runid*.nc")
nclist.sort()
print("Found %i files!" % len(nclist))

# Load files
ds_all  = xr.open_mfdataset(nclist,combine='nested',concat_dim='ens')[vname].load()
print("Loaded data in %.2fs" % (time.time()-st))

# Deseason
dsa     = proc.xrdeseason(ds_all)


# Do a low pas filter

lpfilt = lambda x: proc.lp_butter(x,10,6,btype='lowpass')
st = time.time()
dalpfilt = xr.apply_ufunc(
    lpfilt,  # Pass the function
    dsa,  # The inputs in order that is expected
    # Which dimensions to operate over for each argument...
    input_core_dims=[['time'],],
    output_core_dims=[['time'],],  # Output Dimension
    vectorize=True,  # True to loop over non-core dims
)
print("Finished pointwise spectra calculation in %.2fs" % (time.time()-st))
#ts1  = dsa.isel(ens=0,lon=0,lat=0).values

#%% Examine Pattern of low pass filter

dalpvar = dalpfilt.var('time').mean('ens')

#%%






#%%





#%%






# Indicate CESM files


#cesm_name   = "CESM1_1920to2005_%sACF_lag00to60_ALL_ensALL.nc"
# Plotting Parameters



#%% Load the files (computed by pointwise_autocorrelation_smoutput) # Took 30.12s

st       = time.time()
sm_sss   = xr.open_dataset(procpath+sss_expname).SSS.load()        # (lon: 65, lat: 48, mons: 12, thres: 1, lags: 61)
sm_sst   = xr.open_dataset(procpath+sst_expname).SST.load()

cesm_sst = xr.open_dataset(procpath+cesm_name % "SST").acf.load()  #  (lon: 89, lat: 96, mons: 12, thres: 1, ens: 42, lags: 61)
cesm_sss = xr.open_dataset(procpath+cesm_name % "SSS").acf.load()
print("Loaded all data in %.2fs" % (time.time()-st))

#%% Do some Preprocessing (slice to same size, etc)
acfs_in = [cesm_sst,sm_sst,cesm_sss,sm_sss]

acfs_in_rsz = proc.resize_ds(acfs_in)

explabs = ["SST (CESM)","SST (Stochastic Model)","SSS (CESM)","SSS (Stochastic Model)"]
