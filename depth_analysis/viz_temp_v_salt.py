#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize depth properties of temp and salt

Works with data from get_point_data_stormtrack.py

Created on Mon Mar 25 13:55:27 2024

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob
import time

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
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
procpath    = pathdict['procpath']




# %% Set data paths

# Select Point
lonf   = -36 + 360#330
latf   = 62#50
locfn, loctitle = proc.make_locstring(lonf, latf)


# For certain points, load profile analysis data (Dummy Fix)
dir_index             = 0 # 'Center' 'N' 'S' 'E' 'W'
load_profile_analysis = True # Dummy fix to load output from profile analysis
profilepath           = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"
dsname                = "IrmingerAllEns_SALT_TEMP.nc"



# Calculation Settings
lags   = np.arange(0,37,1)
lagmax = 3 # Number of lags to fit for exponential function 

# Indicate Paths
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (
    lonf, latf)
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240628/"
proc.makedir(figpath)
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % locfn

# Other toggles
debug = True # True to make debugging plots

# Plotting Stuff
mons3 = proc.get_monstr(nletters=3)
mpl.rcParams['font.family'] = 'JetBrains Mono'

vnames = ["SALT","TEMP"]
vunits = ["psu",r"$\degree C$"]
vmaps  = ["cmo.haline","cmo.thermal"]

vcolors = ["navy","hotpink"]
# --------------------------------------------------------------------
#%% 1. Load necessary files (see repair_file_SALT_CESM1.py)
# --------------------------------------------------------------------
# Largely copied from calc_detrainment_damping_pt.py

# Load SALT/TEMP ----------------------------------------------------------------

if load_profile_analysis: # Assume to be on Astraeus
    
    dsprof          = xr.open_dataset(profilepath + dsname).isel(dir=dir_index)
    lonf            = dsprof.TLONG.data.item()
    latf            = dsprof.TLAT.data.item()
    locfn, loctitle = proc.make_locstring(lonf, latf)
    dsprof['z_t']   = dsprof['z_t']/100 # Convert to Meters
    dsprof          = dsprof.rename(dict(ensemble='ens'))
    dsvar           = [dsprof[vnames[0]],dsprof[vnames[1]]]
    ds = dsprof
    
    
else:
    
    dsvar = []
    for v in range(2):
        ncname = outpath + "CESM1_htr_%s_repaired.nc" % vnames[v]
        ds  = xr.open_dataset(ncname).load()[vnames[v]]
        dsvar.append(ds)
    
    



#%%
# Get some variables
z_t         = ds.z_t.values
times       = ds.time.values

# Load Mixed Layer Depths
mldpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
mldnc       = "CESM1_HTR_FULL_HMXL_NAtl.nc"


# Load and select point
dsh         = xr.open_dataset(mldpath+mldnc)
hbltpt      = dsh.sel(lon=lonf-360, lat=latf,
                 method='nearest').load()  # [Ens x time x z_t]

# Compute Mean Climatology [ens x mon]
hclim           = hbltpt.h.values
lags            = np.arange(61)

# Compute Detrainment month
kprev, _        = scm.find_kprev(hclim.mean(-1)) # Detrainment Months #[12,]
hmax            = hclim.max()#hclim.mean(1).max() # Maximum MLD of seasonal cycle # [1,]


# Compute mean profiles (ensembe)
mean_profiles   = [ds.groupby('time.month').mean('time').mean('ens') for ds in dsvar]


# Also load the full mixed layer
mldpath2        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
mldnc2          = "CESM1LE_HMXL_NAtl_19200101_20050101_bilinear.nc"
dsmld2          = xr.open_dataset(mldpath2+mldnc2).HMXL.sel(lon=lonf-360, lat=latf,method='nearest').load() 

hmxl = (dsmld2/100).rename(dict(ensemble='ens',))
hmxl['ens'] = np.arange(1,43,1)

#%% Do preprocessing (deseason and detrend)

def preproc(ds):
    ds = proc.fix_febstart(ds)
    dsa = ds - ds.mean('ens')
    dsa = proc.xrdeseason(dsa)
    return dsa

dsanom    = [preproc(ds) for ds in dsvar]
ds_monstd = [ds.groupby('time.month').std('time') for ds in dsanom]

#%% Visualize monthly stdev for each variable

ylim = [0,550]
fig,axs = plt.subplots(1,2,constrained_layout=True,figsize=(12,4.5))


#xx,yy = np.meshgrid(z_t,np.arange(12))
for a in range(2):
    
    ax      = axs[a]
    plotvar = ds_monstd[a].mean('ens')
    
    pcm = ax.pcolormesh(mons3,z_t,plotvar.T,cmap=vmaps[a])
    ax.set_title(vnames[a])
    
    mu    = hclim.mean(-1)
    sigma = hclim.std(-1) 
    ax.plot(mons3,mu,color="k")
    ax.fill_between(mons3,mu-sigma,mu+sigma,alpha=0.3,color="k")
    ax.set_ylim(ylim)
    #ax.scatter(xx,yy,marker=".")
    
    cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.045,pad=0.01)
    cb.set_label("$1\sigma(%s)$, [%s]" % (vnames[a],vunits[a]))
    ax.invert_yaxis()
    #ax.set_aspect('equal')
    
    if a == 0:
        ax.set_ylabel("Depth [m]")
        ax.legend()

plt.suptitle("Ens. Avg. Monthly Standard Deviation of TEMP/SALT Anomalies (CESM1 Historical)")
    
#%% Plot mean monthly profiles and mixed layer depth





#%% Look at how persistent conditions are at each level
# Section below has been functionized. can skip or delete

# Input variables



# # Start Script
# nlags         = len(lags)
# nens,ntime,nz = invars[0].shape
# nyr           = int(ntime/12)
# acf_z         = np.zeros((nens*nz,nlags,12)) # Preallocate

# # Reshape to [ens*z x yr x mon]
# invars_rs     = [v.transpose('ens','z_t','time').values.reshape(nens*nz,nyr,12) for v in invars]


# # Perform loop by month
# for im in tqdm(range(12)):
#     #varmons        = [v[:,:,im] for v in invars_rs] # [Ens*Depth x Year X Mon
#     flipvars        = [proc.flipdims(v) for v in invars_rs]
#     basevar,lagvar  = flipvars # [Mon x Year X Npts]
    
    
#     lagcovar        = proc.calc_lagcovar_nd(basevar,lagvar,lags,im+1,0) # Should be mon x yrx npts
#     acf_z[:,:,im] = lagcovar.T.copy()
    

# acf_z = acf_z.reshape(nens,nz,nlags,12)

# # Place back into a data array
# coords   = dict(ens=np.arange(1,43,1),z_t=z_t,lag=lags,mon=np.arange(1,13,1))
# acf_calc = xr.DataArray(acf_z,coords=coords,dims=coords)

#%% Make above into a function

def calc_acf_z(invars,lags,):
    nlags         = len(lags)
    nens,ntime,nz = invars[0].shape
    nyr           = int(ntime/12)
    acf_z         = np.zeros((nens*nz,nlags,12)) # Preallocate
    
    # Reshape to [ens*z x yr x mon]
    invars_rs     = [v.transpose('ens','z_t','time').values.reshape(nens*nz,nyr,12) for v in invars]
    
    # Perform loop by month
    for im in tqdm(range(12)):
        #varmons        = [v[:,:,im] for v in invars_rs] # [Ens*Depth x Year X Mon
        flipvars        = [proc.flipdims(v) for v in invars_rs]
        basevar,lagvar  = flipvars # [Mon x Year X Npts]
        
        lagcovar        = proc.calc_lagcovar_nd(basevar,lagvar,lags,im+1,0) # Should be mon x yrx npts
        acf_z[:,:,im] = lagcovar.T.copy()
    
    # Reshape and put into a data array
    acf_z  = acf_z.reshape(nens,nz,nlags,12)
    coords = dict(ens=np.arange(1,43,1),z_t=z_t,lag=lags,mon=np.arange(1,13,1))
    ds_acf = xr.DataArray(acf_z,coords=coords,dims=coords)
    return ds_acf


# Compute Pointwise Autocorrelation with each variable
ds_acfs = []
for vv in range(2):
    invars = [dsanom[vv],dsanom[vv]]
    ds_acf = calc_acf_z(invars,lags)
    ds_acfs.append(ds_acf)

#%% Now visualize the mean persistence of wintertime anomalies

t2_all = []
for vv in range(2):
    ds_in = ds_acfs[vv].mean('ens')
    t2 = proc.calc_T2(ds_in,axis=1)
    t2_all.append(t2)
    
    
#%% Plot the persistence timescale at different depths/months


ylim    = [0,500]
pmesh   = False
t2lims  = [[0,30],[0,30]]
t2ints  = np.arange(0,32,1)

fig,axs = plt.subplots(1,2,constrained_layout=True,figsize=(12,4.5))

#xx,yy = np.meshgrid(z_t,np.arange(12))
for a in range(2):
    
    ax      = axs[a]
    plotvar = t2_all[a].T
    vlm = t2lims[a]
    if pmesh:
        pcm     = ax.pcolormesh(mons3,z_t,plotvar.T,cmap=vmaps[a],vmin=vlm[0],vmax=vlm[-1])
    else:
        pcm     = ax.contourf(mons3,z_t,plotvar.T,cmap=vmaps[a],levels=t2ints,extend='both')
        cl      = ax.contour(mons3,z_t,plotvar.T,colors='lightgray',linewidths=0.75,levels=t2ints,extend='both',)
        ax.clabel(cl)
    ax.set_title(vnames[a])
    
    mu      = hclim.mean(-1)
    sigma   = hclim.std(-1) 
    ax.plot(mons3,mu,color="k",label="Ens. Mean MLD")
    ax.fill_between(mons3,mu-sigma,mu+sigma,alpha=0.3,color="k")
    ax.set_ylim(ylim)
    
    cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.045,pad=0.01)
    cb.set_label("$1\sigma(%s)$, [%s]" % (vnames[a],vunits[a]))
    ax.invert_yaxis()
    
    ax.scatter(kprev[kprev!=0.]-1,hclim.mean(1)[kprev!=0.],marker="x",color="royalblue",zorder=5,
               label="Detrainment time")
    if a == 0:
        ax.set_ylabel("Depth [m]")
        ax.legend()
    
plt.suptitle("Ens. Avg. Persistence Timescale ($T^2$) of TEMP/SALT Anomalies (CESM1 Historical)")

savename = "%sTEMP_SALT_T2_Monthly_EnsAvg_CESM1HTR_%s.png" % (figpath,locfn)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#%% Visualize monthly mean profiles and mixed layer depth (copied from visualize_profiles_hmxl)

#% Plot Monthly Mean Profiles
im   = 0
ylim = 1000 #None

for im in range(12):
    # Initialize Plot
    fig,ax=plt.subplots(1,1,constrained_layout=True,figsize=(4,8))
    if ylim is not None:
        ax.set_ylim([0,ylim])
    axs=viz.init_profile(ax)

    
    # Plot the Profile
    lns = []
    for vv in range(2):
        ax          = axs[vv]
        plotprof    = mean_profiles[vv].isel(month=im)
        z_t         = plotprof.z_t
        
        l,=ax.plot(plotprof,z_t,label=vnames[vv],c=vcolors[vv])
        lns.append(l)
    
    # Plot the MLD
    ploth = hclim.mean(1)[im]
    hl    = ax.axhline([ploth],lw=2,label=u"$\overline{h}$ = %.2fm" % (ploth),c='k',ls='solid')
    lns.append(hl)
    
    # Adjustments and legend
    labs = [l.get_label() for l in lns]
    ax.legend(lns,labs,loc="lower right")
    ax.set_title("%s Mean Profile @ %s" % (mons3[im],loctitle))
    
    savename = '%sCESM1_HTR_Mean_SALT_TEMP_Profile_%s_mon%02i.png' % (figpath,locfn,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')


#%%

# Moved this to proc so I can remove it theoretically
# def rep_ds(ds,repdim,dimname):
#     nreps  = len(repdim)
#     dsrep  = [ds for n in range(nreps)]
#     dsrep  = xr.concat(dsrep,dim=dimname)
#     dsrep  = dsrep.reindex(**{ dimname :repdim})
#     return dsrep

# Select which variables to input

acfs_surf = []
for vv in range(2):

    varin1        = dsanom[vv].sel(z_t=slice(0,50)).mean('z_t')
    varin1        = proc.rep_ds(varin1,z_t,'z_t').transpose('ens','time','z_t')

    # varin1        = [varin1 for n in range(nz)]
    # varin1        = xr.concat(varin1,dim='z_t')
    # varin1rep2        = varin1.reindex({'z_t':z_t})

    varin2        = dsanom[vv]
    invars        = [varin1,varin2]
    ds_acf        = calc_acf_z(invars,lags)
    acfs_surf.append(ds_acf)

# Compute Rho crtiical maps for each ariable
rhocrit_var = []
for vv in range(2):
    rhocrit_ens = []
    for e in range(42):
        ts_in = dsanom[vv].sel(z_t=slice(0,50)).mean('z_t').isel(ens=e).values
        neff  = proc.calc_dof(ts_in,ntotal=86)
        rcrit  = proc.ttest_rho(0.05,2,neff) 
        rhocrit_ens.append(rcrit)
    rhocrit_var.append(rhocrit_ens)
rhocrit_var = np.array(rhocrit_var)
        
    
        
    # #r1    = acfs_surf[ii].isel(z_t=0,lag=1).mean('ens').values # mon
    # #neff  = proc.calc_dof(r1,calc_r1=False,ntotal=86)
    # #neff   = 86 * (1-r1)/(1+r1)
    # rcrit  = proc.ttest_rho(0.05,2,86) 

#%% Plot lag correlation with 1-50 meter average (surface re-emergence plots)

im   = 1
for im in range(12):
    
    xtks   = np.arange(0,37,3)
    ylim   = [0,1000]
    pmesh  = False
    
    t2ints = np.arange(-1,1.05,0.05)
    
    
    fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,10))
    
    for a in range(2):
        
        ax      = axs[a]
        ax,_ = viz.init_acplot(im,xtks,lags,ax=ax,title=vnames[a])
        plotvar = acfs_surf[a].mean('ens').isel(mon=im).T
        vlm = t2lims[a]
        if pmesh:
            pcm     = ax.pcolormesh(lags,z_t,plotvar.T,cmap=vmaps[a],vmin=vlm[0],vmax=vlm[-1])
        else:
            pcm     = ax.contourf(lags,z_t,plotvar.T,cmap='cmo.balance',levels=t2ints,extend='both')
            cl      = ax.contour(lags,z_t,plotvar.T,colors='lightgray',linewidths=0.75,levels=t2ints,extend='both',)
            ax.clabel(cl)
        #ax.set_title(vnames[a],fontsize=14)
        
        mu          = np.roll(np.array([hclim.mean(-1),]*5).flatten(),-im)
        sigma       = np.roll(np.array([hclim.std(-1),]*5).flatten(),-im)
        #mons3rep    = np.array([mons3,]*3).flatten()
        ax.plot(lags[:-1],mu,color="k",label="Ens. Mean MLD")
        ax.fill_between(lags[:-1],mu-sigma,mu+sigma,alpha=0.3,color="k")
        ax.set_ylim(ylim)
        
        viz.plot_mask(lags,z_t,plotvar>rhocrit_var[a].mean(),reverse=True,marker=".",color='midnightblue')
    
        ax.invert_yaxis()
        
        # ax.scatter(kprev[kprev!=0.]-1,hclim.mean(1)[kprev!=0.],marker="x",color="royalblue",zorder=5,
        #            label="Detrainment time")
        if a == 0:
            ax.set_xlabel("")
            
            ax.legend()
        ax.set_ylabel("Depth [m]")
        
    cb      = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.055,pad=0.01)
    cb.set_label("Correlation with %s Anomalies" % (mons3[im]))
    
    plt.suptitle("Ens. Avg. Lagged ACF of TEMP/SALT Anomalies (CESM1 Historical)\n Lag 0 = %s Anomalies (0-50m average)" % (mons3[im]),
                 y=1.03)
        
    savename = "%sTEMP_SALT_DepthACF_%s_mon%02i_EnsAvg_CESM1HTR.png" % (figpath,locfn,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)
    
#%% Plot ACF at different depths for a selected variable

vv      = 1
im      = 1

#

zid_ranges  = [np.arange(ii*7,ii*7+7+1) for ii in range(6)]

#zid_ranges = np.arange(0,7),np.arange(0,42,7)

acfs_in     = ds_acfs[vv].mean('ens') # [z_t x "lag x basemonth]


fig,axs     = plt.subplots(6,1,constrained_layout=True,figsize=(12,18))

for aa in range(6):
    
    ax   = axs.flatten()[aa]
    ax,_ = viz.init_acplot(im,xtks,lags,ax=ax,title="",vlines=[2,6])
    if aa < 6-1:
        ax.set_xlabel("")
        
    zids = zid_ranges[aa]
    for iz,zz in enumerate(zids):
        pa      = (0.2 + iz/len(zids)) / 1.2
        plotvar = acfs_in.isel(mon=im,z_t=zz)
        ax.plot(plotvar.lag,plotvar,label="z=%.2f [m]" % (z_t[zz]),lw=2.5,alpha=pa)
    ax.legend(ncol=3)
    
plt.suptitle("%s ACF for %s" % (mons3[im],vnames[vv]),y=1.02,fontsize=24)

savename = "%s%s_DepthACF_mon%02i_EnsAvg_CESM1HTR.png" % (figpath,vnames[vv],im+1)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)
        
        
#%% Do composites of (+)/(-) anomalies




# def get_posneg_events(inthres,inref):
#     # Given years for a certain month, identify the anomalies
#     idpos = np.where(inref > inthres)[0]
#     idneg = np.where(inref < inthres)[0]
#     return [idpos,idneg]

# def get_composite_len(ids_sel,window,ints):
#     # Given the years in ids_sel and ints in [yr x mon]
    
#     #Issue here is that you need to find the month and date, which 
#     # would require the unraveled timeseries
#     nevents = len(ids_sel)
#     for ie in range(nevents):
#         ievent = ids_sel[ie]
#         istart = ievent - window


im          = 1 # Select a month to do analysis
vv          = 0 # Select a variable
window      = 36

thres       = 1 # threshold in standard deviation
ref_depth   = 5 # reference depth to identify the anomalies (in meters)
ds_in       = dsanom[vv] # Dataset containing anomalies [ens x time x z_t]

# =============================================================================
# Turn this section below into a function -------------------------------------
# Slice to reference depth
ds_ref      = ds_in.sel(z_t=slice(0,ref_depth)).mean('z_t')
ntime       = len(ds_ref.time)


# Get Thresholds (take mean across ensembles for simplicity)
mon_std     = ds_ref.groupby('time.month').std('time').mean('ens') # [ens x mon]


# Identify ievents for a given month
#for im in range(12):
# Get the threshold
monthres           = (mon_std.isel(month=im) * thres).item() * thres

# Month Indices
monids             = np.where(ds_ref.time.dt.month.isin(im+1))[0]

# Loop for each sign
composites_sn      = []
composite_times_sn = []
composite_ens_sn   = []
event_times_sn     = []
event_ens_sn       = []
depth_comp_sn      = []

for sgn in range(2):
    
    if sgn == 0:
        # Exceedance Indices (positive)
        idens,idtime=np.where(ds_ref.data > monthres)
    else:
        # Exceedance Indices (negative)
        idens,idtime=np.where(ds_ref.data < -monthres)
    
    # Restrict To Time points 
    idtime_mon = [True if ii in monids else False for ii in idtime]
    idens_sel  = idens[idtime_mon]
    idtime_sel = idtime[idtime_mon]
    
    # Quickly Cull events at the boundaries
    nevents = len(idens_sel)
    within_range = []
    idcull = 0
    for nn in tqdm(range(nevents)):
        
        itime = idtime_sel[nn]
        
        # Set up window
        istart = itime-window
        iend   = itime+window
        #id_nan = []
        if istart < 0: # Skip edge cases for now
            within_range.append(False)
            idcull +=1
            continue
            ##ontinue#id_nan = id_nan + list(np.arange(0,np.abs(istart)))
            #istart = 0
        if iend > ntime-1:
            within_range.append(False)
            idcull +=1
            continue
            #continue
            #iend   = ntime-1
        within_range.append(True)
    print("Dropping %i events at the boundaries" % idcull)
    idens_sel = idens_sel[within_range]
    idtime_sel = idtime_sel[within_range]
    
    # For each event, grab the number of requested timesteps
    nevents = len(idens_sel)
    
    composites      = np.zeros((window*2+1,nevents))  * np.nan# [Window, Event]
    composite_times = np.zeros((window*2+1,nevents),dtype='object')  * np.nan
    depth_comp      = np.zeros((window*2+1,len(z_t),nevents)) * np.nan
    event_times     = []
    event_ens       = []
    for nn in tqdm(range(nevents)):
            
        # Get Indices
        iens  = idens_sel[nn]
        itime = idtime_sel[nn]
        event_ens.append(iens+1)
        
        # Get Event Time
        event_time = ds_ref.isel(ens=iens,time=itime).time.item()
        event_times.append(event_time)
        
        # Set up window
        istart = itime-window
        iend   = itime+window
        #id_nan = []
        if istart < 0: # Skip edge cases for now
            continue#id_nan = id_nan + list(np.arange(0,np.abs(istart)))
            istart = 0
        if iend > ntime-1:
            continue
            iend   = ntime-1
        
        # Get composite data
        comp_val = ds_ref.isel(ens=iens,time=np.arange(istart,iend+1))#.data
        composites[:,nn] = comp_val.data.copy()
        composite_times[:,nn] = comp_val.time.data.copy()
        
        # Get composite data for full depth
        comp_snapshot = ds_in.isel(ens=iens,time=np.arange(istart,iend+1))
        depth_comp[:,:,nn] = comp_snapshot.data.copy()
        # End Event Loop >>
    
    # Append to Output
    composites_sn.append(composites)
    composite_times_sn.append(composite_times)
    event_times_sn.append(event_times)
    event_ens_sn.append(event_ens)
    depth_comp_sn.append(depth_comp)
    
    # End Sign Loop >>

# Turn this section above into a function -------------------------------------   
# =============================================================================
#%% Add Helper Function

targ_ds = hmxl # [needs to have ens and time dimension]

def composite_variable(targ_ds,composite_times_sn,event_ens_sn):
    
    composite_var_sn = []
    for sn in range(2):
        
        # Get Time Range and Ensemble
        times_sn      = composite_times_sn[sn]
        ens_sn        = event_ens_sn[sn]
        
        nevents       = times_sn.shape[1]
        composite_var = []
        for nn in range(nevents):
            trange = [times_sn[:,nn][0],times_sn[:,nn][-1]]
            ens    = ens_sn[nn]
            
            var_select = targ_ds.sel(ens=ens,time=slice(trange[0],trange[1]))
            composite_var.append(var_select.data)
        composite_var_sn.append(np.array(composite_var))
    return composite_var_sn # [sign][leadlag x event x otherdims??]
            

hmxl_sn = composite_variable(hmxl,composite_times_sn,event_ens_sn)
   
#%% Do some visualizations of identified events (timeseries at the reference level)

sncolors    = ['firebrick','cornflowerblue']
sncolors_mn = ['darkred','darkblue']


snnames  = ["Positive Event","Negative Event"]
leadlags       = np.arange(-window,window+1)
lagticks = leadlags[::3]


leadlag_labels = viz.prep_monlag_labels(im,lagticks,1)


fig,ax = plt.subplots(1,1,figsize=(12,4))
for sn in range(2):
    
    compplot = composites_sn[sn]
    nevents  = compplot.shape[1]
    for nn in range(nevents):
        ax.plot(leadlags,compplot[:,nn],alpha=0.05,c=sncolors[sn],label="")
        
    ax.plot(leadlags,np.nanmean(compplot,1),c=sncolors_mn[sn],label="%s, n=%i" % (snnames[sn],nevents))
    
ax.axhline(-monthres,ls='dashed',lw=0.75,c='k')
ax.axhline(monthres,ls='dashed',lw=0.75,c='k',label="Threshold = %.2f %s" % (monthres,vunits[vv]))
ax.axvline(0,ls='solid',lw=0.55,c='k',)
ax.axhline(0,ls='solid',lw=0.55,c='k',)


ax.legend(ncol=3)
ax.set_xticks(lagticks,labels=leadlag_labels)
ax.set_xlim([leadlags[0],leadlags[-1]])

ax.set_xlabel("Event Lead/lag (months)")
ax.set_ylabel("%s Anomaly %s" % (vnames[vv],vunits[vv]))
ax.set_title("Composites for %s %s Anomalies (Avg 0 to %.2fm) @ %s" % (mons3[im],vnames[vv],ref_depth,loctitle))


savename = "%s%s_%s_Composites_Ref_thres%istd_depth%04i_mon%02i.png" % (figpath,locfn,vnames[vv],thres,ref_depth,im+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot a composite hovmuller (Depth vs LeadLag for Composite)

#cints   = 
if vv == 0:
    #cints = np.arange(-.2,.21,0.01)
    cints = np.arange(-.1,.11,0.01)
else:
    cints = np.arange(-1.5,1.6,0.1)
ylim   = [0,500]
fig,axs = plt.subplots(2,1,figsize=(16,11))


for sn in range(2):
    
    ax      = axs[sn]
    #etimes  = composite_times_sn[sn] #[leadlag x event]
    #nevents = etimes.shape[1]
    
    depth_comp = np.nanmean(depth_comp_sn[sn],2)
    nevents    = depth_comp_sn[sn].shape[2]
    #pcm = ax.pcolormesh(leadlags,z_t,depth_comp.T)
    pcm        = ax.contourf(leadlags,z_t,depth_comp.T,levels=cints,cmap='cmo.balance')
    cl         = ax.contour(leadlags,z_t,depth_comp.T,levels=cints,colors="dimgray",linewidths=0.75)
    ax.clabel(cl,cints[::2])
    
    # Plot the Mixed layer depth composites
    mu = np.nanmean(hmxl_sn[sn],0)
    sigma = np.nanstd(hmxl_sn[sn],0)
    ax.plot(leadlags,mu,label="HMXL",c='k',ls='dotted')
    ax.fill_between(leadlags,mu-sigma,mu+sigma,alpha=0.10,color="k")
    
    # 
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.set_title("%s Anomalies (n=%i)" % (snnames[sn],nevents),fontsize=18)
    
    ax.axvline([0],ls='dashed',c='k',lw=0.55)
    ax.set_xticks(lagticks,labels=leadlag_labels)
    
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.015)    
cb.set_label("%s Anomaly [%s]" % (vnames[vv],vunits[vv]),fontsize=18)   

plt.suptitle("3-D Composites for %s %s Anomalies @ %s" % (mons3[im],vnames[vv],loctitle),fontsize=22)

savename = "%s%s_%s_3DComposites_Ref_thres%istd_depth%04i_mon%02i.png" % (figpath,locfn,vnames[vv],thres,ref_depth,im+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% 

#
#     #exceed_id = np.where(ds_ref.data.flatten() > monthres)

#     # Identify the events
#     pos_events = xr.where(ds_ref > monthres,ds_ref.time.dt.month,False)
#     pos_events = xr.where(pos_events == im+1, pos_events.time,False)

#     mon_sel    = xr.where(ds_ref.time.month == (im+1),True,False)
#     pos_events = xr.where(ds_ref > monthres,True,False)
#     pos_events = pos_events.sel(time=pos_events.time.dt.month.isin([im+1]))
#     pos_time   = pos_events.time

#     idens_pos,idtime_pos = np.where((ds_ref > monthres)) # and ds_ref.time.month == 2)
#     idmon,       = np.where(ds_ref.time.month == (im+1))
#     id_pos       = 

# ds_mon     = ds_ref.groupby('time.month').sel()

# # Separate by month
# ds_ref      = ds_ref.values # [ens x time]
# nens,ntime  = ds_ref.shape
# nyrs        = int(ntime/12)
# ds_ref      = ds_ref.reshape(nens,nyrs,12)

# # for im in range(12):
# ds_ref_mon = ds_ref[:,:,im]
# thres_std  = np.std(ds_ref_mon,1) # [ens]

# # Write a stupid loop
# for e in range(nens):
#     inthres = thres_std[e]
#     inref   = ds_ref_mon[e,:]

# # Find positive events
# id_pos = np.where(ds_ref_mon > thres_std)
#
#viz.prep_monlag_labels(im,leadlags,label_interval=2)







