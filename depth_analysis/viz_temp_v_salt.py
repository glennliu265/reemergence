#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize depth properties of temp and salt

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
lonf   = 330
latf   = 50
locfn, loctitle = proc.make_locstring(lonf, latf)

# Calculation Settings
lags   = np.arange(0,37,1)
lagmax = 3 # Number of lags to fit for exponential function 

# Indicate Paths
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (
    lonf, latf)
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240322/"
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
# --------------------------------------------------------------------
#%% 1. Load necessary files (see repair_file_SALT_CESM1.py)
# --------------------------------------------------------------------
# Largely copied from calc_detrainment_damping_pt.py

# Load SALT/TEMP ----------------------------------------------------------------
dsvar = []
for v in range(2):
    ncname = outpath + "CESM1_htr_%s_repaired.nc" % vnames[v]
    ds  = xr.open_dataset(ncname).load()[vnames[v]]
    dsvar.append(ds)

# Get some variables
z_t       = ds.z_t.values
times     = ds.time.values

# Load Mixed Layer Depths
mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl.nc"

# Load and select point
dsh         = xr.open_dataset(mldpath+mldnc)
hbltpt      = dsh.sel(lon=lonf-360, lat=latf,
                 method='nearest').load()  # [Ens x time x z_t]

# Compute Mean Climatology [ens x mon]
hclim       = hbltpt.h.values
lags        = np.arange(61)

# Compute Detrainment month
kprev, _    = scm.find_kprev(hclim.mean(-1)) # Detrainment Months #[12,]
hmax        = hclim.max()#hclim.mean(1).max() # Maximum MLD of seasonal cycle # [1,]
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

savename = "%sTEMP_SALT_T2_Monthly_EnsAvg_CESM1HTR.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#%%


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
    
    
    
    
    
#%% Plot lag correlation with 1-50 meter average 

im   = 1
for im in range(12):
    
    xtks   = np.arange(0,37,3)
    ylim   = [0,500]
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
        
        mu          = np.roll(np.array([hclim.mean(-1),]*3).flatten(),-im)
        sigma       = np.roll(np.array([hclim.std(-1),]*3).flatten(),-im)
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
        
    savename = "%sTEMP_SALT_DepthACF_mon%02i_EnsAvg_CESM1HTR.png" % (figpath,im+1)
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
        
        
    







