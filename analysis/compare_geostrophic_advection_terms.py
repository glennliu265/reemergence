#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied from compute_geostrophic_advection_term

Created on Wed Aug 14 13:23:31 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import scipy as sp

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

import cartopy.crs as ccrs

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

#%% Plotting Information

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28
proj                        = ccrs.PlateCarree()

#%% Load Land/Ice Mask

# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply  = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

# Load Gulf Stream
ds_gs           = dl.load_gs()
ds_gs           = ds_gs.sel(lon=slice(-90,-50))
ds_gs2          = dl.load_gs(load_u2=True)

# Load Mean SSH
ds_ssh          = dl.load_bsf(ensavg=False,ssh=True).mean('ens').SSH /100

# Load mean SST/SSS
ds_sss          = dl.load_monmean('SSS')
ds_sst          = dl.load_monmean('SST')

# Load Centered Differences
nc_gradT = "CESM1_HTR_FULL_Monthly_gradT2_SST.nc"
nc_gradS = "CESM1_HTR_FULL_Monthly_gradT2_SSS.nc"
ds_gradT = xr.open_dataset(rawpath + nc_gradT).load()
ds_gradS = xr.open_dataset(rawpath + nc_gradS).load()

ds_grad_bar = [ds_gradT,ds_gradS]

#%% Load SST/SSS for plotting (and anomalize)

st = time.time()
ds_sssprime = xr.open_dataset(rawpath + "CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc").SSS.load()
ds_sstprime = xr.open_dataset(rawpath + "CESM1LE_SST_NAtl_19200101_20050101_bilinear.nc").SST.load()

ds_proc = [ds_sssprime,ds_sstprime]
ds_anom = [proc.xrdeseason(ds) for ds in ds_proc]
ds_anom = [ds - ds.mean('ensemble') for ds in ds_anom]

ds_sssprime,ds_sstprime = ds_anom
print("Loaded and processed SST/SSS anomalies in %.2fs" % (time.time()-st))

# ================
#%% Load the files
# ================

# Load the geostrophic currents
st               = time.time()
nc_ugeo          = "CESM1LE_ugeo_NAtl_19200101_20050101_bilinear.nc"
path_ugeo        = rawpath
ds_ugeo          = xr.open_dataset(path_ugeo + nc_ugeo).load()
print("Loaded ugeo in %.2fs" % (time.time()-st))

# Compute monthly means
ugeo_monmean     = ds_ugeo.groupby('time.month').mean('time')#.mean('ens') # [lat x lon x month]
ugeo_mod_monmean = (ugeo_monmean.ug**2 + ugeo_monmean.vg**2)**0.5

# Fix Ensemble dimension
ugeo_monmean     = ugeo_monmean.rename(dict(ens='ensemble'))






#%% Compute the amplitude

ugeo_mod         = (ds_ugeo.ug ** 2 + ds_ugeo.vg ** 2)**0.5

# ==================================================
#%% Try to Compute the )anomalous) geostrophic advection Terms
# ==================================================

# Compute anomalous geostrophic advection
st        = time.time()
ugeoprime = proc.xrdeseason(ds_ugeo)
print("Computed anomalous advection in %.2fs" % (time.time()-st))

#%% Read out the files

# Get Anomalous Geostrophic Advection
ug                  = ugeoprime.ug#.data # (lat, lon, ens, time)
vg                  = ugeoprime.vg#.data # (lat, lon, ens, time)

# Get Mean Gradients
def preproc_grads(ds):
    if 'ensemble' in list(ds.dims):
        ds = ds.rename(dict(ensemble='ens'))
    ds = ds.transpose('lat','lon','ens','month')
    return ds
ds_gradT = preproc_grads(ds_gradT)
ds_gradS = preproc_grads(ds_gradS)

# Apply monthly gradients to each month
st          = time.time()
ugprime_dTx = ug.groupby('time.month') * ds_gradT.dTdx2
vgprime_dTy = vg.groupby('time.month') * ds_gradT.dTdy2

ugprime_dSx = ug.groupby('time.month') * ds_gradS.dTdx2
vgprime_dSy = vg.groupby('time.month') * ds_gradS.dTdy2
print("Computed in %.2f" % (time.time()-st))

# ++++++++++++++++++++++++++++++++++++++++++++++++
#%% Compute (and save) terms, and monthly variance
# ++++++++++++++++++++++++++++++++++++++++++++++++



ugeoprime_gradTbar        = -1 * (ugprime_dTx + vgprime_dTy)
ugeoprime_gradSbar        = -1 * (ugprime_dSx + vgprime_dSy)

ugeoprime_gradTbar_monvar = ugeoprime_gradTbar.groupby('time.month').var('time').rename("SST")
ugeoprime_gradSbar_monvar = ugeoprime_gradSbar.groupby('time.month').var('time').rename("SSS")

geoterm_T_savg            = proc.calc_savg(ugeoprime_gradTbar_monvar.rename(dict(month='mon')),ds=True)
geoterm_S_savg            = proc.calc_savg(ugeoprime_gradSbar_monvar.rename(dict(month='mon')),ds=True)

ugeo_grad_monvar          = xr.merge([ugeoprime_gradTbar_monvar,ugeoprime_gradSbar_monvar,mask.rename('mask')])

savename                  = "%sugeoprime_gradT_gradS_NATL_Monvar.nc" % rawpath
edict                     = proc.make_encoding_dict(ugeo_grad_monvar)
ugeo_grad_monvar.to_netcdf(savename,encoding=edict)



#%% Also Save the full term


ugeoprime_gradbar  = xr.merge([ugeoprime_gradTbar.rename("SST"),ugeoprime_gradSbar.rename("SSS")])
savename           = "%sugeoprime_gradT_gradS_NATL.nc" % rawpath
edict              = proc.make_encoding_dict(ugeoprime_gradbar)
ugeoprime_gradbar.to_netcdf(savename,encoding=edict)

# ==========================================================
#%% Try to compute mean geostrophic adv. of anomalous uprime
# ==========================================================

"""
    Mean Advection of the anomalous temperature/salinity gradient...
    Term = u_bar_geo * \grad T_prime
"""

# Load anomalous gradients (~41 sec)
st         = time.time()
gradSprime = xr.open_dataset(rawpath + "CESM1_HTR_FULL_Monthly_gradT_SSSprime.nc").load()
gradTprime = xr.open_dataset(rawpath + "CESM1_HTR_FULL_Monthly_gradT_SSTprime.nc").load()
print("Loaded Anomalous Gradients in %.2fs" % (time.time()-st))


ds_grad_prime = [gradTprime,gradSprime,]


#%% Compute the terms

# gradTprime (x)

# Compute Temperature 
st                  = time.time()
ug_bar,vg_bar       = ugeo_monmean.ug,ugeo_monmean.vg
dTdx, dTdy          = gradTprime.dx, gradTprime.dy
ugeobar_gradTprime     = ug_bar * dTdx.groupby('time.month') + vg_bar * dTdy .groupby('time.month')
print("Computed Temperature Values in %.2fs" % (time.time()-st))

# Compute Salinity
st                  = time.time()
dSdx,dSdy           = gradSprime.dx, gradSprime.dy
ugeobar_gradSprime     = ug_bar * dSdx.groupby('time.month') + vg_bar * dSdy .groupby('time.month')
print("Computed Salinity Values in %.2fs" % (time.time()-st))

# Multiply by -1
ugeobar_gradTprime = -1 * ugeobar_gradTprime.rename(dict(ensemble='ens'))
ugeobar_gradSprime = -1 * ugeobar_gradSprime.rename(dict(ensemble='ens'))

#%% Compute and save the monthly variance

ugeobar_gradTprime_monvar   = ugeobar_gradTprime.groupby('time.month').var('time').rename("SST")
ugeobar_gradSprime_monvar   = ugeobar_gradSprime.groupby('time.month').var('time').rename("SSS")

ugeo_gradprime_monvar       = xr.merge([ugeobar_gradTprime_monvar,ugeobar_gradSprime_monvar])

savename                    = "%sugeobar_gradTprime_gradSprime_NATL_Monvar.nc" % rawpath
edict                       = proc.make_encoding_dict(ugeo_gradprime_monvar)
ugeo_gradprime_monvar.to_netcdf(savename,encoding=edict)

#%% Save the full term

ugeobar_gradprime = xr.merge([ugeobar_gradTprime.rename('SST'),ugeobar_gradSprime.rename('SSS')])
savename                    = "%sugeobar_gradTprime_gradSprime_NATL.nc" % rawpath
edict                       = proc.make_encoding_dict(ugeobar_gradprime)
ugeobar_gradprime.to_netcdf(savename,encoding=edict)


# ======================================================
#%% Compute the full terms (u dot nable (Tbar + Tprime))
# ======================================================

debug           = True
recalc_fullgrad = True
save_fullgrad   = True
vnames = ["SST","SSS"]

# Needed Terms
# ds_ugeo (full ugeo_terms), (lat, lon, ens, time), ug, vg
# ds_grad_bar   = [ds_gradT,ds_gradS],  (ensemble, month, lat, lon), dTdx2,dTdy2
# ds_grad_prime = [gradTprime,gradSprime,], (lat: 96, lon: 89, time: 1032, ensemble: 42), dx,dy

# First, recalculate or load the full gradient
if recalc_fullgrad:
    rename_dict     = dict(dTdx2="dx",dTdy2="dy")
    ds_grad_bar     = [ds.rename(rename_dict) for ds in ds_grad_bar]
    
    
    # Groupby and sum by month
    ds_grad_full = [ds_grad_prime[ii].groupby('time.month') + ds_grad_bar[ii] for ii in range(2)]
    
    # Sanity Check at one point (add up manually and check...)
    if debug:
        lonf = -20
        latf = 55
        ii   = 1
        ie   = 33
        it   = 25
        
        test_prime   = ds_grad_prime[ii].isel(ensemble=ie,time=it).sel(lon=lonf,lat=latf,method='nearest')
        im           = test_prime.time.dt.month.item() - 1
        test_bar     = ds_grad_bar[ii].isel(ensemble=ie,month=im).sel(lon=lonf,lat=latf,method='nearest')
        
        test_add     = test_prime + test_bar
        
        test_combine = ds_grad_full[ii].isel(ensemble=ie,time=it).sel(lon=lonf,lat=latf,method='nearest')
        
        chk          = test_combine.dx.item() == test_add.dx.item() 
        print("%f + %f = %f, (compare %f, %s)" % (test_prime.dx.item(),test_bar.dx.item(),test_add.dx.item(),
                                              test_combine.dx.item(),chk))
        if chk is False:
            print("Warning, gradient combination has failed the check...")
        
        
    if save_fullgrad:
        
        
        for vv in range(2):
            vname = vnames[vv]
            savename = rawpath + "CESM1_HTR_FULL_Monthly_gradT_FULL_%s.nc" % vname
            save_ds  = ds_grad_full[vv]
            edict    = proc.make_encoding_dict(save_ds)
            save_ds.to_netcdf(savename,encoding=edict)
            print("Saved output to %s" % savename)
else: # Or just load it
    ds_grad_full = []
    for vv in range(2):
        st    = time.time()
        vname = vnames[vv]
        savename = rawpath + "CESM1_HTR_FULL_Monthly_gradT_FULL_%s.nc" % vname
        ds = xr.open_dataset(savename).load()
        ds_grad_full.append(ds)
        print("Loaded fullgrad in %.2fs" % (time.time()-st))


    

#%% Now compute Ugeo for S and T

ds_grad_full = [proc.format_ds_dims(ds) for ds in ds_grad_full]
ds_ugeo_in   = proc.format_ds_dims(ds_ugeo) # ds_ugeo has ens 0...41, renumber it



for vv in range(2):
    vname      = vnames[vv]
    ds_grad_in = ds_grad_full[vv]
    
    u_transport = ds_ugeo_in.ug * ds_grad_in.dx
    v_transport = ds_ugeo_in.vg * ds_grad_in.dy
    
    total_transport = xr.merge([u_transport.rename('UET'),v_transport.rename('VNT')])
    savename        = rawpath + "CESM1_HTR_FULL_Ugeo_%s_Transport_Full.nc" % (vname)
    edict    = proc.make_encoding_dict(total_transport)
    total_transport.to_netcdf(savename,encoding=edict)
    print("Saved output to %s" % savename)

#ds_ugeo_total = ds_ugeo[0]

        
#%% End calculation of full ugeo ==============================
        
    
    


    
    



# <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0>s <0> <0> <0>  <0> <0> <0> <0>  <0> <0>
#%% Do a sanity check for ugeo_bar * grad_Tprime (Mean Advection of Anomalous Gradient)
# <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0>  <0>

t       = 6
e       = 0
qint    = 2
scale   = 5
dtmon   = 3600*24*30
vname   = "SSS"

for t in range(100):
    
    if vname == "SST":
        
        interm = ugeobar_gradTprime
        ingrad = gradTprime
        inanom = ds_sstprime
        cblab  = r"$\overline{u_{geo}'} \cdot \nabla T'$" + r" [$ \frac{\degree C}{month}$]"
        vlims  = [-1,1]
        cints  = np.arange(-2.8,3.1,0.1)
        
    else:
        
        interm = ugeobar_gradSprime
        ingrad = gradSprime
        inanom = ds_sssprime
        cblab  = r"$\overline{u_{geo}'} \cdot \nabla S'$" + r" [$ \frac{psu}{month}$]"
        vlims  = [-0.1,0.1]
        cints  = np.arange(-.20,.22,0.02)
        
        
    # Initialize
    fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,14.5),)
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)
    
    # Plot Forcing Term
    plotvar     = interm.isel(time=t,ens=e) * dtmon
    pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                cmap='cmo.balance',vmin=vlims[0],vmax=vlims[1],
                                transform=proj,zorder=-1)
    cb          = viz.hcbar(pcm,ax=ax,fraction=0.035)
    cb.set_label(cblab,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
    # Determine the month
    timestep = proc.noleap_tostr(plotvar.time)
    im = plotvar.time.dt.month.item() - 1
    
    
    # Plot mean Geostrophic Current
    plotu   = ug_bar.isel(month=im,ensemble=e) * mask #ds_ugeo.ug.mean('time').mean('ens') * mask
    plotv   = vg_bar.isel(month=im,ensemble=e) * mask#ds_ugeo.vg.mean('time').mean('ens') * mask
    lon     = plotu.lon.data
    lat     = plotu.lat.data
    qv      = ax.quiver(lon[::qint],lat[::qint],
                        plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                        transform=proj,scale=scale,color='navy')
    qk      = ax.quiverkey(qv,.9,.85,.2,r"0.2 $\frac{m}{s}$",fontproperties=dict(size=fsz_axis))
    
    # Plot the Anomalous Gradients
    plotvar = inanom.isel(ensemble=e,time=t) * mask
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cints,colors='k',linewidths=2.5)
    ax.clabel(cl,fontsize=fsz_tick)
    
    # Plot Gulf Stream Position
    ax.plot(ds_gs2.lon.isel(mon=im),ds_gs2.lat.isel(mon=im),transform=proj,lw=2.5,c='red',ls='dashdot')
    
    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
    ax.set_title("Anomalous %s and Mean Geostrophic Currents\nt=%s, Ens=%02i" % (vname,timestep,e+1),fontsize=fsz_title)
    
    savename = "%sCESM1_ugeobar_%sprime_Term_ens%02i_time%03i.png" % (figpath,vname,e+1,t)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    
#%%

#%% Compute monthly variances of each term





#%%









#%% ===========================================================================
#%%

#%% Examine the area average contribution of this term and plot monhtly mean

nens            = 42
bbox_sel        = [-40,-30,40,50] # NAC
ugeo_term_reg   = proc.sel_region_xr(ugeo_grad_monvar,bbox_sel).mean('lon').mean('lat')
fig,axs         = viz.init_monplot(1,2,figsize=(12.5,4.5))#plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

for ii in range(2):
    ax = axs[ii]
    if ii == 0:
        plotvar = ugeo_term_reg.SST
        vname   = "SST"
        vunit   = "$\degree C$"
    else:
        plotvar = ugeo_term_reg.SSS
        vname   = "SSS"
        vunit   = "psu"
        
        
    for e in range(nens):
        ax.plot(mons3,plotvar.isel(ens=e) * dtmon**2,alpha=0.5)
    ax.plot(mons3,plotvar.mean('ens') * dtmon**2,color="k")
    ax.set_title(r"$u'_{geo} \nabla \overline{%s} $ " % vname)
    ax.set_ylabel("Interannual Variability (%s per mon)" % vunit)

#%% Plot Mean Variance of the geostrophic forcing term

vnames          = ["SST","SSS"]
vunits          = ["\degree C","psu"]
vmaxes_var      = [0.5,0.01]
vcmaps          = ["cmo.thermal",'cmo.haline']

qint            = 2

fig,axs,_        = viz.init_orthomap(1,2,bboxplot,figsize=(24,14.5),)

for ax in axs:
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)
    
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='red',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
    # Plot the mean Geostrophic Currents
    # Plot the anomalous Ekman Advection
    plotu   = ds_ugeo.ug.mean('time').mean('ens') * mask
    plotv   = ds_ugeo.vg.mean('time').mean('ens') * mask
    lon     = plotu.lon.data
    lat     = plotu.lat.data
    qv      = ax.quiver(lon[::qint],lat[::qint],
                        plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                        transform=proj,scale=scale,color='cyan')
    
for vv in range(2):
    
    ax      = axs[vv]
    vname   = vnames[vv]
    vunit   = vunits[vv]
    plotvar = (ugeo_grad_monvar[vname] * dtmon**2).mean('month').mean('ens') * mask
    
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmax=vmaxes_var[vv],zorder=-2,cmap=vcmaps[vv])
    
    # Plot Contours
    if vname == "SST":
        plotvar = ds_sst.SST.isel(ens=e).mean('mon') # Mean Gradient
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='red',levels=cints_sst)
        ax.clabel(cl,fontsize=fsz_tick)
    else:
        plotvar = ds_sss.SSS.isel(ens=e).mean('mon') # Mean Gradient
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='violet',levels=cints_sss)
        ax.clabel(cl,fontsize=fsz_tick)
        
    
    
    cb      = viz.hcbar(pcm,ax=ax)
    ax.set_title(r"$u_{geo}' \cdot \nabla \overline{%s}$" % (vname),fontsize=fsz_title)
    cb.set_label(r"Forcing ($u_{geo}' \cdot \nabla \overline{%s}$) [$%s^2$  per month]" % (vname,vunit),fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)

    
savename = "%sCESM1_HTR_EnsAvg_Ugeoprime_ForcingTermVariance.png" % (figpath,)
plt.savefig(savename,dpi=200,bbox_inches='tight')



#%%
#%% Plot Variance




invar   = ugeo_grad_monvar.SST * dtmon**2


plot_sid = []

fig,axs,_        = viz.init_orthomap(1,4,bboxplot,figsize=(24,14.5),)

for ax in axs.flatten():
    
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='red',ls='dashdot')
    
    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
for sid in range(4):
    
    
    ax      = axs.flatten()[sid]
    plotvar = invar.isel(ens=0,month=0)#geoterm_T_savg.isel(season=sid).mean('ens')
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmax=1,zorder=-2)
    
    
    cb      = viz.hcbar(pcm,ax=ax)
    cb.set_label(r"Forcing ($u_{geo} \nabla \overline{T}$) [$\degree$C  per month]",fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
    

#%% Test Removal of Ugeo Term from Salinty and Temperature

#%% Load Salinity and temperature

# (Copied from compute coherence SST/SSS)
st     = time.time()
nc_ssti = "CESM1LE_SST_NAtl_19200101_20050101_bilinear.nc"
nc_sssi = "CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc"
ds_ssti = xr.open_dataset(path_ugeo + nc_ssti).SST.load()
ds_sssi = xr.open_dataset(path_ugeo + nc_sssi).SSS.load()
print("Loaded SST and SSS in %.2fs" % (time.time()-st))

def preproc_var(ds):
    ds  = ds.rename(dict(ensemble='ens'))
    dsa = proc.xrdeseason(ds)
    dsa = dsa - dsa.mean('ens')
    return dsa

invars      = [ds_ssti,ds_sssi]
invars_anom = [preproc_var(ds) for ds in invars]

sst,sss     = invars_anom

#%% Remove the term from salinity and temperature

sst_nogeo = sst - (ugeoprime_gradTbar * dtmon) # Flip sign for negative advection
sss_nogeo = sss - (ugeoprime_gradSbar * dtmon) # ibid 


#%%

# Select data for a pt
lonf      = -30
latf      = 48
ds_all_in = [sst,sst_nogeo,sss,sss_nogeo]
dspt      = [proc.selpt_ds(ds,lonf,latf).data for ds in ds_all_in]

expnames  = ["SST","SST (no $u_{geo}$)","SSS","SSS (no $u_{geo}$)"]
expcolors = ["firebrick","hotpink","navy","cornflowerblue"]

for ii in range(4):
    
    if np.any(np.isnan(dspt[ii])):
        idens,idtime = np.where(np.isnan(dspt[ii].data))
        #idcomb       = np.where(np.isnan(arr_pt_flatten[ii]))[0][0]
        print("NaN Detected in arr %02i (ens=%02i, t=%s)" % (ii,idens+1,idtime))
        dspt[ii][idens[0],idtime[0]] = 0.

#%% Compute some metrics

tsms = []
for ii in range(4):
    
    ints = dspt[ii]
    ints = [ints[ee,:] for ee in range(42)]
    tsm  = scm.compute_sm_metrics(ints)
    tsms.append(tsm)

#%% Check Persistence

kmonth  = 1
lags    = np.arange(37)
xtks    = lags[::3]

fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
ax,_    = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")

for ii in range(4):
    
    acfs = np.array(tsms[ii]['acfs'][kmonth])
    
    for e in range(42):
        ax.plot(lags,acfs[e,:],alpha=0.05,c=expcolors[ii])
    
    ax.plot(lags,acfs.mean(0),alpha=1,c=expcolors[ii],label=expnames[ii])
        
ax.legend()

#%% Examine the change in variance

fig,axs = viz.init_monplot(1,2,figsize=(12,4.5))

for ii in range(4):
    
    if ii < 2:
        ax = axs[0]
    else:
        ax = axs[1]
    
    # Get the Monthly Variance
    monvar = np.array(tsms[ii]['monvars']) # [Ens x Mon]
    
    # Plot it
    mu = monvar.mean(0)
    ax.plot(mons3,mu,label=expnames[ii],c=expcolors[ii])
    
for ax in axs:
    ax.legend()
    


# +++++++++++++++++++++++++++
#%% Compute and Check Spectra
# +++++++++++++++++++++++++++

nsmooth   = 2
pct       = 0.10
dtmon     = 3600*24*30

specdicts = []
for ii in range(4):
    in_ts    = dspt[ii]
    specdict = scm.quick_spectrum(in_ts,nsmooth,pct,dt=dtmon,return_dict=True,dim=0,make_arr=True)
    specdicts.append(specdict)

    

#%% Visualize the spectra


xlims       = [1/(86*12*dtmon),1/(2*dtmon)]
xtks_per    = np.array([50,25,10,5,2,1,0.5]) # in Years
xtks_freq   = 1/(xtks_per * 12 * dtmon)

fig,axs      = plt.subplots(2,1,constrained_layout=True,figsize=(12,10))


for ii in range(4):
    
    if ii < 2:
        ax = axs[0]
    else:
        ax = axs[1]
    
    specdict = specdicts[ii]
    
    # -----------------------------------------------------------------------------
    # # Plot each Ensemble
    # nens = specdict['specs'].shape[0]
    # for e in range(nens):
    #     plotspec = specdict['specs'][e,:]
    #     plotfreq = specdict['freqs'][e,:]
    #     ax.loglog(plotfreq,plotspec,alpha=0.25,label="",color=ecols[ii],ls=els[ii])
     
    # PLot Ens Avg
    plotspec = specdict['specs'].mean(0)
    plotfreq = specdict['freqs'].mean(0)
    ax.loglog(plotfreq,plotspec,alpha=1,label=expnames[ii] + ", [smooth=%i adj. bands]" % (nsmooth),color=expcolors[ii])
    
    # Plot AR(1) Null Hypothesis
    plotcc0 = specdict['CCs'][:,:,0].mean(0)
    plotcc1 = specdict['CCs'][:,:,1].mean(0)
    #ax.loglog(plotfreq,plotcc1,alpha=1,label="",color=ecols[ii],lw=0.75,ls='dashed')
    #ax.loglog(plotfreq,plotcc0,alpha=1,label="",color=ecols[ii],lw=0.75)

    
# # Draw some line

for ax in axs:
    ax.axvline([1/(6*dtmon)],label="Semiannual",color='lightgray',ls='dotted')
    ax.axvline([1/(12*dtmon)],label="Annual",color='gray',ls='dashed')
    ax.axvline([1/(10*12*dtmon)],label="Decadal",color='dimgray',ls='dashdot')
    ax.axvline([1/(50*12*dtmon)],label="50-yr",color='k',ls='solid')
    ax.legend(ncol=3)
    
    # Label Frequency Axis
    ax.set_xlabel("Frequency (Cycles/Sec)",fontsize=fsz_axis)
    ax.set_xlim(xlims)
    
    # Label y-axis
    ax.set_ylabel("Power ([m/s]$^2$/cps)",fontsize=fsz_axis)
    ax.tick_params(labelsize=fsz_tick)
    # # Plot reference Line
    
    #ax.loglog(plotfreq,1/plotfreq,c='red',lw=0.5,ls='dotted')
    
    ax2 = ax.twiny()
    ax2.set_xlim(xlims)
    ax2.set_xscale('log')
    ax2.set_xticks(xtks_freq,labels=xtks_per,fontsize=fsz_tick)
    ax2.set_xlabel("Period (Years)",fontsize=fsz_axis)
    #ax.set_title("Power Spectra for Geostrophic Currents @ %s" % loctitle)





# ---------------------------------------
#%% Case Study, since things seem funky...
# ---------------------------------------

"""
Terms Needed

 >>> Variables
- sst
- sss

 >>> Geostrophic Advection Term
- ugeoprime_gradTbar
- ugeoprime_gradSbar

 >>> Currents
 
 ug
 vg
 
 >>> Gradients
 
 ds_gradT.dTdx2
 ds_gradT.dTdy2
 
 ds_gradS.dTdx2
 ds_gradS.dTdy2

"""

e               = 0
lonf            = -30
latf            = 48

locfn,loctitle = proc.make_locstring(lonf,latf)

#%% Choose a particular Timepoint

zoom_time = True
itime     = 548 # Select A specific Instant

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12.5,8),sharex=True)

sstpt = proc.selpt_ds(sst.isel(ens=e),lonf,latf)
ssspt = proc.selpt_ds(sss.isel(ens=e),lonf,latf)
times = [proc.noleap_tostr(ssspt.time[ii]) for ii in range(1032)]
plott = np.arange(1032)
xtks  = plott[::(12*10)]

for ax in axs:
    ax.set_xticks(xtks,labels=np.array(times)[xtks],fontsize=fsz_tick)
    if zoom_time:
        ax.set_xlim([itime-120,itime+120])
    else:
        ax.set_xlim([xtks[0],plott[-1]])
    
    ax.grid(True,ls='dashed')
    ax.tick_params(labelsize=fsz_tick)

ax = axs[0]
ax.plot(plott,sstpt,c='magenta')
#sstpt.plot(ax=ax)
ax.axvline([itime],c='k')
ax.set_ylabel("SST ($\degree C$)",fontsize=fsz_axis)

ax = axs[1]
ax.plot(plott,ssspt,c='firebrick')
#ssspt.plot(ax=ax)

ax.axvline([itime],c='k')
ax.set_ylabel("SSS (psu)",fontsize=fsz_axis)

plt.suptitle("Anomaly Timeseries @ %s, Ens %02i" % (loctitle,e+1),fontsize=fsz_title)

savename = "%sCESM1_HTR_EnsAvg_Ugeoprime_Event_Timeseries_Ens%02i_t%3i_zoom%i.png" % (figpath,e+1,itime,zoom_time)
plt.savefig(savename,dpi=200,bbox_inches='tight')
#ax.set_xlim([350,390])

#%%


t               = itime
dtmon           = 3600*24*30
qint            = 1
scale           = 2.5
cints_sst       = np.arange(250,310,2)
cints_sss       = np.arange(33,39,.3)
zoom            = None # [-60,-20,40,55]
# Initialize figure

if zoom is not None:
    bbin = zoom
    zoomflag=True
else:
    bbin = bboxplot
    zoomflag=False

fig,axs,_        = viz.init_orthomap(1,2,bbin,figsize=(24,10),)
    
for ax in axs:
    ax              = viz.add_coast_grid(ax,bbox=bbin,fill_color="lightgray",fontsize=24)
    
    # Plot Anomalous Advection
    plotu   = ugeoprime.ug.isel(time=t,ens=e) * mask
    plotv   = ugeoprime.vg.isel(time=t,ens=e) * mask
    lon     = plotu.lon.data
    lat     = plotu.lat.data
    qv      = ax.quiver(lon[::qint],lat[::qint],
                        plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                        transform=proj,scale=scale,color='blue')
    
    # Plot target point
    ax.plot(lonf,latf,transform=proj,marker="*",markersize=15,c='yellow',markeredgecolor='k')

for vv in range(2):
    ax = axs[vv]
    timestep = sstpt.time.isel(time=t)
    im       = sstpt.time.isel(time=t).month.item()
    
    
    if vv == 0: # SST
        
        # Mean Values
        vname       = "SST"
        vunit       = "$\degree C$"
        plotgrad    = ds_sst.SST.isel(ens=e,mon=im)
        cints_grad  = cints_sst
        gradcol     = 'magenta'
        
        # Anomaly
        plotanom    = sst.isel(ens=e,time=t)
        vlims_anom  = [-3,3]
        cmap_diff   = 'cmo.balance'
        
        # Forcing
        plotvar     = ugeoprime_gradTbar.isel(ens=e,time=t) * dtmon * mask
        cints_frc   = np.arange(-2,2.2,0.2)
        
        
    elif vv == 1: # SSS
        
        vname       = "SSS"
        vunit       = "$psu$"
        plotgrad    = ds_sss.SSS.isel(ens=e,mon=im)
        cints_grad  = cints_sss
        gradcol     = 'firebrick'
        
        plotanom    = sss.isel(ens=e,time=t)
        vlims_anom  = [-.5,0.5]
        cmap_diff   = 'cmo.delta'
        
        plotvar     = ugeoprime_gradSbar.isel(ens=e,time=t) * dtmon * mask
        cints_frc   = np.arange(-.5,.55,0.05)
    
        
    # Plot the Mean Contours
    cl = ax.contour(plotgrad.lon,plotgrad.lat,plotgrad * mask,transform=proj,colors=gradcol,levels=cints_grad)
    ax.clabel(cl,fontsize=fsz_tick)
    
    # Plot the Anomaly
    pcm = ax.pcolormesh(plotanom.lon,plotanom.lat,plotanom*mask,transform=proj,vmin=vlims_anom[0],vmax=vlims_anom[1],
                        cmap=cmap_diff,zorder=-2)
    
    
    # Add Colorbar
    cb = viz.hcbar(pcm,ax=ax)
    cb.set_label("%s Anomaly (%s)" % (vname,vunit),fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
    # Add Forcing Contours
    clf = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,colors="cyan",
                     levels=cints_frc,linewidths=.75)
    ax.clabel(clf,fontsize=fsz_tick)
    
    ax.set_title(r"$u_{geo}' \cdot \nabla \overline{%s}$ [%s per month]" % (vname[0],vunit),fontsize=fsz_axis)
    
    
    # pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
    #                  )
    # cb = viz.hcbar(pcm,ax=ax)
    # cb.set_label("%s Geostrophic Advection Forcing (%s)" % (vname,vunit),fontsize=fsz_axis)
    # cb.ax.tick_params(labelsize=fsz_tick)
    
plt.suptitle("Anomalous Geostrophic Advection @ %s, Ens. Member %02i" % (proc.noleap_tostr(timestep),e+1),fontsize=fsz_title,y=1.05)
savename = "%sCESM1_HTR_EnsAvg_Ugeoprime_Event_Ens%02i_t%3i_zoom%i.png" % (figpath,e+1,t,zoomflag)
plt.savefig(savename,dpi=200,bbox_inches='tight')

# # Plot the forcing
# plotvar = (ugprime_dTx + vgprime_dTy).isel(time=t,ens=e) * dtmon * -1
# pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#                         cmap='cmo.balance',vmin=-2.5,vmax=2.5)
# cb      = viz.hcbar(pcm,ax=ax)
# cb.set_label(r"Forcing ($u_{geo} \nabla \overline{T}$) [$\degree$C  per month]",fontsize=fsz_axis)

# # Plot the contours
# timestep = plotvar.time
# im       = plotvar.time.month.item() # Get the Month
# cints_grad = np.arange(-30,33,3)
# if plotmode == "u":
#     plotvar = ds_gradT.dTdx2.isel(ens=e,month=im) * dtmon
#     cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='firebrick',levels=cints_grad)
#     ax.clabel(cl,fontsize=fsz_tick)
# elif plotmode == "v":
#     plotvar = ds_gradT.dTdy2.isel(ens=e,month=im) * dtmon
#     cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='firebrick',levels=cints_grad)
#     ax.clabel(cl,fontsize=fsz_tick)
# else:
    
#     plotvar = ds_sst.SST.isel(ens=e,mon=im) # Mean Gradient
#     cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='firebrick',levels=cints_sst)
#     ax.clabel(cl,fontsize=fsz_tick)






#%%





#%%


# Get Gradients
dTdx = ds_gradT.dTdx2 # (lat, lon, ens, month)
dTdy = ds_gradT.dTdy2 # (lat, lon, ens, month)




#ugeoprime = ugeoprime





ugeoprime_gradTbar = 




#%% ---------------------------------------------------------------------------

# #%% Load the Ekman currents

# st          = time.time()
# nc_uek      = "CESM1_HTR_FULL_Uek_NAO_nomasklag1_nroll0_NAtl.nc"
# path_uek    = rawpath
# ds_uek      = xr.open_dataset(path_uek + nc_uek)#.load()
# print("Loaded uek in %.2fs" % (time.time()-st))

