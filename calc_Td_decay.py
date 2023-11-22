#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute theoretical decay timescale of anomalies below the mixed layer
Works with anomalies retrieved with get_Td.py
Copied section from [est_damping_fit]

Created on Thu Oct 12 19:18:33 2023

@author: gliu
"""

import xarray as xr
import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmocean as cmo

from tqdm import tqdm
import scipy as sp
from scipy import signal
#%% Modules

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% User Edits

datpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/"
outpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
figpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20231118/"
proc.makedir(figpath)

e        = 1

# Plotting Parameters
bbox   = [-80,0,20,65]
mons3  = proc.get_monstr(nletters=3)

#%% Load the files

#CESM1_HTR_Sd_ens02.nc
Sd = xr.open_dataset("%sCESM1_HTR_Sd_ens%02i.nc" % (datpath,e+1))['Sd']
Td = xr.open_dataset("%sCESM1_HTR_Td_ens%02i.nc" % (datpath,e+1))['Td']
ld = np.load("%sCESM1_HTR_hmax_ens%02i.npz" % (datpath,e+1),allow_pickle=True)
print(ld.files)

lon = Sd.lon.values
lat = Sd.lat.values
times = Sd.time.values

Sd = Sd.values
Td = Td.values

#%% Remove seasonal cycle and simple linear detrend

rawvars = [Td,Sd]
scycles = []
manoms  = []
for ts in rawvars:
    scycle,tsa=proc.calc_clim(ts,0,returnts=1)
    tsa = tsa - scycle[None,:,:,:]
    tsa = proc.detrend_dim(tsa,0)
    tsa = tsa[0]
    scycles.append(scycle)
    manoms.append(tsa)

#%% Quick Plot of seasonal averages

Td     = manoms[0]
im     = 5

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
pcm    = ax.pcolormesh(lon,lat,Td[:,im,:,:].var(0))
ax.coastlines()
fig.colorbar(pcm)
ax.set_title("Variance of Td in month %i" % (im+1))

#%% Make a mask

landmask = np.sum(Td.copy(),(0,1))
landmask[~np.isnan(landmask)] = 1

#%% Clean dataset for ACF calculations

nyrs,_,nlat,nlon = manoms[0].shape
ntime = len(times)


# Clean Data
nandicts  = []
cleandata = []
for v in range(2):
    invar = manoms[v]
    
    # Remove NaN Pts
    invar   = invar.reshape(ntime,nlat*nlon)
    nandict = proc.find_nan(invar,0,return_dict=True) 
    okdata  = nandict['cleaned_data']
    oksize  = okdata.shape[1]
    cleandata.append(okdata.reshape(nyrs,12,oksize).transpose(1,0,2)) # [mon x yr x npts]
    nandicts.append(nandict)

#%% Compute ACF

basemonths = np.arange(1,13,1)
detrendopt = 1
lags       = np.arange(0,61)
nlags      = len(lags)

# Preallocate
corr_lags  = np.zeros((2,nlags,12,oksize))


# Basemonth Loop
for v in range(2):
    
    for bm in range(12):
        
        basemonth          = basemonths[bm]
        lagcorr            = proc.calc_lagcovar_nd(cleandata[v],cleandata[v],lags,basemonth,detrendopt) # first variable is lagged
        
        corr_lags[v,:,bm,:]  = lagcorr.copy()



# Covariance
full_corrlags  = np.zeros((2,nlags,12,nlat*nlon))
for i in range(2):
    full_corrlags[i,:,:,nandicts[i]['ok_indices']] = corr_lags[i,:,:,:].transpose(2,0,1)
    

full_corrlags = full_corrlags.reshape((2,nlags,12,nlat,nlon))

#%% Test PLots


im     = 5
ilag   = 33
plotvar = full_corrlags[0,ilag,im,:,:]

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
pcm    = ax.pcolormesh(lon,lat,plotvar)
ax.coastlines()
fig.colorbar(pcm)
ax.set_title("Variance of Td in month %i" % (im+1))


#%%

klon,klat = proc.find_latlon(-30,50,lon,lat)

fig,ax    = plt.subplots(1,1)
ax.plot(lags,full_corrlags[1,:,im,klat,klon])


#%% Fit an expnential FUNC TO ACF
import scipy as sp

recalculate=False
lagmaxes = [7,13,61]
savename = "%sCESM1_LENS_Td_Sd_lbd_exponential_fit.nc" % datpath
exists   = proc.checkfile(savename)
nmon     = 12
varnames = ["Td","Sd"]

if (not exists) or recalculate:

    #% Fit The exponential (a dumb loop...) ----------------------------------
    expf3     = lambda t,b: np.exp(b*t)         # No c and A
    # 
    lm        = len(lagmaxes)
    
    lbd_fit   = np.zeros((2,lm,nmon,nlat,nlon)) * np.nan # [variable,]
    funcin    = expf3
    problem_y = []
    for v in range(2):
        for a in tqdm(range(nlat)):
            for o in range(nlon):
                
                acpt = full_corrlags[v,:,:,a,o]#ac_all[v][e,:,:,a,o] # Lag x Month
                
                # Skip Land Points
                if np.all(np.isnan(acpt)):
                    continue
                
                for im in range(nmon):
                    for l in range(lm):
                        lagmax = lagmaxes[l]
                        x = lags[:lagmax]
                        y = acpt[:lagmax,im]
                        
                        try:
                            popt, pcov = sp.optimize.curve_fit(funcin, x[:lagmax], y[:lagmax])
                        except:
                            print("Issue with ilat %i ilon %i"% (a,o))
                            problem_y.append(y)
                            continue
                        lbd_fit[v,l,im,a,o] = popt[0]

    #% Save the exponential Fit
    
    coords_dict = {
        'vars'   : list(varnames),
        'lag_max': lagmaxes,
        'mon'    : np.arange(1,13),
        'lat'    : ds_all[0].lat.values,
        'lon'    : ds_all[0].lon.values
        }
    
    da       = xr.DataArray(lbd_fit,coords=coords_dict,dims=coords_dict,name="lbd")
    
    
    da.to_netcdf(savename,encoding={'lbd': {'zlib': True}})
else:
    da = xr.open_dataset(savename)
    lbd_fit = da.lbd.values
    
# #%% Load data from abive

# savename = "%sCESM1_LENS_Td_Sd_lbd_exponential_fit.nc" % datpath
# exists   = proc.checkfile(savename)
# da       = xr.open_dataset(savename)



damping = da.lbd.values # [variable, lagmax, month x lat x lon]
#%% Plot some Td ' timescales (Ann Avg)

ivar         = varnames.index('Td')
ilag         = lagmaxes.index(13)
Tdexp_scycle = damping[ivar,ilag,:,:,:]

plotcontours = [3,6,12,18,24,36,48,60]
vlms         = [0,75]


# Plot Annual Average
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4),
                       constrained_layout=True)

ax      = viz.add_coast_grid(ax,bbox=bbox,fill_color='gray')
plotvar = 1/np.abs(Tdexp_scycle.mean(0))
plotvar[plotvar<1] = np.nan

# Plot Pcolormesh
pcm     = ax.pcolormesh(lon,lat,plotvar,cmap='cmo.deep_r',vmin=vlms[0],vmax=vlms[-1])
cl      = ax.contour(lon,lat,plotvar,levels=plotcontours,colors="w",linewidths=0.5)
ax.clabel(cl,fontsize=10)
cb = fig.colorbar(pcm,ax=ax,pad=0.01)
cb.set_label("e-folding Timescale (Months)")
ax.set_title("$T_d'$ Damping Timescale (Ann. Avg, %i-Lag Fit)" % (lagmaxes[ilag]-1))

savename = "%s%s_Damping_Timescale_%02lagfit_AnnAvg.png" % (figpath,varnames[ivar],lagmaxes[ilag],)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Same as Above, but for each month

for im in range(12):
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4),
                           constrained_layout=True)
    
    ax      = viz.add_coast_grid(ax,bbox=bbox,fill_color='gray')
    plotvar = 1/np.abs(Tdexp_scycle[im,:,:]) * landmask
    #plotvar[plotvar<1] = np.nan
    
    # Plot Pcolormesh
    pcm     = ax.pcolormesh(lon,lat,plotvar,cmap='cmo.deep_r',vmin=vlms[0],vmax=vlms[-1])
    cl      = ax.contour(lon,lat,plotvar,levels=plotcontours,colors="w",linewidths=0.5)
    ax.clabel(cl,fontsize=10)
    cb = fig.colorbar(pcm,ax=ax,pad=0.01)
    cb.set_label("e-folding Timescale (Months)")
    ax.set_title("$T_d'$ Damping Timescale (%s, %i-Lag Fit)" % (mons3[im],lagmaxes[ilag]-1))
    
    savename = "%s%s_Damping_Timescale_%02lagfit_mon%02i.png" % (figpath,varnames[ivar],lagmaxes[ilag],im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Save some timescales for the stochastic Model


# Convert for output
Tddamp =np.abs(Tdexp_scycle) * landmask[None,:,:] # [Mon Lat Lon]
Tddamp = Tddamp.transpose(2,1,0) # [Lon Lat Mon]

outpath_sminput = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
savename        = "%s%s_Damping_HTR_ens%02i_%02ilagfit.npz" % (outpath_sminput,varnames[ivar],e,lagmaxes[ilag])
np.savez(savename,**{'Tddamp':Tddamp,
                     'lon':lon,
                     'lat':lat},allow_pickle=True)

#plt.pcolormesh(test)

#%% Quickly check S ACF calculation at point;

plt.plot(scycles[1][:,klat,klon])


    
    
#%% Look at ACF for a month

expf3     = lambda t,b: np.exp(b*t)         # No c and A
mons3     = proc.get_monstr(nletters=3)
locfn,loctitle = proc.make_locstring(lon[klon],lat[klat])

v         = 1
klon,klat = proc.find_latlon(-30,50,lon,lat)
im        = 1


# Manually calculate ACF to Check,.. -----------
Spt        = manoms[v][:,:,klat,klon]
acf_manual = proc.calc_lagcovar(Spt.T,Spt.T,lags,im+1,detrendopt,debug=False)

    
plotacf = full_corrlags[v,:,im,klat,klon]





fig,ax = plt.subplots(1,1)
viz.add_ticks(ax)

for ilm in range(3):
    
    dampingtime = damping[v,ilm,im,klat,klon]
    modfit      = expf3(lags,dampingtime)
    ax.plot(lags,modfit,label="Fit %i Lags, t = %.2f" % (lagmaxes[ilm],1/np.abs(dampingtime)))
    

ax.plot(lags,plotacf,color="k",label="Td ACF")
ax.plot(lags,acf_manual,color="r",ls='dashed',label="Pointwise")
ax.set_xlabel("Lags (months)")
ax.set_ylabel("Correlation")
ax.legend()
ax.set_title("%s' Exponential Fit @ %s, Lag 0 = %s" % (varnames[v],loctitle,mons3[im]))

savename = "%sExpFit_Example_%s_ens%02i_mon%02i.png" % (figpath,locfn,e+1,im+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')



    
    
#%% Look at damping timescales


im      = 1
ilagmax = 0


for im in range(12):

    fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4),
                           constrained_layout=True)
    
    for v in range(2):
        ax=axs[v]
        ax = viz.add_coast_grid(ax,bbox)
        plotvar = np.abs(1/damping[v,ilagmax,im,:,:])
        pcm = ax.pcolormesh(lon,lat,plotvar* landmask,vmin=0,vmax=40,cmap='cmo.deep_r')
    
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
    cb.set_label("Damping Timescale (months)")
    plt.suptitle("Damping Timescale for Exponential Fit to %i lags\n %s, Ens %02i" % (lagmaxes[ilagmax]-1,mons3[im],e+1))
    savename = "%sDampingTimescale_ens%02i_mon%02i_lagmax%02i.png" % (figpath,e+1,im+1,lagmaxes[ilagmax])
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Visualize relative to the annual mean

im      = 1
ilagmax = 0
bbox   =[-80,0,20,65]

for im in range(12):
    
    fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4),
                           constrained_layout=True)
    
    for v in range(2):
        
        ax=axs[v]
        ax = viz.add_coast_grid(ax,bbox)
        
        ann_mean = np.abs(1/damping[v,ilagmax,im,:,:])
        
        plotvar  = np.abs(1/damping[v,ilagmax,:,:,:]).mean(0) - ann_mean
        pcm = ax.pcolormesh(lon,lat,plotvar* landmask,vmin=-30,vmax=30,cmap="cmo.balance")
    
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
    cb.set_label("Damping Timescale (months)")
    plt.suptitle("Damping Timescale Diff from Ann. Mean for Exponential Fit to %i lags\n %s, Ens %02i" % (lagmaxes[ilagmax]-1,mons3[im],e+1))
    savename = "%sDampingTimescale_diff_ens%02i_mon%02i_lagmax%02i.png" % (figpath,e+1,im+1,lagmaxes[ilagmax])
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Visualize the actual annual mean


    
fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4),
                       constrained_layout=True)

lvls = np.arange(0,43,3)
pcolor = True
for v in range(2):
    
    ax=axs[v]
    ax = viz.add_coast_grid(ax,bbox)
    
    
    plotvar  = np.abs(1/damping[v,ilagmax,im,:,:])#np.abs(1/damping[v,ilagmax,:,:,:]).mean(0) - ann_mean
    if pcolor:
        pcm = ax.pcolormesh(lon,lat,plotvar* landmask,vmin=lvls[0],vmax=lvls[-1],cmap='cmo.deep_r')
    else:
        pcm = ax.contourf(lon,lat,plotvar* landmask,levels=lvls)
    

cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
cb.set_label("Damping Timescale (months)")
plt.suptitle("Damping Timescale for Exponential Fit to %i lags\n Ann. Mean, Ens %02i" % (lagmaxes[ilagmax]-1,e+1))
savename = "%sDampingTimescale_diff_ens%02i_AnnMean_lagmax%02i.png" % (figpath,e+1,lagmaxes[ilagmax])
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Visualize the mixed layer depth where Td and Sd was taken from

bbox = [-80,0,20,65]

fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4),
                       constrained_layout=True)

for v in range(2):
    
    ax=axs[v]
    ax = viz.add_coast_grid(ax,bbox)
    
    if v == 0:
        plotvar = ld['z_t']
        title   = "Mixed-Layer Base (m)"
        clm     = [0,200]
        lvls    = np.arange(200,1500,300)
        cmap    = 'cmo.dense'
        clcolor = "w"
    elif v == 1:
        plotvar = ld['hmax_monid'] + 1
        title   = "Month of Max MLD"
        clm     = [1,12]
        cmap    = "twilight"
        lvls    = np.arange(1,13,1)
        clcolor = "k"
    
    ax.set_title(title)
    #plotvar  = np.abs(1/damping[v,ilagmax,im,:,:])#np.abs(1/damping[v,ilagmax,:,:,:]).mean(0) - ann_mean
    pcm = ax.pcolormesh(lon,lat,plotvar* landmask,vmin=clm[0],vmax=clm[1],cmap=cmap)
    cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05)
    
    cl = ax.contour(lon,lat,plotvar* landmask,colors=clcolor,levels=lvls,linewidths=0.55)
    ax.clabel(cl)

#cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
#cb.set_label("Damping Timescale (months)")
#plt.suptitle("Damping Timescale for Exponential Fit to %i lags\n Ann. Mean, Ens %02i" % (lagmaxes[ilagmax]-1,e+1))
savename = "%sMLD_Base_ens%02i.png" % (figpath,e+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')


