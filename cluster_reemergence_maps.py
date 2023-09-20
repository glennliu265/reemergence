#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Cluster Re-emergence Maps

Copied sections from [stochmod_point.py], [predict_amv/viz_LRP_predictor.py]


Psuedo-code

1. Load in ACF data calculated for the North Atlantic
2. Take Ensemble Average, left with [ens x mon x lag] maps
3. Can first try clustering for 1 month, then for the whole "map"
4. Perform k-means clustering (try different amounts of groups)
5. Examine characteristic clusters

Created on Wed Sep 20 10:25:54 2023

@author: gliu

"""

from amv import proc, viz
import amv.loaders as dl
import scm
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib as mpl

import cmocean as cmo
# %% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/"  # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)


# %% Set Paths

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20230922/"
proc.makedir(figpath)

debug = True


# ----------------------------------------------------------
# %% 0. Retrieve the autocorrelation for SSS and SST from CESM1
# ----------------------------------------------------------
datpath_ac = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
varnames = ["SST", "SSS"]
ac_colors = ["darkorange", "lightseagreen"]

# Read in datasets
ds_all = []
for v in range(2):
    ds = xr.open_dataset("%sHTR-FULL_%s_autocorrelation_thres0.nc" %
                         (datpath_ac, varnames[v]))  # [Thres x Ens x Lag x Mon x Lat x Lon]
    ds = ds.isel(thres=2).load()
    # ds  = ds.sel(lon=lonf-360,lat=latf,method='nearest').sel(thres="ALL").load()# [ens lag mon]
    ds_all.append(ds)

# ----------------------------------------------------------
# %% 1. Preprocess for clustering  take ensemble average
# ----------------------------------------------------------
# Take Ensemble Average
ds_ensavg = [ds.mean('ens') for ds in ds_all]  # [lag x mon x lat x lon]

# Remove NaN Points
nan_dicts = []
input_maps = []
n_notnans = []
for v in range(2):

    # Read out ACFs, combine dims
    vname = varnames[v]
    acfs = ds_ensavg[v][vname].values
    nlags, nmon, nlat, nlon = acfs.shape
    acfs = acfs.reshape(nlags*nmon, nlat*nlon)

    # Remove NaN Points and Save Indices for later
    nan_dict = proc.find_nan(acfs, 0, return_dict=True)
    nan_dicts.append(nan_dict)
    n_notnan = nan_dict['cleaned_data'].shape[1]
    input_maps.append(nan_dict['cleaned_data'].reshape(
        nlags, nmon, n_notnan))  # [lag,mon, space]
    n_notnans.append(n_notnan)


lon = ds_ensavg[0].lon.values
lat = ds_ensavg[0].lat.values
lags = ds_ensavg[0].lag.values
mons3 = proc.get_monstr(nletters=3)
# ----------------------------------------------------------
# %% Double check to see if things are ok
# ----------------------------------------------------------

if debug:
    v = 0
    lonf = -30
    latf = 52
    klon, klat = proc.find_latlon(lonf, latf, lon, lat)
    locfn, loctitle = proc.make_locstring(lonf, latf)
    id_notnan = nan_dicts[v]['ok_indices']
    test_maps = np.zeros((nlags, nmon, nlat*nlon))*np.nan
    test_maps[:, :, id_notnan] = input_maps[v]
    test_maps = test_maps.reshape(nlags, nmon, nlat, nlon)

    plotvar = test_maps[:, :, klat, klon]
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 8))
    pcm = ax.pcolormesh(mons3, lags, plotvar)
    fig.colorbar(pcm, ax=ax)
    ax.set_title("ACF Map for %s" % loctitle)
    ax.set_ylabel("Lag (Months)")
    ax.set_xlabel("Base Month (Lag 0)")


# ----------------------------------------------------------
# %% 2. Actually Do the Clustering now... (for Lag x Month)
# ----------------------------------------------------------

nclusts = 8

# Preallocate
cluster_labels = np.zeros((2, nlat*nlon)) * np.nan
cluster_centers = np.zeros((2, nclusts, nlags*nmon)) * np.nan
cluster_outputs = []
# Perform clustering for each variable
for v in range(2):

    # Get and prepare input
    inputs = input_maps[v]
    inputs = inputs.reshape(nlags*nmon, n_notnans[v])  # [lag*mon, space]
    # [space x lag*mon] (cluster over spatial dim), must be [samples x features]
    inputs = inputs.T

    # Perform clustering and retrieve output
    cluster_output = KMeans(n_clusters=nclusts, random_state=0).fit(inputs)
    ccenters = cluster_output.cluster_centers_  # [cluster x features]
    # [cluster x space].reshape(nmod*nsamples)
    clabels = cluster_output.labels_

    # Save
    id_notnan = nan_dicts[v]['ok_indices']
    cluster_outputs.append(cluster_output)
    cluster_labels[v, id_notnan] = clabels.copy()
    cluster_centers[v, :, :] = ccenters.copy()

cluster_labels = cluster_labels.reshape(2, nlat, nlon)
cluster_centers = cluster_centers.reshape(2, nclusts, nlags, nmon)

# ----------------------------------------------------------
# %%  Check the output
# ----------------------------------------------------------

vlims = [0, 1]


ytks = np.arange(0, 37, 3)
xtks = np.arange(0, 12, 1)
xtklbls = proc.get_monstr(nletters=1)
mpl.rcParams.update({'font.size': 18})
mpl.rc('xtick', labelsize=16)


# Clustering Output
for v in range(2):
    fig, axs = plt.subplots(1, nclusts, figsize=(18, 10), constrained_layout=True)
    
    for c in range(nclusts):
    
        # Plot data
        ax = axs[c]
        plotmap = cluster_centers[v, c, :, :]
        pcm = ax.pcolormesh(mons3, lags, plotmap,
                            vmin=vlims[0], vmax=vlims[-1], cmap='inferno')
    
        # Set Labels
        ax.set_title("Cluster %i" % (c+1))
        ax.set_yticks(ytks)
        ax.set_xticks(xtks, labels=xtklbls)
        ax.axhline([12], c="w", ls='dotted')
        ax.axhline([24], c="w", ls='dotted')
        ax.axhline([36], c="w", ls='dotted')
    
        if c == 0:
            ax.set_ylabel("Lag (months)")
    
    cb = fig.colorbar(pcm, ax=axs.flatten(), fraction=0.05, pad=0.01)
    cb.set_label("Correlation")
    savename = "%sACFClustering_%s_nclust%i_Cluster_Centers.png" % (
        figpath, varnames[v], nclusts)
    plt.savefig(savename, dpi=150, bbox_inches="tight")


# %% Plot cluster labels


bbox    = [-80, 0, 0, 65]

cmap_in = mpl.cm.get_cmap("PuOr",nclusts)

fig, axs = viz.init_fig(1, 2, figsize=(12, 6))

for v in range(2):
    ax = axs[v]
    ax      = viz.add_coast_grid(ax, bbox,fill_color='lightgray', line_color=None)
    pcm = ax.pcolormesh(lon, lat, cluster_labels[v, ...]+1,cmap=cmap_in)
    fig.colorbar(pcm,ax=ax,orientation='horizontal',pad=0.06,fraction=0.026)
    ax.set_title("%s Clusters" % (varnames[v]))
    
savename = "%sACFClustering_SSTandSSS_nclust%i_Cluster_Centers.png" % (
    figpath, nclusts)
plt.savefig(savename, dpi=150, bbox_inches="tight")

#%% for each cluster, examine the autocorrelation function

kmonth = 1

xtks = np.arange(0,37,3)

for v in range(2):
    vname    = varnames[v]
    clabels  = cluster_labels[v,:,:]
    acs_in   = ds_ensavg[v][vname].values # [lag,mon,lat,lon]
    
    
    # Look for label
    for lbl in range(nclusts):
        idlabel  = np.where(clabels == lbl)
        acslabel = acs_in[:,kmonth,idlabel[0],idlabel[1]] 
        
        title  = "ACF Spaghetti for %s, Cluster %i, Lag 0 = %s" % (vname,lbl+1,mons3[kmonth])
        fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,6))
        ax,ax2     = viz.init_acplot(kmonth,xtks,lags,ax=ax,title=title)
        
        npts = acslabel.shape[1]
        for p in range(npts):
            ax.plot(lags,acslabel[:,p],alpha=0.2,color='gray',label="")
        meanlabel="cluster %i, n=%i" % (lbl+1,npts)
        ax.plot(lags,acslabel[:,:].mean(1),alpha=1,color='k',label=meanlabel)
        ax.plot(lags,cluster_centers[v,lbl,:,kmonth],label="Cluster Center",color='blue',ls='dashed')
        
        savename = "%sACFClustering_%s_nclust%i_ACF_clust%i_kmonth%02i.png" % (figpath,vname,nclusts,lbl+1,kmonth+1)
        plt.savefig(savename, dpi=150, bbox_inches="tight")

# ----------------------------------------------------------------------
#%% Repeat for above, but just focusing on february Autocorrelation
# ----------------------------------------------------------------------


nclusts = 8
kmonth  = 1

# Preallocate
cluster_labels  = np.zeros((2, nlat*nlon)) * np.nan
cluster_centers = np.zeros((2, nclusts, nlags)) * np.nan
cluster_outputs = []

# Perform clustering for each variable
for v in range(2):
    
    # Get and prepare input
    inputs = input_maps[v][:,kmonth,:]
    inputs = inputs.reshape(nlags, n_notnans[v])  # [lag*mon, space]
    # [space x lag*mon] (cluster over spatial dim), must be [samples x features]
    inputs = inputs.T
    
    # Perform clustering and retrieve output
    cluster_output = KMeans(n_clusters=nclusts, random_state=0).fit(inputs)
    ccenters = cluster_output.cluster_centers_  # [cluster x features]
    # [cluster x space].reshape(nmod*nsamples)
    clabels = cluster_output.labels_

    # Save
    id_notnan = nan_dicts[v]['ok_indices']
    cluster_outputs.append(cluster_output)
    cluster_labels[v, id_notnan] = clabels.copy()
    cluster_centers[v, :] = ccenters.copy()

cluster_labels = cluster_labels.reshape(2, nlat, nlon)
cluster_centers = cluster_centers.reshape(2, nclusts, nlags)

#%% Plot cluster labels

cmap_in = mpl.cm.get_cmap("PuOr",nclusts)

fig, axs = viz.init_fig(1, 2, figsize=(12, 6))

for v in range(2):
    ax = axs[v]
    ax      = viz.add_coast_grid(ax, bbox,fill_color='lightgray', line_color=None)
    pcm = ax.pcolormesh(lon, lat, cluster_labels[v, ...]+1,cmap=cmap_in)
    fig.colorbar(pcm,ax=ax,orientation='horizontal',pad=0.06,fraction=0.026)
    ax.set_title("%s Clusters" % (varnames[v]))
    
savename = "%sACFClustering_monthwise_kmonth%i_SSTandSSS_nclust%i_Cluster_Centers.png" % (
    figpath, kmonth+1,nclusts)
plt.savefig(savename, dpi=150, bbox_inches="tight")


#%% Plot Cluster Centers and ACFs

xtks = np.arange(0,37,3)

for v in range(2):
    
    vname    = varnames[v]
    clabels  = cluster_labels[v,:,:]
    acs_in   = ds_ensavg[v][vname].values # [lag,lat,lon]
    
    # Look for label
    for lbl in range(nclusts):
        idlabel  = np.where(clabels == lbl)
        acslabel = acs_in[:,kmonth,idlabel[0],idlabel[1]] 
        
        title  = "ACF Spaghetti for %s, Cluster %i, Lag 0 = %s" % (vname,lbl+1,mons3[kmonth])
        fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,6))
        ax,ax2     = viz.init_acplot(kmonth,xtks,lags,ax=ax,title=title)
        
        npts = acslabel.shape[1]
        for p in range(npts):
            ax.plot(lags,acslabel[:,p],alpha=0.2,color='gray',label="")
        meanlabel="cluster %i, n=%i" % (lbl+1,npts)
        ax.plot(lags,acslabel[:,:].mean(1),alpha=1,color='k',label=meanlabel)
        ax.plot(lags,cluster_centers[v,lbl,:],label="Cluster Center",color='blue',ls='dashed')
        
        savename = "%sACFClustering_monthwise_kmonth%i_%s_nclust%i_ACF_clust%i_kmonth%02i.png" % (figpath,kmonth+1,vname,nclusts,lbl+1,kmonth+1)
        plt.savefig(savename, dpi=150, bbox_inches="tight")