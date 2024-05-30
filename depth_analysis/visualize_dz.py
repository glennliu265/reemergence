#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize dz

Visualize the vertical gradient of TEMP/SALT at the detrainment depth
to see if we can explain the differences in detrainment damping


# General Procedure (for basin loop)


>> For each ens. member

    >> Load Detrainment Depth
    
    >> For each variable
    
        >> Load 3D variable
        >> Compute Centered Difference/Gradient
        
        >> For each point (xr ufunc?)
        
            >> Select detrainment depth
            >> Take mean during detrainment depths (or set gradients to zero otherwise)
            
            
        
        
        
3) Load TEMP and SALT

for each point:
    4) Compute the gradient at the point

    
    
    
# Pointwise Procedire using data loaded from ___

 
Created on Mon May 20 14:46:55 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
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

#%% Work with data postprocessed by [viz_icefrac.py]

# Load the data
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"
ncname  = "IrmingerAllEns_SALT_TEMP.nc"
ds      = xr.open_dataset(outpath+ncname).load()

# Load the mixed layer depth
mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl.nc"
ds_mld  = xr.open_dataset(mldpath + mldnc).load()

# Load detrainment depth computed from [calc_detrainment_depth]
hdtnc = "CESM1_HTR_FULL_hdetrain_NAtl.nc"
ds_hdetrain = xr.open_dataset(mldpath+hdtnc).load()

# Load estimates of SALT and TEMP


vnames  = ["TEMP","SALT"]
vcolors = ["hotpink","navy"]
vunits  = ["$\degree C$","$psu$"]
vmarkers = ['o','x']
#%% Load the depth

e = 0
d = 0 

# Input: ds (time x z_t), lonf, latf, mld

dspt = ds.isel(dir=d,ensemble=e) # (time, z_t)  # Time x Depth

dspt['z_t'] = dspt['z_t']/100 # Convert to meters
lonf = dspt.TLONG.values - 360 #  convert to degrees west
latf = dspt.TLAT.values

mldpt = ds_mld.sel(lon=lonf,lat=latf,method='nearest').isel(ens=e)
kprev,_ = scm.find_kprev(mldpt.h.values)

mons3 = proc.get_monstr()

#%% Approach (1) Compute the mean seasonal cycle of vertical temperature and salt gradients

debug       = True
dspt_scycle = dspt.groupby('time.month').mean('time') # [depth x month]

# For each detrainment depth 
dz_byvar      = [] # [entrain month][variable]
detrain_depth = []
for im in range(12):
    
    # Get the detrainment depth (outsourced this for calc_detrainemnt_depth)
    #mondet   = kprev[im]
    #dtfloor  = int(np.floor(mondet))
    #dtceil   = int(np.ceil(mondet))
    hdetrain = ds_hdetrain.sel(lon=lonf,lat=latf,method='nearest').isel(mon=im,ens=e).h.values.item()#np.interp(mondet,[dtloor,dtceil],[mldpt.sel(mon=dtfloor).h.values,mldpt.sel(mon=dtceil).h.values])
    #detrain_depth.append(hdetrain)
    
    dz_var = []
    for vv in range(2):
        
        # Compute gradient (centered difference)
        dsvar = dspt[vnames[vv]]
        
        # Get depth gradient (centered difference)
        z_t      = dsvar.z_t.values/100
        iz       = np.argmin(np.abs(z_t-hdetrain))
        dz       =  z_t[iz+1] - z_t[iz-1]
        
        # Method (1) : Compute gradients, then take seasonal avg of gradient
        dx               = dsvar.isel(z_t=iz+1) - dsvar.isel(z_t=iz-1) # Difference values above and below detrainemtn depth
        zgrad            = dx/dz
        zgrad_scycle1    = zgrad.groupby('time.month').mean('time')
        
        # Method (2) : Take seasonal avg, then compute gradient (it seems to make no difference)
        if debug:
            # Compute vertical gradients
            dsvar_scycle     = dsvar.groupby('time.month').mean('time')
            dx_1             = dsvar_scycle.isel(z_t=iz+1) - dsvar_scycle.isel(z_t=iz-1)
            zgrad_scycle2    = dx_1 / dz
        
            # Check the Difference
            fig,ax = viz.init_monplot(1,1)
            ax.plot(mons3,zgrad_scycle1,label="gradient, then seasonal cycle")
            ax.plot(mons3,zgrad_scycle2,label="seasonal cycle,then gradient",ls='dashed')
            ax.legend()
            ax.set_title("Lon %f, Lat %f, Month %s" % (lonf,latf,mons3[im]))
            
            
        dz_var.append(zgrad_scycle1)
    dz_byvar.append(dz_var)
    
#%% next, take mean over the detrained period


dz_byvar_avg = np.ones((2,12)) * np.nan

dz_detrain_fin_byvar = []
for im in range(12):
    
    # Get the months to average over
    if kprev[im] == 0.:
        dxdz_avg = np.nan
        continue
    else:
        imdetrain = int(np.floor(kprev[im])) - 1
        
        if imdetrain > im:
            sel_mons = np.hstack([np.arange(imdetrain,12),np.arange(0,im+1)])
        elif im == imdetrain:
            sel_mons = np.arange(12)
        else:
            sel_mons = np.arange(imdetrain,im+1)
            
        # Debug Print
        print("For im = %i" % (im))
        print(sel_mons)
        print("\n")
        
    
    # Loop by variabe
    dz_detrain_fin_byvar = []
    for vv in range(2):
        
        dzin  = dz_byvar[im][vv] # Get gradients at depth
        
        dzavg = dzin.isel(month=sel_mons).mean('month').values.item()
        
        dz_byvar_avg[vv,im] = dzavg
        
    #dzvar
#%%


#%% Plot the mean gradient


fig,axs = viz.init_monplot(2,1)
for a in range(2):
    ax = axs[a]
    ax.plot(mons3,dz_byvar_avg[a,:],label=vnames[a])
    #ax.plot(mons3,dz_byvar_avg[a,:],label=vnames[a],ls='dashed')
    ax.legend()
    #ax.set_title("Lon %f, Lat %f, Month %s" % (lonf,latf,mons3[im]))
        
#%% Double check this by looking at the actual profile

vv = 0



plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.values # Plot the actual mean seasonal cycle
plotdetrain = ds_hdetrain.sel(lon=lonf,lat=latf,method='nearest').isel(ens=e).h
plotmld     = ds_mld.sel(lon=lonf,lat=latf,method='nearest').isel(ens=e)
plot_dz     = np.gradient(plotvar,z_t,axis=0)

fig,ax  = plt.subplots(1,1)

# Plot the actual variable
cl     = ax.contour(mons3,z_t,plotvar,colors="k")
ax.clabel(cl)
plt.gca().invert_yaxis()

# Contour vertical gradient (centered difference)
pcm      = ax.pcolormesh(mons3,z_t,plot_dz)
fig.colorbar(pcm,ax=ax,fraction=0.045,pad=0.01)
#ax.clabel(cl)


# Plot the detrainment depths
ax.plot(mons3,plotdetrain,marker="x",c="red",label="detrain depths")

# Plot vertical gradient estimates
ax.scatter(mons3,plotdetrain.h,c=dz_byvar_avg[vv,:])



# Plot Estimated gradient

ax.legend(loc='lower center')

#%% Just Check a single vertical profile

im = 9

locfn,loctitle = proc.make_locstring(lonf,latf,)

for im in range(12):
    
    
    fig,ax=plt.subplots(1,1,constrained_layout=True,figsize=(4,8))
    
    ax2 = ax.twiny()
    
    axs = [ax,ax2]
    for vv in range(2):
        
        axin = axs[vv]
        
        plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.isel(month=im)#values 
        axin.plot(plotvar,z_t,c=vcolors[vv],marker=vmarkers[vv])
        axin.set_xlabel("%s [%s]" % (vnames[vv],vunits[vv]),c=vcolors[vv])
        axin.tick_params(axis='x', colors=vcolors[vv])
    
        
        #ax.spines['bottom'].set_color
    plt.gca().invert_yaxis()
    
    # Adjust Axis Colors
    ax2.spines['bottom'].set_color(vcolors[0])
    ax2.spines['top'].set_color(vcolors[1])
    
    ax.axhline([plotdetrain[im]],ls='dashed',label="Detrainment Depth",c="gray")
    ax.axhline([mldpt.h.isel(mon=im)],lw=0.75,label="Mixed Layer Depth",c='k')
    ax.legend(loc="lower center")
    
    # Compute Detrainment Gradient
    vv          = 0
    functexts   = []
    for vv in range(2):
        plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.isel(month=im)
        idetrain = np.argmin(np.abs(z_t-plotdetrain[im].values))
        Xm1 = plotvar[idetrain-1]
        Xp1 = plotvar[idetrain+1]
        dX  = Xp1-Xm1
        zm1 = z_t[idetrain-1]
        zp1 = z_t[idetrain+1] 
        dz  = zp1-zm1
        outgrad = dX/dz
        functext = r"$\frac{d%s}{dz} = \frac{%.3f - %.3f}{%.1f - %.1f} = \frac{%.3f}{%.1f} = %f$ [%s/m]" % (vnames[vv],Xp1,Xm1,zp1,zm1,dX,dz,outgrad,vunits[vv])
        functexts.append(functext)
    
    
    ax.grid(True,ls='dotted',c='gray')
    ax.set_title("%s Vertical Profiles @ Lon (%.2f), Lat (%.2f)\n%s\n%s" % (mons3[im],lonf,latf,functexts[0],functexts[1]))
    
    #ax.set_ylim([150,0])
    savename = "%sVertical_Profile_%s_mon%02i.png" % (figpath,locfn,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
#%% Compare Gradient Functions (Debug)


plot_dz      = (np.roll(plotvar,1,axis=0) - np.roll(plotvar,-1,axis=0)) / (np.roll(z_t,1) - np.roll(z_t,-1))[:,None]


fig,axs = plt.subplots(1,2)

ax  = axs[0]
pcm = ax.pcolormesh(auto_dz)
fig.colorbar(pcm,ax=ax)
ax.set_title("Gradient Function")

ax  = axs[1]
pcm = ax.pcolormesh(plot_dz)
fig.colorbar(pcm,ax=ax)
ax.set_title("Roll Gradient")



#%% Next, load the output from [compute_dz_ens]

ocnpath        = rawpath + "ocn_var_3d/"
dstemp_dtgrad  = xr.open_dataset(ocnpath + "CESM1_HTR_TEMP_Detrain_Gradients.nc").load()
dssalt_dtgrad  = xr.open_dataset(ocnpath + "CESM1_HTR_SALT_Detrain_Gradients.nc").load()


# Visualize the vertical gradient
#%% Check the second derivative

for im in range(12):

    
    
    ddnames = [[],
        [r"$\frac{dTEMP}{dz}$ ($\frac{\degree C}{m}$)",r"$\frac{dSALT}{dz}$ ($\frac{psu}{m}$)"],
        [r"$\frac{d^2 TEMP}{dz^2}$ ($\frac{\degree C}{m^2}$)",r"$\frac{d^2 SALT}{dz^2}$ ($\frac{psu}{m^2}$)"]
        ]
    
    
    
    
    fig,axsall=plt.subplots(1,3,constrained_layout=True,figsize=(12,9))
    
    for dd in range(3):
        ax = axsall[dd]
        ax2 = ax.twiny()
        
        axs = [ax,ax2]
        for vv in range(2):
            
            axin        = axs[vv]
            plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.isel(month=im)#values 
            
            if dd == 0:
                axin.set_xlabel("%s (%s)" % (vnames[vv],vunits[vv]),c=vcolors[vv],fontsize=16)
            
            elif dd == 1: # Compute the first derivative
            
                plotvar = plotvar.differentiate('z_t')
                lab = ddnames[dd][vv]#r"$\frac{d%s}{dz}$" % (vnames[vv]) #+ " ($\frac{%s}{m}$)" % (vnames[vv],vunits[vv])
                axin.set_xlabel(lab,c=vcolors[vv],fontsize=16)
                
            elif dd == 2: # Compute the second derivative
                
                plotvar = plotvar.differentiate('z_t').differentiate('z_t')
                lab = ddnames[dd][vv]#r"$\frac{d^2 %s}{dz^2}$ ($\frac{%s}{m^2}$)" % (vnames[vv],vunits[vv])
                axin.set_xlabel(lab,c=vcolors[vv],fontsize=16)
                
            axin.plot(plotvar,z_t,c=vcolors[vv],marker=vmarkers[vv])
            axin.tick_params(axis='x', colors=vcolors[vv])
        
            
            #ax.spines['bottom'].set_color
        plt.gca().invert_yaxis()
        
        # Adjust Axis Colors
        ax2.spines['bottom'].set_color(vcolors[0])
        ax2.spines['top'].set_color(vcolors[1])
        
        ax.axhline([plotdetrain[im]],ls='dashed',label="Detrainment Depth",c="gray")
        ax.axhline([mldpt.h.isel(mon=im)],lw=0.75,label="Mixed Layer Depth",c='k')
        ax.legend(loc="lower center")
        
        # Compute Detrainment Gradient
        vv          = 0
        functexts   = []
        for vv in range(2):
            plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.isel(month=im)
            idetrain = np.argmin(np.abs(z_t-plotdetrain[im].values))
            Xm1 = plotvar[idetrain-1]
            Xp1 = plotvar[idetrain+1]
            dX  = Xp1-Xm1
            zm1 = z_t[idetrain-1]
            zp1 = z_t[idetrain+1] 
            dz  = zp1-zm1
            outgrad = dX/dz
            functext = r"$\frac{d%s}{dz} = \frac{%.3f - %.3f}{%.1f - %.1f} = \frac{%.3f}{%.1f} = %f$ [%s/m]" % (vnames[vv],Xp1,Xm1,zp1,zm1,dX,dz,outgrad,vunits[vv])
            functexts.append(functext)
        
        
        ax.grid(True,ls='dotted',c='gray')
        #ax.set_title("%s Vertical Profiles @ Lon (%.2f), Lat (%.2f)\n%s\n%s" % (mons3[im],lonf,latf,functexts[0],functexts[1]))
        
        #ax.set_ylim([150,0])
    
    
    plt.suptitle("%s Gradients @ %s" % (mons3[im],loctitle),fontsize=20)
    savename = "%sVertical_Profile_Gradients_%s_mon%02i.png" % (figpath,locfn,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
#%% For second deriative, Try different number of spline functions

im = 1

ylm = [600,900]
    

fig,axsall=plt.subplots(1,3,constrained_layout=True,figsize=(12,9))

for dd in range(3):
    ax = axsall[dd]
    ax2 = ax.twiny()
    
    axs = [ax,ax2]
    for vv in range(2):
        
        axin        = axs[vv]
        plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.isel(month=im)#values 
        
        if dd == 0:
            axin.set_xlabel("%s (%s)" % (vnames[vv],vunits[vv]),c=vcolors[vv],fontsize=16)
        
        elif dd == 1: # Compute the first derivative
        
            plotvar = plotvar.differentiate('z_t')
            lab = ddnames[dd][vv]#r"$\frac{d%s}{dz}$" % (vnames[vv]) #+ " ($\frac{%s}{m}$)" % (vnames[vv],vunits[vv])
            axin.set_xlabel(lab,c=vcolors[vv],fontsize=16)
            
        elif dd == 2: # Compute the second derivative
            # Take Second Derivative
            plotvar = plotvar.differentiate('z_t').differentiate('z_t')
            lab     = ddnames[dd][vv]#r"$\frac{d^2 %s}{dz^2}$ ($\frac{%s}{m^2}$)" % (vnames[vv],vunits[vv])
            axin.set_xlabel(lab,c=vcolors[vv],fontsize=16)
            
        axin.plot(plotvar,z_t,c=vcolors[vv],marker=vmarkers[vv])
        axin.tick_params(axis='x', colors=vcolors[vv])
    
        
        #ax.spines['bottom'].set_color
        
    # Adjust axis
    if ylm is not None:
        ax.set_ylim(ylm)
    plt.gca().invert_yaxis()
    
    # Adjust Axis Colors
    ax2.spines['bottom'].set_color(vcolors[0])
    ax2.spines['top'].set_color(vcolors[1])
    
    ax.axhline([plotdetrain[im]],ls='dashed',label="Detrainment Depth",c="gray")
    ax.axhline([mldpt.h.isel(mon=im)],lw=0.75,label="Mixed Layer Depth",c='k')
    ax.legend(loc="lower center")
    
    # Compute Detrainment Gradient
    vv          = 0
    functexts   = []
    for vv in range(2):
        plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.isel(month=im)
        idetrain = np.argmin(np.abs(z_t-plotdetrain[im].values))
        Xm1 = plotvar[idetrain-1]
        Xp1 = plotvar[idetrain+1]
        dX  = Xp1-Xm1
        zm1 = z_t[idetrain-1]
        zp1 = z_t[idetrain+1] 
        dz  = zp1-zm1
        outgrad = dX/dz
        functext = r"$\frac{d%s}{dz} = \frac{%.3f - %.3f}{%.1f - %.1f} = \frac{%.3f}{%.1f} = %f$ [%s/m]" % (vnames[vv],Xp1,Xm1,zp1,zm1,dX,dz,outgrad,vunits[vv])
        functexts.append(functext)
    
    
    ax.grid(True,ls='dotted',c='gray')
    #ax.set_title("%s Vertical Profiles @ Lon (%.2f), Lat (%.2f)\n%s\n%s" % (mons3[im],lonf,latf,functexts[0],functexts[1]))
    
    #ax.set_ylim([150,0])

# def check_centdiff(z,invar,idsel,order=1,vname="x",vunit="units"):
#     Xm1         = invar[idsel-1]
#     Xp1         = invar[idsel+1]
#     dX          = Xp1-Xm1
#     zm1         = z[idsel-1]
#     zp1         = z[idsel+1] 
#     dz          = zp1-zm1
#     outgrad     = dX/dz
#     functext    = r"$\frac{d%s}{dz} = \frac{%.3f - %.3f}{%.1f - %.1f} = \frac{%.3f}{%.1f} = %f$ [%s/m]" % (vname,Xp1,Xm1,zp1,zm1,dX,dz,outgrad,vunit)
        
        
#     return functext


    plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.isel(month=im)
    idetrain = np.argmin(np.abs(z_t-plotdetrain[im].values))
    Xm1 = plotvar[idetrain-1]
    Xp1 = plotvar[idetrain+1]
    dX  = Xp1-Xm1
    zm1 = z_t[idetrain-1]
    zp1 = z_t[idetrain+1] 
    dz  = zp1-zm1
    outgrad = dX/dz
    functext = r"$\frac{d%s}{dz} = \frac{%.3f - %.3f}{%.1f - %.1f} = \frac{%.3f}{%.1f} = %f$ [%s/m]" % (vnames[vv],Xp1,Xm1,zp1,zm1,dX,dz,outgrad,vunits[vv])
    return functext

#%%
deriv1 = plotvar.differentiate('z_t')


#%% Look close to the detrain depth