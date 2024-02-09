#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze output produced by run_SSS_basinwide.py
Copied upper section of that script on Feb 1 Thu

Created on Thu Feb  1 22:48:44 2024

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

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

sys.path.append(scmpath + '../')

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

import stochmod_params as sparams

#%% General Variables/User Edits, Set Experiment Name

# # Ekman Advection (no EOF)
compare_name  = "Qek_compare_simple"
expnames      = ["Test_Td0.1_SPG_noroll","SST_OSM_Tddamp","SST_OSM_Tddamp_Qek_monvar"]
expnames_long = ["SST","SST ($\lambda^d$)","SST ($\lambda^d$,. $Q_{ek}$)"] 
varname       = "SST"


# # OSM Comparison (SST/SSS, with and without entrainment)
# compare_name  = "OSM_entrain"
# expnames      = ["SST_OSM_Tddamp_noentrain","SST_OSM_Tddamp","SSS_OSM_Tddamp_noentrain","SSS_OSM_Tddamp",]
# expnames_long = ["SST","SST (entrain)","SSS","SSS (entrain)"] 
# varname       = "BOTH" # #["SST","SSS"]

# Comparing Shift Effects
#compare_name = "shift_test"
#expnames_long = ["No Shift","Half Shift","Shift Forcing and MLD"]
#expnames      = ["Test_Td0.1_SPG_noroll","Test_Td0.1_SPG_froll1-mroll1","Test_Td0.1_SPG_allroll1_halfmode",]
#varname        = "SSS"

# # Comparing Expfit damping vs. our statistical method
# compare_name  = "damping_expfit"
# expnames      = ["SST_covariance_damping_20to65","SST_expfit_damping_20to65","SST_expfit_SST_damping_20to65"]
# expnames_long = ["Covariance-based","$\lambda^a$ fit","SST fit"]
# varname       = "SST"

# expparams   = {
#     'bbox_sim'      : [-65,0,45,65],
#     'nyrs'          : 1000,
#     'runids'        : ["test%02i" % i for i in np.arange(0,11,1)],
#     'PRECTOT'       : "CESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc",
#     'LHFLX'         : "CESM1_HTR_FULL_Eprime_nroll0_NAtl_EnsAvg.nc",
#     'h'             : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'         : 0.10,
#     'Sbar'          : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
#     'beta'          : None, # If None, just compute entrainment damping
#     'kprev'         : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'         : None, # NEEDS TO BE ALREADY CONVERTED TO 1/Mon !!!
#     'froll'         : 0,
#     'mroll'         : 0,
#     'droll'         : 0,
#     'halfmode'      : False,
#     }

# expparams   = {
#     'bbox_sim'      : [-65,0,45,65],
#     'nyrs'          : 1000,
#     'runids'        : ["test%02i" % i for i in np.arange(0,11,1)],
#     'PRECTOT'       : "CESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc",
#     'LHFLX'         : "CESM1_HTR_FULL_Eprime_nroll0_NAtl_EnsAvg.nc",
#     'h'             : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'         : 0.10,
#     'Sbar'          : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
#     'beta'          : None, # If None, just compute entrainment damping
#     'kprev'         : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'         : None, # NEEDS TO BE ALREADY CONVERTED TO 1/Mon !!!
#     'froll'         : 0,
#     'mroll'         : 0,
#     'droll'         : 0,
#     'halfmode'      : False,
#     }
# Parameters needed from expparam
# Constants
dt  = 3600*24*30 # Timestep [s]
cp  = 3850       # 
rho = 1026       # Density [kg/m3]
B   = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L   = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

debug = False

#%% Load output

output_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240205/"
proc.makedir(figpath)

paramdicts = []
nexps      = len(expnames)
ds_out     = [] 
for ex in range(nexps):
    expname = expnames[ex]
    
    # Load NC Files
    expdir       = output_path + expname + "/Output/"
    nclist       = glob.glob(expdir +"*.nc")
    nclist.sort()
    #print(nclist)
    print("Found %i files for %s" % (len(nclist),expname))
    
    ds_all = xr.open_mfdataset(nclist,concat_dim="run",combine='nested').load()
    ds_out.append(ds_all)
    
    # Load Param Dictionary
    dictpath = output_path + expname + "/Input/expparams.npz"
    ld = np.load(dictpath,allow_pickle=True)
    paramdicts.append(ld)

# Debugging stuff
if compare_name == 'damping_expfit': # Check to see if different damping was loaded
    [print( paramdicts[ii]['lbd_a']) for ii in range(3)] 

#%% Load CESM1 Output for SSS/SST

# Loading old anomalies
#ncpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
#ncname  = "%s_FULL_HTR_lon-80to0_lat0to65_DTEnsAvg.nc" % varname
#anom_cesm = True

# Loading anomalies used in recent scripts (find origin, I think its prep_var_monthly, etc)
ncpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
ncname    = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % varname
anom_cesm = False # Set to false to anomalize

# Load DS
ds_cesm  = xr.open_dataset(ncpath+ncname).squeeze()

# Slice to region
bbox_sim = paramdicts[0]['bbox_sim']
ds_cesm  = proc.sel_region_xr(ds_cesm,bbox_sim)

# Correct Start time
ds_cesm  = proc.fix_febstart(ds_cesm)
ds_cesm  = ds_cesm.sel(time=slice('1920-01-01','2005-12-31')).load()

# Anomalize if necessary
if anom_cesm is False:
    print("Detrending and deseasonalizing variable!")
    ds_cesm = proc.xrdeseason(ds_cesm) # Deseason
    ds_cesm = ds_cesm[varname] - ds_cesm.SST.mean('ensemble')
else:
    ds_cesm = ds_cesm[varname]

#%% Load some dims for plotting
ds             = ds_out[0]
lon            = ds.lon.values
lat            = ds.lat.values
tsim           = ds.time.values

latf           = 50
lonf           = -30
klon,klat      = proc.find_latlon(lonf,latf,lon,lat)
locfn,loctitle = proc.make_locstring(lonf,latf)


#%% Set up for analysis

# Combine DS for Analysis, Rename run/ensemble --> ens
ds_sm       = [ds.SST for ds in ds_out] # ('run', 'time', 'lat', 'lon')
ds_sm       = [ds.rename({'run':'ens'}) for ds in ds_sm]
ds_cesm_in  = ds_cesm.rename({'ensemble':'ens'}) 
ds_all      = ds_sm + [ds_cesm_in,]
[print(ds.dims) for ds in ds_all]


# Make a mask common to all
mask_sum  = np.array([ds.sum('ens').sum('time').values for ds in ds_all]) # 0. seems to be anom
mask_zeros = (mask_sum !=0.).prod(0) # If I do .sum() instead of .prod() this isolates the ice mask....
mask = mask_zeros.copy().astype('float')
mask[mask==0.] = np.nan
plt.pcolormesh(mask),plt.colorbar()

# for ii in range(4):
#     plt.pcolormesh(mask[ii,:,:],plt.colorbar(),plt.show())
    
# mask  = []
# ds_mask  = [np.isnan()] 
#%% Set up strings & plotting paraemters

enames    = expnames_long + ["CESM1 Historical",]
cols      = ["salmon","violet","darkblue","black"]
lss       = ["solid",'dashed','dotted','solid']
mks       = ["x","s","d","o"]
bbox_plot = [-80,0,20,65]

# Adjust rcparams
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['font.family']      = 'JetBrains Mono'#'Courier'#'STIXGeneral'

fsz_title  = 16
fsz_axis   = 14 






#%% Updated variance plots (overall stdev, 4x4)


plotdiff = True
ds_ref   = ds_all[-1].std('time').mean("ens") * mask # Cesm as reference mask

fig,axs  = viz.geosubplots(2,2,figsize=(10,7.1),)

pcms = []
for a,ax in enumerate(axs.flatten()):
    
    # Formatting
    blabels=[0,0,0,0]
    if a%2 == 0:
        blabels[0]  = 1
    if a>1:
        blabels[-1] = 1
    ax=viz.add_coast_grid(ax,bbox=bbox_plot,fill_color="lightgray",blabels=blabels)

# Now loop for each experiment
for a in range(len(enames)):
    ax = axs.flatten()[a]
    ax.set_title(enames[a],fontsize=fsz_title)
    
    # Plot Variables
    if plotdiff and a < len(enames)-1:
        plotvar = (ds_all[a].std('time').mean("ens") - ds_ref) * mask
        vlm      = [-.5,.5]
        cmap     = 'cmo.balance' 
        
    else:
        plotvar = ds_all[a].std('time').mean("ens") * mask
        vlm      = [0,1]
        cmap     = 'cmo.thermal' 
        
    pcm     = ax.pcolormesh(lon,lat,plotvar,vmin=vlm[0],vmax=vlm[-1],cmap=cmap)
    pcms.append(pcm)

# Colorbar
if plotdiff:
    cblabel = "Difference in SST Standard Deviation, SM - CESM1 ($\degree C^2$)"
    pcm_choose=0
else:
    cblabel = "SST Standard Deviation ($\degree C^2$)"
    pcm_choose=-1
cb = fig.colorbar(pcms[pcm_choose],ax=axs.flatten(),orientation='horizontal',fraction=0.040,pad=0.02)
cb.set_label(cblabel,fontsize=fsz_axis)

savename = "%s%s_Variance_%s_diff%0i.png" % (figpath,varname,compare_name,plotdiff)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Compare Summer and winter

plotmode = "Winter"
plotdiff = True
#ds_ref   = ds_all[-1].std('time').mean("ens") * mask # Cesm as reference mask

if plotmode == "Summer":
    #plotmons = ["AMJJAS"]
    plotmons = [4,5,6,7,8,9]
elif plotmode == "Winter":
    plotmons = [1,2,3,10,11,12]
    
ds_monvar = [ds.sel(time=ds.time.dt.month.isin(plotmons)).std('time').mean('ens') for ds in ds_all]
ds_ref    = ds_monvar[-1] * mask

fig,axs = viz.geosubplots(2,2,figsize=(10,7.1),)

pcms = []
for a,ax in enumerate(axs.flatten()):
    
    # Formatting
    blabels=[0,0,0,0]
    if a%2 == 0:
        blabels[0]  = 1
    if a>1:
        blabels[-1] = 1
    ax=viz.add_coast_grid(ax,bbox=bbox_plot,fill_color="lightgray",blabels=blabels)
    ax.set_title(enames[a],fontsize=fsz_title)
    
    # Plot Variables
    if plotdiff and a < len(enames)-1:
        
        plotvar  = (ds_monvar[a] - ds_ref) * mask
        vlm      = [-.5,.5]
        cmap     = 'cmo.balance' 
        
    else:
        plotvar = ds_monvar[a] * mask
        vlm      = [0,1]
        cmap     = 'cmo.thermal' 
        
    pcm     = ax.pcolormesh(lon,lat,plotvar,vmin=vlm[0],vmax=vlm[-1],cmap=cmap)
    pcms.append(pcm)

# Colorbar
if plotdiff:
    cblabel = "Difference in SST Standard Deviation, SM - CESM1 ($\degree C^2$)"
    pcm_choose=0
else:
    cblabel = "SST Standard Deviation ($\degree C^2$)"
    pcm_choose=-1
cb = fig.colorbar(pcms[pcm_choose],ax=axs.flatten(),orientation='horizontal',fraction=0.040,pad=0.02)
cb.set_label(cblabel,fontsize=fsz_axis)

savename = "%s%s_Variance_%s_%s_diff%0i.png" % (figpath,varname,compare_name,plotmode,plotdiff)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Do some regional analysis

bbxall      = sparams.bboxes
regionsall  = sparams.regions 
rcolsall    = sparams.bbcol

# Select Regions
regions_sel = ["SPG","NNAT","STGe","STGw"]
bboxes      = [bbxall[regionsall.index(r)] for r in regions_sel]
rcols       = [rcolsall[regionsall.index(r)] for r in regions_sel]

#%% 


# Area weights from xr demon (move this to proc) https://docs.xarray.dev/en/latest/examples/area_weighted_temperature.html

def area_avg_cosweight(ds):
    weights     = np.cos(np.deg2rad(ds.lat))
    ds_weighted = ds.weighted(weights)
    return ds_weighted.mean(('lat','lon'))



# loop for region
r     = 1
tsm_regs = []
ssts_reg = []
for r in range(len(regions_sel)):
    #rshapes = []
    
    # Take the area weighted average
    bbin  = bboxes[r]
    rssts = [proc.sel_region_xr(ds,bbin) for ds in ds_all]
    ravgs = [area_avg_cosweight(ds) for ds in rssts]
    
    # Compute some metrics
    rsstin = [rsst.values.flatten() for rsst in ravgs]
    rsstin = [np.where((np.abs(rsst)==np.inf) | np.isnan(rsst),0.,rsst) for rsst in rsstin]
    tsmr   = scm.compute_sm_metrics(rsstin)
    
    tsm_regs.append(tsmr)
    ssts_reg.append(ravgs)


#%% examine ACF over different regions

nregs = len(bboxes)

kmonth = 1
xtksl  = np.arange(0,37,3)
lags   = np.arange(37)

fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(16,8))

for rr in range(nregs):
    
    ax   = axs.flatten()[rr]
    ax,_ = viz.init_acplot(kmonth,xtksl,lags,title="",ax=ax)
    
    for ii in range(4):
        plotvar = tsm_regs[rr]['acfs'][kmonth][ii]
        ax.plot(lags,plotvar,label=enames[ii],c=cols[ii],ls=lss[ii],marker=mks[ii])
    ax.legend()
    ax.set_title(regions_sel[rr])

#%% Plot MOnthly variance
mons3=proc.get_monstr()

fig,axs = plt.subplots(1,4,constrained_layout=True,figsize=(16,4.5))

for rr in range(nregs):
    
    ax   = axs[rr]
    #ax,_ = viz.init_monplot()
    
    for ii in range(4):
        plotvar = tsm_regs[rr]['monvars'][ii]
        ax.plot(mons3,plotvar,label=enames[ii],c=cols[ii],ls=lss[ii],marker=mks[ii])
    ax.legend()
    ax.set_title(regions_sel[rr])

#%%




"#%% Quickly plot simulations
r = 0

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,6.5))

ax = axs[0]
for ii in range(3):
    for nr in range(5):
        if nr == 0:
            lab = "%s "% (enames[ii])#"#(run %i)" % (enames[ii],nr+1)
        else:
            lab = ""
        ax.plot(ravgs[ii].isel(ens=nr),color=cols[ii],label=lab,alpha=0.7)
ax.legend(ncol=3)
ax.set_title("Stochastic Model")
ax.set_ylim([-3,3])
ax.set_ylabel("SST ($\degree C$)")

ax = axs[1]
ii = 3
for nr in range(42):
    if nr == 0:
        lab = "%s "% (enames[ii])#"#(ens %i)" % (enames[ii],nr+1)
    else:
        lab = ""
    ax.plot(ravgs[ii].isel(ens=nr),color=cols[ii],label=lab,alpha=0.1)
ax.set_title("CESM1 Historical")
ax.legend()
ax.set_ylabel("SST ($\degree C$)")
ax.set_xlabel("Timestep (Month)")
plt.suptitle("SST Timeseries")


#%% Plot ACF






#%%



#%%




#%%  Old Scripts, can delete evemtually.

#%% Compare Variance Across Simulations (Old plot for SSS, delete)


mpl.rcParams['mathtext.fontset'] = 'stix'#'custom' 
mpl.rcParams['font.family'] = 'STIXGeneral'#'Courier'#'STIXGeneral'

plotdiff = False


if compare_name == "shift_test_SSS":
    vlms = [0,0.25]
fig,axs = viz.geosubplots(2,2,figsize=(10,5),)

for a,ax in enumerate(axs.flatten()):
    
    blabels=[0,0,0,0]
    if a%2 == 0:
        blabels[0] = 1
    if a>1:
        blabels[-1] =1 
    # Formatting
    ax=viz.add_coast_grid(ax,bbox=bbox_plot,fill_color="darkgray",blabels=blabels)
    
    # PLotting
    if a > 0:
        
        ds_in   = ds_out[a-1]
        if plotdiff:
            plotvar_sm = ds_in.std('time').mean('run')
            plotvar_ce = ds_cesm.std('time').mean('ensemble')
            plotvar = plotvar_sm - plotvar_ce
            vlms = [-.1,.1]
            
            cmap = 'RdBu_r'
        else:
            plotvar = ds_in.std('time').mean('run')
        title   = "Stochastic Model (%s) - CESM1" % expnames_long[a-1] 
    else:
        ds_in = ds_cesm
        plotvar = ds_in.std('time').mean('ensemble')
        title   = "CESM1 Historical"
        cmap = 'cmo.haline'
        
    #plotvar = plotvar[varname]
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=vlms[0],vmax=vlms[-1],cmap=cmap)
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
    ax.set_title(title)
    
plt.suptitle(r"%s Std. Dev. (psu)" % varname)

savename = "%s%s_Variance_ShiftComparison_%s_diff%0i.png" % (figpath,varname,comparename,plotdiff)
plt.savefig(savename,dpi=150)
plt.show()


#%% Merge DS for comparison

nrun,ntime,nlat,nlon  = ds_out[0].SSS.shape
nens,ntimec,nlat,nlon = ds_cesm.SSS.shape

smflatten   = [ds.SSS.values.reshape(nrun*ntime,nlat,nlon) for ds in ds_out]
cesmflatten = ds_cesm.SSS.values.reshape(nens*ntimec,nlat,nlon)
cesmflatten[np.isnan(cesmflatten)] = 0

dsloop     = smflatten + [cesmflatten,]
nameloops  = ["Stochastic Model (%s)" % a for a in expnames_long] + ["CESM1 Historical",]
loopcolors = ["blue","orange","violet","k"]

dspt = [ds[:,klat,klon] for ds in dsloop]
tsm  = scm.compute_sm_metrics(dspt)

#%% Compute T2 (winter)





#%% Plot ACF

lags   = np.arange(37)
xtks   = np.arange(0,37,3)
kmonth = 1
fig,ax = viz.init_acplot(kmonth,xtks,lags)

for ii in range(4):
    ax.plot(lags,tsm['acfs'][kmonth][ii],label=nameloops[ii],lw=3.5,c=loopcolors[ii])
    
ax.legend()
plt.show()

#%% Plot Monvar
mons3 = proc.get_monstr(nletters=3)

fig,ax = viz.init_monplot(1,1)

for ii in range(4):
    ax.plot(mons3,tsm['monvars'][ii],label=nameloops[ii],lw=3.5,c=loopcolors[ii])
    
ax.legend()
plt.show()


#%%
#%%
# Below is still scrap

#%%
#%%

#%% Set Paths

# Path to Experiment Data
output_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
expdir       = output_path + expname + "/"

# Named Paths
outpath_sm = expdir + "Output/" # path to stochastic model output
figpath    = expdir + "Figures/" # path to figures

proc.makedir(expdir + "Input")
proc.makedir(outpath_sm)
proc.makedir(expdir + "Metrics")
proc.makedir(figpath)




#%%



#%%

# Paths and Experiment
input_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/"
output_path= "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"

expname    = "Test_Td0.1_SPG"

expparams   = {
    'bbox_sim'      : [-65,0,45,65],
    'nyrs'          : 1000,
    'runids'        : ["test%02i" % i for i in np.arange(1,11,1)],
    'PRECTOT'       : "CESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc",
    'LHFLX'         : "CESM1_HTR_FULL_Eprime_nroll0_NAtl_EnsAvg.nc",
    'h'             : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'         : 0.10,
    'Sbar'          : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
    'beta'          : None, # If None, just compute entrainment damping
    'kprev'         : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'         : None, # NEEDS TO BE ALREADY CONVERTED TO 1/Mon !!!
    }

# Constants
dt  = 3600*24*30 # Timestep [s]
cp  = 3850       # 
rho = 1026       # Density [kg/m3]
B   = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L   = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

debug = False


