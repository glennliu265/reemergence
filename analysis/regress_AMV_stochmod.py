#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regress AMV Index to SST/SSS in the stochastic model vs CESM1

Copied upper section of point_metrics_paper

Created on Mon Aug 12 14:47:48 2024

@author: gliu

"""


import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time


#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm



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

proc.makedir(figpath)

#%%

bboxplot                        = [-80,0,20,65]
mpl.rcParams['font.family']     = 'Avenir'
mons3                           = proc.get_monstr(nletters=3)

fsz_tick                        = 18
fsz_axis                        = 20
fsz_title                       = 16

rhocrit                         = proc.ttest_rho(0.05,2,86)
proj                            = ccrs.PlateCarree()

# Get Region Info
regionset = "SSSCSU"
rdict                       = rparams.region_sets[regionset]
regions                     = rdict['regions']
bboxes                      = rdict['bboxes']
rcols                       = rdict['rcols']
rsty                        = rdict['rsty']
regions_long                = rdict['regions_long']
nregs                       = len(bboxes)

regions_long = ('Sargasso Sea', 'N. Atl. Current',  'Irminger Sea')


#%% Load Land Ice Mask

# Load the currents
ds_uvel,ds_vvel = dl.load_current()
ds_bsf          = dl.load_bsf(ensavg=False)
ds_ssh          = dl.load_bsf(ensavg=False,ssh=True)

# Convert Currents to m/sec instead of cmsec
ds_uvel = ds_uvel/100
ds_vvel = ds_vvel/100
tlon  = ds_uvel.TLONG.mean('ens').values
tlat  = ds_uvel.TLAT.mean('ens').values

# Load Land Ice Mask
icemask    = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")
mask       = icemask.MASK.squeeze()
mask_plot  = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values

# Get A region mask
mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub

ds_gs2          = dl.load_gs(load_u2=True)


#%%  Indicate Experients (copying upper setion of viz_regional_spectra )
regionset       = "SSSCSU"
comparename     = "Draft3" #"Paper_Draft02_AllExps"#"Paper_Draft02_AllExps"

# Take single variable inputs from compare_regional_metrics and combine them
if comparename == "Paper_Draft02_AllExps": # Draft 2
    # # #  Same as comparing lbd_e effect, but with Evaporation forcing corrections !!
    # SSS Plotting Params
    comparename_sss         = "SSS_Paper_Draft02"
    expnames_sss            = ["SSS_Draft01_Rerun_QekCorr", "SSS_Draft01_Rerun_QekCorr_NoLbde",
                           "SSS_Draft01_Rerun_QekCorr_NoLbde_NoLbdd", "SSS_CESM"]
    expnames_long_sss       = ["Stochastic Model ($\lambda^e$, $\lambda^d$)","Stochastic Model ($\lambda^d$)","Stochastic Model","CESM1"]
    expnames_short_sss      = ["SM_lbde","SM_no_lbde","SM_no_lbdd","CESM"]
    ecols_sss               = ["magenta","forestgreen","goldenrod","k"]
    els_sss                 = ['dotted',"solid",'dashed','solid']
    emarkers_sss            = ['+',"d","x","o"]
    
    # # SST Comparison (Paper Draft, essentially Updated CSU) !!
    # SST Plotting Params
    comparename_sst     = "SST_Paper_Draft02"
    expnames_sst        = ["SST_Draft01_Rerun_QekCorr","SST_Draft01_Rerun_QekCorr_NoLbdd","SST_CESM"]
    expnames_long_sst   = ["Stochastic Model ($\lambda^d$)","Stochastic Model","CESM1"]
    expnames_short_sst  = ["SM","SM_NoLbdd","CESM"]
    ecols_sst           = ["forestgreen","goldenrod","k"]
    els_sst             = ["solid",'dashed','solid']
    emarkers_sst        = ["d","x","o"]
elif comparename == "Draft3":
    # SSS Plotting Params
    comparename_sss         = "SSS_Paper_Draft03"
    expnames_sss            = ["SSS_Draft03_Rerun_QekCorr", "SSS_Draft03_Rerun_QekCorr_NoLbde",
                               "SSS_Draft03_Rerun_QekCorr_NoLbde_NoLbdd", "SSS_CESM"]
    expnames_long_sss       = ["Stochastic Model ($\lambda^e$, $\lambda^d$)","Stochastic Model ($\lambda^d$)","Stochastic Model","CESM1"]
    expnames_short_sss      = ["SM_lbde","SM_no_lbde","SM_no_lbdd","CESM"]
    ecols_sss               = ["magenta","forestgreen","goldenrod","k"]
    els_sss                 = ['dotted',"solid",'dashed','solid']
    emarkers_sss            = ['+',"d","x","o"]
    
    # # SST Comparison (Paper Draft, essentially Updated CSU) !!
    # SST Plotting Params
    comparename_sst     = "SST_Paper_Draft03"
    expnames_sst        = ["SST_Draft03_Rerun_QekCorr","SST_Draft03_Rerun_QekCorr_NoLbdd","SST_CESM"]
    expnames_long_sst   = ["Stochastic Model ($\lambda^d$)","Stochastic Model","CESM1"]
    expnames_short_sst  = ["SM","SM_NoLbdd","CESM"]
    ecols_sst           = ["forestgreen","goldenrod","k"]
    els_sst             = ["solid",'dashed','solid']
    emarkers_sst        = ["d","x","o"]



expnames        = expnames_sst + expnames_sss
expnames_long   = expnames_long_sst + expnames_long_sss
ecols           = ecols_sst + ecols_sss
els             = els_sst + els_sss
emarkers        = emarkers_sst + emarkers_sss
expvars         = ["SST",] * len(expnames_sst) + ["SSS",] * len(expnames_sss)

cesm_exps       = ["SST_CESM","SSS_CESM","SST_cesm2_pic","SST_cesm1_pic",
                  "SST_cesm1le_5degbilinear","SSS_cesm1le_5degbilinear",]
#%% Load the Dataset (us sm output loader)
# Hopefully this doesn't clog up the memory too much

nexps = len(expnames)
ds_all = []
for e in tqdm.tqdm(range(nexps)):
    
    # Get Experiment information
    expname         = expnames[e]
    varname         = expvars[e]
    
    # For stochastic model output
    ds = dl.load_smoutput(expname,output_path)
    
    if expname in cesm_exps:
        print("Detrending and deseasoning")
        ds = proc.xrdeseason(ds[varname])
        if 'ens' in list(ds.dims):
            ds = ds - ds.mean('ens')
        else:
            ds = proc.xrdetrend(ds)
        ds = xr.where(np.isnan(ds),0,ds) # Sub with zeros for now
    else:
        ds = ds[varname]
    
    ds_all.append(ds)

#%% Select the AMV Region

# Wrapper Function
def calc_amv_ens(ds,bbox_amv):
    # Format dimensions
    if 'ensemble' in list(ds.dims):
        ds = ds.rename(dict(ensemble='ens'))
    if 'ens' in list(ds.dims):
        ds = ds.rename(dict(ens='run'))

    
    # ds = [run x time x lat x lon]
    ds      = ds.transpose('run','lon','lat','time')
    nrun,nlon,nlat,ntime = ds.shape
    nyr = int(ntime/12)
    amvpats = []
    amvids  = []
    for rr in range(nrun):
        dsrun = ds.isel(run=rr).data
        amvid,amvpat = proc.calc_AMVquick(dsrun,ds.lon.data,ds.lat.data,
                                          bbox_amv,dropedge=5)
        
        amvpats.append(amvpat)
        amvids.append(amvid)
    amvpats     = np.array(amvpats).transpose(0,2,1) # [Run x Lon x Lat] --> 3 [Run x Lat x Lon]
    amvids      = np.array(amvids) # [Run x Time]
    
    coords_pat  = dict(run=np.arange(1,nrun+1),lat=ds.lat.data,lon=ds.lon.data)
    coords_id   = dict(run=np.arange(1,nrun+1),time=np.arange(1,nyr+1))
    
    amvpats     = xr.DataArray(amvpats,coords=coords_pat,dims=coords_pat)
    amvids      = xr.DataArray(amvids,coords=coords_id,dims=coords_id)
    
    return amvids,amvpats

bbox_amv = [-80,0,20,60]

cutoff = 5

amvid_exp   = []
amvpat_exp  = []
for ex in range(nexps):
    invar = ds_all[ex] * mask
    
    # Format dimensions
    if 'ensemble' in list(invar.dims):
        invar = invar.rename(dict(ensemble='ens'))
    if 'ens' in list(invar.dims):
        invar = invar.rename(dict(ens='run'))
    invar_maxvar = invar.std('time').max('run')
    invar_maxvar,mask = proc.resize_ds([invar_maxvar,mask])
    mask_var = xr.where(invar_maxvar > cutoff,np.nan,mask)
    
    # Apply NaN if variance exceeds cutoff
    invar = invar * mask_var

    amvid,amvpat=calc_amv_ens(invar,bbox_amv)
    
    amvid_exp.append(amvid)
    amvpat_exp.append(amvpat)
    #dsreg = proc.sel_region_xr(invar,bbin).mean('lat').mean('lon')
    #reg_avgs.append(dsreg)


#%% First Plot (AMV Pattern and Index), Based on REI Plot

pmesh       = False
fsz_title   = 32
cmap_in     = 'cmo.balance'
vnames      = ["SST","SSS"]
vunits      = ["\degree C","psu"]
cints_byvar = [np.arange(-.6,.64,0.04),np.arange(-0.080,0.085,0.005)]

plotids     = [2,0,6,3]
inpats      = [amvpat_exp[ex].mean('run') for ex in plotids]
plotnames   = [expnames_long[ex] for ex in plotids]

plotindex   = [amvid_exp[ex] for ex in plotids]
plotstds    = [ts.var('time').mean('run').data.item() for ts in plotindex]

#inpats  = [amvpat_exp[2],amvpat[]]


fig,axs,_   = viz.init_orthomap(2,2,bboxplot,figsize=(20,14.5))
ii          = 0
for vv in range(2):
    cints = cints_byvar[vv]
    vname=vnames[vv]
    vunit=vunits[vv]
    
    for yy in range(2): # Loop for experinent
        
        # Select Axis
        ax  = axs[vv,yy]
        
        # Set Labels
        blb = viz.init_blabels()
        if yy !=0:
            blb['left']=False
        else:
            blb['left']=True
            viz.add_ylabel(vnames[vv],ax=ax,rotation='horizontal',fontsize=fsz_title)
        if vv == 1:
            blb['lower'] =True
        
        ax.set_title("%s\n$\sigma^2$=%.5f $%s$"% (plotnames[ii],plotstds[ii],vunit),fontsize=fsz_title)
        ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=fsz_tick,blabels=blb,
                                        fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        plotvar = inpats[ii] * mask
        plotvar = xr.where(np.abs(plotvar)>10,np.nan,plotvar)
        
        if pmesh:
            pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj)
            cb      = viz.hcbar(pcm,ax=ax)
        else:
            pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                                  cmap=cmap_in,levels=cints,extend='both')
            cl =ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                                  colors="k",linewidths=0.75,levels=cints)
            ax.clabel(cl,fontsize=fsz_tick)
            
            if yy == 1:
                cb      = fig.colorbar(pcm,ax=axs[vv,:].flatten(),fraction=0.025,pad=0.01)
        
                cb.ax.tick_params(labelsize=fsz_tick)
                cb.set_label("$%s$ per $\sigma_{AMV,%s}$" % (vunit,vname),fontsize=fsz_axis)
        viz.label_sp(ii,ax=ax,fontsize=fsz_title,alpha=0.75)
        ii += 1
        #if vv == 0
savename = "%sAMV_Patterns_CESM_v_SM_Draft03.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Plot The Indices

fig,axs = plt.subplots(4,1,constrained_layout=True,figsize=(12.5,10))

for ex in range(4):
    vunit = vunits[ex%2]
    ax = axs[ex]
    tsin = plotindex[ex]
    nrun = tsin.shape[0]
    for rr in range(nrun):
        ax.plot(tsin.isel(run=rr),alpha=0.55)
    ax.plot(tsin.mean('run'),alpha=1,color="k")  
    ax.set_title("%s ($\sigma^2$=%.5f $%s$)"% (plotnames[ex],plotstds[ex],vunit),fontsize=12)
    #ax.set_title(plotnames[ex])
    ax.set_xlim([0,86])
savename = "%sAMV_Indices_first86yr_CESM_v_SM_Draft03.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Perform Regression of AMV Index to SSS


amvid_sm = amvid_exp[0] # SST
print("Using AMV Indices from %s" % expnames_long[0])
regvar_sm = ds_all[3]
print("Regressing SSS from %s" % expnames_long[3])

nrun = amvid_sm.shape[0]
sss_pats = []
for rr in range(nrun):
    ts = amvid_sm.isel(run=rr)
    varrun = regvar_sm.sel(run=rr).groupby('time.year').mean('time')
    varrun = varrun.transpose('lon','lat','year').data
    
    regrout = proc.regress2ts(varrun,ts.data,)
    coords  = dict(lon=regvar_sm.lon,lat=regvar_sm.lat)
    da_out  = xr.DataArray(regrout,coords=coords,dims=coords)
    sss_pats.append(da_out.transpose('lat','lon'))
sss_pats = xr.concat(sss_pats,dim='run')

sss_pats.mean('run').plot(vmin=-.8,vmax=.8,cmap='cmo.balance')

#%% Compute the pattern correlation to the ensemble average

# SST
amvpat_sm   = amvpat_exp[0]
amvpat_cesm = amvpat_exp[2].mean('run')

# Compute the pattern correlation by run
pcorr_sm_sst=[]
for rr in range(10):
    pcc=proc.patterncorr(amvpat_cesm.data,amvpat_sm.isel(run=rr).data)
    pcorr_sm_sst.append(pcc)

plt.bar(np.arange(1,11,1),pcorr_sm_sst),plt.title("PatternCorr with CESM1 Ens Avg by Run (SST)")


# Copied for SSS
amvpat_sm   = amvpat_exp[3]
amvpat_cesm = amvpat_exp[-1].mean('run')

# Compute the pattern correlation by run
pcorr_sm_sss=[]
for rr in range(10):
    pcc=proc.patterncorr(amvpat_cesm.data,amvpat_sm.isel(run=rr).data)
    pcorr_sm_sss.append(pcc)

plt.bar(np.arange(1,11,1),pcorr_sm_sss),plt.title("PatternCorr with CESM1 Ens Avg by Run (SSS)")


#%% Pattern Difference between members


vv = 1

cints       = cints_byvar[vv]
fig,axs,_   = viz.init_orthomap(2,5,bboxplot,figsize=(28,6.5))

for rr in range(10):
    ax = axs.flatten()[rr]
    ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=fsz_tick,
                            fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    

    if vv == 0:
        plotvar = amvpat_exp[0].isel(run=rr)
    elif vv == 1:
        plotvar = amvpat_exp[3].isel(run=rr)
    pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                          cmap=cmap_in,levels=cints,extend='both')
    cl =ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                          colors="k",linewidths=0.75,levels=cints)
    ax.clabel(cl,fontsize=fsz_tick)
    ax.set_title("Run %02i" % (rr+1))
savename = "%sAMV_Pat_byRun_%s_SM_Draft03.png" % (figpath,vnames[vv])
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%%

pointmode  = True  # Set to False to take a regional average

if pointmode:
    # # Max REI (except in Sargasso Sea)
    # points  = [[-65,36],
    #            [-34,46],
    #            [-36,58],
    #            ]
    
    # Intraregional Output (High REI, low Advection)
    points = [[-65,36], #SAR
              [-39,44], #NAC
              [-35,53], #IRM
              ]
    
    locstring_all = [proc.make_locstring(pt[0],pt[1]) for pt in points]
    npts    = len(points)
else:
    
    # Same Bounding Boxes as the rparams
    bboxes  = [
        [-70,-55,35,40],
        [-40,-30,40,50],
        [-40,-25,50,60],
        ]
    
    locstring_all = [proc.make_locstring_bbox(bb) for bb in bboxes]
    npts    = len(bboxes)

pointnames = ["SAR",
              "NAC",
              "IRM"]



# # Make a Locator Plot
# fig,ax,mdict = viz.init_orthomap(1,1,bboxplot=bboxplot,figsize=(28,10))
# ax           = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

# for nn in range(npts):
#     origin_point = points[nn]
#     ax.plot(origin_point[0],origin_point[1],transform=proj,marker="d",markersize=15)

# invars       = ds_all + ds_ugeos
# invars_names = expnames_long + ugeos_names 

#%% lets Perform the analysis
# First, Compute the Regional Averages
#totalvars_pt = np.zeros((npts,len(invars)))
#monvars_pt  = np.zeros((npts,len(invars),12))

ravg_all = []
for ex in tqdm.tqdm(range(nexps)): # Looping for each dataset
    
    reg_avgs = []
    for nn in range(npts): # Looping for each region or point
    
        invar = ds_all[ex] * mask # Added a mask application prior to computations
        
        # Format dimensions
        if 'ensemble' in list(invar.dims):
            invar = invar.rename(dict(ensemble='ens'))
        if 'run' in list(invar.dims):
            invar = invar.rename(dict(run='ens'))
        
        if pointmode:
            bbin = points[nn]
            dsreg = proc.selpt_ds(invar,bbin[0],bbin[1])
        else:
            bbin  = bboxes[nn]
            dsreg = proc.sel_region_xr(invar,bbin).mean('lat').mean('lon')
        reg_avgs.append(dsreg)
        
    reg_avgs = xr.concat(reg_avgs,dim='region')
    reg_avgs['region'] = pointnames
    
    ravg_all.append(reg_avgs)