#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Instead of Selecting a Whole Region, Just look at features at individual points.

Copied sections from [viz_missing_terms]

Created on Fri Aug 23 10:33:58 2024

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
comparename     = "Draft3"#"Paper_Draft02_AllExps"

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


# Set Plotting Options


darkmode = True
if darkmode:
    dfcol = "w"
    transparent = True
    plt.style.use('dark_background')
else:
    dfcol = "k"
    transparent = False
    plt.style.use('default')
for cc in range(len(ecols)):
    if "CESM" in expnames[cc]:
        ecols[cc] = dfcol
    

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

#%% Select a few points

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


#%% Compute the metrics

dtin     = 3600*24*365

tsm_all  = []
spec_all = []
for ex in tqdm.tqdm(range(nexps)):
    expname = expnames[ex]
    
    in_ds = ravg_all[ex]
    print(in_ds.shape)
    
    tsm_byreg  = []
    spec_byreg = []
    for nn in range(npts):
        
        ds_reg = in_ds.isel(region=nn) # [Ens x Time]
        nens   = ds_reg.shape[0]
        invar  = [ds_reg.isel(ens=e).data for e in range(nens)]
        if expname in cesm_exps:
            nsmooth = 2
        else:
            nsmooth = 20
        tsm    = scm.compute_sm_metrics(invar,nsmooth=nsmooth)
        
        tsm_byreg.append(tsm)
        
        # Take annual average
        rsst_ann    = ds_reg.groupby('time.year').mean('time')
        tsens       =  [rsst_ann.isel(ens=e).values for e in range(nens)]
        specout     = scm.quick_spectrum(tsens, nsmooth, 0.10, dt=dtin,make_arr=True,return_dict=True)
        spec_byreg.append(specout)
        
    spec_all.append(spec_byreg)
    tsm_all.append(tsm_byreg)

# ====================================
#%% Visualize ACF (Draft 02)
# ====================================

fsz_leg = 10

lags    = np.arange(37)
xtks    = np.arange(0,37,3)
kmonth  = 1
lw      = 2.5
vnames  = ["SST","SSS"]
vunits  = ["\degree C","psu"]
fig,axs = plt.subplots(2,3,constrained_layout=True,figsize=(16,8.5))

#dfcol   = "k"

# Set up Axes First
ii = 0
for vv in range(2):
    vname = vnames[vv]
    for rr in range(3):
        ax   = axs[vv,rr]
        ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")
        # SEt up plot and axis labels
        if rr != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("%s Correlation" % (vname),fontsize=fsz_axis)
        if not (rr == 1 and vv == 1):
            ax.set_xlabel("")
        if vv == 0:
            ax.set_title("%s \n%s" % (regions_long[rr],locstring_all[rr][1]),fontsize=fsz_axis)
        
        ax.set_ylim([-0.1,1.1])
        
        viz.label_sp(ii,alpha=0,ax=ax,fontsize=fsz_title,fontcolor=dfcol)
        # Labe with the region name, which is not necessary
        # ax=viz.label_sp(regions_long[rr],ax=ax,x=0,y=.125,alpha=0,fig=fig,
        #              labelstyle="%s",usenumber=True,fontsize=fsz_title,fontcolor=dfcol,)
        ii+=1


# Plot the Variables
legflag = False
for ex in range(nexps):
    
    vname = expvars[ex]
    
    if vname == 'SST':
        vv = 0
    elif vname == 'SSS':
        vv = 1 
    
    for rr in range(3):
        
        ax     = axs[vv,rr]
        tsm_in = tsm_all[ex][rr]
        
        # Set up the plot and axis labels
        plotacf = np.array(tsm_in['acfs'][kmonth]) # [Ens x Lag]
        mu       = plotacf.mean(0)
        std      = proc.calc_stderr(plotacf,0)
        ax.plot(lags,mu,color=ecols[ex],label=expnames_long[ex],lw=lw,marker=emarkers[ex],markersize=4)
        ax.fill_between(lags,mu-std,mu+std,color=ecols[ex],alpha=0.15,zorder=2,)
        

ax = axs[0,0]
ax.legend(fontsize=fsz_leg,loc='upper right')

ax = axs[1,0]
ax.legend(fontsize=fsz_leg,loc='upper right')

savename = "%sPoint_Metrics_ACF_mon%02i.png" % (figpath,kmonth+1)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
    transparent=True
else:
    transparent=False
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)


# =========================
#%% Plot the ACF Maps
# =========================

# Copied from compare_regional_metrics
fsz_tick         = 15
fsz_axis         = 16

vlms             = [-0.5,0.5]
xtks             = np.arange(0,37,3)

comparename_acf  = "SM_v_CESM" #"lbd_e"#"SM_v_CESM"

fig,axs = plt.subplots(2,3,constrained_layout=True,figsize=(22,6.5))

# Set up axes
# Set up Axes First
ii = 0
for vv in range(2):
    
    vname = vnames[vv]
    for rr in range(3):
        
        ax   = axs[vv,rr]
        
        ax.set_xticks(xtks)
        ax.tick_params(labelsize=fsz_tick)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        if rr != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Month of\n%s Anomaly" % (vname),fontsize=fsz_axis)
        if not (rr == 1 and vv == 1):
            ax.set_xlabel("")
        if vv == 0:
            ax.set_title("%s \n%s" % (regions_long[rr],locstring_all[rr][1]),fontsize=fsz_axis)
        
        viz.label_sp(ii,alpha=0,ax=ax,fontsize=fsz_title,fontcolor=dfcol)
        # Labe with the region name, which is not necessary
        # ax=viz.label_sp(regions_long[rr],ax=ax,x=0,y=.125,alpha=0,fig=fig,
        #              labelstyle="%s",usenumber=True,fontsize=fsz_title,fontcolor=dfcol,)
        ii+=1


# Plot the variables
for vv in range(2):
    
    if vv == 0:
        id_sm       = 0 # SST (with lbdd)
        id_cesm     = 2 # CESM (SST)
    elif vv == 1:
        id_sm       = 3 # SST (with lbdd)
        id_cesm     = 6 # CESM (SSS)
        
    diffstr    = "%s - %s" % (expnames_long[id_sm],expnames_long[id_cesm])
    diffstr_fn = "%s_v_%s" % (expnames[id_sm],expnames[id_cesm])
    print("Computing differences for:  %s" % (diffstr))
    
    for rr in range(3):
        ax = axs[vv,rr]
        
        acfmap_sm   = np.nanmean(np.array(tsm_all[id_sm][rr]['acfs']),1) # [Kmonth x run x lag] --> [kmonth x lag]
        acfmap_cesm = np.nanmean(np.array(tsm_all[id_cesm][rr]['acfs']),1) # [Kmonth x run x lag] --> [kmonth x lag]
          
        
        pv          = acfmap_sm - acfmap_cesm
        
        pcm = ax.pcolormesh(lags,mons3,pv,
                            cmap='cmo.balance',vmin=vlms[0],vmax=vlms[1],
                            edgecolors="lightgray")
        
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Difference in Monthly Lag Correlation",fontsize=fsz_axis)


savename = "%sPoint_Metrics_REMMap.png" % (figpath)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

# ====================================
#%% Monthly Variance (Draft 02)
# ====================================

skip_exps = [5,]
fig,axs   = viz.init_monplot(2,3,constrained_layout=True,figsize=(16,8.5))
share_ylm = False
barcol = "cornflowerblue"
add_bar = True

# Set up Axes First
ii = 0
for vv in range(2):
    vname = vnames[vv]
    for rr in range(3):
        ax   = axs[vv,rr]
        
        # SEt up plot and axis labels
        if rr != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("%s Variance $[%s]^2$" % (vname,vunits[vv]),fontsize=fsz_axis)
        if not (rr == 1 and vv == 1):
            ax.set_xlabel("")
        
        if vv == 0:
            ax.set_title("%s \n%s" % (regions_long[rr],locstring_all[rr][1]),fontsize=fsz_axis)
        
        ax.tick_params(labelcolor=dfcol,color=dfcol)
        ax.spines['left'].set_color(dfcol)
        ax.spines['bottom'].set_color(dfcol)
        #ax.spines['right'].set_color(barcol)
        
        # Add Bar plots
        if add_bar: # Add Bars in the Background
            
            if vv == 0: # SST
                cesm_mv = np.array(tsm_all[2][rr]['monvars']).mean(0)
                sm_mv   = np.array(tsm_all[0][rr]['monvars']).mean(0)
                barcol  = "forestgreen"
            elif vv == 1: # SSS
                cesm_mv = np.array(tsm_all[6][rr]['monvars']).mean(0)
                sm_mv   = np.array(tsm_all[3][rr]['monvars']).mean(0)
                barcol  = "magenta"
            plotvar = sm_mv/cesm_mv
            
            ax2 = ax.twinx()
            
            ax2.bar(mons3,plotvar*100,color=barcol,alpha=0.25,edgecolor=dfcol)
            ax2.set_ylim([0,200])
            ax2.axhline([100],ls='dashed',color=barcol,lw=0.75)
            ax2.set_zorder(ax.get_zorder()-1)
            ax2.tick_params(labelsize=fsz_tick-2)
            # Adjust COlor
            ax2.yaxis.label.set_color(barcol)  
            ax2.tick_params(axis='y', colors=barcol)
            #ax2.spines['right'].set_color(barcol)
                            
            ax.patch.set_visible(False)
            
            if rr == 2:
                ax2.set_ylabel("%"+r" Explained ($\frac{Stochastic \,\, Model}{CESM1}$)",fontsize=fsz_axis-6)
                
        viz.label_sp(ii,alpha=0,ax=ax,fontsize=fsz_title,fontcolor=dfcol)
        # Labe with the region name, which is not necessary
        # ax=viz.label_sp(regions_long[rr],ax=ax,x=0,y=.125,alpha=0,fig=fig,
        #              labelstyle="%s",usenumber=True,fontsize=fsz_title,fontcolor=dfcol,)
        ii+=1
        
        
        

# Plot the Variables
legflag = False
for ex in range(nexps):
    
    if ex in skip_exps:
        continue
    
    vname = expvars[ex]
    
    if vname == 'SST':
        vv = 0
        ylm     = [0,1]
    elif vname == 'SSS':
        vv = 1 
        ylm     = [0,0.030]
   
    
    for rr in range(3):
        
        ax     = axs[vv,rr]
        tsm_in = tsm_all[ex][rr]
        
        # Set up the plot and axis labels
        plotvar = np.array(tsm_in['monvars']) # [Ens x Lag]
        mu       = plotvar.mean(0)
        std      = proc.calc_stderr(plotvar,0)
        ax.plot(mons3,mu,color=ecols[ex],label=expnames_long[ex],lw=lw,marker=emarkers[ex],markersize=4)
        ax.fill_between(mons3,mu-std,mu+std,color=ecols[ex],alpha=0.15,zorder=2,)
        
        if share_ylm:
            ax.set_ylim(ylm)
    
ax = axs[0,1]
ax.legend(fontsize=fsz_leg,loc='upper right',framealpha=0.1)

ax = axs[1,0]
ax.legend(fontsize=fsz_leg,loc=(.025,.65),framealpha=0.1)

# for vv in range(2):
#     if vv == 0:
#         id_cesm = 0
#         id_sm   = 2
#         vname   = "SST"
#         smcol   = 'forestgreen'
#         vunit   = "\degree C"
        
#         ylm     = [0,1.0]
        
#     elif vv == 1:
#         id_cesm = 1
#         id_sm   = 3
#         vname   = "SSS"
#         smcol   = 'violet'
#         vunit   = "psu"
#         ylm     = [0,0.030]
    
#     for rr in range(3):
        
#         # Set up the plot and axis labels
#         ax = axs[vv,rr]
#         if rr != 0:
#             ax.set_ylabel("")
#         else:
#             ax.set_ylabel("%s Variance ($%s^2$)" % (vname,vunit),fontsize=fsz_axis)
#         if not (rr == 1 and vv == 1):
#             ax.set_xlabel("")
#         if vv == 0:
#             ax.set_title(locstring_all[rr][1],fontsize=fsz_axis)
        
#         # Plot CESM1
#         cesm_plot = np.array(tsm_all[id_cesm][rr]['monvars']) # [Ens x Lag]
#         mu       = cesm_plot.mean(0)
#         std      = proc.calc_stderr(cesm_plot,0)
#         ax.plot(mons3,mu,color='k',label="CESM1",lw=lw,marker="d",markersize=4)
#         ax.fill_between(mons3,mu-std,mu+std,color="k",alpha=0.15,zorder=2,)
        
#         # Plot Stochastic Model
#         sm_plot = np.array(tsm_all[id_sm][rr]['monvars']) # [Ens x Lag]
#         mu       = sm_plot.mean(0)
#         std      = proc.calc_stderr(sm_plot,0)
#         ax.plot(mons3,mu,color=smcol,label="Stochastic Model",lw=lw,marker='s',markersize=4)
#         ax.fill_between(mons3,mu-std,mu+std,color=smcol,alpha=0.15,zorder=2,)
#         ax.set_ylim(ylm)  
#         if rr == 0:
#             ax.legend(fontsize=fsz_tick-6)

savename = "%sPoint_Metrics_Monvar.png" % (figpath)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
    transparent=True
else:
    transparent=False
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)

# ====================================
#%% Power Spectra 
# ====================================

def init_logspec(nrows,ncols,figsize=(10,4.5),ax=None,
                 xtks=None,dtplot=None,
                 fsz_axis=16,fsz_ticks=14,toplab=True,botlab=True):
    if dtplot is None:
        dtplot     = 3600*24*365  # Assume Annual data
    if xtks is None:
        xpers      = [100, 50,25, 20, 15,10, 5, 2]
        xtks       = np.array([1/(t) for t in xpers])
    
    if ax is None:
        newfig = True
        fig,ax = plt.subplots(nrows,ncols,constrained_layout=True,figsize=figsize)
    else:
        newfig = False
        
    ax = viz.add_ticks(ax)
    
    #ax.set_xticks(xtks,labels=xpers)
    ax.set_xscale('log')
    ax.set_xlim([xtks[0], xtks[-1]])
    if botlab:
        ax.set_xlabel("Frequency (Cycles/Year)",fontsize=fsz_axis)
    ax.tick_params(labelsize=fsz_ticks)
    
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xticks(xtks,labels=xpers,fontsize=fsz_ticks)
    ax2.set_xlim([xtks[0], xtks[-1]])
    if toplab:
        ax2.set_xlabel("Period (Years)",fontsize=fsz_axis)
    ax2.grid(True,ls='dotted',c="gray")
    
    if newfig:
        return fig,ax
    return ax
        


dtplot=dtin

fig,axs = plt.subplots(2,3,constrained_layout=True,figsize=(16,8.5))

# Set up Axes First
ii = 0
for vv in range(2):
    for rr in range(3):
        
        # Set up the plot and axis labels
        ax = axs[vv,rr]
        
        if vv == 0:
            toplab=True
            botlab=False
        else:
            toplab=False
            botlab=True
        
        ax = init_logspec(1,1,ax=ax,toplab=toplab,botlab=botlab)
        
        if vv == 0:
            ax.set_title("%s \n%s" % (regions_long[rr],locstring_all[rr][1]),fontsize=fsz_axis)
            
        if rr == 0:
            ax.set_ylabel("Power ($%s^2 \, cpy^{-1}$)" % (vunits[vv]),fontsize=fsz_axis)
        
        viz.label_sp(ii,alpha=0,ax=ax,fontsize=fsz_title,fontcolor=dfcol)
        # Labe with the region name, which is not necessary
        # ax=viz.label_sp(regions_long[rr],ax=ax,x=0,y=.125,alpha=0,fig=fig,
        #              labelstyle="%s",usenumber=True,fontsize=fsz_title,fontcolor=dfcol,)
        ii+=1


# Now Plot the Variables
legflag = False
for ex in range(nexps):
    
    if ex in skip_exps:
        continue
    
    vname = expvars[ex]
    
    if vname == 'SST':
        vv = 0
    elif vname == 'SSS':
        vv = 1
    
    for rr in range(3):
        
        ax     = axs[vv,rr]
        
        c       = ecols[ex]
        emk     = emarkers[ex]
        ename   = expnames_long[ex]
        
        # Read out spectra variables
        svarsin = spec_all[ex][rr]
        P       = svarsin['specs']
        freq    = svarsin['freqs']
        cflab   = "Red Noise"
        CCs     = svarsin['CCs']
        
        print(P.shape)
        print(freq.shape)
        
        # Convert units
        freq     = freq[0,:] * dtplot
        P        = P / dtplot
        Cbase    = CCs.mean(0)[:, 0]/dtplot
        Cupbound = CCs.mean(0)[:, 1]/dtplot
        
        # Plot Ens Mean
        mu    = P.mean(0)
        sigma = P.std(0)
        
        # Plot Spectra
        ax.loglog(freq, mu, c=c, lw=2.5,
                label=ename, marker=emk, markersize=1)
        
        # Plot Significance
        if ex == 2:
            labc1 = cflab
            labc2 = "95% Confidence"
        else:
            labc1=""
            labc2=""
        ax.loglog(freq, Cbase, color=c, ls='solid', lw=1.2, label=labc1)
        ax.loglog(freq, Cupbound, color=c, ls="dotted",
                lw=2, label=labc2)
        
        
ax = axs[0,0]
ax.legend(fontsize=fsz_leg,loc='lower left')

ax = axs[1,0]
ax.legend(fontsize=fsz_leg,loc='lower left')

savename = "%sPoint_Metrics_Spectra.png" % (figpath)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=False)



    

        
        
        





