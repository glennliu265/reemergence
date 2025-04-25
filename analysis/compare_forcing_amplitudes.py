#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load in forcing amplitudes with the missing terms
Written for SSS Paper Revisions

Copied upper section of viz_inputs_paper_draft.py

Created on Fri Apr 25 09:11:45 2025

@author: gliu

"""



import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
import matplotlib.patheffects as PathEffects
from cmcrameri import cm

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
machine    = "Astraeus"
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
rawpath     = pathdict['raw_path']

# ----------------------------------
#%% User Edits
# ----------------------------------

# Indicate the experiment
expname_sss         = "SSS_Revision_Qek_TauReg"#"#"SSS_Draft03_Rerun_QekCorr"#SSS_EOF_LbddCorr_Rerun_lbdE_neg" #"SSS_EOF_Qek_LbddEnsMean"#"SSS_EOF_Qek_LbddEnsMean"
expname_sst         = "SST_Revision_Qek_TauReg"#"SST_Draft03_Rerun_QekCorr"#"SST_EOF_LbddCorr_Rerun"


# Constants
dt          = 3600*24*30 # Timestep [s]
cp          = 3850       # 
rho         = 1026    #`23      # Density [kg/m3]
B           = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L           = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

fsz_tick    = 18
fsz_title   = 24
fsz_axis    = 22


debug       = False



#%% Add some functions to load (and convert) inputs

def stdsqsum(invar,dim):
    return np.sqrt(np.nansum(invar**2,dim))

def stdsq(invar):
    return np.sqrt(invar**2)

def stdsqsum_da(invar,dim):
    return np.sqrt((invar**2).sum(dim))

def convert_ds(invar,lat,lon,):
    
    if len(invar.shape) == 4: # Include mode
        nmode = invar.shape[0]
        coords = dict(mode=np.arange(1,nmode+1),mon=np.arange(1,13,1),lat=lat,lon=lon)
    else:
        coords = dict(mon=np.arange(1,13,1),lat=lat,lon=lon)
    
    return xr.DataArray(invar,coords=coords,dims=coords)

def compute_detrain_time(kprev_pt):
    
    detrain_mon   = np.arange(1,13,1)
    delta_mon     = detrain_mon - kprev_pt#detrain_mon - kprev_pt
    delta_mon_rev = (12 + detrain_mon) - kprev_pt # Reverse case 
    delta_mon_out = xr.where(delta_mon < 0,delta_mon_rev,delta_mon) # Replace Negatives with 12+detrain_mon
    delta_mon_out = xr.where(delta_mon_out == 0,12.,delta_mon_out) # Replace deepest month with 12
    delta_mon_out = xr.where(kprev_pt == 0.,np.nan,delta_mon_out)
    
    return delta_mon_out


#%% Plotting Params

mpl.rcParams['font.family'] = 'Avenir'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
#lon                         = daspecsum.lon.values
#lat                         = daspecsum.lat.values
mons3                       = proc.get_monstr()


plotver = "rev1" # [sub1]

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


# Get Point Info
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']


#%% Check and Load Params (copied from run_SSS_basinwide.py on 2024-03-04)

# Load the parameter dictionary
expparams_byvar     = []
paramset_byvar      = []
convdict_byvar      = []
convda_byvar        = []
for expname in [expname_sst,expname_sss]:
    
    print("Loading inputs for %s" % expname)
    
    expparams_raw   = np.load("%s%s/Input/expparams.npz" % (output_path,expname),allow_pickle=True)
    
    expparams       = scm.repair_expparams(expparams_raw)
    
    # Get the Variables (I think only one is really necessary)
    #expparams_byvar.append(expparams.copy())
    
    # Load Parameters
    paramset = scm.load_params(expparams,input_path)
    inputs,inputs_ds,inputs_type,params_vv = paramset
    

    # Convert to the same units
    convdict                               = scm.convert_inputs(expparams,inputs,return_sep=True)
    
    # Get Lat/Lon
    ds = inputs_ds['h']
    lat = ds.lat.data
    lon = ds.lon.data
    
    # Convert t22o DataArray
    varkeys = list(convdict.keys())
    nk = len(varkeys)
    conv_da = {}
    for nn in range(nk):
        #print(nn)
        varkey = varkeys[nn]
        invar  = convdict[varkey]
        conv_da[varkey] =convert_ds(invar,lat,lon)
        
    
    # Append Output
    expparams_byvar.append(expparams)
    paramset_byvar.append(paramset)
    convdict_byvar.append(convdict)
    convda_byvar.append(conv_da)

# --------------------------------------
#%% Load kprev and compute convert lbd-d
# --------------------------------------

lbdd_sst    = paramset_byvar[0][1]['lbd_d']
lbdd_sss    = paramset_byvar[1][1]['lbd_d']

# Compute Detrainment Times
ds_kprev    = xr.open_dataset(input_path + "mld/CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc")
delta_mon   = xr.apply_ufunc(
        compute_detrain_time,
        ds_kprev.kprev,
        input_core_dims=[['mon']],
        output_core_dims=[['mon']],
        vectorize=True,
        )

lbdd_sst_conv = -delta_mon / np.log(lbdd_sst)
lbdd_sss_conv = -delta_mon / np.log(lbdd_sss)

#%% Load (or compute) the SST Evaporation Feedback

# Load lbd_e
lbd_e    = xr.open_dataset(input_path + "forcing/" + expparams_byvar[1]['lbd_e']).lbd_e.load() # [mon x lat x lon]
lbd_e    = proc.sel_region_xr(lbd_e,bbox=expparams_byvar[1]['bbox_sim'])

# Convert [sec --> mon]
lbd_emon = lbd_e * dt
#lbd_emon = lbd_emon.transpose('lon','lat','mon')#.values

# End copy of viz_inputs_paper_draft ------------------------------------------

#%% Also Load geostrophic term (calculated in scrap/viz_total/ugeo/)

revpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/revision_data/"
vnames  = ["SST","SSS"]
ugeo_monvars = []
for vv in range(2):
    outname    = "%sCESM1_Ugeo_Transport_MonStd_%s.nc" % (revpath,vnames[vv])
    ds = xr.open_dataset(outname).load()[vnames[vv]]
    ugeo_monvars.append(ds)
    

# Get the amplitude of the forcing
ugeo_amp        = [ds.mean('ens').mean('month') for ds in ugeo_monvars]
ugeo_amp_FM     = [ds.mean('ens').isel(month=[1,2]).mean('month') for ds in ugeo_monvars]

#%% Check Amplitude of geostrophic transport term

dtmon   = 3600*24*30
vv      = 1
pmesh   = False
if vv == 0:
    cints = np.arange(-.5,.55,.05)
else:
    cints = np.arange(-.1,0.11,0.01)
    

fig,ax  = viz.init_regplot(bboxin=bboxplot)

plotvar = ugeo_amp[vv] * dtmon
if pmesh:
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                            vmin = cints[0],vmax=cints[-1],
                            transform=proj)
else:
    pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                            levels=cints,
                            transform=proj)
    
cb      = viz.hcbar(pcm,ax=ax)


    
#%% Get Corresponding amplitudes of each of the forcings (Start Copy from Viz_input_parameters)

viz_total_include_correction = True # Set to True to include correction in total forcing visualization

selmons        = [1,2] # Indices
monstr         = proc.mon2str(selmons)

Fprime          = convda_byvar[0]['Fprime']                  # [Mode x Mon x Lat x Lon]
Fprime_corr     = convda_byvar[0]['correction_factor']       # [Mon x Lat x Lon]
lhflx           = convda_byvar[1]['LHFLX']                   # [Mode x Mon x Lat x Lon]
lhflx_corr      = convda_byvar[1]['correction_factor_evap']  # [Mon x Lat x Lon]
prec            = convda_byvar[1]['PRECTOT']                 # [Mode x Mon x Lat x Lon]
prec_corr       = convda_byvar[1]['correction_factor_prec']  # [Mon x Lat x Lon]
qek_sst         = convda_byvar[0]['Qek']                     # [Mode x Mon x Lat x Lon]
qek_sst_corr    = convda_byvar[0]['correction_factor_Qek']   # [Mon x Lat x Lon]
qek_sss         = convda_byvar[1]['Qek']                     # [Mode x Mon x Lat x Lon]
qek_sss_corr    = convda_byvar[1]['correction_factor_Qek']   # [Mon x Lat x Lon]

# Compute the Percentage of the correction (Corr% = Corr / (Corr + EOF))
Fprime_std_total  = stdsqsum_da(Fprime.isel(mon=selmons).mean('mon'),'mode')
Fprime_corr_perc  = (Fprime_corr.isel(mon=selmons).mean('mon')) / (Fprime_corr.isel(mon=selmons).mean('mon') + Fprime_std_total) *100

lhflx_std_total   = stdsqsum_da(lhflx.isel(mon=selmons).mean('mon'),'mode')
lhflx_corr_perc   = (lhflx_corr.isel(mon=selmons).mean('mon')) / (lhflx_corr.isel(mon=selmons).mean('mon') + lhflx_std_total) *100

prec_std_total    = stdsqsum_da(prec.isel(mon=selmons).mean('mon'),'mode')
prec_corr_perc    = (prec_corr.isel(mon=selmons).mean('mon')) / (prec_corr.isel(mon=selmons).mean('mon') + prec_std_total) *100

qek_sst_std_total = stdsqsum_da(qek_sst.isel(mon=selmons).mean('mon'),'mode')
qek_sst_corr_perc = (qek_sst_corr.isel(mon=selmons).mean('mon')) / (qek_sst_corr.isel(mon=selmons).mean('mon') + qek_sst_std_total ) *100

qek_sss_std_total = stdsqsum_da(qek_sss.isel(mon=selmons).mean('mon'),'mode')
qek_sss_corr_perc = (qek_sss_corr.isel(mon=selmons).mean('mon')) / (qek_sss_corr.isel(mon=selmons).mean('mon') + qek_sss_std_total ) *100


# Try plotting the total forcing (eof + correction) for each case
if viz_total_include_correction:
    Fprime_std_total  = stdsqsum_da(Fprime.isel(mon=selmons).mean('mon'),'mode') + Fprime_corr.isel(mon=selmons).mean('mon')
    qek_sst_std_total = stdsqsum_da(qek_sst.isel(mon=selmons).mean('mon'),'mode') + qek_sst_corr.isel(mon=selmons).mean('mon')
    
    lhflx_std_total  = stdsqsum_da(lhflx.isel(mon=selmons).mean('mon'),'mode') + lhflx_corr.isel(mon=selmons).mean('mon')
    prec_std_total   = stdsqsum_da(prec.isel(mon=selmons).mean('mon'),'mode') + prec_corr.isel(mon=selmons).mean('mon')
    qek_sss_std_total = stdsqsum_da(qek_sss.isel(mon=selmons).mean('mon'),'mode') + qek_sss_corr.isel(mon=selmons).mean('mon')
    
    
# Take EOF1, EOF2, Conversion Factor, and Total
Fprime_in = [Fprime.isel(mode=0,mon=selmons).mean('mon'),
             Fprime.isel(mode=1,mon=selmons).mean('mon'),
             Fprime_std_total,
             Fprime_corr_perc,
             ]

evap_in = [lhflx.isel(mode=0,mon=selmons).mean('mon'),
           lhflx.isel(mode=1,mon=selmons).mean('mon'),
           lhflx_std_total,
           np.abs(lhflx_corr_perc),
           ]

prec_in = [prec.isel(mode=0,mon=selmons).mean('mon'),
           prec.isel(mode=1,mon=selmons).mean('mon'),
           prec_std_total,
           prec_corr_perc,
           ]

qek_sst_in = [qek_sst.isel(mode=0,mon=selmons).mean('mon'),
              qek_sst.isel(mode=1,mon=selmons).mean('mon'),
              qek_sst_std_total,
              qek_sst_corr_perc,
              ]

qek_sss_in = [qek_sss.isel(mode=0,mon=selmons).mean('mon'),
              qek_sss.isel(mode=1,mon=selmons).mean('mon'),
              qek_sss_std_total,
              qek_sss_corr_perc,
              ]

rownames       = ["EOF 1", "EOF 2", "EOF Total", r"$\frac{Correction \,\, Factor}{Total \,\, Forcing}$"]
if plotver == "rev1":
    
    vnames_force   = ["Stochastic Heat Flux Forcing\n"+r"($\frac{1}{\rho C_p h} F_N'$, SST)",
                      "Ekman Forcing\n($Q_{ek,T},SST)$",
                      "Evaporation\n"+r"($\frac{\overline{S}}{\rho h L} F_L'$,SSS)",
                      "Precipitation\n"+r"($\frac{\overline{S}}{\rho h} P'$,SSS)",
                      "Ekman Forcing\n($Q_{ek,S},SSS)'$"]
    
else:
    vnames_force   = ["Stochastic Heat Flux Forcing\n"+r"($\frac{F'}{\rho C_p h}$, SST)",
                      "Ekman Forcing\n($Q_{ek}'$, SST)",
                      "Evaporation\n"+r"($\frac{\overline{S} q_L'}{\rho h L}$,SSS)",
                      "Precipitation\n"+r"($\frac{\overline{S} P'}{\rho h}$,SSS)",
                      "Ekman Forcing\n($Q_{ek}'$, SSS)"]
plotvars_force = [Fprime_in,qek_sst_in,evap_in,prec_in,qek_sss_in,]

#%% Compare Forcing Ratios for SST
fsz_ticks    = 12
sst_forcings = [Fprime_std_total,qek_sst_std_total]
sst_fnames   = [r"F_N'",r"Q_{ek,T}"]
sst_fnames_short = ["Fprime","Qek"]
refvar       = ugeo_amp_FM[0] * dtmon

cintsrat = np.array([0,0.25,0.5,1,1.5,2,3,5,10,25,50])
clabs    = ["%.1fx" % (ss) for ss in cintsrat]
for ii in range(2):
    
    fig,ax  = viz.init_regplot(bboxin=bboxplot)
    
    # Compute Ratio (i.e. how much larger is geostrophic adv compared to the forcing)
    plotvar = refvar/ sst_forcings[ii]
    
    # pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
    #                       #levels=cintsrat,
    #                       cmap='cmo.balance',
    #                       transform=proj)
    
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                            vmin=0,vmax=2,
                          #levels=cintsrat,
                          cmap='cmo.balance',
                          transform=proj)
    
    cl     = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                          levels=cintsrat,
                          colors='gray',
                          transform=proj)
    
    
    fmt= {}
    for l, s in zip(cl.levels, clabs):
        fmt[l] = s
    clb = ax.clabel(cl,fmt=fmt,fontsize=fsz_ticks)
    
    viz.add_fontborder(clb,w=2)
    
    # Plot Ice Mask
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=-1)
    
    # Plot Gulf Stream Position
    gss = ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
    gss[0].set_path_effects([PathEffects.withStroke(linewidth=4, foreground='lightgray')])
    
    cb = viz.hcbar(pcm,ax=ax)
    cb.set_label("Forcing Ratio",fontsize=fsz_axis)
    title = r"$U_{geo}/%s$" % (sst_fnames[ii])
    ax.set_title(title,fontsize=fsz_title)
    
    figname = "%sForcing_Ratio_SST_Ugeo_v_%s.png" % (figpath,sst_fnames_short[ii])
    if viz_total_include_correction:
        savename = proc.addstrtoext(savename,"_addCorrToTotal")
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
#%% Also Compare Forcing Ratios for SSS

sss_forcings        = [lhflx_std_total,prec_std_total,qek_sss_std_total]
sss_fnames          = [r"F_L'",r"P'",r"Q_{ek,S}"]
sss_fnames_short    = ["Eprime","P","Qek"]
refvar              = ugeo_amp_FM[1] * dtmon


for ii in range(3):
    
    fig,ax  = viz.init_regplot(bboxin=bboxplot)
    
    # Compute Ratio (i.e. how much larger is geostrophic adv compared to the forcing)
    plotvar = refvar/ sss_forcings[ii]
    
    # pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
    #                       #levels=cintsrat,
    #                       cmap='cmo.balance',
    #                       transform=proj)
    
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                            vmin=0,vmax=2,
                          #levels=cintsrat,
                          cmap='cmo.balance',
                          transform=proj)
    
    cl     = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                          levels=cintsrat,
                          colors='gray',
                          transform=proj)
    
    
    fmt= {}
    for l, s in zip(cl.levels, clabs):
        fmt[l] = s
    clb = ax.clabel(cl,fmt=fmt,fontsize=fsz_ticks)
    
    viz.add_fontborder(clb,w=2)
    
    # Plot Ice Mask
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=-1)
    
    # Plot Gulf Stream Position
    gss = ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
    gss[0].set_path_effects([PathEffects.withStroke(linewidth=4, foreground='lightgray')])
    
    cb = viz.hcbar(pcm,ax=ax)
    cb.set_label("Forcing Ratio",fontsize=fsz_axis)
    title = r"$U_{geo}/%s$" % (sss_fnames[ii])
    ax.set_title(title,fontsize=fsz_title)
    
    figname = "%sForcing_Ratio_SSS_Ugeo_v_%s.png" % (figpath,sss_fnames_short[ii])
    if viz_total_include_correction:
        savename = proc.addstrtoext(savename,"_addCorrToTotal")
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
#%% Quickly Check Seasonal Variation in the forcing

maskreg         = proc.sel_region_xr(mask,bboxplot)
frcname         = "QekSSS"


if frcname == "LHFLX":
    input_forcing   = lhflx.copy() + lhflx_corr.copy() #* mask_apply
    vname           = "SSS"
    cints_in        = np.arange(0,0.11,0.005)
    vmax_in         = 0.05
    
elif frcname == "P":
    input_forcing   = prec.copy() + prec_corr.copy() #* mask_apply
    vname           = "SSS"
    
    cints_in        = np.arange(0,1,0.1)
    vmax_in         = 0.5
    
elif frcname == "QekSSS":
    input_forcing   = qek_sss.copy() + qek_sss_corr.copy() #* mask_apply
    vname           = "SSS"
    
    cints_in        = np.arange(0,0.11,0.005)
    vmax_in         = 0.05
    

lon             = input_forcing.lon
lat             = input_forcing.lat




if vname == "SSS":
    vunit           = "[psu/mon]"
    #cints_in        = np.arange(0,0.11,0.005)
    #vmax_in         = 0.05

elif vname == "SST":
    vunit           = "[degC/mon]"
    cints_in        = np.arange(0,0.55,0.05)
    vmax_in         = 0.5
    
for im in range(12):
    
    fig,ax          = viz.init_regplot(bboxin=bboxplot)
    
    # Plot the Variable
    plotvar         = stdsqsum(input_forcing.isel(mon=im),0) 
    pcm             = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                                    vmin=0,vmax=vmax_in,cmap="cmo.haline",zorder=-4)
    
    # Plot Contour Lines
    cl              = ax.contour(lon,lat,plotvar,transform=proj,linewidths=0.55,
                                    levels=cints_in,colors="k",zorder=1)
    clb = ax.clabel(cl,fontsize=fsz_tick)
    viz.add_fontborder(clb,w=2)
    
    # Plot Ice Mask
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=-1)
    
    # Plot Gulf Stream Position
    gss             = ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot',zorder=1)
    gss[0].set_path_effects([PathEffects.withStroke(linewidth=4, foreground='lightgray')])
    
    title           = "%s" % (mons3[im])
    ax.set_title(title,fontsize=fsz_axis)
    
        
    cb = viz.hcbar(pcm,ax=ax)
    cb.set_label("%s %s" % (frcname,vunit),fontsize=fsz_axis)
    
    figname = "%sForcing_Maps_%s_%s_mon%02i.png" % (figpath,vname,frcname,im+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    
#%% One Big Plot (in a line)

fig,axs,_ = viz.init_orthomap(1,12,bboxplot,figsize=(60,20))
for im in range(12):
    
    ax = axs[im]
    ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=fsz_tick,
                            fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    
    
    # Plot the Variable
    plotvar         = stdsqsum(input_forcing.isel(mon=im),0) * maskreg
    pcm             = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                                    vmin=0,vmax=vmax_in,cmap="cmo.haline",zorder=-4)
    
    # Plot Contour Lines
    cl              = ax.contour(lon,lat,plotvar,transform=proj,linewidths=0.55,
                                    levels=cints_in,colors="k",zorder=1)
    clb = ax.clabel(cl,fontsize=8)
    viz.add_fontborder(clb,w=2)
    
    # Plot Ice Mask
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                transform=proj,levels=[0,1],zorder=-1)
    
    # Plot Gulf Stream Position
    gss             = ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot',zorder=1)
    gss[0].set_path_effects([PathEffects.withStroke(linewidth=4, foreground='lightgray')])
    
    title           = "%s" % (mons3[im])
    ax.set_title(title,fontsize=fsz_axis)
    
        
cb = fig.colorbar(pcm,ax=axs.flatten(),pad=0.01,fraction=0.0025)
cb.set_label("%s %s" % (frcname,vunit),fontsize=fsz_axis)
    
figname = "%sForcing_Maps_%s_%s_AllMon.png" % (figpath,vname,frcname)
plt.savefig(figname,dpi=150,bbox_inches='tight')