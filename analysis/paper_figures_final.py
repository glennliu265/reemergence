#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:33:37 2025

@author: gliu
"""





import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import tqdm
import os
import matplotlib as mpl
import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.patheffects as PathEffects
from cmcrameri import cm
import colorcet as cc


# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
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


#%% Indicate Paths

#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240411/"
datpath   = pathdict['raw_path']
outpath   = pathdict['input_path']+"forcing/"
rawpath   = pathdict['raw_path']

# Where the revision data is located
revpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/revision_data/"

# Save publication ready PDF version of the figures 
pubready  = True

#%% Indicate some plotting variables

darkmode = False
if darkmode:
    dfcol = "w"
    bgcol = np.array([15,15,15])/256
    transparent = True
    plt.style.use('dark_background')
    mpl.rcParams['font.family']     = 'Avenir'
else:
    dfcol = "k"
    bgcol = "w"
    transparent = False
    plt.style.use('default')


bboxplot                        = [-80,0,20,65]
mpl.rcParams['font.family']     = 'Avenir'
mons3                           = proc.get_monstr(nletters=3)

fsz_tick                        = 18
fsz_axis                        = 20
fsz_title                       = 32

rhocrit                         = proc.ttest_rho(0.05,2,86)
proj                            = ccrs.PlateCarree()

# Get Region Info
regionset                       = "SSSCSU"
rdict                           = rparams.region_sets[regionset]
regions                         = rdict['regions']
bboxes                          = rdict['bboxes']
rcols                           = rdict['rcols']
rsty                            = rdict['rsty']
regions_long                    = rdict['regions_long']
nregs                           = len(bboxes)

regions_long = ('Sargasso Sea', 'N. Atl. Current',  'Irminger Sea')

# Get Point Info
pointset                    = "PaperDraft02"
ptdict                      = rparams.point_sets[pointset]
ptcoords                    = ptdict['bboxes']
ptnames                     = ptdict['regions']
ptnames_long                = ptdict['regions_long']
ptcols                      = ptdict['rcols']
ptsty                       = ptdict['rsty']


#%% Load Land Ice Mask

# Load the currents
ds_uvel,ds_vvel = dl.load_current()
ds_bsf          = dl.load_bsf(ensavg=False)
ds_ssh          = dl.load_bsf(ensavg=False,ssh=True)

# Convert Currents to m/sec instead of cmsec
ds_uvel         = ds_uvel/100
ds_vvel         = ds_vvel/100
tlon            = ds_uvel.TLONG.mean('ens').values
tlat            = ds_uvel.TLAT.mean('ens').values

# Load Land Ice Mask
icemask         = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")
mask            = icemask.MASK.squeeze()
mask_plot       = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply      = icemask.MASK.squeeze().values

# Load data processed by [calc_monmean_CESM1.py]
ds_sss          = dl.load_monmean('SSS')
ds_sst          = dl.load_monmean('SST')

# Get A region mask
mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub
ds_gs2          = dl.load_gs(load_u2=True)

# =============================================================================
#%% Figure (1) Plot Mean States of the Currents, and the mixed-layer depth
# =============================================================================

# Process Mean SST and SSS
sst_mean            = ds_sst.SST.mean('ens').mean('mon').transpose('lat','lon')  - 273.15 #  Covert to Celsius
sss_mean            = ds_sss.SSS.mean('ens').mean('mon').transpose('lat','lon') 

# Set some colorbar limits
cints_sstmean_degC  = np.arange(6,27)
cints_sssmean       = np.arange(34,37.6,0.2)

# Load the mixed-layer depth
mldpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
ncmld       = mldpath + "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc"
ds_mld      = xr.open_dataset(ncmld).load()

# Set plotting options for mld
vlms        = [0,200]# None
cints_sp    = np.arange(200,1500,100)# None
cmap_mld    = 'cmo.dense'
vlabel      = "Max Seasonal Mixed Layer Depth (meters)"

#%% Do some plotting

fsz_axis        = 24
fsz_title       = 28
fsz_tick        = 22
fsz_leg         = 18
lw_plot         = 4.5
use_star        = True
qint            = 2
plot_point      = True

fig,axs,_       = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))
ii              = 0

for vv in range(2):
    
    ax      = axs[vv]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    
    # Plot Gulf Stream Position
    #ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=1.75,c="k",ls='dashed')
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='k',ls='dashdot',zorder=1)

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=1)
    
    # Label Subplot
    viz.label_sp(ii,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.08,x=-.02,
                 fontcolor=dfcol)
    ii +=1


# (1) Mean State Plot ---------------------------------------------------------
ax                 = axs[0]

# Plot SST
plotvar            = sst_mean * mask_reg
pcm                = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
            cmap="RdYlBu_r",levels=cints_sstmean_degC,extend='both')
cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.set_label("SST ($\degree C$)",fontsize=fsz_axis)
cb.ax.tick_params(labelsize=fsz_tick)

# Plot SSS
plotvar            = sss_mean * mask_reg
cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
            linewidths=1.5,colors="darkviolet",levels=cints_sssmean,linestyles='dashed',zorder=1)
ax.clabel(cl,fontsize=fsz_tick)

# Plot Currents
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='gray',transform=proj,alpha=0.55,zorder=-1)

if plot_point:
    nregs = len(ptnames)
    for ir in range(nregs):
        pxy   = ptcoords[ir]
        ax.plot(pxy[0],pxy[1],transform=proj,markersize=20,markeredgewidth=.5,c=ptcols[ir],
                marker='*',markeredgecolor='k',zorder=4)
    
# (2) Wintertime MLD Plot -----------------------------------------------------
ax = axs[1]

plot_dots = True
# fsz_tick = 26
# fsz_axis = 32


# Plot maximum MLD
plotvar     = ds_mld.h.max('mon') * mask_reg


# Just Plot the contour with a colorbar for each one
if vlms is None:
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,cmap=cmap_mld,zorder=-1)
    cb = fig.colorbar(pcm,ax=ax)
else:
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        cmap=cmap_mld,vmin=vlms[0],vmax=vlms[1],zorder=-4)

pcm.set_rasterized(True) 

# Do special contours
if cints_sp is not None:
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    levels=cints_sp,colors="w",linewidths=1.1,zorder=2)
    cl_lab = ax.clabel(cl,fontsize=fsz_tick,zorder=6)
    
    viz.add_fontborder(cl_lab,w=4,c='k')

if plot_dots:
    bboxr       = proc.get_bbox(mask_reg)
    hreg        = proc.sel_region_xr(ds_mld.h,bboxr)
    hmax        = hreg.argmax('mon').values
    # Special plot for MLD (mark month of maximum)
    hmask_feb  = (hmax == 1) * mask_reg#ds_mask # Note quite a fix, as 0. points will be rerouted to april
    hmask_mar  = (hmax == 2) * mask_reg #ds_mask
    
    smap = viz.plot_mask(hmask_feb.lon,hmask_feb.lat,hmask_feb.T,reverse=True,
                          color="mediumseagreen",markersize=4,marker="+",
                          ax=ax,proj=proj,geoaxes=True)
    
    smap = viz.plot_mask(hmask_mar.lon,hmask_mar.lat,hmask_mar.T,reverse=True,
                          color="palegoldenrod",markersize=3,marker="o",
                          ax=ax,proj=proj,geoaxes=True)
    
    ax.scatter(-100,-100,c="mediumseagreen",s=125,marker="+",label="February Max",transform=proj)
    ax.scatter(-100,-100,c="palegoldenrod",s=125,marker="o",label="March Max",transform=proj)
    
    leg = ax.legend(fontsize=fsz_leg,framealpha=0.85,
                    bbox_to_anchor=(.35,1.08), loc="upper left",  bbox_transform=ax.transAxes)
    leg.get_frame().set_linewidth(0.0)
    
if vlms is not None:
    cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
savename        = "%sFig01MeanState.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')
if pubready:
    savename        = "%sFig01MeanState.pdf" % figpath
    plt.savefig(savename,format='pdf',bbox_inches='tight')


# =============================================================================
#%% Figure (2) REI Index
# =============================================================================
"""

Figure 2 of the SSS_paper
 - Output generated by compare_rei.py
 - Feb/March Ensemble Average Re-emergence Index

"""

# Load the data
nc_rei      = "%sREI_Patterns_Paper_FM.nc" % (revpath)
dsrei       = xr.open_dataset(nc_rei).load()

# Specific Parameters
bbplot      = [-80,0,20,65]
expvars     = ["SST","SST","SSS","SSS"]
rrsel       = ["SAR","NAC","IRM"]



#%% Make the REI plot

fsz_axis    = 32
fsz_title   = 36
fsz_ticks   = 24

if darkmode:
    dfcol = "w"
    transparent = True
    plt.style.use('dark_background')
    mpl.rcParams['font.family'] = 'Avenir'
else:
    dfcol = "k"
    transparent = False
    plt.style.use('default')
    mpl.rcParams['font.family'] = 'Avenir'

plot_point          = True
pubready            = True

# Plotting choice
levels      = np.arange(0,0.55,0.05)
fig,axs,_   = viz.init_orthomap(2,2,bbplot,figsize=(26,18.5),centlat=45,)

iicb = 0

for ex in range(4):
    
    # Prepare Plotting Variable (by Model and Variable)
    if ex < 2: # Plot for SST
        vname = 'SST'
    else:
        vname = 'SSS'
    
    if ex%2 == 0:
        simname     = "CESM1"
        modelname   = "CESM1"
    else:
        simname     = "StochasticModel"
        modelname   = "Stochastic Model"
    reiname = "%s_%s" % (vname,simname)
    print(reiname)
    
    
    # Set up Axis
    ax           = axs.flatten()[ex]
    ax           = viz.add_coast_grid(ax,bbplot,fill_color="lightgray",fontsize=20,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color='k')
    
    # Add title
    if ex < 2:
        ax.set_title(modelname,fontsize=fsz_title)
    
    # Set plotting options
    if vname == "SSS":
        cmap_in = cmo.cm.deep #"cmo.deep"
        cmap_in.set_under('lightyellow')
        cmap_in.set_over('royalblue')
    elif vname == "SST":
        cmap_in = cmo.cm.dense#cmo.dense"
        cmap_in.set_under('lightcyan')
        cmap_in.set_over('darkorchid')
    
    # Get REI
    plotvar = dsrei[reiname]
    lon     = plotvar.lon
    lat     = plotvar.lat
    plotvar = plotvar * mask_reg #* mask2 #* mask3
    
    # Add contours
    pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,cmap=cmap_in,levels=levels,transform=proj,extend='both',zorder=-2)
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=proj,zorder=-2)
    cl_lab = ax.clabel(cl,fontsize=fsz_ticks,inline_spacing=2)
    
    viz.add_fontborder(cl_lab,w=2,c='lightgray')
    
    # Add Colorbar
    if ex%2 == 1:
        
        cb = fig.colorbar(pcm,ax=axs[iicb,:].flatten(),fraction=0.015,pad=0.015)
        cb.ax.tick_params(labelsize=fsz_ticks)
        cb.set_label("%s Re-emergence Index" % vname,fontsize=fsz_axis)
        iicb +=1
    
    # Add Variable Label
    if ex%2 == 0:
        viz.add_ylabel("%s" % vname,ax=ax,y=0.65,x=0.01,
                       fontsize=fsz_title)
    
    # Add Subplot Label
    ax=viz.label_sp(ex,ax=ax,x=0.05,y=1.01,alpha=0,fig=fig,
                 labelstyle="%s)",usenumber=False,fontsize=fsz_title,fontcolor=dfcol)
    
    
    # Add additional features
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='k',ls='dashdot')
    
    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
    # Plot the Bounding Boxes or Points
    if plot_point: # Plot points
        
        nreg = len(ptnames)
        for rr in range(nreg):
            pxy   = ptcoords[rr]
            pname = ptnames[rr]
            if pname not in rrsel:
                continue
            if ex == 0 and pname == "IRM":
                ax.plot(pxy[0],pxy[1],transform=proj,markersize=46,markeredgewidth=.5,c=ptcols[rr],
                        marker='*',markeredgecolor='k')
            elif ex == 2 and pname != "IRM":
                ax.plot(pxy[0],pxy[1],transform=proj,markersize=46,markeredgewidth=.5,c=ptcols[rr],
                        marker='*',markeredgecolor='k')

# Save the Figure
savename    = "%sFig02REI.png" % (figpath,)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)
if darkmode:
    savename = proc.addstrtoext(savename,"_darkmode")
if pubready:
    savename    = "%sFig02REI.pdf" % (figpath,)
    plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent,format="pdf")
else: 
    plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)

# =============================================================================
#%% Figure (3) Damping 
# =============================================================================

# Load files
ncname_damping  = "%sParameters_plot_damping_feedbacks.nc" % revpath
ds_damping      = xr.open_dataset(ncname_damping).load()

# Prep variable
ds_damping_savg = proc.calc_savg(ds_damping,ds=True)

# Declare some variables
dampvars_cints  = [np.arange(0,48,3),np.arange(0,0.055,0.005)]
cints_taudamp   = np.arange(0,63,3)
cints_hff       = np.arange(-40,44,4)
dampvars_cmap   = ['cmo.amp_r','cmo.matter']

# Sizes
fsz_title       = 32 #before
fsz_axis        = 32 #before
fsz_tick        = 28 #before

figsize         = (30,14)
dampvars_name   = ["Net Heat Flux Damping Timescale\n($\lambda^N$)","SST-Evaporation Feedback \n on SSS ($\lambda^e$)"]
ylabs           = ["Net Heat Flux Damping ($\lambda^N$)","SST-Evaporation Feedback on SSS ($\lambda^e$)"]
dampvars_units  = [r"[$\frac{W}{m^{-2} \,\, \degree C}]$",r'[$\frac{psu}{\degree C \,\, mon}$]' ] 

#%% Make the Plot

fig,axs,mdict = viz.init_orthomap(2,4,bboxplot=bboxplot,figsize=figsize)
ii              = 0
for vv in range(2):
    
    cints  = dampvars_cints[vv]
    
    for ss in range(4):
        
        ax = axs[vv,ss]
        ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=fsz_tick,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        if vv == 0: # Plot Damping and 
        
            plotvar = ds_damping_savg.tau_a.isel(season=ss) #* mask 
            plothff = ds_damping_savg.lbd_a.isel(season=ss)
            
            # Plot Damping as Values
            pcm       = ax.contourf(plothff.lon,plothff.lat,plothff,
                                  transform=proj,levels=cints_hff,extend='both',cmap='cmo.balance')
            
            # Plot Timescale as Lines
            cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                  transform=proj,levels=cints_taudamp,extend='both',colors="navy",linewidths=0.75)
            cl_lab  = ax.clabel(cl,fontsize=fsz_tick)
            [tt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')]) for tt in cl_lab]
            
            dunits = "$W m^{-2} \degree C^{-1}$"
        
        else:
            
            plotvar = ds_damping_savg.lbd_e.isel(season=ss) * mask
            pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                                  transform=proj,levels=cints,extend='both',cmap=dampvars_cmap[vv])
            
            
            cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                  transform=proj,levels=cints,extend='both',colors="lightgray",linewidths=0.75)
            cl_lab = ax.clabel(cl,fontsize=fsz_tick,)
            [tt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='dimgrey')]) for tt in cl_lab]
            
            dunits = dampvars_units[vv]
            
        if vv == 0:
            ax.set_title(plotvar.season.data.item(),fontsize=fsz_axis)
            
        pcm.set_rasterized(True) 
         
        # Plot Ice Mask
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=1,
                   transform=mdict['noProj'],levels=[0,1],zorder=-1)
        
        # Plot Gulf Stream Position
        gss = ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
        #gss[0].set_path_effects([PathEffects.withStroke(linewidth=4, foreground='lightgray')])
            
        # Label Subplot
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
        ii+=1
    
    cb      = viz.hcbar(pcm,ax=axs[vv,:].flatten(),pad=0.05,fraction=0.055)
    cb.set_label("%s, %s" % (ylabs[vv],dunits),fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
savename = "%sFig03Damping.png" % (figpath)
if pubready:
    savename = "%sFig03Damping.pdf" % (figpath)
    plt.savefig(savename,format='pdf',bbox_inches='tight') 
else:
    plt.savefig(savename,dpi=150,bbox_inches='tight') 

# =============================================================================
#%% Figure (4) Forcing
# =============================================================================
"""

Plot with all the forcings
Originall from viz_inputs_paper_draft

"""


# Load Inputs
ncname_inputs   = "%sRevision01_SM_Input_Parameters_Forcing.nc" % revpath
ds_forcings     = xr.open_dataset(ncname_inputs).load()

# Set to True to include correction in total forcing visualization
viz_total_include_correction=False

# Select months
selmons        = [1,2] # Indices
monstr         = proc.mon2str(selmons)
dsfmon         = ds_forcings.isel(mon=selmons).mean('mon')

# Compute totals and correction percentages
# Take EOF1, EOF2, Conversion Factor, and Total
def get_total_corr(vname,viz_total_include_correction=False):
    mode1       = dsfmon[vname].isel(mode=0)
    mode2       = dsfmon[vname].isel(mode=1)
    std_total   = proc.stdsqsum_da(dsfmon[vname],'mode')
    correction  = np.abs(dsfmon[vname+"_corr"])
    corr_perc   = correction / (std_total + correction) * 100
    if viz_total_include_correction:
        std_total = std_total + correction # Include this in the total
    return mode1,mode2,std_total,corr_perc

Fprime_in   = get_total_corr('Fprime'   ,viz_total_include_correction=viz_total_include_correction)
evap_in     = get_total_corr('lhflx'    ,viz_total_include_correction=viz_total_include_correction)
prec_in     = get_total_corr('prec'     ,viz_total_include_correction=viz_total_include_correction)
qek_sst_in  = get_total_corr('qek_sst'  ,viz_total_include_correction=viz_total_include_correction)
qek_sss_in  = get_total_corr('qek_sss'  ,viz_total_include_correction=viz_total_include_correction)


# Declare Names, etc
plotvars_force = [Fprime_in,qek_sst_in, #SST
                  evap_in,prec_in,qek_sss_in] #SSS

vnames_force   = ["Stochastic Heat Flux Forcing\n"+r"($\frac{1}{\rho C_p h} F_N'$, SST)",
                  "Ekman Forcing\n($Q_{ek,T},SST)$",
                  "Evaporation\n"+r"($\frac{\overline{S}}{\rho h L} F_L'$,SSS)",
                  "Precipitation\n"+r"($\frac{\overline{S}}{\rho h} P'$,SSS)",
                  "Ekman Forcing\n($Q_{ek,S},SSS)$"]

rownames       = ["EOF 1", "EOF 2", "EOF Total", r"$\frac{Correction \,\, Factor}{EOF \,\, Total \,\, + Correction \,\, Factor }$"]

#%% Make the Plot


pubready        = True

mult_SSS_factor = 1e3 # Default is 1

fsz_tick        = 26
fsz_title       = 32
fsz_axis        = 28

sss_vlim        = np.array([-.01,.01]) * mult_SSS_factor
sss_vlim_var    = np.array([0,.015])   * mult_SSS_factor
sst_vlim        = [-.20,.20]
sst_vlim_var    = [0,.5]

plotover        = False
if plotover:
    cints_sst_lim = np.arange(0.5,1.6,0.25)
    cints_sss_lim = np.arange(0.015,1.5,0.015)  * mult_SSS_factor
else:
    cints_sst_lim = np.arange(0,0.55,0.05)
    cints_sss_lim = np.arange(0,0.028,0.004)  * mult_SSS_factor
    

fig,axs,mdict = viz.init_orthomap(4,5,bboxplot=bboxplot,figsize=(30,22))
ii = 0
for rr in range(4):
    
    for vv in range(5):
        
        ax          = axs[rr,vv]
        
        # Get Variable and vlims and Clear Axis where needed
        plotvar     = plotvars_force[vv][rr]
        
        if plotvar is None:
            ax.clear()
            ax.axis('off')
            continue
        else:
            plotvar = plotvar * mask
        
        f_vname = vnames_force[vv]
        if "SST" in f_vname:
            vunit = rparams.vunits[0]
            if rr == 2: # Use variance
                clab = "Total Standard Deviation"
                vlim = sst_vlim_var
                cmap = cm.lajolla_r
                cints_lim = cints_sst_lim
                if plotover:
                    ccol      = "k"
                else:
                    ccol      = 'dimgray'
                
            elif rr == 3: # % Correction
                clab = "% Correction"
                vlim = [0,100]
                cmap = 'cmo.amp'
                
            else:
                clab = "SST Forcing"
                vlim = sst_vlim
                cmap = 'cmo.balance'
                
        
        elif "SSS" in f_vname:
            if rr < 3:
                plotvar =  plotvar * mult_SSS_factor
            vunit = rparams.vunits[1]
            if rr == 2: # Use variance
                clab = "Total Standard Deviation"
                vlim = sss_vlim_var
                cmap = cm.acton_r#'cmo.rain'
                if plotover:
                    ccol = "lightgray"
                else:
                    ccol = 'lightgray'
                cints_lim = cints_sss_lim
            elif rr == 3: # % Correction
                clab = "% Correction"
                vlim = [0,100]
            else:
                clab = "SSS Forcing"
                vlim = sss_vlim
                cmap = 'cmo.delta'
        
        # Set Up Axes and Labeling ---
        blb = viz.init_blabels()
        if vv == 0:
            blb['left']  = True
            viz.add_ylabel(rownames[rr],ax=ax,fontsize=fsz_axis)
        if rr == 3:
            blb['lower'] = True
        
        ax          = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                          fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        if rr == 0:
            ax.set_title(vnames_force[vv],fontsize=fsz_axis)
        # ------------
        
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                            transform=proj,vmin=vlim[0],vmax=vlim[1],cmap=cmap)
        pcm.set_rasterized(True) 
        
        # Plot additional contours
        if rr == 2:
            
            #ccol = "lightgray"
            cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,levels=cints_lim,
                                colors=ccol,linewidths=0.75,)
            
            
            
            cl_lab = ax.clabel(cl,levels=cints_lim[::2],fontsize=fsz_tick)
            if ccol == "lightgray":
                fbcol = "k"
            else:
                fbcol = "w"
                
            viz.add_fontborder(cl_lab,w=4,c=fbcol)
        
        # Make Colorbars (and Adjust)
        makecb = False
        if (rr==1) and (vv==0): # SST Plots (EOF)
            axcb = axs[:2,:2]
            makecb = True
            frac=0.03
            pad =0.04
        if (rr==2) and (vv==1): # SST Plots (total Variance)
            axcb = axs[2,:2]
            makecb = True
            frac=0.06
            pad =0.04
        if (rr==1) and (vv==3): # SSS Plots (EOF)
            axcb = axs[:2,2:]
            makecb = True
            frac=0.03
            pad =0.04
        if (rr==2) and (vv==3): # SSS Plots (total Variance)
            axcb = axs[2,2:]
            makecb = True
            frac=0.06
            pad =0.04
        if (rr==3) and (vv==4): # SST Plots (Total Variance)
            axcb = axs[3,:]
            makecb = True
            frac   = 0.065
            pad     =0.04
        
        if makecb:
            cb  = viz.hcbar(pcm,ax=axcb.flatten(),fraction=frac,pad=pad)
            cb.ax.tick_params(labelsize=fsz_tick)
            if rr == 3: # Correction Factor
                
                cb.set_label(r"%s" % (clab),fontsize=fsz_axis)
            else:
                if vunit == "psu" and mult_SSS_factor > 1: # Add Multiplcation Factor
                    mult_factor = (np.log10(1/mult_SSS_factor))
                    
                    cb.set_label(r"%s [$\frac{%s}{mon}$ $\times$10$^{%i}$]" % (clab,vunit,mult_factor),fontsize=fsz_axis)
                else:
                    cb.set_label(r"%s [$\frac{%s}{mon}$]" % (clab,vunit),fontsize=fsz_axis)
        
        # Plot Ice Mask
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                   transform=mdict['noProj'],levels=[0,1],zorder=-1)
        
        # Plot Gulf Stream Position
        gss = ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
        gss[0].set_path_effects([PathEffects.withStroke(linewidth=6, foreground='lightgray')])
        
        
        
        # Label Subplot
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
        ii+=1
        
savename = "%sFig04Forcing.png" % (figpath)
if viz_total_include_correction:
    savename = proc.addstrtoext(savename,"_addCorrToTotal")
    
if pubready:
    plt.savefig(savename,dpi=150,bbox_inches='tight')  
    
    savename = "%sFig04Forcing.pdf" % (figpath)
    plt.savefig(savename,format="pdf",bbox_inches='tight')
    
else:
    plt.savefig(savename,dpi=150,bbox_inches='tight')  



# =============================================================================
#%% Figure (5) Detrain
# =============================================================================
"""

Detrainment example at 50N, 30W
Originally from analysis/viz_detrainment

"""

# Load Inputs
ncdetrain   = revpath + "Detrain_Plot_Data.nc"
dsdetrain   = xr.open_dataset(ncdetrain).load()
hclim       = dsdetrain.h.data
kprev       = dsdetrain.kprev.data

# Prep some other inputs
def monstacker(scycle):
    return np.hstack([scycle,scycle[:1]])
mons3stack = monstacker(mons3)
hstack     = monstacker(hclim)
plotx      = np.arange(1,14)

# Fontsize
fsz_title   = 20
fsz_ticks   = 14
fsz_axis    = 16

# Toggles
pubready    = True

#%% Make the Plot

fig,ax     = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

ax         = viz.viz_kprev(hclim,kprev,ax=ax,lw=3,
                   fsz_lbl=fsz_axis,fsz_axis=fsz_axis,plotarrow=False,msize=15,
                   shade_layers=True)


ax.set_xticklabels(mons3stack,fontsize=fsz_ticks)
ax          = viz.add_ticks(ax,minorx=False,grid_col="w",grid_ls='dotted')
ax.set_title("Mixed-Layer Seasonal Cycle and Detrainment Months",fontsize=fsz_title)

ax.tick_params(axis='both', which='major', labelsize=fsz_ticks)

ax.set_xlabel("Month",fontsize=fsz_axis)
ax.set_ylabel("Mixed-Layer Depth [meters]",fontsize=fsz_axis)
ax.set_ylim([0,175])
ax.invert_yaxis()


savename = "%sFig05Detrain.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)
if pubready:
    savename = "%sFig05Detrain.pdf" % (figpath)
    plt.savefig(savename,format='pdf',bbox_inches='tight')

# =============================================================================
#%% Figure (6) Deep Damping
# =============================================================================
"""
- Plot with the subsurface/deep memory timescale and deep damping parameter
 during the entrainment season
 
"""

# Load and set up the variables
ncname_lbdd     = "%sParameters_plot_subsurface_damping.nc" % revpath
dslbd           = xr.open_dataset(ncname_lbdd).load()
plotvars        = [dslbd.SST_taud,dslbd.SSS_taud]
plotvars_corr   = [dslbd.SST_lbdd,dslbd.SSS_lbdd]

# Plotting pArameters
fsz_title       = 42 
fsz_axis        = 32 
fsz_tick        = 25 
figsize         = (28,15)
clab            = r"Subsurface Memory Timescale [$\tau^d$,months]"


selmons         = [[6,7,8],[9,10,11],[0,1,2]]
vlms            = [0,60]
cints_corr      = np.arange(0,1.1,.1)
tau_cints       = np.arange(0,66,6)
cmap            = 'inferno'
ylabs_dd        = [u"$\lambda^d_T$" + " (SST)", u"$\lambda^d_S$" + " (SSS)"]

#%% Make the figure

ii              = 0
fig,axs,mdict   = viz.init_orthomap(2,3,bboxplot=bboxplot,figsize=figsize)
for vv in range(2):
    
    for mm in range(3):
        
        ax = axs[vv,mm]
        selmon = selmons[mm]
        
        ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=fsz_tick,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        if vv == 0:
            ax.set_title(proc.mon2str(selmon),fontsize=fsz_title)
        if mm == 0:
            viz.add_ylabel(ylabs_dd[vv],ax=ax,fontsize=fsz_title,x=-.065,y=0.6)
        
        # Plot the Timescales
        plotvar = plotvars[vv].isel(mon=selmon).mean('mon') * mask
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
        pcm.set_rasterized(True) 
        
        # Plot the Correlation
        plotvar = plotvars_corr[vv].isel(mon=selmon).mean('mon') * mask
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,levels=cints_corr,
                                colors="lightgray",linewidths=1.5)
        cl_lab = ax.clabel(cl,fontsize=fsz_tick,colors='k')
        [tt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')]) for tt in cl_lab]
        
        # Add other features ----
        
        # Plot Ice Mask
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=1,
                   transform=mdict['noProj'],levels=[0,1],zorder=-1)
        
        # Plot Gulf Stream Position
        gss = ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
        gss[0].set_path_effects([PathEffects.withStroke(linewidth=4, foreground='lightgray')])
                             
        # Label Subplot
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)# y = y=1.08
        ii+=1

cb = viz.hcbar(pcm,ax=axs.flatten())
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label(clab,fontsize=fsz_axis)

figname         = "%sFig06DeepDamping.png" % (figpath)
plt.savefig(figname,dpi=150,bbox_inches='tight')  
if pubready:
    figname     = "%sFig06DeepDamping.pdf" % (figpath)
    plt.savefig(figname,format='pdf',bbox_inches='tight')  

# =============================================================================
#%% Figure (7) ACF ~ Figure (9) Spectra, see other script
# =============================================================================


# =============================================================================
#%% Figure (8) MonVar
# =============================================================================

# =============================================================================
#%% Figure (9) Spectra
# =============================================================================

# =============================================================================
#%% Figure (10) CrossCorr
# =============================================================================

# =============================================================================
#%% Figure (11) AMV
# =============================================================================



# =============================================================================
#%% Figure (12) LagDiff
# =============================================================================

# =============================================================================
#%% Figure (13) VarRatio
# =============================================================================

# =============================================================================
#%% Figure (14) MLDVar
# =============================================================================

# =============================================================================
#%% Figure (15) Ugeo
# =============================================================================

# =============================================================================
#%% Figure (10) DeCorr
# =============================================================================




