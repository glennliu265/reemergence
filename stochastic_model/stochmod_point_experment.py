#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Experiment with the stochastic SST/SSS Model
Created on Thu Nov 30 11:51:53 2023

@author: gliu

"""

#%% 


"""

Constants:
 - rho  [kg/m3]  : Density
 - L    [J/kg]   : Specific Heat of Evaporation 
 - B             : Bowen Ratio
 - cp  [J/kg/K]  : Specific Heat

Both:
 - MLD [meters] : (ARRAY: [12,]) - Mixed-layer depth


SST Eqn:
 - Atmospheric Damping [W/m2/C] : [12] Can be Net, or specific flux
 - Forcing Amplitude   [W/m2]   : [12 x EOF_Mode]

SSS Eqn:
 - Precipitation Forcing : 
 - Evaporation Forcing   : 
 - Sbar                  : (ARRAY: [12,]) Mean Salinity

Process

1. Do Unit Conversions + Preprocessing (calculate kprev, make forcingetc)
2. Run SST Model
3. Run SSS Model
4. Compute Metrics
5. Compare with CESM1

"""

#%% Import Stuff

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

#%% Load Everything

# Output Paths
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240126/"
proc.makedir(figpath)


# MLD (Meters)
h = np.array([137.26482 , 154.96883 , 129.62602 ,  77.24637 ,  34.46067 ,
        23.328617,  23.019062,  27.13498 ,  36.598366,  51.93153 ,
        73.71345 , 102.56731 ])

# Stochastic Evaporation (W/m2)
E =  np.array([37.00598542, 34.54504215, 27.06711451, 23.50952673, 19.62948819,
       17.42450815, 17.86913774, 20.72606242, 24.08597815, 28.77556982,
       34.2877225 , 37.36201712])

# Precip
P = np.array([1.48168970e-08, 1.49428860e-08, 1.37938265e-08, 1.19365797e-08,
       1.01685842e-08, 9.56785762e-09, 9.26983557e-09, 9.89661064e-09,
       1.22099308e-08, 1.39296770e-08, 1.49598200e-08, 1.43752539e-08])

# Heat Flux Feedback (LH)
hff_l = np.array([16.96781049, 13.30814792, 11.72234556, 11.34872113, 11.5776733 ,
       10.75794417, 11.08759759, 12.94189574, 14.99567006, 17.29924383,
       18.58622274, 18.9057527 ])

# Tddamp
Tddamp_pt = np.array([0.08535528, 0.08491756, 0.07404652, 0.06366363, 0.06505359,
       0.06773722, 0.07149485, 0.07389011, 0.07524264, 0.07866281,
       0.07309676, 0.07428371])

# Load Output from other simulations ------
lonf =-30
latf = 50
locfn,loctitle=proc.make_locstring(lonf,latf)
locfn360,_ = proc.make_locstring(lonf+360,latf)
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % (locfn360)
proc.makedir(outpath)
cesm_savename = "%sCESM1_SSS_Metrics.npz" % outpath
ld_cesm = np.load(cesm_savename,allow_pickle=True)

# Loadout some output
cesm_metrics_pic = ld_cesm['tsmetrics'][0]
cesm_metrics_htr = ld_cesm['tsmetrics'][1]#item()

#
nens = len(cesm_metrics_htr['acfs'][0])


# Load HBLT (Copied from calc detrainment dmaping pt script)
# Paths and Names
mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
mldnc   = "HBLT_FULL_HTR_lon-80to0_lat0to65_DTFalse.nc"
dsh         = xr.open_dataset(mldpath+mldnc)
hbltpt      = dsh.sel(lon=lonf, lat=latf,
                 method='nearest').load()  # [Ens x time x z_t]
hclim       = hbltpt.groupby('time.month').mean('time').squeeze().HBLT.values/100  # Ens x month, convert cm --> m
hblt_in     = hclim.mean(1)
# PIC HBLT
hbltpic    = xr.open_dataset(outpath+"CESM1_FULL_PIC_HBLT.nc").HBLT.load()#np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/FULL_PIC_HBLT_hclim.npy")
hbltpic_in = proc.calc_clim(hbltpic.values,0) 

# Load Sd' Damping (calculated in )
lagmax   = 3
lags     = np.arange(0,37,1)
savename = "%sSd_damping_CESM1_HTR_FULL_%s_HBLT_%ilagfig_lags%02i.npz" % (outpath,locfn360,lagmax,lags[-1])
ld       = np.load(savename,allow_pickle=True)
# Load some variables
lbd_d_ensavg   = ld['lbd_d'].mean(0) # [ens x mon] => [mon]
#tau_est = ld['tau_est']


# Constants ------
dt           = 3600*24*30 # Timestep [s]
cp           = 3850       # 
rho          = 1026       # Density [kg/m3]
B            = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L            = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document
lbde_mult    = -1
Sbar         = 35.287624

mons3 = proc.get_monstr(nletters=3)
#%% Vis HBLT vs. H
fig,ax =viz.init_monplot(1,1)
ax.plot(mons3,h,label="HMXL (Ens. Avg)")
for e in range(nens):
    ax.plot(mons3,hclim[:,e],label="",color="gray",alpha=0.1)
ax.plot(mons3,hblt_in,label="HBLT (Ens. Avg)",color="k")
ax.plot(mons3,hbltpic_in/100,label="HBLT (PiControl)",color='red')
ax.set_title("MLD Seasonal Cycle @ %s"%loctitle)
ax.legend()

#%%  Do a silly Integration
# Copied from stochmod_point

dumb_entrain= True
Tddamp      = True
reset_eta   = False
nyrs        = 10000
lags        = np.arange(0,37)
kmonth      = 1


#(qL_in) / (rho*L*h_denom) * dt * sbar

# Set Parameters
h0          = hblt_in.copy()#h.copy()
alpha0      = (E.copy()/(rho*L*h0)*dt*Sbar + P*dt ) ##1     * np.ones(12)#( E.copy()/(rho*cp*h0)*dt + P*dt )
beta0       = scm.calc_beta(h[None,None,:]).squeeze() # 0     * np.ones(12) # 
damping     = 0  * np.ones(12) + beta0                #hff_l/(rho*cp*h0) *dt + beta0
lbd_Td_mon  = 1e-1 * np.ones(12) # Units are 1/mon

if reset_eta or "eta0" not in locals():
    print("Regenerating White Noise")
    eta0        =np.random.normal(0,1,nyrs*12)

kprev       =scm.find_kprev(h0)[0]
FAC         =scm.calc_FAC(damping[None,None,:]).squeeze()

# SST
# alpha0  = paramset['alpha']/(rho*cp*h0) * dt
# damping = paramset['hff']/(rho*cp*h0)*dt +beta0
# FAC     = scm.calc_FAC(damping[None,None,:]).squeeze()

# List Append Style
S_all1 = []
S_all1.append(0)

# Array Style
S_all = np.zeros((nyrs*12))
fterm = []
dterm = []
eterm = []

for t in tqdm(range(nyrs*12)):
    
    im = t%12
    
    # fterm.append(( eta0[t] *alpha0[im] )*FAC[im])
    # dterm.append(np.exp(-damping[im])*S_all[t-1])
    
    # List Append
    S1       = np.exp(-damping[im]) * S_all1[t] +  ( eta0[t]*alpha0[im] )*FAC[im] # Had to be careful to use t instead of t-1
    S_all1.append(S1)
    
    # Array Style
    if t == 0:
        S0 = 0
    else:
        S0 = S_all[t-1]
        
    fterm.append(eta0[t]*alpha0[im]*FAC[im]) 
    dterm.append(S0*np.exp(-damping[im]) )
    
    S1       = np.exp(-damping[im]) * S0 +  ( eta0[t]*alpha0[im] )*FAC[im]
    
    if dumb_entrain:
        # Very Basic Implementation of Entrainment 
        if (h0[im] > h0[im-1]) and (t>11):
            kp = kprev[im]
            Sd = S_all[int(t-np.round(kp))]
            
            
            
            if Tddamp:
                Td0        = Sd
                # Implement decay to Td
                lbd_Td     = lbd_Td_mon[im]
                detrain_m  = np.round(kp)               # Month of Detrainment for that anomaly
                detrain_dt = (im - (detrain_m-1))%12
                Td0        = Td0 * np.exp(-detrain_dt * lbd_Td)
                Sd         = Td0
            
            S1 = S1 + beta0[im] *Sd * FAC[im]
            eterm.append(beta0[im] *Sd * FAC[im])
    
            
            
    S_all[t] = S1.copy()

S_all1 = np.array(S_all1)[1:]
fterm  = np.array(fterm)
dterm  = np.array(dterm)

# Calculate the Theoretical ACF
varF   = alpha0.mean()**2 # Note quite, the delta t seems too high
#theoacf = 1 - lags**2 * varF/ np.var(S_all1)


lags    = np.arange(0,37)
S_all = S_all - S_all.mean()
acS     = scm.calc_autocorr([S_all[None,None,:],fterm[None,None,:],dterm[None,None,:]],lags,kmonth)
#acS   = scm.calc_autocorr([Tdict_base['T'],],lags,2)
#%%
theoacf = 1 - lags**2 * varF/ np.var(S_all1)
theoacf1 = np.exp(-damping.mean()*lags)

xtks    = np.arange(0,37,1)

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(10,4))
ax,ax2 = viz.init_acplot(kmonth,xtks,lags,ax=ax)
ax.plot(lags,acS[0],label="ACF (SSS)",marker="o")
ax.plot(lags,acS[1],label="ACF (Forcing)",marker="x")
#ax.plot(lags,theoacf1,label=r"$e^{-\lambda \tau}$")
#ax.plot(lags,theoacf,label="Theoretical ACF",marker="+")
#ax.plot(lags,acS[2],label="ACF (Damping)",marker="d")
#ax.set_ylim([.990,1])

#ax.set_ylim([.990,1])
ax.legend()
#plt.plot(lags,acS[0])
#plt.plot(lags,theoacf)


#%% Look at timeseries

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(10,4))
ax.plot(S_all)


#%% Plot Parameters
mons3 = proc.get_monstr(nletters=3)
fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(12,4))

ax = axs[0]
ax.plot(mons3,alpha0)
ax.set_title("Forcing Amplitude (n'), $\degree C$")

ax = axs[1]
ax.plot(mons3,beta0,color="red")
ax.set_title(r"Entrainment Damping ($\frac{w_e}{h}$), $mon^{-1}$")

ax = axs[2]
ax.plot(mons3,h)
ax.set_title("MLD (h), $m$")


#%%

varS     = np.var(S_all1)
varN     = np.var(fterm)

lbd_mean = damping.mean()



def theoacfdoc(lags,varN,varS):
    return 1 - (lags**2 * (varN/varS))


def theoacfexp(lags,lbd):
    return np.exp(lags*-lbd)

    

fig,ax= plt.subplots(1,1,constrained_layout=True,figsize=(6,3.5))


#ax.set_title("$Var(n'(t)=%.2f$ | " % (varN) + "$Var(S'(t+\Delta) $"))
#ax.plot(lags,theoacfdoc(lags,varN,varS),label=r"$1-\Delta t^2 \frac{var(n'(t)')}{var(S'(t+\Delta t))}$")
ax.plot(lags,theoacfexp(lags,lbd_mean),label=r"$e^{-\lambda \tau}$")
ax.plot(lags,acS[0],label="SSS Integration")
ax.legend()


#%% Rewrite an experimental section below, based on what is above (but using smconfig)
# Copied the cell "Do a silly Integration"


# Part (1) Unit Conversions, combine MLD, Damping, Forcing ----------------------

# Set Parameters

# Simulation
expname_base = "Default_Tddamp10Neg1_NoLbdA_EPforcing_Mldvar_HTRensavg_Params"
nyrs         = 10000
eta_in       = np.random.normal(0,1,nyrs*12)

# Forcing/Damping (with unit conversions)
h0          = np.roll(hblt_in.copy(),0)# h.copy() # Mixed Layer depth (month), units: [meters]
alpha       = np.roll((E.copy()/(rho*L*h0)*dt*Sbar + P*dt ),0)
damping_in  = np.zeros(12) # (qL_in) / (rho*L*h_denom) * dt * sbar
lbd_d       = 1e-1 * np.ones(12) 

# Toggles
reset_eta   = False
use_entrain = True

# Set up Forcing (tile and multiple)
if reset_eta or "eta_in" not in locals():
    print("Regenerating White Noise")
    eta_in   = np.random.normal(0,1,nyrs*12)
forcing_EP   = np.tile(alpha,nyrs) * eta_in

# Additional calculations
beta       = scm.calc_beta(h[None,None,:]).squeeze()
kprev      = scm.find_kprev(h0)[0]

# Set Configuration Parameters (already converted)
smconfig = {}
smconfig["entrain"] = use_entrain                # True to use entrainment             BOOL
smconfig["h"]       = h0[None,None,:]            # Mixed-Layer Depth (used for beta)   (month,) units: [meters]
smconfig["forcing"] = forcing_EP[None,None,:]    # Stochastic Forcing Amplitude.       (time,)  units: [psu/mon]
smconfig["lbd_a"]   = damping_in[None,None,:]    # Atmospheric Heat Flux Feedback      (month,) units: [1/mon]
smconfig["beta"]    = beta[None,None,:]          # Entrainment Damping                 (month,) units: [1/mon]. By default, this is calculated in the function and added to the overall damping
smconfig["lbd_d"]   = lbd_d[None,None,:]         # Damping of value below mixed-layer. (month,) units: [1/mon]
smconfig["kprev"]   = kprev[None,None,:]         # Detrainment Month                   (month,) units: [month index]

# Part (2) Do the Actual Integration --------------------------------------------
if smconfig['entrain']:
    outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                    Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],
                                    return_dict=True)
else:
    outdict = scm.integrate_noentrain(smconfig['lbd_a'],smconfig["F"],)


# Compute basic metrics
proc_ts = [outdict['T'].squeeze(),]
tsmetrics_default = scm.compute_sm_metrics(proc_ts,nsmooth=100)


#%% Quick Viz (ACF)
kmonth = 1
xtksl  = np.arange(0,37,3)
lags   = np.arange(37)

fig,ax =  plt.subplots(1,1,figsize=(8,3.5),constrained_layout=True)
ax,_ = viz.init_acplot(kmonth,xtksl,lags,title="",ax=ax)
ax.plot(lags,tsmetrics_default['acfs'][kmonth][0],label="Default Run SSS",color="mediumblue",lw=2.5,ls='dashed')
ax.plot(lags,cesm_metrics_pic['acfs'][kmonth][0],color="k",label=ld_cesm['dataset_names'][0])

acf_ensavg = np.zeros(37)
for e in range(nens):
    if e == 0:
        lab=ld_cesm['dataset_names'][1] + " (Indv. Member)"
    else:
        lab = ""
    plotvar = cesm_metrics_htr['acfs'][kmonth][e]
    ax.plot(lags,plotvar,color="gray",label=lab,alpha=0.15)
    acf_ensavg += plotvar
acf_ensavg = acf_ensavg/nens
lab = ld_cesm['dataset_names'][1] + " (Ens. Avg.)"
ax.plot(lags,acf_ensavg,color="gray",label=lab,alpha=1)
ax.legend()


savename = "%sSSS_Model_%s_mon%02i.png" % (figpath,expname_base,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Quick Viz (Monthly Variance)

fig,ax = viz.init_monplot(1,1,)

# Plot SM Run
lab = "%s (var=%.3f)" % ("Default Run",np.var(outdict['T'].squeeze()))
ax.plot(mons3,tsmetrics_default['monvars'][0],label=lab,color="mediumblue",lw=2.5,ls='dashed')

# Plot PIC 
lab = "%s (var=%.3f)" % (ld_cesm['dataset_names'][0],np.var(ld_cesm['sss'][0]))
ax.plot(mons3,cesm_metrics_pic['monvars'][0],color="k",label=lab)

# Plot Historical (Ens Member)
monvar_ensavg = np.zeros(12)
for e in range(nens):
    lab = ""
    plotvar = cesm_metrics_htr['monvars'][e]
    ax.plot(mons3,plotvar,color="gray",label=lab,alpha=0.15)
    monvar_ensavg += plotvar

# Plot Historical (Ens Avg)
monvar_ensavg = monvar_ensavg/nens
lab = "%s (var=%.3f)" % (ld_cesm['dataset_names'][1],np.nanmean(np.nanvar(ld_cesm['sss'][1],1)))
ax.plot(mons3,monvar_ensavg,color="gray",label=lab,alpha=1)

# Legends, Labeling
ax.legend(ncol=2)
ax.set_ylabel("$psu^2$")
ax.set_title("SSS Monthly Variance @ %s" % loctitle)

# Save Figure
savename = "%sSSS_Model_MonVar_CESM1_Default.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Quick Viz (Spectra)

# Plotting Params
dtplot  = 3600*24*30
plotyrs = [100,50,25,10,5]
xtks    = 1/(np.array(plotyrs)*12)

fig,ax = plt.subplots(1,1,constrained_layout=1,figsize=(8,3))

# Plot SM Run
freqs   = tsmetrics_default['freqs']
specs   = tsmetrics_default['specs']
lab     = "%s (var=%.3f)" % ("Default Run",np.var(outdict['T'].squeeze()))
ax.plot(freqs[0]*dtplot,specs[0]/dtplot,label=lab,color="mediumblue",lw=2.5,ls='solid')


# Plot PIC
lab = "%s (var=%.3f)" % (ld_cesm['dataset_names'][0],np.var(ld_cesm['sss'][0]))
ax.plot(cesm_metrics_pic['freqs'][0]*dtplot,cesm_metrics_pic['specs'][0]/dtplot,label=lab,c="k")


# # Plot Historical (Ens Avg)
specs_htr = np.array([cesm_metrics_htr['specs'][e] for e in range(nens)])
spec_eavg = specs_htr[1:,].mean(0)
lab = "%s (var=%.3f)" % (ld_cesm['dataset_names'][1],np.nanmean(np.nanvar(ld_cesm['sss'][1],1)))
ax.plot(cesm_metrics_htr['freqs'][e]*dtplot,spec_eavg/dtplot,label=lab,c="gray")

#ax.plot(cesm_metrics_htr['freqs'][e]*dtplot,specs_htr[0,:]/dtplot,label=lab,c="red")

for e in range(nens):
    
    if e == 0:
        continue
    else:
        spec_eavg = cesm_metrics_htr['specs'][e]
    
    ax.plot(cesm_metrics_htr['freqs'][e]*dtplot,cesm_metrics_htr['specs'][e]/dtplot,label="",c="gray",alpha=0.1)
    
    
# spec_eavg = spec_eavg/nens

# lab = "%s (var=%.3f)" % (ld_cesm['dataset_names'][1],np.nanmean(np.nanvar(ld_cesm['sss'][1],1)))
# ax.plot(cesm_metrics_htr['freqs'][e]*dtplot,spec_eavg/dtplot,label=lab,c="gray")


ax.set_xticks(xtks,labels=plotyrs)
ax.set_xlim([xtks[0],xtks[-1]])
ax.legend()
ax.set_xlabel("Period (Years)")
ax.set_ylabel("$psu^2$/cpy")
ax.set_title("SSS Power Spectra @ %s" % (loctitle))
savename = "%sSSS_Model_Power_Spectra_CESM1_Default.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


# ----------------------------------------------------------------------------- # Now, test an experiment
#%% Experiment 1 (Vary Tddamping)

test_param       = "lbd_d"
test_param_name  = r"$\lambda^d$"
test_values      = [1e-3,1e-2,0.05,0.075,0.1,0.12,0.15,0.2,0.3]#[0,1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.3,0.5,0.9]
smconfig_default = smconfig.copy()
expname          = "VarySddamp"

alphas = 0.2 + 0.8*np.linspace(0,1,len(test_values))
exmarkers = ["x","o","+","d",".","v","s","^","+","|","X","*","1","8","2"]


nexps   = len(test_values)
sss_out = []
for ex in range(nexps):
    
    # Retrieve Test Value and assign to dictionary
    test_val = test_values[ex] 
    print("Now Running %s = %f" % (test_param,test_val))
    
    # Assign to Dict
    smconfig = smconfig_default.copy()
    smconfig[test_param] = (test_val * np.ones(12))[None,None,:]
    
    
    # Part (2) Do the Actual Integration --------------------------------------------
    if smconfig['entrain']:
        outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                        Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],
                                        return_dict=True)
    else:
        outdict = scm.integrate_noentrain(smconfig['lbd_a'],smconfig["F"],)
        
    # ---------------------
    
    sss_out.append(outdict['T'].squeeze())

# Compute some metrics
tsmetrics_exp = scm.compute_sm_metrics(sss_out,nsmooth=100)

#%% Examine ACF

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#% Plot the Defaults, copied from above 

kmonth = 1
xtksl  = np.arange(0,37,3)
lags   = np.arange(37)

for kmonth in range(12):
    

    title = "%s SSS ACF @ %s" % (mons3[kmonth],loctitle)
    fig,ax =  plt.subplots(1,1,figsize=(8,4),constrained_layout=True)
    ax,_ = viz.init_acplot(kmonth,xtksl,lags,title=title,ax=ax)
    ax.plot(lags,tsmetrics_default['acfs'][kmonth][0],label="Default Run SSS",color="mediumblue",lw=2,ls='solid')
    
    ax.plot(lags,cesm_metrics_pic['acfs'][kmonth][0],color="k",label=ld_cesm['dataset_names'][0],lw=2)
    
    acf_ensavg = np.zeros(37)
    for e in range(nens):
        if e == 0:
            lab=ld_cesm['dataset_names'][1] + " (Indv. Member)"
        else:
            lab = ""
        plotvar = cesm_metrics_htr['acfs'][kmonth][e]
        ax.plot(lags,plotvar,color="gray",label="",alpha=0.15)
        acf_ensavg += plotvar
    acf_ensavg = acf_ensavg/nens
    lab = ld_cesm['dataset_names'][1] + " (Ens. Avg.)"
    ax.plot(lags,acf_ensavg,color="gray",label=lab,alpha=1,lw=2)
    
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    # Plot the experiment
    
    for ex in range(nexps):
        
        
        plotvar = tsmetrics_exp['acfs'][kmonth][ex]
        lab     = "%s = %f" % (test_param_name,test_values[ex],)
        ax.plot(lags,plotvar,label=lab,lw=.5,ls='dashed',alpha=1,marker=exmarkers[ex],zorder=1)#alpha=alphas[ex])
    
    fig.legend(ncol=4,bbox_to_anchor=(0.975, 1.25), loc="upper right")
    
    savename = "%sSSS_Model_%s_mon%02i.png" % (figpath,expname,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot Input Values

fig,axs = viz.init_monplot(3,1,figsize=(6,8))

ax = axs[0]
ax.plot(mons3,beta,label="Entrainment Damping",c="magenta")
ax.plot(mons3,(hff_l) / (rho*L*h0) * dt * Sbar,label=r"LHFLX Damping ($\lambda^{LHFLX} \, \Delta t \, S_{avg}) / (\rho  L  h}$)",c="red")
ax.set_ylabel("Damping ($mon^{-1}$)")
ax.legend()

ax = axs[1]
ax.plot(mons3,E.copy()/(rho*L*h0)*dt*Sbar,label=r"Evaporation Forcing ($(E \, \Delta t \, S_{avg}) / (\rho  L  h}$)",c="goldenrod")
ax.plot(mons3,P.copy()*dt,label="Precipitation Forcing $(P \Delta t)$",c="blue")
ax.set_ylabel("Effective Forcing ($psu/mon^{-1}$)")
ax.legend()

ax = axs[2]
ax.plot(mons3,h0,label="HMXL (CESM-HTR Ens. Avg.)",c="forestgreen")
ax.plot(mons3,hblt_in/100,label="HBLT (CESM-PIC)",c="cornflowerblue")
ax.set_ylabel("Mixed Layer Depth ($m$)")
ax.legend()


savename = "%sSSS_Model_Default_Variables.png" % (figpath,)
plt.savefig(savename,dpi=150,bbox_inches='tight')


# ----------------------------------------------------------------------------- # Experiment 2
#%% SST Model, does this improve?

# Load some variables (inputs)
Fstd           = np.load(outpath+"CESM1_htr_Fprime_forcing.npy").mean(0) # Take Ensemble Average
ds_qnet        = xr.open_dataset(outpath+"CESM1_htr_qnet_damping.nc").qnet_damping
hff_qnet       = ds_qnet.isel(lag=0).mean('ens').values

# Load some SST ACFs
#% Retrieve the autocorrelation for SST (CESM1 - HTR)
datpath_ac  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
ds          = xr.open_dataset("%sHTR-FULL_%s_autocorrelation_thres0.nc" % (datpath_ac,"SST"))
ds          = ds.sel(lon=lonf,lat=latf,method='nearest').sel(thres="ALL").load()
sst_acf    = ds.SST.values # [Ens x Lag x Mon]
# From Synth Stochmod
cesmauto = np.array([1.        , 0.87030124, 0.70035406, 0.56363062, 0.39430458,
       0.31124585, 0.29375811, 0.27848886, 0.31501385, 0.38280692,
       0.42196468, 0.46813908, 0.4698927 , 0.40145464, 0.32138829,
       0.27078197, 0.19005614, 0.15290666, 0.16341374, 0.15956728,
       0.17240383, 0.1911096 , 0.2213633 , 0.25900566, 0.25901782,
       0.24166312, 0.17903541, 0.14626024, 0.09599903, 0.06320866,
       0.06668568, 0.10627938, 0.10912452, 0.10387345, 0.08380565,
       0.09628464, 0.09792761])


# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#  Copied Section from above

# Forcing/Damping (with unit conversions)
#h0          = h.copy() # Mixed Layer depth (month), units: [meters] # Use the same MLD
alpha       = Fstd     / (rho*cp*h0) * dt #(E.copy()/(rho*L*h0)*dt*Sbar + P*dt ) 
damping_in  = hff_qnet / (rho*cp*h0) * dt #np.zeros(12) # (qL_in) / (rho*L*h_denom) * dt * sbar
lbd_d       = 1e-1 * np.ones(12) 

# Toggles
reset_eta   = False
#use_entrain = True

# Set up Forcing (tile and multiple)
if reset_eta or "eta_in" not in locals():
    print("Regenerating White Noise")
    eta_in        =np.random.normal(0,1,nyrs*12)
forcing_fprime = np.tile(alpha,nyrs) * eta_in

# Additional calculations
beta       = scm.calc_beta(h[None,None,:]).squeeze()
kprev      = scm.find_kprev(h0)[0]

# Set Configuration Parameters (already converted)
tsmconfig = {}
tsmconfig["entrain"] = use_entrain                # True to use entrainment             BOOL
tsmconfig["h"]       = h0[None,None,:]            # Mixed-Layer Depth (used for beta)   (month,) units: [meters]
tsmconfig["forcing"] = forcing_fprime[None,None,:]    # Stochastic Forcing Amplitude.       (time,)  units: [psu/mon]
tsmconfig["lbd_a"]   = damping_in[None,None,:]    # Atmospheric Heat Flux Feedback      (month,) units: [1/mon]
tsmconfig["beta"]    = beta[None,None,:]          # Entrainment Damping                 (month,) units: [1/mon]. By default, this is calculated in the function and added to the overall damping
tsmconfig["lbd_d"]   = lbd_d[None,None,:]         # Damping of value below mixed-layer. (month,) units: [1/mon]
tsmconfig["kprev"]   = kprev[None,None,:]         # Detrainment Month                   (month,) units: [month index]

# Part (2) Do the Actual Integration --------------------------------------------
if smconfig['entrain']:
    outdict = scm.integrate_entrain(tsmconfig['h'],tsmconfig['kprev'],tsmconfig['lbd_a'],tsmconfig['forcing'],
                                    Tdexp=tsmconfig['lbd_d'],beta=tsmconfig['beta'],
                                    return_dict=True)
else:
    outdict = scm.integrate_noentrain(tsmconfig['lbd_a'],tsmconfig["F"],)


# Compute basic metrics
proc_ts = [outdict['T'].squeeze(),]
tsmetrics_default = scm.compute_sm_metrics(proc_ts,nsmooth=100)
#
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#%% Examine ACF (Default SST)

nens = 42
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#% Plot the Defaults, copied from above 

kmonth = 1
xtksl  = np.arange(0,37,3)
lags   = np.arange(37)

for kmonth in range(12):
    
    title = "%s SST ACF @ %s" % (mons3[kmonth],loctitle)
    fig,ax =  plt.subplots(1,1,figsize=(8,4),constrained_layout=True)
    ax,_ = viz.init_acplot(kmonth,xtksl,lags,title=title,ax=ax)
    
    # Plot Default Run
    ax.plot(lags,tsmetrics_default['acfs'][kmonth][0],label="Default Run SST",color="mediumblue",lw=2,ls='solid')
    
    # Plot
    ax.plot(lags,cesmauto,color="k",label=ld_cesm['dataset_names'][0],lw=2)
    
    for e in range(nens):
        if e == 0:
            lab=ld_cesm['dataset_names'][1] + " (Indv. Member)"
        else:
            lab = ""
        plotvar = sst_acf[e,:,kmonth]
        ax.plot(lags,plotvar,color="gray",label="",alpha=0.15)

    acf_ensavg = sst_acf.mean(0)[:,kmonth]
    lab = ld_cesm['dataset_names'][1] + " (Ens. Avg.)"
    ax.plot(lags,acf_ensavg,color="gray",label=lab,alpha=1,lw=2)
    
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    # # Plot the experiment
    
    # for ex in range(nexps):
        
        
    #     plotvar = tsmetrics_exp['acfs'][kmonth][ex]
    #     lab     = "%s = %f" % (test_param_name,test_values[ex],)
    #     ax.plot(lags,plotvar,label=lab,lw=.5,ls='dashed',alpha=1,marker=exmarkers[ex],zorder=1)#alpha=alphas[ex])
    
    fig.legend(ncol=4,bbox_to_anchor=(0.975, 1.25), loc="upper right")
    
    savename = "%sSST_Model_%s_mon%02i.png" % (figpath,expname,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% plot the timeseries
fig,ax = plt.subplots(1,1,figsize=(8,2))
plt.plot(outdict['T'].squeeze())
#plt.xlim([11800,12000])

#%%
#%% Experiment : Compare addition of Sd' damping


test_param       = "lbd_d"
test_param_name  = r"$\lambda^d$"
smconfig_default = smconfig.copy()
expname          = "Sddamp_empirical"

# alphas = 0.2 + 0.8*np.linspace(0,1,len(test_values))
# exmarkers = ["x","o","+","d",".","v","s","^","+","|","X","*","1","8","2"]


nexps   = 4
sss_out = []
for ex in range(nexps):
    
    # Assign to Dict
    smconfig = smconfig_default.copy()
    
    
    if ex == 0:
        smconfig['lbd_d'] = (0.1 * np.ones(12))[None,None,:]
    elif ex == 1:
        smconfig['lbd_d'] = np.abs(lbd_d_ensavg.copy())[None,None,:]
    elif ex == 2:
        #smconfig['lbd_d'] = (0.6 * np.ones(12))[None,None,:]
        smconfig['lbd_d'] = np.abs(lbd_d_ensavg.copy()*10)[None,None,:]
    elif ex == 3:
        lbd_in =(0.13 * np.ones(12))#np.abs(lbd_d_ensavg.copy())
        #lbd_in[lbd_in==0.] = 0.1
        smconfig['lbd_d'] = lbd_in[None,None,:]
    
    # Part (2) Do the Actual Integration --------------------------------------------
    if smconfig['entrain']:
        outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                        Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],
                                        return_dict=True)
    else:
        outdict = scm.integrate_noentrain(smconfig['lbd_a'],smconfig["F"],)
        
    # ---------------------
    
    sss_out.append(outdict['T'].squeeze())

# Compute some metrics
tsmetrics_exp = scm.compute_sm_metrics(sss_out,nsmooth=100)

#%%

nens = 41
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#% Plot the Defaults, copied from above 

kmonth = 1
xtksl  = np.arange(0,37,3)
lags   = np.arange(37)

for kmonth in range(12):
    
    if kmonth != 1:
        continue
    title = "%s SSS ACF @ %s" % (mons3[kmonth],loctitle)
    fig,ax =  plt.subplots(1,1,figsize=(8,4),constrained_layout=True)
    ax,_ = viz.init_acplot(kmonth,xtksl,lags,title=title,ax=ax)
    
    # Plot Default Run
    ax.plot(lags,tsmetrics_exp['acfs'][kmonth][0],label="$\lambda^d=0.1$",color="mediumblue",lw=2,ls='solid')
    
    ax.plot(lags,tsmetrics_exp['acfs'][kmonth][1],label="$\lambda^d=Empirical \,  Estimate$",color="red",lw=2,ls='dashed')
    #ax.plot(lags,tsmetrics_exp['acfs'][kmonth][2],label="$\lambda^d=0.5$",color="orange",lw=2,ls='dashed')
    ax.plot(lags,tsmetrics_exp['acfs'][kmonth][2],label="$\lambda^d=Empirical$ x10",color="orange",lw=2,ls='dashed')
    #ax.plot(lags,tsmetrics_exp['acfs'][kmonth][3],label="$\lambda^d=Empirical \,  Estimate$ Fill Zero",color="violet",lw=2,ls='dashed')
    ax.plot(lags,tsmetrics_exp['acfs'][kmonth][3],label="$\lambda^d=0.2$",color="violet",lw=2,ls='dashed')
    # Plot
    #ax.plot(lags,cesmauto,color="k",label=ld_cesm['dataset_names'][0],lw=2)
    
    acf_ensavg = np.zeros(37)
    for e in range(nens):
        if e == 0:
            lab=ld_cesm['dataset_names'][1] + " (Indv. Member)"
        else:
            lab = ""
        plotvar = cesm_metrics_htr['acfs'][kmonth][e]
        ax.plot(lags,plotvar,color="gray",label="",alpha=0.15)
        acf_ensavg += plotvar
    acf_ensavg = acf_ensavg/nens
    lab = ld_cesm['dataset_names'][1] + " (Ens. Avg.)"
    ax.plot(lags,acf_ensavg,color="k",label=lab,alpha=1,lw=2)
    
    #lab = ld_cesm['dataset_names'][1] + " (Ens. Avg.)"
    #ax.plot(lags,acf_ensavg,color="gray",label=lab,alpha=1,lw=2)
    
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    # # Plot the experiment
    
    # for ex in range(nexps):
        
        
    #     plotvar = tsmetrics_exp['acfs'][kmonth][ex]
    #     lab     = "%s = %f" % (test_param_name,test_values[ex],)
    #     ax.plot(lags,plotvar,label=lab,lw=.5,ls='dashed',alpha=1,marker=exmarkers[ex],zorder=1)#alpha=alphas[ex])
    
    fig.legend(ncol=4,bbox_to_anchor=(0.995, 1.15), loc="upper right")
    #fig.legend(ncol=4,bbox_to_anchor=(0.995, 1.10), loc="upper right")
    #fig.legend(ncol=4,bbox_to_anchor=(0.91, 1.10), loc="upper right")
    savename = "%sSST_Model_%s_mon%02i.png" % (figpath,expname,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

# ------------------------------------------------------------------------------------
#%% Experiment 5. HBLT vs HMXL
# ------------------------------------------------------------------------------------




smconfig_default = smconfig.copy()
expname          = "hblt_vhmxl"

# alphas = 0.2 + 0.8*np.linspace(0,1,len(test_values))
# exmarkers = ["x","o","+","d",".","v","s","^","+","|","X","*","1","8","2"]


nexps   = 2
sss_out = []
for ex in range(nexps):
    
    # Assign to Dict
    smconfig = smconfig_default.copy()
    
    if ex == 0:
        h0 = h.copy() # Mixed Layer depth (month), units: [meters]
    elif ex == 1:
        h0 = np.roll(hblt_in.copy(),0)# 
    
    # ------ Copied from Above (Default Run)
    
    # Forcing/Damping (with unit conversions)
    
    alpha       = np.roll((E.copy()/(rho*L*h0)*dt*Sbar + P*dt ),0)
    damping_in  = np.zeros(12) # (qL_in) / (rho*L*h_denom) * dt * sbar
    lbd_d       = np.abs(lbd_d_ensavg.copy())#1e-1 * np.ones(12) 

    # Toggles
    reset_eta   = False
    use_entrain = True

    # Set up Forcing (tile and multiple)
    if reset_eta or "eta_in" not in locals():
        print("Regenerating White Noise")
        eta_in   = np.random.normal(0,1,nyrs*12)
    forcing_EP   = np.tile(alpha,nyrs) * eta_in

    # Additional calculations
    beta       = scm.calc_beta(h[None,None,:]).squeeze()
    kprev      = scm.find_kprev(h0)[0]

    # Set Configuration Parameters (already converted)
    smconfig = {}
    smconfig["entrain"] = use_entrain                # True to use entrainment             BOOL
    smconfig["h"]       = h0[None,None,:]            # Mixed-Layer Depth (used for beta)   (month,) units: [meters]
    smconfig["forcing"] = forcing_EP[None,None,:]    # Stochastic Forcing Amplitude.       (time,)  units: [psu/mon]
    smconfig["lbd_a"]   = damping_in[None,None,:]    # Atmospheric Heat Flux Feedback      (month,) units: [1/mon]
    smconfig["beta"]    = beta[None,None,:]          # Entrainment Damping                 (month,) units: [1/mon]. By default, this is calculated in the function and added to the overall damping
    smconfig["lbd_d"]   = lbd_d[None,None,:]         # Damping of value below mixed-layer. (month,) units: [1/mon]
    smconfig["kprev"]   = kprev[None,None,:]         # Detrainment Month                   (month,) units: [month index]
    # ------ Copied from Above
    
    
    # Part (2) Do the Actual Integration --------------------------------------------
    if smconfig['entrain']:
        outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                        Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],
                                        return_dict=True)
    else:
        outdict = scm.integrate_noentrain(smconfig['lbd_a'],smconfig["F"],)
        
    # ---------------------
    
    sss_out.append(outdict['T'].squeeze())

# Compute some metrics
tsmetrics_exp = scm.compute_sm_metrics(sss_out,nsmooth=100)

#%%

nens = 41
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#% Plot the Defaults, copied from above 

kmonth = 1
xtksl  = np.arange(0,37,3)
lags   = np.arange(37)

for kmonth in range(12):
    
    if kmonth != 1:
        continue
    title = "%s SSS ACF @ %s" % (mons3[kmonth],loctitle)
    fig,ax =  plt.subplots(1,1,figsize=(8,4),constrained_layout=True)
    ax,_ = viz.init_acplot(kmonth,xtksl,lags,title=title,ax=ax)
    
    # Plot Default Run
    ax.plot(lags,tsmetrics_exp['acfs'][kmonth][0],label="$HMXL$",color="blue",lw=2,ls='solid')
    ax.plot(lags,tsmetrics_exp['acfs'][kmonth][1],label="$HBLT$",color="k",lw=2,ls='dashed')

    # Plot
    #ax.plot(lags,cesmauto,color="k",label=ld_cesm['dataset_names'][0],lw=2)
    
    acf_ensavg = np.zeros(37)
    for e in range(nens):
        if e == 0:
            lab=ld_cesm['dataset_names'][1] + " (Indv. Member)"
        else:
            lab = ""
        plotvar = cesm_metrics_htr['acfs'][kmonth][e]
        ax.plot(lags,plotvar,color="gray",label="",alpha=0.15)
        acf_ensavg += plotvar
    acf_ensavg = acf_ensavg/nens
    lab = ld_cesm['dataset_names'][1] + " (Ens. Avg.)"
    ax.plot(lags,acf_ensavg,color="k",label=lab,alpha=1,lw=2)
    
    #lab = ld_cesm['dataset_names'][1] + " (Ens. Avg.)"
    #ax.plot(lags,acf_ensavg,color="gray",label=lab,alpha=1,lw=2)
    
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    # # Plot the experiment
    
    # for ex in range(nexps):
        
        
    #     plotvar = tsmetrics_exp['acfs'][kmonth][ex]
    #     lab     = "%s = %f" % (test_param_name,test_values[ex],)
    #     ax.plot(lags,plotvar,label=lab,lw=.5,ls='dashed',alpha=1,marker=exmarkers[ex],zorder=1)#alpha=alphas[ex])
    
    #fig.legend(ncol=4,bbox_to_anchor=(0.995, 1.15), loc="upper right")
    #fig.legend(ncol=4,bbox_to_anchor=(0.995, 1.10), loc="upper right")
    fig.legend(ncol=4,bbox_to_anchor=(0.84, 1.10), loc="upper right")
    savename = "%sSST_Model_%s_mon%02i.png" % (figpath,expname,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------


#%%

nens = 41
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#% Plot the Defaults, copied from above 

kmonth = 1
xtksl  = np.arange(0,37,3)
lags   = np.arange(37)

for kmonth in range(12):
    
    if kmonth != 1:
        continue
    title = "%s SSS ACF @ %s" % (mons3[kmonth],loctitle)
    fig,ax =  plt.subplots(1,1,figsize=(8,4),constrained_layout=True)
    ax,_ = viz.init_acplot(kmonth,xtksl,lags,title=title,ax=ax)
    
    # Plot Default Run
    ax.plot(lags,tsmetrics_exp['acfs'][kmonth][0],label="$HMXL$",color="blue",lw=2,ls='solid')
    ax.plot(lags,tsmetrics_exp['acfs'][kmonth][1],label="$HBLT$",color="k",lw=2,ls='dashed')

    # Plot
    #ax.plot(lags,cesmauto,color="k",label=ld_cesm['dataset_names'][0],lw=2)
    
    acf_ensavg = np.zeros(37)
    for e in range(nens):
        if e == 0:
            lab=ld_cesm['dataset_names'][1] + " (Indv. Member)"
        else:
            lab = ""
        plotvar = cesm_metrics_htr['acfs'][kmonth][e]
        ax.plot(lags,plotvar,color="gray",label="",alpha=0.15)
        acf_ensavg += plotvar
    acf_ensavg = acf_ensavg/nens
    lab = ld_cesm['dataset_names'][1] + " (Ens. Avg.)"
    ax.plot(lags,acf_ensavg,color="k",label=lab,alpha=1,lw=2)
    
    #lab = ld_cesm['dataset_names'][1] + " (Ens. Avg.)"
    #ax.plot(lags,acf_ensavg,color="gray",label=lab,alpha=1,lw=2)
    
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    # # Plot the experiment
    
    # for ex in range(nexps):
        
        
    #     plotvar = tsmetrics_exp['acfs'][kmonth][ex]
    #     lab     = "%s = %f" % (test_param_name,test_values[ex],)
    #     ax.plot(lags,plotvar,label=lab,lw=.5,ls='dashed',alpha=1,marker=exmarkers[ex],zorder=1)#alpha=alphas[ex])
    
    #fig.legend(ncol=4,bbox_to_anchor=(0.995, 1.15), loc="upper right")
    #fig.legend(ncol=4,bbox_to_anchor=(0.995, 1.10), loc="upper right")
    fig.legend(ncol=4,bbox_to_anchor=(0.84, 1.10), loc="upper right")
    savename = "%sSST_Model_%s_mon%02i.png" % (figpath,expname,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
#%% Experiment 6 Try shifting forcing damping
# ------------------------------------------------------------------------------------


smconfig_default = smconfig.copy()
expname          = "forcingshift"

# alphas = 0.2 + 0.8*np.linspace(0,1,len(test_values))
# exmarkers = ["x","o","+","d",".","v","s","^","+","|","X","*","1","8","2"]

nexps = 4

sss_out = []
for ex in range(nexps):
    
    # Assign to Dict
    #smconfig    = smconfig_default.copy()
    
    # Forcing/Damping (with unit conversions)
    alpha       = np.roll((E.copy()/(rho*L*h0)*dt*Sbar + P*dt ),0)
    damping_in  = np.zeros(12) # (qL_in) / (rho*L*h_denom) * dt * sbar
    h0          = hblt_in.copy() # Mixed Layer depth (month), units: [meters]
    lbd_d       = lbd_d_ensavg#1e-1 * np.ones(12) 
    
    
    if ex == 0:
        alpha = np.roll(alpha,1)
    elif ex == 1:
        alpha = np.roll(alpha,1)
        h0 = np.roll(h0,1)
    
    
    # ------ Copied from Above (Default Run)
    
    

    
    

    # Toggles
    reset_eta   = False
    use_entrain = True

    # Set up Forcing (tile and multiple)
    if reset_eta or "eta_in" not in locals():
        print("Regenerating White Noise")
        eta_in   = np.random.normal(0,1,nyrs*12)
    forcing_EP   = np.tile(alpha,nyrs) * eta_in

    # Additional calculations
    beta       = scm.calc_beta(h[None,None,:]).squeeze()
    kprev      = scm.find_kprev(h0)[0]

    # Set Configuration Parameters (already converted)
    smconfig = {}
    smconfig["entrain"] = use_entrain                # True to use entrainment             BOOL
    smconfig["h"]       = h0[None,None,:]            # Mixed-Layer Depth (used for beta)   (month,) units: [meters]
    smconfig["forcing"] = forcing_EP[None,None,:]    # Stochastic Forcing Amplitude.       (time,)  units: [psu/mon]
    smconfig["lbd_a"]   = damping_in[None,None,:]    # Atmospheric Heat Flux Feedback      (month,) units: [1/mon]
    smconfig["beta"]    = beta[None,None,:]          # Entrainment Damping                 (month,) units: [1/mon]. By default, this is calculated in the function and added to the overall damping
    smconfig["lbd_d"]   = lbd_d[None,None,:]         # Damping of value below mixed-layer. (month,) units: [1/mon]
    smconfig["kprev"]   = kprev[None,None,:]         # Detrainment Month                   (month,) units: [month index]
    # ------ Copied from Above
    
    
    # Part (2) Do the Actual Integration --------------------------------------------
    if smconfig['entrain']:
        outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                        Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],
                                        return_dict=True)
    else:
        outdict = scm.integrate_noentrain(smconfig['lbd_a'],smconfig["F"],)
        
    # ---------------------
    
    sss_out.append(outdict['T'].squeeze())

# Compute some metrics
tsmetrics_exp = scm.compute_sm_metrics(sss_out,nsmooth=100)
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
#%% Experiment 7. Try parameter sets for all ensembl emebers
# ------------------------------------------------------------------------------------


smconfig_default = smconfig.copy()
expname          = "hblt_vhmxl"

# alphas = 0.2 + 0.8*np.linspace(0,1,len(test_values))
# exmarkers = ["x","o","+","d",".","v","s","^","+","|","X","*","1","8","2"]


nexps   = 42
sss_out = []
for ex in range(nexps):
    
    # Assign to Dict
    #smconfig    = smconfig_default.copy()
    
    h0          = hclim[:,e].copy() # Mixed Layer depth (month), units: [meters]
    lbd_d       = ld['lbd_d'][e,:].copy()#1e-1 * np.ones(12) 
    
    # ------ Copied from Above (Default Run)
    
    # Forcing/Damping (with unit conversions)
    
    alpha       = np.roll((E.copy()/(rho*L*h0)*dt*Sbar + P*dt ),0)
    damping_in  = np.zeros(12) # (qL_in) / (rho*L*h_denom) * dt * sbar
    

    # Toggles
    reset_eta   = False
    use_entrain = True

    # Set up Forcing (tile and multiple)
    if reset_eta or "eta_in" not in locals():
        print("Regenerating White Noise")
        eta_in   = np.random.normal(0,1,nyrs*12)
    forcing_EP   = np.tile(alpha,nyrs) * eta_in

    # Additional calculations
    beta       = scm.calc_beta(h[None,None,:]).squeeze()
    kprev      = scm.find_kprev(h0)[0]

    # Set Configuration Parameters (already converted)
    smconfig = {}
    smconfig["entrain"] = use_entrain                # True to use entrainment             BOOL
    smconfig["h"]       = h0[None,None,:]            # Mixed-Layer Depth (used for beta)   (month,) units: [meters]
    smconfig["forcing"] = forcing_EP[None,None,:]    # Stochastic Forcing Amplitude.       (time,)  units: [psu/mon]
    smconfig["lbd_a"]   = damping_in[None,None,:]    # Atmospheric Heat Flux Feedback      (month,) units: [1/mon]
    smconfig["beta"]    = beta[None,None,:]          # Entrainment Damping                 (month,) units: [1/mon]. By default, this is calculated in the function and added to the overall damping
    smconfig["lbd_d"]   = lbd_d[None,None,:]         # Damping of value below mixed-layer. (month,) units: [1/mon]
    smconfig["kprev"]   = kprev[None,None,:]         # Detrainment Month                   (month,) units: [month index]
    # ------ Copied from Above
    
    
    # Part (2) Do the Actual Integration --------------------------------------------
    if smconfig['entrain']:
        outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                        Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],
                                        return_dict=True)
    else:
        outdict = scm.integrate_noentrain(smconfig['lbd_a'],smconfig["F"],)
        
    # ---------------------
    
    sss_out.append(outdict['T'].squeeze())

# Compute some metrics
tsmetrics_exp = scm.compute_sm_metrics(sss_out,nsmooth=100)



#%%

nens = 41
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#% Plot the Defaults, copied from above 

kmonth = 1
xtksl  = np.arange(0,37,3)
lags   = np.arange(37)

for kmonth in range(12):
    
    if kmonth != 1:
        continue
    
    title = "%s SSS ACF @ %s" % (mons3[kmonth],loctitle)
    fig,ax =  plt.subplots(1,1,figsize=(8,4),constrained_layout=True)
    ax,_ = viz.init_acplot(kmonth,xtksl,lags,title=title,ax=ax)
    
    # Plot Default Run
    for e in range(nens):
        ax.plot(lags,tsmetrics_exp['acfs'][kmonth][e],label="",color="blue",lw=2,ls='solid',alpha=0.4)
    #ax.plot(lags,tsmetrics_exp['acfs'][kmonth][0],label="$HMXL$",color="blue",lw=2,ls='solid')
    #ax.plot(lags,tsmetrics_exp['acfs'][kmonth][1],label="$HBLT$",color="k",lw=2,ls='dashed')

    # Plot
    #ax.plot(lags,cesmauto,color="k",label=ld_cesm['dataset_names'][0],lw=2)
    
    acf_ensavg = np.zeros(37)
    for e in range(nens):
        if e == 0:
            lab=ld_cesm['dataset_names'][1] + " (Indv. Member)"
        else:
            lab = ""
        plotvar = cesm_metrics_htr['acfs'][kmonth][e]
        ax.plot(lags,plotvar,color="gray",label="",alpha=0.15)
        acf_ensavg += plotvar
    acf_ensavg = acf_ensavg/nens
    lab = ld_cesm['dataset_names'][1] + " (Ens. Avg.)"
    ax.plot(lags,acf_ensavg,color="k",label=lab,alpha=1,lw=2)
    
    #lab = ld_cesm['dataset_names'][1] + " (Ens. Avg.)"
    #ax.plot(lags,acf_ensavg,color="gray",label=lab,alpha=1,lw=2)
    
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    # # Plot the experiment
    
    # for ex in range(nexps):
        
        
    #     plotvar = tsmetrics_exp['acfs'][kmonth][ex]
    #     lab     = "%s = %f" % (test_param_name,test_values[ex],)
    #     ax.plot(lags,plotvar,label=lab,lw=.5,ls='dashed',alpha=1,marker=exmarkers[ex],zorder=1)#alpha=alphas[ex])
    
    #fig.legend(ncol=4,bbox_to_anchor=(0.995, 1.15), loc="upper right")
    #fig.legend(ncol=4,bbox_to_anchor=(0.995, 1.10), loc="upper right")
    fig.legend(ncol=4,bbox_to_anchor=(0.84, 1.10), loc="upper right")
    savename = "%sSST_Model_%s_mon%02i.png" % (figpath,expname,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')


