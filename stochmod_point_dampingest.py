#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Different ways to estimate damping at the stochastic model point

Compare the following damping estimates

1. Direct Fit to SSS
2. Covariance based (normalized by Cov(T,S))
3. Covariance based (normalized by Cov(S,S))
4. Lambda_E (theoretical damping)

Copied structure of stochmod_point.py on 2023.09.21

Created on Thu Sep 21 14:20:33 2023

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% User Edits

# Location
lonf           = 330
latf           = 50
locfn,loctitle = proc.make_locstring(lonf,latf)

datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (lonf,latf)
figpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20230929/"
proc.makedir(figpath)

# Variable Information
flxs           = ["LHFLX","SHFLX","FLNS","FSNS","qnet"]
prcps          = ["PRECC","PRECL","PRECSC","PRECSL",]
varnames_ac    = ["SST","SSS"]

# ENSO Information
ensopath       = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/hff_calculations/htr_ENSO_pcs3/"
ensoname       = "htr_ENSO_detrend1_pcs3_1920to2006_ens%02i.npz"
ensolag        = 1
monwin         = 3
reduceyr       = True

# Other plotting information
mons3          = proc.get_monstr(nletters=3)

debug          = True

#%% Preprocessing Function

def preproc_pt(ds):
    
    # Detrend (by removing ensemble mean)
    ds_dt = ds - ds.mean('ens')
    
    # Remove seasonal average
    varpt             = ds_dt.values
    scycle,varpt_ds = proc.calc_clim(varpt,dim=1,returnts=True)
    varpt_dtds        = varpt_ds - scycle[:,None,:]
    return varpt_dtds

def init_annplot(figsize=(6,4),ax=None,figflag=False):
    if ax is None:
        figflag=True
        fig,ax = plt.subplots(1,1,figsize=figsize,constrained_layout=True)
    ax.set_xlim([0,11])
    ax.set_xticks(np.arange(0,12))
    ax.set_xticklabels(mons3)
    ax = viz.add_ticks(ax)
    if figflag:
        return fig,ax
    else:
        return fig
# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><
#%% Parameter Loading Section
# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><

# Get the <Fluxes>
flxs_ds = {}
flxa_ds = {}
for f in range(5):
    vname = flxs[f]
    if vname == "qnet":
        ds_new = flxs_ds['FSNS'] - (flxs_ds['LHFLX'] + flxs_ds['SHFLX'] + flxs_ds['FLNS'])
        ds     = ds_new.rename(vname)
    else:
        ds = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,vname))[vname].load() # [ens x time]

    # Compute the values
    flxs_ds[vname] = ds.copy()
    
    # Compute the anomaly
    dsa = preproc_pt(ds)
    flxa_ds[vname] = dsa.copy()

#%% Get the <SSS and SST> fields
acvars_ds = {}
acvars_np = []
for f in range(len(varnames_ac)):
    vname = varnames_ac[f]
    ds = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,vname))[vname].load() # [ens x time]
    times = ds.time.values
    ds = preproc_pt(ds)
    acvars_ds[vname] = ds.copy() # [ens x year x mon]
    acvars_np.append(ds)
    
#%% Load the ENSO information

enso_load = []
for e in range(42):
    ld = np.load(ensopath + ensoname % (e+1),allow_pickle=True)
    enso_load.append(ld)
    
ensoids = np.array([enso_load[e]['pcs'] for e in range(42)]) # [ens x year x mon x pc]

# -----------------------------------------------------------------------------
#%% REMOVE ENSO
# -----------------------------------------------------------------------------

varanoms = acvars_ds.copy()
varanoms.update(flxa_ds)
varnames = list(varanoms.keys())

# Loop over each variable and remove ENSO
nvars    = len(varnames)
nens     = ensoids.shape[0]
varanoms_noenso = {}
ensocomp_rem    = {}
for v in range(nvars):
    vname = varnames[v]
    
    # Remove enso, one ariable at a time
    vproc     = []
    ensocomps = []
    for e in range(nens):
        
        # Get the variable and enso index for the given ensemble member
        ensoid = ensoids[e,:,:,:] # [year x mon x pc]
        varin  = varanoms[vname][e,...].squeeze() # [year x mon]
        
        # Do some silly preprocessing to fit into the function
        nyr,nmon = varin.shape
        varin    = varin.reshape(nyr*nmon)[:,None,None] # [Time x Lat x Lon]
        
        # Remove ENSO
        times_in = np.arange(0,len(times)) # New times will be [24:-12]
        vout,ensopattern,times_rem = scm.remove_enso(varin,ensoid,ensolag,monwin,reduceyr=reduceyr,times=times_in)
        
        # Recover ENSO component [yr x mon x pc] * [None x mon x pc]
        ensocomp = (ensoid * ensopattern.squeeze()[None,...]).reshape(nyr*nmon,3) # [time x pc]
        
        # Save Output
        vproc.append(vout.squeeze()) # [time]
        ensocomps.append(ensocomp) # [time x pc]
        
        # Check removal
        if debug and (v == 0) and (e == 0):
            fig,ax =plt.subplots(1,1)
            ax.plot(times_in,varin.squeeze(),label="Raw")
            ax.plot(times_rem,vout.squeeze(),label="Proc")
            ## Original variable is one month ahead of enso component, and first+last year dropped [24:-12]
            ## ENSO component is one year behind (time lag for Atl --> Pac communication. forst + last year dropped [12:-24])
            ax.plot(times_rem,varin.squeeze()[24:-12] -ensocomp.sum(1)[12:-24],label="Subtr")
            ax.set_xticks(np.arange(0,len(times_in),12))
            ax.set_xlim([24,72])
            ax.grid(True)
            ax.legend()
        # <END Ensemble Loop>
    varanoms_noenso[vname] = np.array(vproc)     # [ens x time]
    ensocomp_rem[vname]    = np.array(ensocomps) # [ens x time x pc]
    # <END Variable Loop>

#%% Save the anoms (ENSO removed)


savename = "%sCESM1_htr_vars_noenso.npz" % datpath
np.savez(savename,**varanoms_noenso,allow_pickle=True)


# -----------------------------------------------------------------------------
#%% Estimate HFF for 1 ens mem.
# -----------------------------------------------------------------------------
e = 0

# First estimate the temperature damping --------------------------------------
lbda_dict = {}
for f in range(len(flxs)):
    
    # Get variables
    fname = flxs[f]
    flxin = varanoms_noenso[fname][e,:] # [Time]
    sstin = varanoms_noenso['SST'][e,:]
    
    # Reshape to [yr x mon x lat x lon]
    invars = [flxin,sstin]
    invars = [iv.reshape(int(iv.shape[0]/12),12,1,1) for iv in invars]
    flxin,sstin = invars
    
    hff_dict = scm.calc_HF(sstin,flxin,[1,2,3],monwin,return_dict=True,return_cov=True)
    lbda_dict[fname] = hff_dict
print(hff_dict.keys())


#%% Next, estimate the damping for SSS directly

lbds_dict = {}
for f in range(len(flxs)):
    
    # Get variables
    fname = flxs[f]
    flxin = varanoms_noenso[fname][e,:] # [Time]
    sssin = varanoms_noenso['SSS'][e,:]
    
    # Reshape to [yr x mon x lat x lon]
    invars = [flxin,sssin]
    invars = [iv.reshape(int(iv.shape[0]/12),12,1,1) for iv in invars]
    flxin,sssin = invars
    
    hff_dict = scm.calc_HF(sssin,flxin,[1,2,3],monwin,return_dict=True,return_cov=True)
    lbds_dict[fname] = hff_dict
print(hff_dict.keys())

#%% Next, estimate SSS damping replace S in the denominator

lbdts_dict = {}
for f in range(len(flxs)):
    
    # Get variables
    fname = flxs[f]
    flxin = varanoms_noenso[fname][e,:] # [Time]
    sssin = varanoms_noenso['SSS'][e,:]
    sstin = varanoms_noenso['SST'][e,:]
    
    # Reshape to [yr x mon x lat x lon]
    invars = [flxin,sssin,sstin]
    invars = [iv.reshape(int(iv.shape[0]/12),12,1,1) for iv in invars]
    flxin,sssin,sstin = invars
    
    hff_dict = scm.calc_HF(sssin,flxin,[1,2,3],monwin,return_dict=True,return_cov=True,var_denom=sstin)
    lbdts_dict[fname] = hff_dict
print(hff_dict.keys())

#%% Compare the values (for that ensemble member)


varin = lbds_dict['SHFLX']['damping'].squeeze() # [mon x lag]

#lbdts_dict['LHFLX']['damping'].squeeze() # [mon x lag]

fig,ax = fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))

for l in range(3):
    
    ax.plot(mons3,varin[:,l],label="Lambda TS (lag%i)" % (l+1))
    
ax.legend()

#%% Understand the [covariance] value differences


dnames = ['damping', 'autocorr', 'crosscorr', 'autocovall', 'covall']
vname = 'LHFLX'


for v in range(len(flxs)):
    for d in range(len(dnames)):
        
        
        dname = dnames[d]
        vname = flxs[v]
        #dname = "autocorr" # dict_keys(['damping', 'autocorr', 'crosscorr', 'autocovall', 'covall'])
        if dname == "covall":
            dname_plot="Numerator"
        elif dname == "autocovall":
            dname_plot="Denominator"
        elif dname == "damping":
            dname_plot="Damping"
        elif dname == "crosscorr":
            dname_plot="Cross-Correlation"
        elif dname == "autocorr":
            dname_plot="Autocorrelation"
            
        
        
        fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(12,12))
        
        for a in range(3):
            ax = axs[a]
            
            if a == 0:
                varin = lbda_dict[vname][dname].squeeze()
                title = "$Cov(T_0,Q_1)$ / $Cov(T_0,T_1)$"
            elif a == 1:
                varin = lbds_dict[vname][dname].squeeze()
                title = "$Cov(S_0,Q_1)$ / $Cov(S_0,S_1)$"
            elif a == 2:
                varin = lbdts_dict[vname][dname].squeeze()
                title = "$Cov(S_0,Q_1)$ / $Cov(S_0,T_1)$"
                
            for l in range(3):
                ax.plot(mons3,varin[:,l],label="lag=%i" % (l+1))
            ax.legend()
            ax.set_title(title)
            ax.grid(True,ls='dotted')
            
        plt.suptitle("Flux: %s, Variable = %s" % (vname,dname_plot))
        savename = "%sPoint_Estimate_HFF_%s_%s.png" % (figpath,vname,dname)
        plt.savefig(savename,dpi=150,bbox_inches='tight')
#%% Check values

# Cov(T,Q)
np.cov(varanoms_noenso['SST'][0,:],varanoms_noenso['LHFLX'][0,:])[0,1]
np.cov(varanoms_noenso['SST'][0,:-1],varanoms_noenso['LHFLX'][0,1:])[0,1]

# Cov(S,Q)
np.cov(varanoms_noenso['SSS'][0,:],varanoms_noenso['LHFLX'][0,:])[0,1]
np.cov(varanoms_noenso['SSS'][0,:-1],varanoms_noenso['LHFLX'][0,1:])[0,1]

# Cov(S,S)
np.cov(varanoms_noenso['SSS'][0,:],varanoms_noenso['SSS'][0,:])[0,1]
np.cov(varanoms_noenso['SSS'][0,:-1],varanoms_noenso['SSS'][0,1:])[0,1]

# Cov(T,T)
np.cov(varanoms_noenso['SST'][0,:],varanoms_noenso['SST'][0,:])[0,1]
np.cov(varanoms_noenso['SST'][0,:-1],varanoms_noenso['SST'][0,1:])[0,1]

# Cov(T,S)
np.cov(varanoms_noenso['SSS'][0,:],varanoms_noenso['SST'][0,:])[0,1]
np.cov(varanoms_noenso['SSS'][0,:-1],varanoms_noenso['SST'][0,1:])[0,1]

#%% Repeat lambda_a damping estimation above, but do it for all ens (copying above re-estimate)


# First estimate the temperature damping --------------------------------------
lbda_dict = {}  # Dictionary of calc_HFF output
lbda_est  = {}  # Just the damping
for f in range(len(flxs)):
    
    # Get variables
    fname = flxs[f]
    hffs_ens = []
    hffs_np  = np.zeros((nens,12,3))
    for e in range(42):
        
        flxin = varanoms_noenso[fname][e,:] # [Time]
        sstin = varanoms_noenso['SST'][e,:]
        
        # Reshape to [yr x mon x lat x lon]
        invars = [flxin,sstin]
        invars = [iv.reshape(int(iv.shape[0]/12),12,1,1) for iv in invars]
        flxin,sstin = invars
        
        hff_dict = scm.calc_HF(sstin,flxin,[1,2,3],monwin,return_dict=True,return_cov=True)
        hffs_ens.append(hff_dict)
        hffs_np[e,:,:] = hff_dict['damping'].squeeze()
    lbda_dict[fname] = hffs_ens
    lbda_est[fname]  = hffs_np
print(hff_dict.keys())


#%% Compute stochastic LHF forcing (q' = Q'_L + lambda_LH * T')

# Get the variables (just take the first lag)
Q_L     = varanoms_noenso['LHFLX'] # [Ens x Time]
Tprime  = varanoms_noenso['SST'] # [Ens x Time]
lbd_lhf = lbda_est['LHFLX'].squeeze()[:,:,0] # [ens x mon x lag]

# Tile the damping
nyrs_rem = int(Tprime.shape[1]/12)
lbd_tile = np.tile(lbd_lhf,nyrs_rem)

# Calculate stochastic q'
q_L     = Q_L + lbd_tile[None,:] * Tprime

# Check the amplitude
q_L_monyear = q_L.reshape(nens,nyrs_rem,12) # [Ens x Yr x Mon]
q_L_std     = q_L_monyear.std(1) # [Ens x mon]

# Save the forcing
savename = "%sCESM1_htr_qL_forcing.npy" % datpath
np.save(savename,q_L_std)

#%% Also compute Fprime for consistency

# Load variables
QNET     = varanoms_noenso['qnet'] # [Ens x Time]
lbd_qnet = lbda_est['qnet'].squeeze()[:,:,0] # [ens x mon x lag]

# Tile the damping
nyrs_rem = int(Tprime.shape[1]/12)
lbd_tile = np.tile(lbd_qnet,nyrs_rem)

# Calculate stochastic F'
Fprime     = QNET + lbd_tile[None,:] * Tprime

# Check the amplitude
Fprime_monyear = Fprime.reshape(nens,nyrs_rem,12) # [Ens x Yr x Mon]
Fprime_std     = Fprime_monyear.std(1) # [Ens x mon]
print(Fprime_std)

# Save the forcing
savename = "%sCESM1_htr_Fprime_forcing.npy" % datpath
np.save(savename,Fprime_std)


#%% Save Atmospheric T damping
savename = "%sCESM1_htr_lbda_reestimate.npz" % datpath
np.savez(savename,**lbda_est,allow_pickle=True)

# -----------------------------------------------------------------------------
#%% Additional Analysis
# -----------------------------------------------------------------------------
# Visualize the relationship between T and S
e = 0
rhos = []
for e in range(42):
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,8))
    ts_T = varanoms_noenso['SST'][e,:]
    ts_S = varanoms_noenso['SSS'][e,:]
    rho  = np.corrcoef(ts_T,ts_S)[0,1]
    ax.scatter(ts_T,ts_S,alpha=0.25)
    ax.set_ylabel("S'")
    ax.set_xlabel("T'")
    ax.set_title(r"T' vs S' (ens%02i, $\rho$=%.2f)" % (e+1,rho))
    
    
    #ax.plot([-3,3],[-3,3],color="k",ls='dotted')
    ax.axhline([0],ls='dotted',color="k")
    ax.axvline([0],ls='dotted',color="k")
    
    ax.set_xlim([-3,3])
    ax.set_ylim([-.4,.4])
    figname = "%sTS_Scatter_ens%02i.png" % (figpath,e+1)
    plt.savefig(figname,bbox_inches='tight',dpi=150)
    rhos.append(rho)
rhos = np.array(rhos)

#%% Plot a histogram of T-S Correlation

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,8))
ax.hist(rhos,bins=5,edgecolor='w',alpha=0.5)
ax.set_xlabel("T-S Correlation")
ax.set_ylabel("Frequency")

#%% Esimate SSS Feedback, but for all ensemble members

lbds_dict = {}  # Dictionary of calc_HFF output
lbds_est  = {}  # Just the damping
for f in range(len(flxs)):
    
    # Get variables
    fname = flxs[f]
    hffs_ens = []
    hffs_np  = np.zeros((nens,12,3))
    for e in range(42):
        
        flxin = varanoms_noenso[fname][e,:] # [Time]
        sssin = varanoms_noenso['SSS'][e,:]
        
        # Reshape to [yr x mon x lat x lon]
        invars = [flxin,sssin]
        invars = [iv.reshape(int(iv.shape[0]/12),12,1,1) for iv in invars]
        flxin,sssin = invars
        
        hff_dict = scm.calc_HF(sssin,flxin,[1,2,3],monwin,return_dict=True,return_cov=True)
        hffs_ens.append(hff_dict)
        hffs_np[e,:,:] = hff_dict['damping'].squeeze()
    lbds_dict[fname] = hffs_ens
    lbds_est[fname]  = hffs_np
print(hff_dict.keys())

#%% Save SSS Damping Feedback
savename = "%sCESM1_htr_lbds_reestimate.npz" % datpath
np.savez(savename,**lbds_est,allow_pickle=True)


#%% Visualize estimated values, compare with T


cp      = 3996
rho     = 1026
L       = 2.5e6
h       = 70
Sbar    = 36
dt      = 3600*24*30

lbds_lhf = lbds_est['LHFLX'][:,:,0]
lbda_lhf = lbda_est['LHFLX'][:,:,0]

sconv    = Sbar / (rho*L*h)  * dt
tconv    = 1    / (rho*cp*h) * dt



fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,8),sharey=False)

ax = axs[0]
for e in range(42):
    ax.plot(mons3,1/(lbds_lhf[e,:]*sconv))
    #ax.axhline([1/(lbds_lhf[e,:]*sconv).mean()])
    #ax.set_ylim([2,550])
    
ax = axs[1]
for e in range(42):
    ax.plot(mons3,1/(lbda_lhf[e,:]*tconv))

#%% SCATTERPLOT: Correlation/Covariance between T and QL, S and QL

make_plot = False
covs_t  = []
corrs_t = []

covs_s  = []
corrs_s = []

for e in range(42):
    
    sst_in = varanoms_noenso['SST'][e,:-1]
    lhf_in = varanoms_noenso['LHFLX'][e,1:]
    sss_in =  varanoms_noenso['SSS'][e,:-1]
    
    
    # Calculate T,Q correlation/covariance
    covt  = np.cov(sst_in,lhf_in)[0,1]
    corrt = np.corrcoef(sst_in,lhf_in)[0,1]
    
    # Repeat for S
    covs  = np.cov(sss_in,lhf_in)[0,1]
    corrs = np.corrcoef(sss_in,lhf_in)[0,1]
    
    # Save output
    covs_t.append(covt)
    corrs_t.append(corrt)
    covs_s.append(covs)
    corrs_s.append(corrs)
    
    # Make the Plot
    if make_plot:
        fig,axs = plt.subplots(1,2,constrained_layout=True,sharey=True,figsize=(12,6))
        
        ax = axs[0]
        ax.scatter(sst_in,lhf_in,alpha=0.5)
        ax.set_title("Cov(T,Q) = %f\nCorr(T,Q) = %f" % (covt,corrt))
        ax.set_xlim([-4,4])
        ax.axhline([0],lw=0.5,c="k")
        ax.axvline([0],lw=0.5,c="k")
        ax.set_ylabel("Latent Heat Flux $\tau=-1$(W/m2), ens%02i" % (e+1))
        ax.set_xlabel("SST ($\degree C$)")
        
        ax = axs[1]
        ax.scatter(sss_in,lhf_in,alpha=0.5)
        ax.set_title("Cov(S,Q) = %f\nCorr(S,Q) = %f" % (covs,corrs))
        ax.set_xlabel("SSS (psu)")
        
        ax.set_ylim([-100,100])
        ax.set_xlim([-.4,.4])
        ax.axhline([0],lw=0.5,c="k")
        ax.axvline([0],lw=0.5,c="k")
        savename = "%sLHFLX_COV_CORR_scatter_ens%02i.png" % (figpath,e+1)
        plt.savefig(savename,dpi=150,bbox_inches='')


#%% HISTOGRAM of QL vs. S and T Corr/Cov


fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(16,6))

ax = axs[0]
ax.hist(covs_t,edgecolor="w",alpha=0.55,color='orange')

ax = axs[1]
ax.hist(covs_s,edgecolor="w",alpha=0.55,color='b')

ax = axs[2]
ax.hist(corrs_t,edgecolor="w",alpha=0.55,color='orange',label="Corr(T,Q)")
ax.hist(corrs_s,edgecolor="w",alpha=0.55,color='b',label="Corr(S,Q)")
ax.legend()

#%% Calc N_eff for Q, T

p     = 0.05
tails = 2
n_tot   = len(varanoms_noenso['SST'][0,:])
dof_t_all = []
dof_s_all = []

rhocrit_t = []
rhocrit_s = []
for e in range(42):
    n_eff_t = proc.calc_dof(varanoms_noenso['SST'][e,:],ts1=varanoms_noenso['LHFLX'][e,:])
    n_eff_s = proc.calc_dof(varanoms_noenso['SSS'][e,:],ts1=varanoms_noenso['LHFLX'][e,:])
    dof_t_all.append(n_eff_t)
    dof_s_all.append(n_eff_s)
    
    
    rhocrit_t.append(proc.ttest_rho(p,tails,n_eff_t))
    rhocrit_s.append(proc.ttest_rho(p,tails,n_eff_s))


#%%
    
#%% Histogram of Correlations for T and S

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(5,8),sharex=True)

bins = np.arange(0,0.52,0.02)
xtks = np.arange(0,0.6,0.1)

ax   = axs[0]
ax.hist(corrs_t,bins=bins,edgecolor="w",alpha=0.55,color='orange',label="Corr(T,Q)")
ax.axvline([np.nanmax(rhocrit_t)],color="r",label=r"$\rho$=%.3f" % (np.nanmax(rhocrit_t)))
ax.legend(fontsize=14)
ax.set_ylabel("Frequency")
title = r"DOF: [%i,%i], $\rho$: [%.3f,%.3f]" % (np.nanmin(dof_t_all),
                                           np.nanmax(dof_t_all),
                                           np.nanmin(rhocrit_t),
                                           np.nanmax(rhocrit_t),
                                           )
ax.set_title(title)


ax   = axs[1]
ax.hist(corrs_s,bins=bins,edgecolor="w",alpha=0.55,color='b',label="Corr(S,Q)")
ax.axvline([np.nanmax(rhocrit_s)],color="r",label=r"$\rho$=%.3f" % (np.nanmax(rhocrit_s)))
ax.legend(fontsize=14)
ax.set_xticks(xtks)
ax.set_xlabel("Cross-Correlation")
title = r"DOF: [%i,%i], $\rho$: [%.3f,%.3f]" % (np.nanmin(dof_s_all),
                                           np.nanmax(dof_s_all),
                                           np.nanmin(rhocrit_s),
                                           np.nanmax(rhocrit_s),
                                           )
ax.set_title(title)
plt.suptitle("Cross Correlation with LHFLX \n p=%.2f tails=%i" % (p,tails))
figname = "%sTS_LHFLX_Corr.png" % (figpath)
plt.savefig(figname,bbox_inches='tight',dpi=150)

