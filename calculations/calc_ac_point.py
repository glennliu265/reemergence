#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Examine Autocorrelation and thresholds for a single point

copied from pointwise_autocorrelation.py (2022.09.29)

Created on Thu Sep 29 12:06:04 2022

@author: gliu

"""

import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Select dataset to postprocess

# Set Machine
# -----------
stormtrack = 0 # Set to True to run on stormtrack, False for local run

# Autocorrelation parameters
# --------------------------
lags        = np.arange(0,61)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

mconfig    = "PIC-FULL"#"HadISST" #["PIC-FULL","HTR-FULL","PIC_SLAB","HadISST","ERSST"]
runid      = 9
thresholds = [0,]
thresname  = "thres" + "to".join(["%i" % i for i in thresholds])
varname    = "SSS" # ["TS","SSS","SST]

# Set to False to not apply a mask (otherwise specify path to mask)
loadmask   = False #"/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/limask180_FULL-HTR.npy"
glonpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lon180.npy"
glatpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lat.npy"

# Load another variable to compare thresholds (might need to manually correct)
thresvar      = True #
thresvar_name = "HMXL"  
thresvar_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/thresholdvar/HMXL_FULL_PIC_lon-80to0_lat0to65_DTNone.nc"
if thresvar is True:
    loadvar = xr.open_dataset(thresvar_path)
    loadvar = loadvar[thresvar_name].values.squeeze() # [time x lat x lon]
    
    # Adjust dimensions to [lon x lat x time x (otherdims)]
    loadvar = loadvar.transpose(2,1,0)#[...,None]

# Plotting Params
# ---------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]
bboxlim  = [-80,0,0,65]

debug = False # Debug section below script (set to True to run)
#%% Set Paths for Input (need to update to generalize for variable name)

if stormtrack:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    # Input Paths 
    if "SM" in mconfig:
        datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    else:
        datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s/" % varname
        
    # Output Paths
    figpath = "/stormtrack/data3/glliu/02_Figures/20220930/"
    outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/"
    
else:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

    # Input Paths 
    if "SM" in mconfig:
        datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
    elif "PIC" in mconfig:
        datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    elif "HTR" in mconfig:
        datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
    elif mconfig in ["HadISST","ERSST"]:
        datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    
    
    # Output Paths
    figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20221008/'
    outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'
    
# Import modules
from amv import proc,viz
import scm

# Set Input Names
# ---------------
if "SM" in mconfig: # Stochastic model
    # Postprocess Continuous SM  Run
    # ------------------------------
    print("WARNING! Not set up for stormtrack yet.")
    if "Qek" in mconfig:
        fnames      = ["stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0_Qek.npz" %i for i in range(10)]
        mnames      = ["entraining"] 
    else:
        fnames      = ["stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0.npz" %i for i in range(10)]
        mnames      = ["constant h","vary h","entraining"]
elif "PIC" in mconfig:
    # Postproess Continuous CESM Run
    # ------------------------------
    print("WARNING! Not set up for stormtrack yet.")
    fnames     = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
    mnames     = ["FULL","SLAB"] 
elif "HTR" in mconfig:
    # CESM1LE Historical Runs
    # ------------------------------
    fnames     = ["%s_FULL_HTR_lon-80to0_lat0to65_DTEnsAvg.nc" % varname,]
    mnames     = ["FULL",]
elif mconfig == "HadISST":
    # HadISST Data
    # ------------
    fnames = ["HadISST_detrend2_startyr1870.npz",]
    mnames     = ["HadISST",]
elif mconfig == "ERSST":
    fnames = ["ERSST_detrend2_startyr1900_endyr2016.npz"]

# Set Output Directory
# --------------------
proc.makedir(figpath)
savename   = "%s%s_%s_autocorrelation_%s_%s.npz" %  (outpath,mconfig,varname,thresname,lagname)
if "SM" in mconfig:
    savename = proc.addstrtoext(savename,"_runid2%02i" % (runid))
if thresvar is True:
    savename = proc.addstrtoext(savename,"_thresvar%s" % (thresvar_name))

print("Output will save to %s" % savename)


#%% Read in the data (Need to update for variable name)
st = time.time()

if mconfig == "PIC-FULL":
    sst_fn = fnames[0]
elif mconfig == "PIC-SLAB":
    sst_fn = fnames[1]
elif "SM" in mconfig:
    sst_fn = fnames[runid]
else:
    sst_fn = fnames[0]
print("Processing: " + sst_fn)

if ("PIC" in mconfig) or ("SM" in mconfig):
    # Load in SST [model x lon x lat x time] Depending on the file format
    if 'npy' in sst_fn:
        print("Loading .npy")
        sst = np.load(datpath+sst_fn)
        # NOTE: Need to write lat/lon loader
    elif 'npz' in sst_fn:
        print("Loading .npz")
        ld  = np.load(datpath+sst_fn,allow_pickle=True)
        lon = ld['lon']
        lat = ld['lat']
        sst = ld['sst'] # [model x lon x lat x time]
        
        # Transpose to [lon x lat x time x otherdims]
        sst = sst.transpose(1,2,3,0)
        
    elif 'nc' in sst_fn:
        print("Loading netCDF")
        ds  = xr.open_dataset(datpath+sst_fn)
        
        ds  = ds.sel(lon=slice(-80,0),lat=slice(0,65))
            
        lon = ds.lon.values
        lat = ds.lat.values
        sst = ds[varname].values # [lon x lat x time]
        
elif "HTR" in mconfig:
    
    ds  = xr.open_dataset(datpath+fnames[0])
    ds  = ds.sel(lon=slice(-80,0),lat=slice(0,65))
    lon = ds.lon.values
    lat = ds.lat.values
    sst = ds[varname].values # [ENS x Time x Z x LAT x LON]
    sst = sst[:,840:,...].squeeze() # Select 1920 onwards
    sst = sst.transpose(3,2,1,0) # [LON x LAT x Time x ENS]
    
elif mconfig == "HadISST":
    
    # Load the data
    sst,lat,lon=scm.load_hadisst(datpath,startyr=1900) # [lon x lat x time]
    
    # Slice to region
    sst,lon,lat = proc.sel_region(sst,lon,lat,bboxlim)
    
elif mconfig == "ERSST":
    
    # Load the data
    sst,lat,lon=scm.load_ersst(datpath,startyr=1900)
    
    # Fliip the longitude
    lon,sst = proc.lon360to180(lon,sst)
    
    # Slice to region
    sst,lon,lat = proc.sel_region(sst,lon,lat,bboxlim)
    
    
print("Loaded data in %.2fs"% (time.time()-st))

# Apply land/ice mask if needed
if loadmask:
    print("Applying mask loaded from %s!"%loadmask)
    # Load the mask
    msk  = np.load(loadmask) # Lon x Lat (global)
    glon = np.load(glonpath)
    glat = np.load(glatpath)
    
    # Restrict to Region
    bbox = [lon[0],lon[-1],lat[0],lat[-1]]
    rmsk,_,_ = proc.sel_region(msk,glon,glat,bbox)
        
    # Apply to variable
    if "HTR" in mconfig:
        sst *= rmsk[:,:,None,None]
    else:
        sst *= rmsk[:,:,None]
#%%
# Get Dimensions
if len(sst.shape) > 3:
    
    print("%s has more than 3 dimensions. Combining." % varname)
    nlon,nlat,ntime,notherdims = sst.shape
    sst = sst.transpose(0,1,3,2) # [nlon,nlat,otherdims,time]
    npts = nlon*nlat*notherdims # combine ensemble and points
    
else:
    notherdims      = 0
    nlon,nlat,ntime = sst.shape
    npts            = nlon*nlat

nyr             = int(ntime/12)
nlags           = len(lags)
nthres          = len(thresholds)



#%% Select Variables for a point

lonf = -30
latf = 50

th   = 1

klon,klat = proc.find_latlon(lonf,latf,lon,lat)
nlon = len(lon)
nlat = len(lat)
kpt  = np.ravel_multi_index(np.array(([40],[53])),(nlon,nlat))

locfn,loctitle = proc.make_locstring(lonf,latf)

# Get Point Variable and reshape to yr x mon
sstpt  = sst[klon,klat,:]#sstrs[kpt,:]
mldpt  = loadvar[klon,klat,:]#loadvarrs[kpt,:]
sst_in = sstpt.reshape(int(sstpt.shape[0]/12),12) # []
mld_in = mldpt.reshape(sst_in.shape)

# Calculate autocorrelation (no mask)
#acs    = proc.calc_lagcovar(sst_in.T,sst_in.T,lags,im+1,0,yr_mask=None,debug=False)
#plt.plot(acs)

for im in range(12):
    newshape = (int(sstpt.shape[0]/12),12)
    loadvar_mon = mld_in[:,im]/100 # Convert to meters
    sst_mon     = sst_in[:,im]
    
    
    #% Visualize histograms (for the selected month)
    
    
    def add_vlines_std(invar,ax=None):
        if ax is None:
            ax = ax.gca()
        mu    = invar.mean()
        sigma = invar.std() 
        ax.axvline(mu,color="k",lw=1)
        ax.axvline(mu+sigma,color="k",ls='dashed',lw=0.75)
        ax.axvline(mu-sigma,color="k",ls='dashed',lw=0.75)
        ax.axvline(mu+sigma*2,color="gray",ls='dotted',lw=0.55)
        ax.axvline(mu-sigma*2,color="gray",ls='dotted',lw=0.55)
        return mu,sigma
    
    nbins = 20
    
    fig,axs = plt.subplots(2,1,figsize=(6,8),constrained_layout=True)
    
    
    ax = axs[0]
    ax.hist(sst_mon,bins=nbins,alpha=0.75,edgecolor='k',linewidth=0.25,color='darkorange')
    mu,sigma = add_vlines_std(sst_mon,ax=ax)
    ax.set_title("SST ($\mu$ = %.2f, $\sigma$ = %.2f)" % (mu,sigma))
    ax.set_xlim([-3.2,3.2])
    #ax.grid(True,ls='dotted')
    
    ax = axs[1]
    ax.hist(loadvar_mon,bins=nbins,alpha=0.75,edgecolor='k',linewidth=0.25,color='cornflowerblue')
    mu,sigma = add_vlines_std(loadvar_mon,ax=ax)
    ax.set_title("MLD ($\mu$ = %.2f, $\sigma$ = %.2f)" % (mu,sigma))
    ax.set_xlim([-130,130])
    
    plt.suptitle("Histograms at %s, Month %02i, nbins=%i " % (loctitle,im+1,nbins),)
    
    plt.savefig("%sHistogram_MLD_SST_%s_Mon%02i_%ibins.png" % (figpath,locfn,im+1,nbins),dpi=150)

#ax.grid(True,ls='dotted')

#%% Plot the skewness
from scipy import stats

fig,ax=plt.subplots(1,1,figsize=(6,4),constrained_layout=True)

ax.plot(np.arange(1,13,1),stats.skew(sst_in,axis=0),color="darkorange",label="SST")
ax.plot(np.arange(1,13,1),stats.skew(mld_in,axis=0),color="cornflowerblue",label="MLD")
ax.legend()
ax.set_xlim([1,12])
ax.set_xticks(np.arange(1,13,1))
ax.grid(True,ls='dotted')
ax.set_ylabel("Skewness (0=Gaussian)")
ax.set_xlabel("Month")
ax.set_title("Monthly Skewness at %s" % (loctitle))
plt.savefig("%sSkewness_MLD_SST_%s.png" % (figpath,locfn,),dpi=150)
#%%
thresholds = [0]


# Calculate autocorrelation with mask
sst_class = proc.make_classes_nd(sst_mon,thresholds,dim=1,debug=False)#[kpt,:]
mld_class = proc.make_classes_nd(loadvar_mon,thresholds,dim=1,debug=False)#[kpt,:]
    
# Preallocate
nthres      = len(thresholds) +  2
acs_all    = np.zeros((len(lags),nthres,2))*np.nan # [lag x threshold x variable]
counts_all = np.zeros((nthres,2)) # [threshold x variable]

for th in tqdm(range(nthres)): # Loop for each threshold

    # Apply Threshold
    if th < nthres-1:
        mask_sst     = np.where(sst_class.squeeze() == th)[0] # Indices of valid years
        mask_mld     = np.where(mld_class.squeeze() == th)[0] # Indices of valid years
        
        acs_sst,yr_count_sst = proc.calc_lagcovar(sst_in.T,sst_in.T,lags,im+1,0,yr_mask=mask_sst,debug=False)
        acs_mld,yr_count_mld = proc.calc_lagcovar(sst_in.T,sst_in.T,lags,im+1,0,yr_mask=mask_mld,debug=False)
        
        
    else:
        acs_sst = proc.calc_lagcovar(sst_in.T,sst_in.T,lags,im+1,0,yr_mask=None,debug=False)
        acs_mld = proc.calc_lagcovar(sst_in.T,sst_in.T,lags,im+1,0,yr_mask=None,debug=False)
    
        yr_count_sst = sst_in.shape[0]
        yr_count_mld = sst_in.shape[0]
    
    acs_all[:,th,0] = acs_sst.copy()
    acs_all[:,th,1] = acs_mld.copy()
    counts_all[th,0] = yr_count_sst
    counts_all[th,1] = yr_count_mld

#%% Make some plots comparing each (3 anomaly case)

if nthres == 3:

    linestyles = ('dashed','dotted','solid')
    threslabs  = ("-","+","All")
    vcolors    = ('darkorange','cornflowerblue')
    vnames     = ('SST','MLD')
    xtks       = np.arange(0,66,6)
    
    fig,ax=plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
    
    for v in range(2):
        for th in range(3):
            if v == 1 and th == 2:
                ax.plot(lags,acs_all[:,th,v],color=vcolors[v],ls="dashdot",label="%s (%s), count=%i" % (vnames[v],threslabs[th],counts_all[th,v]),alpha=0.7)
            else:
                ax.plot(lags,acs_all[:,th,v],color=vcolors[v],ls=linestyles[th],label="%s (%s), count=%i" % (vnames[v],threslabs[th],counts_all[th,v]),alpha=0.7)
    ax.legend()
    
    
    ax.grid(True,ls='dotted')
    ax.set_xlabel("Lag (Months from %s)" % (proc.get_monstr()[im]))
    ax.set_ylabel("Correlation")
    ax.set_xticks(xtks)
    ax.set_xlim([0,60])
    #ax.plot(lags,acs_sst,label="SST Threshold, count=%i" % yr_count_sst,color='k')
    #ax.plot(lags,acs_mld,label="MLD Threshold, count=%i" % yr_count_mld,color='b')
    
    plt.suptitle("Autocorrelation at %s, Lag 0 = %s" % (loctitle,proc.get_monstr()[im]),)
    
    plt.savefig("%sACF_MLD_SST_%s_Mon%02i.png" % (figpath,locfn,im+1),dpi=150)

#%% 4 Anomaly Case


if thresholds == [-1,1]:

    linestyles = ('dashed','solid','dotted','dashdot')
    threslabs  = ("-","Neutral","+","All")
    vcolors    = ('darkorange','cornflowerblue')
    vnames     = ('SST','MLD')
    xtks       = np.arange(0,66,6)
    
    fig,ax=plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
    
    for v in range(2):
        for th in range(3):
            if v == 1 and th == 2:
                ax.plot(lags,acs_all[:,th,v],color=vcolors[v],ls="dashdot",label="%s (%s), count=%i" % (vnames[v],threslabs[th],counts_all[th,v]),alpha=0.7)
            else:
                ax.plot(lags,acs_all[:,th,v],color=vcolors[v],ls=linestyles[th],label="%s (%s), count=%i" % (vnames[v],threslabs[th],counts_all[th,v]),alpha=0.7)
    ax.legend()
    
    
    ax.grid(True,ls='dotted')
    ax.set_xlabel("Lag (Months from %s)" % (proc.get_monstr()[im]))
    ax.set_ylabel("Correlation")
    ax.set_xticks(xtks)
    ax.set_xlim([0,60])
    #ax.plot(lags,acs_sst,label="SST Threshold, count=%i" % yr_count_sst,color='k')
    #ax.plot(lags,acs_mld,label="MLD Threshold, count=%i" % yr_count_mld,color='b')
    
    plt.suptitle("Autocorrelation at %s, Lag 0 = %s" % (loctitle,proc.get_monstr()[im]),)
    
    plt.savefig("%sACF_MLD_SST_%s_Mon%02i_extreme.png" % (figpath,locfn,im+1),dpi=150)

#%% Compare the distribution for the subsets




#%% 
fig,ax=plt.subplots(1,1)
ax.plot(lags,acs_sst,label="SST Threshold, count=%i" % yr_count_sst,color='k')
ax.plot(lags,acs_mld,label="MLD Threshold, count=%i" % yr_count_mld,color='b')
ax.plot(lags,acs,label="No Threshold, count=%i" % sst_in.shape[0],color='gray')
ax.legend()


