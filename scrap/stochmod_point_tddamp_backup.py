#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 01:46:53 2024

@author: gliu
"""

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


#%% Functions


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


Sbar = 35.287624


# Constants
dt           = 3600*24*30 # Timestep [s]
cp           = 3850       # 
rho          = 1026       # Density [kg/m3]
B            = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L            = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document
lbde_mult    = -1

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
h0          = h.copy()
alpha0      = (E.copy()/(rho*L*h0)*dt*Sbar + P*dt ) ##1     * np.ones(12)#( E.copy()/(rho*cp*h0)*dt + P*dt )
beta0       = scm.calc_beta(h[None,None,:]).squeeze() # 0     * np.ones(12) # 
damping     = 0  * np.ones(12) + beta0 #hff_l/(rho*cp*h0) *dt + beta0
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
ax.plot(lags,theoacfdoc(lags,varN,varS),label=r"$1-\Delta t^2 \frac{var(n'(t)')}{var(S'(t+\Delta t))}$")
ax.plot(lags,theoacfexp(lags,lbd_mean),label=r"$e^{-\lambda \tau}$")
ax.plot(lags,acS[0],label="SSS Integration")
ax.legend()



