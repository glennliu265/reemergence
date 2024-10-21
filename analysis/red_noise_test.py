#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Red Noise Forcing

Created on Thu Oct 10 12:21:58 2024

@author: gliu
"""

import numpy


#%% Toy experiment, adding red noise forcing



simlen = 10000  # Length of simulation
r1     = 0.8    # Autocorrelation
sigma  = 0.1    # Standard Deviation of timeseries


noise_ts  = np.random.normal(0,1,simlen)
noise_ts3 = np.random.normal(0,1,simlen)



ts1 = np.zeros((simlen+1))
ts2 = np.zeros((simlen+1))
ts3 = np.zeros((simlen+1))

for t in range(simlen):
    
    ts1[t+1] = ts1[t] * r1 + noise_ts[t]
    ts2[t+1] = ts2[t] * r1 + noise_ts[t] + ts1[t]
    ts3[t+1] = ts2[t] * r1 + noise_ts3[t] + ts1[t]
    
    
    
#%% Compute lagged persistence

def calc_acf(ts,lag):
    acf    = np.zeros(nlags)
    for l in range(nlags):
        lag = lags[l]
        acf[l] = np.corrcoef(ts[:(simlen+1-lag)],ts[lag:])[0,1]
    return acf
        
    
#calc_acf = lambda ts,lag: 

lags    = np.arange(0,36)
nlags   = len(lags)

ts_in = [ts1,ts2,ts3]
acfs  = [calc_acf(ts,lags) for ts in ts_in]

acf1,acf2,acf3 = acfs
# acf1    = np.zeros(nlags)
# acf2    = np.zeros(nlags)
# for l in range(nlags):
#     lag = lags[l]
#     acf1[l] = np.corrcoef(ts1[:(simlen+1-lag)],ts1[lag:])[0,1]
#     acf2[l] = np.corrcoef(ts2[:(simlen+1-lag)],ts2[lag:])[0,1]

#%%
lw  =   2.5
fig,ax = plt.subplots(1,1,constrained_layout=True)
ax.plot(lags,acf1,label="Timeseries 1",lw=lw)
ax.plot(lags,acf2,label="Timeseries 2 + Timeseries 1",lw=lw)
ax.plot(lags,acf3,label="Timeseries 2 + Timeseries 1 (uncorrelated)",lw=lw)
ax.legend()
ax.set_xlabel("Lag")
ax.set_ylabel("Correlation")

    