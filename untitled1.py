#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test AC Depth

Created on Tue Jun 28 14:57:44 2022

@author: gliu
"""


import numpy as np

#%%

nyr = 1000
nz  = 10

# Make 2 random white noise timeseries
wn1 = np.random.normal(0,1,(12*nyr))
wn2 = np.random.normal(0,1,(12*nyr))
print(np.corrcoef(wn1,wn2)[0,1])

# Make vertical structure
temps = np.zeros((nz,12*nyr))
for z in range(nz):
    if z%2 == 0:# correlated at even depths
        temps[z,:] = wn1
    else: # Uncorrelated at odd depths
        temps[z,:] = wn2

