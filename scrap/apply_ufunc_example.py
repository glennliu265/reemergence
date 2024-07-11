#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Example with xarray ufunc (compute pointwise cross-correlation)

Created on Fri Jun 21 17:08:02 2024

@author: gliu
"""

import xarray as xr
import numpy as np

# Make Dummy variables
coords     = dict(time=np.arange(100),lat=np.arange(-90,100,10),lon=np.arange(-180,190,10))
var1       = xr.DataArray(np.random.normal(0,1,(100,19,37)),coords=coords,dims=coords)
var2       = xr.DataArray(np.random.normal(0,1,(100,19,37)),coords=coords,dims=coords)

# Make the function (you don't have to use lambda, can put any function)
crosscorr = lambda x,y: np.corrcoef(x,y)[0,1]

# Apply ufunc
ccout = xr.apply_ufunc(
    crosscorr, # Pass the function
    var1, # Input 1 for function
    var2, # Input 2 for function,... (and so on)
    input_core_dims=[['time'],['time']], # Which dimensions to operate over for each argument... 
    output_core_dims=[[]], # Output Dimension
    vectorize=True,# True to loop over non-core dims
    # I think you can set a "parallel" argument...
    )


# Here's a version with random differences -----------------------------------
def monte_carlo_diff(var1,p=0.5,mciter=10000):
    
    randsample1 = np.random.choice(var1,size=mciter,replace=True)
    randsample2 = np.random.choice(var1,size=mciter,replace=True)
    diffs       = randsample2 - randsample1
    thres       = np.percentile(np.sort(diffs),q=[(p/2)*100,(1-p/2)*100])
    
    return thres

# Make the function
p       = 0.05
funcin  = lambda x: monte_carlo_diff(x,p=p,mciter=10000)

# Apply ufunc
significant_thres = xr.apply_ufunc(
    funcin, # Pass the function
    var1, # Input 1 for function
    input_core_dims=[['time']], # Which dimensions to operate over for each argument... 
    output_core_dims=[['thres']], # Output Dimension
    vectorize=True,# True to loop over non-core dims
    # I think you can set a "parallel" argument...
    )

significant_thres['thres'] = [(p/2)*100,(1-p/2)*100]

        

