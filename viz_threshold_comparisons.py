#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare the result of subsetting based on different thresholds (SST, MLD, etc.)

Copied from viz_pointwise_autocorrelation on 2022.09.29



Created on Thu Sep 29 11:51:58 2022

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

from amv import proc,viz
import scm
import numpy as np
import xarray as xr
from tqdm import tqdm 
import time
import cartopy.crs as ccrs

#%%
