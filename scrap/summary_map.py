#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Make a summary plot

Created on Thu May  2 11:18:40 2024

@author: gliu
"""


import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec#===
import cartopy.crs as ccrs

bboxplot  = [-80,0,20,65]
fsz_title = 16
fsz_ticks = 14

fig       = plt.figure(figsize=(18,14))
gs        = gridspec.GridSpec(3,3)

# First Row Locator, Monthly Variance, and some histogram

# Locator
ax1 = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax1.set_extent(bboxplot)
ax1.coastlines()
ax1.set_title("Decorrelation Timescale ($T_2$)",fontsize=fsz_title)

# Monthly Variance
ax2 = fig.add_subplot(gs[0,1])
ax2.set_xticks(np.arange(1,13,1))
ax2.set_title("Monthly Variance",fontsize=fsz_title)
#ax2 = viz.init_monplot(ax2)

# Histogram
ax3 = fig.add_subplot(gs[0,2])
ax3.set_title("Histogram",fontsize=fsz_title)
#ax3.axvline([0],lw=0.75,c="k")

# Feb ACF
ax4 = fig.add_subplot(gs[1,:2])
ax4.set_title("February Autocorrelation",fontsize=fsz_title)

# Spectra
ax5 = fig.add_subplot(gs[2,:2])
ax5.set_title("Power Spectra",fontsize=fsz_title)

# ACF fingerprint
ax6 = fig.add_subplot(gs[1:,2])
ax6.set_title("Monthly Lagged Autocorrelation",fontsize=fsz_title)

#%%

