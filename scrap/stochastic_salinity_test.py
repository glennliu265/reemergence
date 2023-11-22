#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Stochastic Salinity Model

Created on Wed Aug 30 15:42:54 2023

@author: gliu

"""

#%% Inputs

rho      = 1026     # Density of Seawater [kg/m3]
#cp       = 3996    # Specific Heat J/(kg*C)
L        = 2.5e6    # Specific Heat of Evaporation [J/kg]
B        = 0.2      # Bowen Ratio (Sensible/Latent Heat Flux)

dt       = 3600*24*30 # sec (Monthly Step)
lambda_a = 70       # Salinity damping [psu/W/m2]
alpha    = 1 # Stochastic Forcing Amplitude (W/m2)
S_bar    = 36         # psu

#%% Generate a timeseries



