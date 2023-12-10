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

#%% Load Everything

#%% 



