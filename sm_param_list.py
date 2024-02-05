#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

List of common parameters/names for the stochastic model project

NoteL moved to stochmod/stochmod_params

Created on Thu Feb  1 09:26:28 2024

@author: gliu

"""


import numpy as np

# Bounding Boxes [LonW, LonE, LatS, latN]
amvbbox       = [-80,0,0,65]   # AMV Calculation box
bbox_crop     = [-90,20,0,90]  # Preprocessing box