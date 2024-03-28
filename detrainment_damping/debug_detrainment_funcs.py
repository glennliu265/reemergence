#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to Debug Detrainment Function
Created on Thu Mar 28 13:07:42 2024

@author: gliu
"""

h     = np.array([137.59690615, 154.78135511, 124.53401863,  68.65643993,
        31.7234189 ,  22.75172748,  22.88946861,  26.89790936,
        36.49577171,  51.96625643,  73.8477436 , 102.84985886])
kprev = np.array([2.52060898, 2.        , 0.        , 0.        , 0.        ,
       0.        , 5.94709726, 5.37415379, 4.80178841, 4.36822303,
       3.86053426, 3.36897631])


hdetrains = scm.get_detrain_depth(kprev,h)

# Visualize detrainment times
fig,ax = plt.subplots(1,1,figsize=(12,4.5),constrained_layout=True)
ax = viz.viz_kprev(h,kprev)
ax.scatter(np.arange(1,13,1),hdetrains,marker="d",s=55,zorder=3,c='w')
