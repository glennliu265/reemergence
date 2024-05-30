#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:51:58 2024

Just trying to figure out how to properly implement kprev_byens
It turns out I had just forgotten to change the kprev variable to kprev_in
for the correlation/interpolation calcualtion.

@author: gliu
"""



corrkp0

kp0ensmean = corrkp0.mean('ens')
kp1ensmean = corrkp1.mean('ens')

# Plot mean
fig,ax = plt.subplots(1,1)
ax.plot(mons3,kp0ensmean,label="kprev0")
ax.plot(mons3,kp1ensmean,label="kprev1")
ax.legend()


# Plot Ens
e = 9
kp0ensmean = corrkp0.isel(ens=e)
kp1ensmean = corrkp1.isel(ens=e)
fig,ax = plt.subplots(1,1)
ax.plot(mons3,kp0ensmean,label="kprev0")
ax.plot(mons3,kp1ensmean,label="kprev1")
ax.legend()
