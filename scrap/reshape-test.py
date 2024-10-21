#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:20:14 2024

@author: gliu
"""


test    = np.tile(np.tile(np.arange(1,13,1)[:,None],2)[...,None],42)

test    = test.transpose(2,1,0) # [Year nx]

test1   = test.reshape((42,2*12))

test12 = test1.reshape((42*2,12))


test1   = test.reshape((42*2,12))