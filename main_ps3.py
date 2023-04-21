# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:55:40 2023

@author: sam low and katherine
"""

# This code starts with Q1(d) and Q2(c): propagation of relative orbits
# using HCW and YA state transition matrices.
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

# HCW state transition matrix.
# Inputs are a `Spacecraft` object and time of propagation [sec].
def stm_hcw(sc, t):
    
    stm = np.zeros((6,6))
    n = sc.n
    nt = n * t
    
    # First row
    stm[0,0] = 4 - 3 * cos(nt)
    stm[0,1] = 0
    stm[0,2] = 0
    stm[0,3] = sin(nt) / n
    stm[0,4] = (2/n) * (1 - cos(nt))
    stm[0,5] = 0
    
    # Second row
    stm[1,0] = (6 * sin(nt)) - (6 * nt)
    stm[1,1] = 1
    stm[1,2] = 0
    stm[1,3] = (2/n) * (cos(nt) - 1)
    stm[1,4] = (4 * sin(nt) / n) - (3 * t)
    stm[1,5] = 0
    
    # Third row
    stm[2,0] = 0
    stm[2,1] = 0
    stm[2,2] = cos(nt)
    stm[2,3] = 0
    stm[2,4] = 0
    stm[2,5] = sin(nt) / n
    
    # Fourth row
    stm[3,0] = 3 * sin(nt) * n
    stm[3,1] = 0
    stm[3,2] = 0
    stm[3,3] = cos(nt)
    stm[3,4] = 2 * sin(nt)
    stm[3,5] = 0
    
    # Fifth row
    stm[4,0] = 6 * n * cos(nt) - 6 * n
    stm[4,1] = 0
    stm[4,2] = 0
    stm[4,3] = -2 * sin(nt)
    stm[4,4] = 4 * cos(nt) - 3
    stm[4,5] = 0
    
    # Sixth row
    stm[5,0] = 0
    stm[5,1] = 0
    stm[5,2] = -n * sin(nt)
    stm[5,3] = 0
    stm[5,4] = 0
    stm[5,5] = cos(nt)
    
    return stm

# HCW state transition matrix.
# Inputs are a `Spacecraft` object and time of propagation [sec].
def stm_ya(sc, t):
    
    stm = np.zeros((6,6))
    
    # First row
    stm[0,0] = 
    stm[0,1] = 
    stm[0,2] = 
    stm[0,3] = 
    stm[0,4] = 
    stm[0,5] = 
    
    # Second row
    stm[1,0] = 
    stm[1,1] = 
    stm[1,2] = 
    stm[1,3] = 
    stm[1,4] = 
    stm[1,5] = 
    
    # Third row
    stm[2,0] = 
    stm[2,1] = 
    stm[2,2] = 
    stm[2,3] = 
    stm[2,4] = 
    stm[2,5] = 
    
    # Fourth row
    stm[3,0] = 
    stm[3,1] = 
    stm[3,2] = 
    stm[3,3] = 
    stm[3,4] = 
    stm[3,5] = 
    
    # Fifth row
    stm[4,0] = 
    stm[4,1] = 
    stm[4,2] = 
    stm[4,3] = 
    stm[4,4] = 
    stm[4,5] = 
    
    # Sixth row
    stm[5,0] = 
    stm[5,1] = 
    stm[5,2] = 
    stm[5,3] = 
    stm[5,4] = 
    stm[5,5] = 
    
    return stm