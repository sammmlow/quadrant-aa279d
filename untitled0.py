# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:28:18 2023

@author: sammm
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt
from numpy.linalg import norm

# Import our own spacecraft library
from source import spacecraft

def stm_yank_propagate(sc, nu0, dt, rv_rtn):
    
    stm = np.zeros((6,6))
    
    # Obtain chief spacecraft angular momentum
    r_vec = np.array([sc.px, sc.py, sc.pz])
    v_vec = np.array([sc.vx, sc.vy, sc.vz])
    h_vec = np.cross(r_vec, v_vec)
    h_abs = np.linalg.norm(h_vec)
    
    # Constants.
    e = sc.e
    I = dt * (sc.GM**2) / (h_abs**3)
    k = 1 + e * cos(sc.nu)
    c = k * cos(sc.nu)
    s = k * sin(sc.nu)
    cdelta = cos(sc.nu - nu0)
    sdelta = sin(sc.nu - nu0)
    kdelta = 1 + e * cos(sc.nu - nu0)
    cprime = -1 * (sin(sc.nu) + e * sin(2 * sc.nu))
    sprime = cos(sc.nu) + e * cos(2 * sc.nu)
    
    sRT1 = np.array([
        [(1-e**2), (3*e*s*((1/k)/(1/k**2))  ), (-e*s*(1+(1/k))), (2-e*c)],
        [(0     ), (-3*s*((1/k)/(e**2/k**2))), (s*(1+(1/k))   ), (2*e-c)],
        [(0     ), (-3*(c/k)+3              ), (e+c*(1+(1/k)) ), (-s)   ],
        [(0     ), (3*k+(e**2)-1            ), (-k**2         ), (e*s)  ]])
    
    sRT2 = np.array([
        [(1), (-c*(1+(1/k))), (s*(1+(1/k))), (3*(k**2)*I              )],
        [(0), (s,          ), (c,         ), (2-3*e*s*I               )],
        [(0), (2*s,        ), (2*c-e,     ), (3*(1-2*e*s*I)           )],
        [(0), (sprime,     ), (cprime,    ), (-3*e*(sprime*I+s/(k**2)))]])
    
    sN = np.array([
        [ cdelta, sdelta],
        [-sdelta, cdelta]]) / kdelta
    
    
