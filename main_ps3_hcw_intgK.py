# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:49:06 2023

@author: sammm annd katherine
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi
from numpy.linalg import norm

# Import our own spacecraft library
from source import spacecraft

GM = 398600.4418  # Earth default (km**3/s**2)

def get_hill_frame(r, v):
    h = np.cross(r, v)  # Angular momentum vector
    r_hat = r / np.linalg.norm(r)  # Local X-axis (R)
    h_hat = h / np.linalg.norm(h)  # Local Z-axis (N)
    y_hat = np.cross(h_hat, r_hat)  # Local Y-axis (T)
    return np.array([r_hat, y_hat, h_hat])

# Define a function that takes in ECI absolute and relative positions (chief
# to deputy) and converts them to positions and velocities in the RTN frame,
# where the velocity is taken as a time derivative as seen in the RTN frame.
# Inputs are 1x6 vector (3 elements of position and velocity each)

def rv_eci_to_rtn(rv_c_eci, rv_cd_eci):
    r = rv_c_eci[0:3]
    v = rv_c_eci[3:6]
    rho = rv_cd_eci[0:3]
    rhoDot = rv_cd_eci[3:6]
    nuDot = norm(np.cross(r, v)) / (norm(r)**2)  # true anomaly derivative
    omega = np.array([0.0, 0.0, nuDot])
    matrix_eci2rtn = get_hill_frame(r, v)
    r_rtn = matrix_eci2rtn @ rho
    v_rtn = matrix_eci2rtn @ rhoDot - np.cross(omega, r_rtn)
    return r_rtn, v_rtn

def intg_constants_hcw(sc1, sc2, dt):
    
    a = sc1.a
    an = a * sc1.n
    nt = sc1.n * dt
    snt = sin(nt)
    cnt = cos(nt)
    
    mat1 = np.array([
        [1, snt, cnt, 0, 0, 0],
        [-1.5*nt, 2*cnt, -2*snt, 1, 0, 0],
        [0, 0, 0, 0, snt, cnt],
        [0, cnt, -snt, 0, 0, 0],
        [-1.5, -2*snt, -2*cnt, 0, 0, 0],
        [0, 0, 0, 0, cnt, -snt]])
    
    mat2 = np.array([
        [a, 0, 0, 0, 0, 0],
        [0, a, 0, 0, 0, 0],
        [0, 0, a, 0, 0, 0],
        [0, 0, 0, an, 0, 0],
        [0, 0, 0, 0, an, 0],
        [0, 0, 0, 0, 0, an]])
    
    # Get the RTN coordinates of the deputy
    sc1_eci = np.array([sc1.px, sc1.py, sc1.pz, sc1.vx, sc1.vy, sc1.vz])
    sc2_eci = np.array([sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz])
    r_rtn, v_rtn = rv_eci_to_rtn(sc1_eci, sc2_eci - sc1_eci)
    rv_rtn = np.concatenate(( r_rtn, v_rtn ))
    
    # Compute integration constants
    invMat = np.linalg.inv( mat2 @ mat1 )
    print(invMat @ rv_rtn)
    
    return None

# Now test and see, based on initial conditions in Table 2 of PS3, if HCW
# approximates motion well, with the non-linear FDERM propagation.
sc1_elements = [7928.137, 0.000001, 97.5976, 0.0, 250.6620, 0.00827]
sc2_elements = [7928.137, 0.000001, 97.5976, 0.0, 250.6703, 0.00413]
sc3_elements = [7928.137, 0.000001, 97.5976, 0.0, 250.6620, 0.00000]

# Create the spacecraft objects.
sc1 = spacecraft.Spacecraft( elements = sc1_elements )
sc2 = spacecraft.Spacecraft( elements = sc2_elements )
sc3 = spacecraft.Spacecraft( elements = sc3_elements )

intg_constants_hcw(sc1, sc2, 30)
intg_constants_hcw(sc1, sc3, 30)