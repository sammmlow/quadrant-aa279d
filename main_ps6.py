# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:43:46 2023

Sam Low and Katherine Cao
"""

import math
import matplotlib.pyplot as plt
import numpy as np

from math import sqrt, sin, cos, tan, pi
from numpy.linalg import norm, pinv

from source import spacecraft

# Initialize both chief and deputy.
sc1_elements = [6918.14, 0.00001, 97.59760, 0.000000, -109.33800,   0.00827]
sc2_elements = [6918.14, 0.00722, 97.5976, 134.94389, -108.5025, -134.71026] 
sc1 = spacecraft.Spacecraft( elements = sc1_elements )
sc2 = spacecraft.Spacecraft( elements = sc2_elements )

# Set the chief of the spacecraft. Enable maneuvers for SC2.
sc2.chief = sc1 # ROEs and RTN states computed w.r.t. SC1
sc2.forces['maneuvers'] = True # Important to switch this on.

# Start the simulation here.
timeNow, duration, timestep = 0.0, 86400.0, 30.0 # Time in seconds
k, samples = 0, int(duration / timestep) # Sample count and total samples

# Matrix to store the data
state_history = np.zeros((samples, 6))

# Keplerian plant matrix.
def build_A(sc):
    A = np.zeros((6,6))
    A[1,0] = sc1.n
    return A

# Control input matrix from GVEs (see Eq (4) in Steindorf, 2017)
def build_B(sc):
    
    ec = sc.e
    ex = ec*cos(sc.w)
    ey = ec*sin(sc.w)
    sfc = sin(sc.nu)
    cfc = cos(sc.nu)
    tic = tan(sc.i)
    swfc = sin(sc.w + sc.nu)
    cwfc = cos(sc.w + sc.nu)
    eta = sqrt(1-sc.e**2)
    B = np.zeros((6,3))
    
    B[0,0] = 2*ec*sfc/eta
    B[0,1] = 2*(1+ec*cfc)/eta
    B[1,0] = -2*eta*eta/(1+ec*cfc)
    B[2,0] = eta*swfc
    B[2,1] = eta*((2+ec*cfc)*cwfc+ex)/(1+ec*cfc)
    B[2,2] = eta*ey*swfc/(tic*(1+ec*cfc))
    B[3,0] = -eta*cwfc
    B[3,1] = eta*((2+ec*cfc)*swfc+ey)/(1+ec*cfc)
    B[3,2] = -eta*ex*swfc/(tic*(1+ec*cfc))
    B[4,2] = eta*cwfc/(1+ec*cfc)
    B[5,2] = eta*swfc/(1+ec*cfc)
   
    return (1/(sc.n * sc.a)) * B

# In the loop, in order for the deputy to properly update its ROEs and RTN, 
# the chief needs to be propagated first...
while timeNow < duration:
    
    # Record states.
    state_history[k,:] = [sc2.pR, sc2.pT, sc2.pN, sc2.vR, sc2.vT, sc2.vN]

    # Compute the plant matrix A (time-varying for non-Keplerian case).
    A = build_A(sc1)
    
    # Compute the control matrix B.
    B = build_B(sc1)
    
    # Build some gain matrix.
    P = 0.0001 * np.eye(6)
    
    # Compute control input. For now assume desired ROE is zero (rendezvous).
    ROE = np.array([sc2.da, sc2.dL, sc2.ex, sc2.ey, sc2.ix, sc2.iy])
    dROE = ROE
    u = -1 * pinv(B) @ ((A @ ROE) + (P @ dROE))
    
    # Apply the control maneuver to SC2.
    sc2.set_thruster_acceleration( u )
    
    # Finally, the chief itself needs to be propagated (in absolute motion)
    sc1.propagate_perturbed(timestep, timestep)
    sc2.propagate_perturbed(timestep, timestep)

    # Update time and sample count.
    timeNow += timestep
    k += 1
    
# Plot the full trajectory below, with chief as a quiver triad.
axisLimit = 1.0 # km
fig1 = plt.figure(1).add_subplot(projection='3d')
fig1.plot(state_history[:,1], state_history[:,2], state_history[:,0])
fig1.quiver(0,0,0,1,0,0, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig1.quiver(0,0,0,0,1,0, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig1.quiver(0,0,0,0,0,1, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig1.set_title('Trajectory in RTN Frame')
fig1.grid()
fig1.set_xlabel('T [km]')
fig1.set_ylabel('N [km]')
fig1.set_zlabel('R [km]')
