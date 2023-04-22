# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:55:40 2023

By: Sam Low and Katherine Cao
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt
from numpy.linalg import norm

# Import our own spacecraft library
from source import spacecraft

##############################################################################
##############################################################################

# Note: states are [xBar, xBarDot, yBar, yBarDot, zBar, zBarDot] in the book
# but states are re-ordered to [xBar, yBar, zBar, xBarDot, yBarDot, zBarDot]
# here in the simulations below (in the while loop near line 190).
# Bar => normalized over the chief's current position norm

# YA STM 1. Compute this STM only after the chief S/C has propagated.
def stm_yank_current(sc, dt):
    
    stm = np.zeros((6,6))
    
    # Obtain chief spacecraft angular momentum
    r_vec = np.array([sc.px, sc.py, sc.pz])
    v_vec = np.array([sc.vx, sc.vy, sc.vz])
    h_vec = np.cross(r_vec, v_vec)
    h_abs = np.linalg.norm(h_vec)
    
    # Constants.
    I = dt * (sc.GM**2) / (h_abs**3)
    k = 1 + sc.e * cos(sc.nu)
    c = k * cos(sc.nu)
    s = k * sin(sc.nu)
    cprime = -1 * (sin(sc.nu) + sc.e * sin(2 * sc.nu))
    sprime = cos(sc.nu) + sc.e * cos(2 * sc.nu)
    
    # Row for xBar
    stm[0,0] = s
    stm[0,1] = c
    stm[0,2] = 2 - (3 * sc.e * s * I)
    stm[0,3] = 0
    stm[0,4] = 0
    stm[0,5] = 0
    
    # Row for xBarDot
    stm[1,0] = sprime
    stm[1,1] = cprime
    stm[1,2] = -3 * sc.e * ((sprime * I) + (s / (k**2)))
    stm[1,3] = 0
    stm[1,4] = 0
    stm[1,5] = 0
    
    # Row for yBar
    stm[2,0] = c * (1+k) # (1 + (1/k))
    stm[2,1] = -s * (1+k) # (1 + (1/k))
    stm[2,2] = -3 * (k**2) * I
    stm[2,3] = 1
    stm[2,4] = 0
    stm[2,5] = 0
    
    # Row for yBarDot
    stm[3,0] = -2 * s
    stm[3,1] = sc.e - (2 * c)
    stm[3,2] = -3 * (1 - (2 * sc.e * s * I))
    stm[3,3] = 0
    stm[3,4] = 0
    stm[3,5] = 0
    
    # Row for zBar
    stm[4,0] = 0
    stm[4,1] = 0
    stm[4,2] = 0
    stm[4,3] = 0
    stm[4,4] = cos(sc.nu)
    stm[4,5] = sin(sc.nu)
    
    # Row for zBarDot
    stm[5,0] = 0
    stm[5,1] = 0
    stm[5,2] = 0
    stm[5,3] = 0
    stm[5,4] = -sin(sc.nu)
    stm[5,5] = cos(sc.nu)
    
    return stm

# YA STM 2. Compute this STM before the chief S/C has propagated (initial).
def stm_yank_initial(sc):
    
    stm = np.zeros((6,6))
    
    # Constants.
    k = 1 + sc.e * cos(sc.nu)
    c = k * cos(sc.nu)
    s = k * sin(sc.nu)
    eta = sqrt(1 - sc.e**2)
    
    # Row for xBar
    stm[0,0] = -3 * s * (k + sc.e**2) / (k**2)
    stm[0,1] = c - (2 * sc.e)
    stm[0,2] = 0
    stm[0,3] = -s * (k + 1) / k
    stm[0,4] = 0
    stm[0,5] = 0
    
    # Row for xBarDot
    stm[1,0] = -3 * (sc.e + (c / k))
    stm[1,1] = -s
    stm[1,2] = 0
    stm[1,3] = -1 * (c * ((k + 1) / k) + sc.e)
    stm[1,4] = 0
    stm[1,5] = 0
    
    # Row for yBar
    stm[2,0] = (3 * k) - eta**2
    stm[2,1] = sc.e * s
    stm[2,2] = 0
    stm[2,3] = k**2
    stm[2,4] = 0
    stm[2,5] = 0
    
    # Row for yBarDot
    stm[3,0] = -3 * sc.e * s * ((k + 1) / (k**2))
    stm[3,1] = -2 + sc.e * c
    stm[3,2] = eta**2
    stm[3,3] = -sc.e * s * (k + 1) / k
    stm[3,4] = 0
    stm[3,5] = 0
    
    # Row for zBar
    stm[4,0] = 0
    stm[4,1] = 0
    stm[4,2] = 0
    stm[4,3] = 0
    stm[4,4] = (eta**2) * cos(sc.nu)
    stm[4,5] = -1 * (eta**2) * sin(sc.nu)
    
    # Row for zBarDot
    stm[5,0] = 0
    stm[5,1] = 0
    stm[5,2] = 0
    stm[5,3] = 0
    stm[5,4] = (eta**2) * sin(sc.nu)
    stm[5,5] = (eta**2) * cos(sc.nu)
    
    return (1/(eta**2)) * stm

##############################################################################
##############################################################################

from main_ps2 import rv_eci_to_rtn, relative_rk4

# Initialize SC osculating elements
sc1_elements = [7928.137, 0.00001, 97.5976, 0.0, 250.6620, 0.00827]
sc2_elements = [7928.137, 0.00001, 97.5976, 0.0, 250.6703, 0.00413]

# Create the spacecraft objects.
sc1 = spacecraft.Spacecraft( elements = sc1_elements )
sc2 = spacecraft.Spacecraft( elements = sc2_elements )
sc2 = spacecraft.Spacecraft( elements = sc2_elements )

# Start the simulation here.
timeNow, duration, timestep = 0.0, 86400.0, 30.0 # Seconds
samples = int(duration / timestep)
k = 0  # Sample count

# Matrix to store the data
rtn_states_yank = np.zeros((samples, 6))
rtn_states_true = np.zeros((samples, 6))

# Initialize the data
rv_c_eci = np.array([sc1.px, sc1.py, sc1.pz, sc1.vx, sc1.vy, sc1.vz])
rv_d_eci = np.array([sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz])
rv_cd_eci = rv_d_eci - rv_c_eci
r_rtn, v_rtn = rv_eci_to_rtn(rv_c_eci, rv_cd_eci)
rv_rtn = np.concatenate((r_rtn, v_rtn)) # Merge the pos and vel

# Make 2 copies of the initial RTN states (one for YA, and one for truth)
rv_rtn_yank = rv_rtn
rv_rtn_true = rv_rtn

# ACTUAL SIMULATION CODE BELOW. Note: the actual SC2 object isn't used below.
while timeNow < duration:

    # Record states for SC2 copy (using YA state transitions)
    rtn_states_yank[k,:] = rv_rtn_yank
    
    # Record states for SC2 copy (using non-linear FDERM)
    rtn_states_true[k,:] = rv_rtn_true
    
    # Compute the initial YA transition matrix
    stm_init = stm_yank_initial(sc1)
    
    # Propagate states for SC2 copy (using non-linear FDERM)
    rv_rtn_true[0:3], rv_rtn_true[3:6] = relative_rk4(timestep, sc1.n,
                                                      rv_rtn_true[0:3],
                                                      rv_rtn_true[3:6], sc1.a)
    
    # Propagate the chief SC
    sc1.propagate_perturbed(timestep, timestep)
    
    # Compute the current YA transition matrix
    stm_curr = stm_yank_current(sc1, timestep)
    
    # Propagate states for SC2 copy (using YA state transitions).
    
    # Note the ordering of states in the book is different. The ordering is:
    # rv_temp = [xBar, xBarDot, yBar, yBarDot, zBar, zBarDot] (in the book)
    # rv_rtn_yank = [xBar, yBar, zBar, xBarDot, yBarDot, zBarDot] (in this sim)
    
    r0 = norm([ sc1.px, sc1.py, sc1.pz ])
    v0 = norm([ sc1.vx, sc1.vy, sc1.vz ])
    rv_temp = [rv_rtn_yank[0], rv_rtn_yank[3], rv_rtn_yank[1],
               rv_rtn_yank[4], rv_rtn_yank[2], rv_rtn_yank[5]] # Re-order
    rv_temp = rv_temp / np.array([r0, v0, r0, v0, r0, v0])
    rv_temp = (stm_curr @ stm_init @ rv_temp) * np.array([r0, v0, r0, v0, r0, v0])
    rv_rtn_yank = [rv_temp[0], rv_temp[2], rv_temp[4],
                   rv_temp[1], rv_temp[3], rv_temp[5]] # Re-order
    
    # print(np.linalg.norm(rv_rtn_true - rv_rtn_yank))
    
    # Update time and sample count.
    timeNow += timestep
    k += 1

##############################################################################
##############################################################################

plt.close('all')
timeAxis = np.linspace(0, duration, samples)


# Plot position errors between YA and truth
plt.figure(1)

plt.subplot(3, 1, 1)
plt.title('`YA` minus `Truth` position error in RTN [km]')
plt.plot(timeAxis, rtn_states_true[:, 0] - rtn_states_yank[:, 0])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('R [km]')

plt.subplot(3, 1, 2)
plt.plot(timeAxis, rtn_states_true[:, 1] - rtn_states_yank[:, 1])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('T [km]')

plt.subplot(3, 1, 3)
plt.plot(timeAxis, rtn_states_true[:, 2] - rtn_states_yank[:, 2])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('N [km]')

plt.show()

###############################################################################
###############################################################################

# Plot velocity errors between YA and truth
plt.figure(2)

plt.subplot(3, 1, 1)
plt.title('`YA` minus `Truth` velocity error in RTN [km/s]')
plt.plot(timeAxis, rtn_states_true[:, 3] - rtn_states_yank[:, 3])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('R [km/s]')

plt.subplot(3, 1, 2)
plt.plot(timeAxis, rtn_states_true[:, 4] - rtn_states_yank[:, 4])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('T [km/s]')

plt.subplot(3, 1, 3)
plt.plot(timeAxis, rtn_states_true[:, 5] - rtn_states_yank[:, 5])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('N [km/s]')

plt.show()

###############################################################################
###############################################################################

# Plot RTN of truth and YA in 3D
fig3 = plt.figure(3).add_subplot(projection='3d')
axisLimit = 1.0 # km

# Plot YA vs truth
fig3.plot(rtn_states_yank[:,1], rtn_states_yank[:,2], rtn_states_yank[:,0],
          'r-', alpha = 0.35)
fig3.plot(rtn_states_true[:,1], rtn_states_true[:,2], rtn_states_true[:,0],
          'b:', alpha = 0.85)

# Plot a vector triad to represent chief at the origin
fig3.quiver(0,0,0,1,0,0, length = axisLimit / 5, color = 'g',
            arrow_length_ratio = 0.3 )
fig3.quiver(0,0,0,0,1,0, length = axisLimit / 5, color = 'g',
            arrow_length_ratio = 0.3 )
fig3.quiver(0,0,0,0,0,1, length = axisLimit / 5, color = 'g',
            arrow_length_ratio = 0.3 )

# Set plotting parameters
fig3.set_title('Trajectory in RTN of YA versus truth')
fig3.grid()
fig3.set_xlabel('T [km]')
fig3.set_ylabel('N [km]')
fig3.set_zlabel('R [km]')
fig3.set_xlim(-axisLimit, axisLimit)
fig3.set_ylim(-axisLimit, axisLimit)
fig3.set_zlim(-axisLimit, axisLimit)
fig3.legend(['YA', 'True'])

###############################################################################
###############################################################################

# Plot of relative orbit planes

plt.figure(4)

# TR plane
plt.subplot(1, 3, 1)
plt.title('TR plane plot')
plt.plot(rtn_states_yank[:,1], rtn_states_yank[:,0], 'r-')
plt.plot(rtn_states_true[:,1], rtn_states_true[:,0], 'b:')
plt.grid()
plt.xlabel('T component [km]')
plt.ylabel('R component [km]')
plt.axis('equal')
plt.legend(['YA', 'True'])

# NR plane
plt.subplot(1, 3, 2)
plt.title('NR plane plot')
plt.plot(rtn_states_yank[:,2], rtn_states_yank[:,0], 'r-')
plt.plot(rtn_states_true[:,2], rtn_states_true[:,0], 'b:')
plt.grid()
plt.xlabel('N component [km]')
plt.ylabel('R component [km]')
plt.axis('equal')
plt.legend(['YA', 'True'])

# TN plane
plt.subplot(1, 3, 3)
plt.title('TN plane plot')
plt.plot(rtn_states_yank[:,1], rtn_states_yank[:,2], 'r-')
plt.plot(rtn_states_true[:,1], rtn_states_true[:,2], 'b:')
plt.grid()
plt.xlabel('T component [km]')
plt.ylabel('N component [km]')
plt.axis('equal')
plt.legend(['YA', 'True'])

plt.show()