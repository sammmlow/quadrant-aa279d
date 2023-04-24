# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:55:40 2023

By: Sam Low and Katherine Cao
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

# Import our own spacecraft library
from source import spacecraft

##############################################################################
##############################################################################

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

##############################################################################
##############################################################################

from main_ps2 import rv_eci_to_rtn, relative_rk4
from main_ps3_roe import compute_roe

# Now test and see, based on initial conditions in Table 2 of PS3, if HCW
# approximates motion well, with the non-linear FDERM propagation.
sc1_elements = [7928.137, 0.6, 97.5976, 0.0, 250.6620, 0.00827]
sc2_elements = [7928.137, 0.6, 97.5976, 0.0, 250.6703, 0.00413]
sc3_elements = [7928.137, 0.6, 97.5976, 0.0, 250.6620, 0.00000]

# Create the spacecraft objects.
sc1 = spacecraft.Spacecraft( elements = sc1_elements )
sc2 = spacecraft.Spacecraft( elements = sc2_elements )
sc3 = spacecraft.Spacecraft( elements = sc3_elements )

# Print out the QSN ROEs for SC2
# compute_roe(sc1, sc2)
# compute_roe(sc1, sc3)

# Start the simulation here.
timeNow, duration, timestep = 0.0, 86400.0, 30.0 # Seconds
samples = int(duration / timestep)
k = 0  # Sample count

# Matrix to store the data
rtn_states_hcw  = np.zeros((samples, 6))
rtn_states_true = np.zeros((samples, 6))

# Initialize the data
rv_c_eci = np.array([sc1.px, sc1.py, sc1.pz, sc1.vx, sc1.vy, sc1.vz])
rv_d_eci = np.array([sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz])
rv_cd_eci = rv_d_eci - rv_c_eci
r_rtn, v_rtn = rv_eci_to_rtn(rv_c_eci, rv_cd_eci)
rv_rtn = np.concatenate((r_rtn, v_rtn)) # Merge the pos and vel

# Make 2 copies of the initial RTN states (one for HCW, and one for truth)
rv_rtn_hcw = rv_rtn
rv_rtn_true = rv_rtn

# ACTUAL SIMULATION CODE BELOW. Note: the actual SC2 object isn't used below.
while timeNow < duration:

    # Record states for SC2 copy (using HCW state transitions)
    rtn_states_hcw[k,:] = rv_rtn_hcw
    
    # Record states for SC2 copy (using non-linear FDERM)
    rtn_states_true[k,:] = rv_rtn_true
    
    # Propagate states for SC2 copy (using HCW state transitions)
    rv_rtn_hcw = stm_hcw(sc1, timestep) @ rv_rtn_hcw
    
    # Propagate states for SC2 copy (using non-linear FDERM)
    rv_rtn_true[0:3], rv_rtn_true[3:6] = relative_rk4(timestep, sc1.n,
                                                      rv_rtn_true[0:3],
                                                      rv_rtn_true[3:6], sc1.a)
    
    # print(np.linalg.norm(rv_rtn_true - rv_rtn_hcw))
    
    # Finally, the chief itself needs to be propagated (in absolute motion)
    sc1.propagate_perturbed(timestep, timestep)

    # Update time and sample count.
    timeNow += timestep
    k += 1

##############################################################################
##############################################################################

plt.close('all')
timeAxis = np.linspace(0, duration, samples)

# Plot position errors between HCW and truth
plt.figure(1)

plt.subplot(3, 1, 1)
plt.title('`HCW` minus `Truth` position error in RTN [km]')
plt.plot(timeAxis, rtn_states_true[:, 0] - rtn_states_hcw[:, 0])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('R [km]')

plt.subplot(3, 1, 2)
plt.plot(timeAxis, rtn_states_true[:, 1] - rtn_states_hcw[:, 1])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('T [km]')

plt.subplot(3, 1, 3)
plt.plot(timeAxis, rtn_states_true[:, 2] - rtn_states_hcw[:, 2])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('N [km]')

plt.show()

###############################################################################
###############################################################################

# Plot velocity errors between HCW and truth
plt.figure(2)

plt.subplot(3, 1, 1)
plt.title('`HCW` minus `Truth` velocity error in RTN [km/s]')
plt.plot(timeAxis, rtn_states_true[:, 3] - rtn_states_hcw[:, 3])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('R [km/s]')

plt.subplot(3, 1, 2)
plt.plot(timeAxis, rtn_states_true[:, 4] - rtn_states_hcw[:, 4])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('T [km/s]')

plt.subplot(3, 1, 3)
plt.plot(timeAxis, rtn_states_true[:, 5] - rtn_states_hcw[:, 5])
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('N [km/s]')

plt.show()

###############################################################################
###############################################################################

# Plot RTN of truth and HCW in 3D
fig3 = plt.figure(3).add_subplot(projection='3d')
axisLimit = 1.0 # km

# Plot HCW vs truth
fig3.plot(rtn_states_hcw[:,1], rtn_states_hcw[:,2], rtn_states_hcw[:,0],
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
fig3.set_title('Trajectory in RTN of HCW versus truth')
fig3.grid()
fig3.set_xlabel('T [km]')
fig3.set_ylabel('N [km]')
fig3.set_zlabel('R [km]')
fig3.set_xlim(-axisLimit, axisLimit)
fig3.set_ylim(-axisLimit, axisLimit)
fig3.set_zlim(-axisLimit, axisLimit)
fig3.legend(['HCW', 'True'])

###############################################################################
###############################################################################

# Plot of relative orbit planes

plt.figure(4)

# TR plane
plt.subplot(1, 3, 1)
plt.title('TR plane plot')
plt.plot(rtn_states_hcw[:,1], rtn_states_hcw[:,0], 'r-')
plt.plot(rtn_states_true[:,1], rtn_states_true[:,0], 'b:')
plt.grid()
plt.xlabel('T component [km]')
plt.ylabel('R component [km]')
plt.axis('equal')
plt.legend(['HCW', 'True'])

# NR plane
plt.subplot(1, 3, 2)
plt.title('NR plane plot')
plt.plot(rtn_states_hcw[:,2], rtn_states_hcw[:,0], 'r-')
plt.plot(rtn_states_true[:,2], rtn_states_true[:,0], 'b:')
plt.grid()
plt.xlabel('N component [km]')
plt.ylabel('R component [km]')
plt.axis('equal')
plt.legend(['HCW', 'True'])

# TN plane
plt.subplot(1, 3, 3)
plt.title('TN plane plot')
plt.plot(rtn_states_hcw[:,1], rtn_states_hcw[:,2], 'r-')
plt.plot(rtn_states_true[:,1], rtn_states_true[:,2], 'b:')
plt.grid()
plt.xlabel('T component [km]')
plt.ylabel('N component [km]')
plt.axis('equal')
plt.legend(['HCW', 'True'])

plt.show()