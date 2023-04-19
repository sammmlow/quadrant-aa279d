# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 21:00:16 2023

@author: sammm
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm as norm

from source import spacecraft

###############################################################################
###############################################################################

GM = 398600.4418  # Earth default (km**3/s**2)

# Initialize orbit elements [a, e, i, w, R, M] in km and degrees
sc1_elements = [6928.137, 0.000001, 97.5976, 0.0, 250.662, 0.827]
# sc2_elements = [6928.137, 0.000001, 97.5976, 0.0, 251.489, 0.413]
# Introduce semi-major difference deltaa = 1 km to deputy
sc2_elements = [6929.137, 0.000001, 97.5976, 0.0, 251.489, 0.413]

# Define the spacecraft itself.
sc1 = spacecraft.Spacecraft(elements=sc1_elements)
sc2 = spacecraft.Spacecraft(elements=sc2_elements)

# # Enable J2 forces on these two spacecraft.
# sc1.forces['j2'] = True
# sc2.forces['j2'] = True

###############################################################################
###############################################################################

# FUNCTIONAL DEFINITIONS BELOW

# Define a function that produces a 3x3 Hill frame matrix given an ECI
# position and velocity vector `r` and `v`.


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


# Define an acceleration function to compute RTN force vectors in RTN basis
# and as seen in RTN frame. Input vectors are also RTN basis with time
# derivatives taken in the RTN frame.


def accel_rtn(n, r_rtn, v_rtn, r0):
    x = r_rtn[0]
    y = r_rtn[1]
    z = r_rtn[2]
    xdot = v_rtn[0]
    ydot = v_rtn[1]
    zdot = v_rtn[2]
    xddot_eci = ((GM*(r0+x))/((r0+x)**2 + y**2 + z**2)**1.5) - (GM/(r0**2))
    yddot_eci = ((GM*y)/((r0+x)**2 + y**2 + z**2)**1.5)
    zddot_eci = ((GM*z)/((r0+x)**2 + y**2 + z**2)**1.5)
    xddot = (2*n*ydot) + (n*n*x) - xddot_eci
    yddot = (n*n*y) - (2*n*xdot) - yddot_eci
    zddot = -zddot_eci
    return np.array([xddot, yddot, zddot])

# Define an RK4 propagator function (Simpson's rule variant).
# Inputs timestep `dt` in seconds, `n` as mean motion (rad/s), `r_rtn` and
# `v_rtn` are 1x3 numpy arrays, and `r0` is the radial distance of the chief
# spacecraft i.e. origin of the RTN frame from center of the primary attractor


def relative_rk4(dt, n, r_rtn, v_rtn, r0):
    c = 1/3  # Constant
    k1p = v_rtn
    k1v = accel_rtn(n, r_rtn, v_rtn, r0)
    k2p = v_rtn + dt * (c*k1v)
    k2v = accel_rtn(n, r_rtn + dt*(c*k1p), v_rtn + dt*(c*k1v), r0)
    k3p = v_rtn + dt * (k2v-c*k1v)
    k3v = accel_rtn(n, r_rtn + dt*(k2p-c*k1p), v_rtn + dt*(k2v-c*k1v), r0)
    k4p = v_rtn + dt * (k1v-k2v+k3v)
    k4v = accel_rtn(n, r_rtn + dt*(k1p-k2p+k3p), v_rtn + dt*(k1v-k2v+k3v), r0)
    r_rtn_f = r_rtn + (dt/8) * (k1p + 3*k2p + 3*k3p + k4p)
    v_rtn_f = v_rtn + (dt/8) * (k1v + 3*k2v + 3*k3v + k4v)
    return r_rtn_f, v_rtn_f

###############################################################################
###############################################################################


# Start the simulation here.
timeNow = 0.0
duration = 86400 * 1
timestep = 30.0  # Seconds
samples = int(duration / timestep) * 3
n = 0  # Sample count

# Initialize the ECI positions and velocities of SC1 and SC2.
rv_c_eci = np.array([sc1.px, sc1.py, sc1.pz, sc1.vx, sc1.vy, sc1.vz])
rv_d_eci = np.array([sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz])
rv_cd_eci = rv_d_eci - rv_c_eci

# Initialize the RTN positions and velocities of SC2 w.r.t. SC1.
# Note that the time derivative for v_rtn is performed in ECI frame, but
# vector coordinates are expressed in RTN basis.
r_rtn, v_rtn = rv_eci_to_rtn(rv_c_eci, rv_cd_eci)

# Store results of the simulation
states_rel_rtn = np.zeros((samples, 6))  # Results of propagating relative ODE
states_abs_rtn = np.zeros((samples, 6))  # Results of propagating absolute ODE

while timeNow < duration:

    # Record states for relative motion propagated by QUADRANT using absolute.
    hill = sc1.get_hill_frame()
    sc1_eci = np.array([sc1.px, sc1.py, sc1.pz, sc1.vx, sc1.vy, sc1.vz])
    sc2_eci = np.array([sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz])
    sc12_eci = sc2_eci - sc1_eci
    r_rtn_abs, v_rtn_abs = rv_eci_to_rtn(sc1_eci, sc12_eci)
    states_abs_rtn[n, 0:3] = r_rtn_abs
    states_abs_rtn[n, 3:6] = v_rtn_abs

    # # Record states for relative motion propagated by QUADRANT using absolute.
    # hill = sc1.get_hill_frame()
    # sc1_eci = np.array([sc1.px, sc1.py, sc1.pz])
    # sc2_eci = np.array([sc2.px, sc2.py, sc2.pz])
    # states_abs_rtn[n, 0:3] = hill @ (sc2_eci - sc1_eci)

    # Propagate the absolute motion of the spacecraft using QUADRANT.
    sc1.propagate_perturbed(timestep, timestep)
    sc2.propagate_perturbed(timestep, timestep)

    # Record states of relative motion propagated by non-linear relative EOMs.
    states_rel_rtn[n, 0:3] = r_rtn
    states_rel_rtn[n, 3:6] = v_rtn

    # Propagate relative motion using non-linear relative EOMs.
    r_rtn, v_rtn = relative_rk4(timestep, sc1.n, r_rtn, v_rtn, sc1.a)

    # Update time and sample count.
    timeNow += timestep
    n += 1

# Perform impulsive maneuver @ tM
# Obtain deputy states right before maneuver to calculate the corresponding delta v_T
# Calculate both Absolute & Relative to compare

dela = -1
adc = sc1_elements[0]
vadc = np.array([adc, 0, 0])
ad0 = sc2_elements[0]
delvT = np.sqrt(GM*ad0)*dela/(2*ad0**2)
vxtMm = sc2.vx
vytMm = sc2.vy
vztMm = sc2.vz
# velocity impulse in ECI
rtM = 1+delvT/(np.sqrt(vxtMm**2+vytMm**2+vztMm**2))
check1 = np.sqrt(vxtMm**2+vytMm**2+vztMm**2)
vxtMp = vxtMm * rtM
vytMp = vytMm * rtM
vztMp = vztMm * rtM
sc2.vx = vxtMp
sc2.vy = vytMp
sc2.vz = vztMp
rv_c_ecitM = np.array([sc1.px, sc1.py, sc1.pz, sc1.vx, sc1.vy, sc1.vz])
rv_d_ecitM = np.array([sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz])
omegac = np.array([0.0, 0.0, np.sqrt(GM/(adc*adc*adc))])
# deputy velocity in inertial, expressed in RTN of chief
v_rtn_I = v_rtn + np.cross(omegac, vadc+r_rtn)
# perform impulse maneuver
v_rtn_I = v_rtn_I * (1 + delvT/(np.linalg.norm(v_rtn_I)))
# calculate corresponding deputy velocity seen by chief RTN, expressed in chief RTN
v_rtn = v_rtn_I - np.cross(omegac, vadc+r_rtn)
# Propagate relative motion
#r_rtn, v_rtn = relative_rk4(timestep, sc1.n, r_rtn, v_rtn, sc1.a)

duration2 = 3 * duration

while timeNow < duration2:

    # Record states for relative motion propagated by QUADRANT using absolute.
    hill = sc1.get_hill_frame()
    sc1_eci = np.array([sc1.px, sc1.py, sc1.pz, sc1.vx, sc1.vy, sc1.vz])
    sc2_eci = np.array([sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz])
    sc12_eci = sc2_eci - sc1_eci
    r_rtn_abs, v_rtn_abs = rv_eci_to_rtn(sc1_eci, sc12_eci)
    states_abs_rtn[n, 0:3] = r_rtn_abs
    states_abs_rtn[n, 3:6] = v_rtn_abs

    # # Record states for relative motion propagated by QUADRANT using absolute.
    # hill = sc1.get_hill_frame()
    # sc1_eci = np.array([sc1.px, sc1.py, sc1.pz])
    # sc2_eci = np.array([sc2.px, sc2.py, sc2.pz])
    # states_abs_rtn[n, 0:3] = hill @ (sc2_eci - sc1_eci)

    # Propagate the absolute motion of the spacecraft using QUADRANT.
    sc1.propagate_perturbed(timestep, timestep)
    sc2.propagate_perturbed(timestep, timestep)

    # Record states of relative motion propagated by non-linear relative EOMs.
    states_rel_rtn[n, 0:3] = r_rtn
    states_rel_rtn[n, 3:6] = v_rtn

    # Propagate relative motion using non-linear relative EOMs.
    r_rtn, v_rtn = relative_rk4(timestep, sc1.n, r_rtn, v_rtn, sc1.a)

    # Update time and sample count.
    timeNow += timestep
    n += 1

###############################################################################
###############################################################################

# Plot results for position.

plt.close('all')

timeAxis = np.linspace(0, duration2, samples)

plt.figure(1)

plt.subplot(3, 1, 1)
plt.title('Plots of RTN position components')
plt.plot(timeAxis, states_abs_rtn[:, 0], '-')
plt.plot(timeAxis, states_rel_rtn[:, 0], '--')
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('R component [km]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.subplot(3, 1, 2)
plt.plot(timeAxis, states_abs_rtn[:, 1], '-')
plt.plot(timeAxis, states_rel_rtn[:, 1], '--')
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('T component [km]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.subplot(3, 1, 3)
plt.plot(timeAxis, states_abs_rtn[:, 2], '-')
plt.plot(timeAxis, states_rel_rtn[:, 2], '--')
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('N component [km]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.show()

###############################################################################
###############################################################################

# Plot results for velocity.

plt.figure(2)

plt.subplot(3, 1, 1)
plt.title('Plots of RTN velocity components')
plt.plot(timeAxis, states_abs_rtn[:, 3], '-')
plt.plot(timeAxis, states_rel_rtn[:, 3], '--')
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('R component [km/s]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.subplot(3, 1, 2)
plt.plot(timeAxis, states_abs_rtn[:, 4], '-')
plt.plot(timeAxis, states_rel_rtn[:, 4], '--')
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('T component [km/s]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.subplot(3, 1, 3)
plt.plot(timeAxis, states_abs_rtn[:, 5], '-')
plt.plot(timeAxis, states_rel_rtn[:, 5], '--')
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('N component [km/s]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.show()

###############################################################################
###############################################################################

''' # Plot results for position error.
plt.figure(3)

plt.subplot(3, 1, 1)
plt.title('Plots of RTN position errors between FODE and FDERM')
plt.plot(timeAxis, states_abs_rtn[:, 0] - states_rel_rtn[:, 0], 'g')
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('R component [km]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.subplot(3, 1, 2)
plt.plot(timeAxis, states_abs_rtn[:, 1] - states_rel_rtn[:, 1], 'g')
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('T component [km]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.subplot(3, 1, 3)
plt.plot(timeAxis, states_abs_rtn[:, 2] - states_rel_rtn[:, 2], 'g')
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('N component [km]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.show()

###############################################################################
###############################################################################

# Plot results for velocity error.

plt.figure(4)

plt.subplot(3, 1, 1)
plt.title('Plots of RTN velocity errors between FODE and FDERM')
plt.plot(timeAxis, states_abs_rtn[:, 3] - states_rel_rtn[:, 3], 'm')
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('R component [km/s]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.subplot(3, 1, 2)
plt.plot(timeAxis, states_abs_rtn[:, 4] - states_rel_rtn[:, 4], 'm')
plt.grid()
plt.xlabel('Simula#tion time [sec]')
plt.ylabel('T component [km/s]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.subplot(3, 1, 3)
plt.plot(timeAxis, states_abs_rtn[:, 5] - states_rel_rtn[:, 5], 'm')
plt.grid()
plt.xlabel('Simulation time [sec]')
plt.ylabel('N component [km/s]')
plt.legend(['Absolute FODE', 'Relative FDERM'])

plt.show() '''
