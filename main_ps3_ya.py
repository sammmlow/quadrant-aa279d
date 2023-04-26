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

# The YA STM follows the original YA paper in terms of matrix ordering.
# Input ordering will be sorted within the YA STM propagation function.
# Transforms are taken from Matthew Willis' PhD thesis

def ya_transform(sc, rv_rtn):
    
    # Constants
    k = 1 + sc.e * cos(sc.nu)
    p = sc.a * (1 - sc.e**2)
    factor = sqrt(p / sc.GM)
    kprime = -sc.e * sin(sc.nu)
    
    # Compute transform
    r = rv_rtn[0:3]
    v = rv_rtn[3:6]
    rt = (k / p) * r
    vt = ((kprime / p) * r) + ((factor / k) * v)
    
    return np.concatenate((rt, vt))

def ya_inverse_transform(sc, rv_rtn_t):
    
    # Constants
    k = 1 + sc.e * cos(sc.nu)
    p = sc.a * (1 - sc.e**2)
    factor = sqrt(p / sc.GM)
    kprime = -sc.e * sin(sc.nu)
    
    # Compute transform
    rt = rv_rtn_t[0:3]
    vt = rv_rtn_t[3:6]
    r = (p / k) * rt
    v = ((k * vt) - kprime * rt) / factor
    return np.concatenate((r, v))

def stm_yank_propagate_willis(sc, dt, f0, rv_rtn):
    
    # Constants.
    e     = sc.e
    f     = sc.nu
    eta   = sqrt(1-e**2)
    etasq = eta**2
    p     = sc.a * (1 - sc.e**2)
    J     = dt * sqrt(sc.GM / (p**3))
    k     = 1 + e * cos(f)
    k0    = 1 + e * cos(f0)
    ksf   = k * sin(f)
    ksf0  = k0 * sin(f0)
    ksfp  = cos(f) + e*cos(2*f)
    kcf   = k * cos(f)
    kcf0  = k0 * cos(f0)
    kcfp  = -sin(f) - e*sin(2*f)
    
    # Compute the transform matrix for pseudo-initial states
    STM1 = (1/etasq) * np.array([
        [(6*k0+(2*e**2)-2),         (0),     (0),             (2*e*ksf0/etasq), (2*k0**2),           (0)             ],
        [(-3*(1+(e**2)/k0)*sin(f0)), (0),     (0),             (kcf0-2*e),       (-sin(f0)*(1+k0)),   (0)             ],
        [(-3*(e+cos(f0))),           (0),     (0),             (-ksf0),          (-e-(1+k0)*cos(f0)), (0)             ],
        [(-3*e*(1+1/k0)*sin(f0)),    (etasq), (0),             (e*kcf0-2),       (-e*sin(f0)*(1+k0)), (0)             ],
        [(0),                        (0),     (etasq*sin(f0)), (0),              (0),                 (etasq*cos(f0)) ],
        [(0),                        (0),     (etasq*cos(f0)), (0),              (0),                 (-etasq*sin(f0))]])
    
    # Compute the state transition matrix for pseudo-final states
    STM2 = np.array([
        [(1-1.5*e*ksf*J),              (ksf),          (kcf),           (0), (0),      (0)      ],
        [(-1.5*(k**2)*J),              ((1+k)*cos(f)), (-(1+k)*sin(f)), (1), (0),      (0)      ],
        [(0),                          (0),            (0),             (0), (sin(f)), (cos(f)) ],
        [(-1.5*e*(ksfp*J+(sin(f)/k))), (ksfp),         (kcfp),          (0), (0),      (0)      ],
        [(1.5*(2*e*J*ksf-1)),          (-2*ksf),       (e-2*kcf),       (0), (0),      (0)      ],
        [(0),                          (0),            (0),             (0), (cos(f)), (-sin(f))]])
    
    # print(STM2 @ STM1)
    
    # Perform the coordinate transform and state transition.
    rv_rtn_trans = ya_transform(sc, rv_rtn)
    rv_rtn_trans_final = STM2 @ STM1 @ rv_rtn_trans
    rv_rtn_final = ya_inverse_transform(sc, rv_rtn_trans_final)
    return rv_rtn_final

# def stm_yank_propagate(sc, dt, nu0, rv_rtn):
    
#     # Perform coordinate transform (see Eq 86 of YA paper)
#     # rv_y  = [-pN, -vN]
#     # rv_xz = [ pT, -pR, vT, -vR]
#     rv_rtn_trans = ya_transform(sc, rv_rtn)
#     rv_y = np.array([-rv_rtn_trans[2], -rv_rtn_trans[5]])
#     rv_xz = np.array([rv_rtn_trans[1], -rv_rtn_trans[0],
#                       rv_rtn_trans[4], -rv_rtn_trans[3]])
    
#     # Obtain chief spacecraft angular momentum
#     r_vec = np.array([sc.px, sc.py, sc.pz])
#     v_vec = np.array([sc.vx, sc.vy, sc.vz])
#     h_vec = np.cross(r_vec, v_vec)
#     h_abs = np.linalg.norm(h_vec)
#     p = sc.a * (1 - sc.e**2)
    
#     # Constants.
#     e = sc.e
#     I = dt * h_abs / p**2 # Alternatively = dt * (sc.GM**2) / (h_abs**3)
#     k = 1 + e * cos(sc.nu)
#     c1 = k * cos(nu0)         # Using initial true anomaly
#     s1 = k * sin(nu0)         # Using initial true anomaly
#     c2 = k * cos(sc.nu)       # Using final true anomaly
#     s2 = k * sin(sc.nu)       # Using final true anomaly
#     cdelta = cos(sc.nu - nu0) # Using change in true anomaly
#     sdelta = sin(sc.nu - nu0) # Using change in true anomaly
#     kdelta = 1 + e * cdelta
#     cprime = -1 * (sin(sc.nu) + e * sin(2 * sc.nu))
#     sprime = cos(sc.nu) + e * cos(2 * sc.nu)
    
#     # In-plane STM1 using nu0
#     sRT1 = (1/(1-e**2)) * np.array([
#         [(1-e**2), (3*e*s1*((1/k)+(1/k**2))  ), (-e*s1*(1+(1/k))), (2-e*c1)],
#         [(0     ), (-3*s1*((1/k)+(e**2/k**2))), (s1*(1+(1/k))   ), (c1-2*e)],
#         [(0     ), (-3*((c1/k)+e)            ), (e+c1*(1+(1/k)) ), (-s1)   ],
#         [(0     ), (3*k+(e**2)-1             ), (-k**2          ), (e*s1)  ]])
    
#     # In-plane STM2 using sc.nu
#     sRT2 = np.array([
#         [(1), (-c2*(1+(1/k))), (s2*(1+(1/k))), (3*(k**2)*I               )],
#         [(0), (s2           ), (c2          ), (2-3*e*s2*I               )],
#         [(0), (2*s2         ), (2*c2-e      ), (3*(1-2*e*s2*I)           )],
#         [(0), (sprime       ), (cprime      ), (-3*e*(sprime*I+s2/(k**2)))]])
    
#     # Out-of-plane STM
#     sN = np.array([
#         [ cdelta, sdelta],
#         [-sdelta, cdelta]]) / kdelta
    
#     # Perform propagation.
#     rv_y_final = sN @ rv_y
#     rv_xz_final = sRT2 @ sRT1 @ rv_xz
    
#     # Re-order the vector back into [pR, pT, pN, vR, vT, vN]
#     r_rtn_t_final = np.array([-rv_xz_final[1], rv_xz_final[0], -rv_y_final[0]])
#     v_rtn_t_final = np.array([-rv_xz_final[3], rv_xz_final[2], -rv_y_final[1]])
#     rv_rtn_t_final = np.concatenate((r_rtn_t_final, v_rtn_t_final))
    
#     # Inverse coordinate transform
#     rv_rtn_final = ya_inverse_transform(sc, rv_rtn_t_final)
#     return rv_rtn_final

##############################################################################
##############################################################################

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


def relative_rk4(sc, dt, r_rtn, v_rtn):
    r = np.array([sc.px, sc.py, sc.pz])
    v = np.array([sc.vx, sc.vy, sc.vz])
    r0 = norm(r)
    n = norm(np.cross(r, v)) / r0**2 # True anomaly rate
    c = 1/3 # Constant
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

##############################################################################
##############################################################################

# Initialize SC osculating elements
sc1_elements = [7928.137, 0.1, 97.5976, 0.0, 250.6620, 0.00827]
sc2_elements = [7928.137, 0.1, 97.5976, 0.0, 250.6703, 0.00413]

# Create the spacecraft objects.
sc1 = spacecraft.Spacecraft( elements = sc1_elements )
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

# ACTUAL SIMULATION CODE BELOW.
while timeNow < duration:

    # Record states for SC2 copy (using YA state transitions)
    rtn_states_yank[k,:] = rv_rtn_yank
    
    # Record states for SC2 copy (using non-linear FDERM)
    rtn_states_true[k,:] = rv_rtn_true
    
    # Propagate the chief SC1 (and compute the YA transition matrices)
    nu0 = sc1.nu
    sc1.propagate_perturbed(timestep, timestep) # Propagate SC1
    
    # Propagate states for SC2 copy (using YA state transitions).
    rv_rtn_yank = stm_yank_propagate(sc1, timestep, nu0, rv_rtn_yank)
    
    # # Propagate states for SC2 copy (using non-linear FDERM)
    # rv_rtn_true[0:3], rv_rtn_true[3:6] = relative_rk4(sc1, timestep,
    #                                               rv_rtn_true[0:3],
    #                                               rv_rtn_true[3:6])
    
    # Note that FDERM is not suitable for non-circular orbits so
    # we will have to use the difference of the two FODE solutions
    sc2.propagate_perturbed(timestep, timestep)
    sc1_eci = np.array([sc1.px, sc1.py, sc1.pz, sc1.vx, sc1.vy, sc1.vz])
    sc2_eci = np.array([sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz])
    r_rtn_true, v_rtn_true = rv_eci_to_rtn(sc1_eci, sc2_eci - sc1_eci)
    rv_rtn_true = np.concatenate(( r_rtn_true, v_rtn_true ))
    
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