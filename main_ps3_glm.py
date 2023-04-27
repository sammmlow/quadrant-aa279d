# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:52:30 2023

By: Sam Low and Katherine Cao
"""

from main_ps2 import rv_eci_to_rtn, relative_rk4
from main_ps3_roe import compute_roe
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt

# Import our own spacecraft library
from source import spacecraft

##############################################################################
##############################################################################


# The Linear Geometric Mapping follows Matthew Willis' PhD thesis
#ROE = [da, dL, dex, dey, dix, diy]

def stm_lgm_propagate_willis(sc, t, roe6):

    # Constants.
    e = sc.e
    f = sc.nu
    u = f + sc.w
    eta = sqrt(1-e**2)
    etasq = eta**2
    etacub = eta * etasq
    n = sc.n
    ex = sc.e * cos(sc.w)
    ey = sc.e * sin(sc.w)
    #k = 1 + ex * cos(u) + ey * sin(u)
    k = 1 + e*cos(f)
    ksq = k**2
    kprime = -ex * sin(u) + ey * cos(u)
    sci = sc.i

    # Compute first matrix
    STM1 = np.array([[(sc.a * etasq), (0), (0), (0), (0), (0)],
                     [(0), (sc.a * etasq), (0), (0), (0), (0)],
                     [(0), (0), (sc.a * etasq), (0), (0), (0)],
                     [(0), (0), (0), (sc.a * n / eta), (0), (0)],
                     [(0), (0), (0), (0), (sc.a * n / eta), (0)],
                     [(0), (0), (0), (0), (0), (sc.a * n / eta)]])
    bx1 = 1/k + 3/2*kprime*n*t/etacub
    bx2 = -kprime/etacub
    bx3 = 1/etacub*(ex * (k-1)/(1+eta)-cos(u))
    bx4 = 1/etacub*(ey * (k-1)/(1+eta)-sin(u))
    bx6 = kprime/etacub * cos(sci)/sin(sci)
    by1 = -3/2*k*n*t/etacub
    by2 = k/etacub
    by3 = 1/etasq*((1+1/k)*sin(u)+ey/k + k/eta*(ey/(1+eta)))
    by4 = -1/etasq*((1+1/k)*cos(u)+ex/k + k/eta*(ex/(1+eta)))
    by6 = (1/k - k/etacub) * cos(sci)/sin(sci)
    bxd1 = kprime/2 + 3/2*ksq*(1-k)*n*t/etacub
    bxd2 = ksq/etacub*(k-1)
    bxd3 = ksq/etacub*(eta*sin(u)+ey*((k-1)/(1+eta)))
    bxd4 = -ksq/etacub*(eta*cos(u)+ex*((k-1)/(1+eta)))
    bxd6 = -ksq/etacub*(k-1)*cos(sci)/sin(sci)
    byd1 = -3/2*k*(1+k*kprime*n*t/etacub)
    byd2 = ksq*kprime/etacub
    byd3 = (1+ksq/etacub)*cos(u) + ex*k/etasq*(1+k/eta*((1-k)/(1+eta)))
    byd4 = (1+ksq/etacub)*sin(u) + ey*k/etasq*(1+k/eta*((1-k)/(1+eta)))
    byd6 = -(1+ksq/etacub)*kprime*cos(sci)/sin(sci)
    bz5 = 1/k*sin(u)
    bz6 = -1/k*cos(u)
    bzd5 = cos(u)+ex
    bzd6 = sin(u)+ey

    # Compute second matrix
    STM2 = np.array([[(bx1), (bx2), (bx3), (bx4), (0), (bx6)],
                     [(by1), (by2), (by3), (by4), (0), (by6)],
                     [(0), (0), (0), (0), (bz5), (bz6)],
                     [(bxd1), (bxd2), (bxd3), (bxd4), (0), (bxd6)],
                     [(byd1), (byd2), (byd3), (byd4), (0), (byd6)],
                     [(0), (0), (0), (0), (bzd5), (bzd6)]])
    STM = STM1@STM2

    return STM @ roe6


##############################################################################
##############################################################################


# Now test and see, based on initial conditions in Table 3of PS3
sc1_elements = [7928.137, 0.1, 97.5976, 0.0, 250.6620, 0.00827]
sc2_elements = [7928.137, 0.1-0.001, 97.5976, 0.0, 250.6703, 0.00413]
sc3_elements = [7928.137, 0.1-0.001, 97.5976, 0.0, 250.6620, 0.00000]

# Create the spacecraft objects.
sc1 = spacecraft.Spacecraft(elements=sc1_elements)
sc2 = spacecraft.Spacecraft(elements=sc2_elements)
sc3 = spacecraft.Spacecraft(elements=sc3_elements)

# Print out the QSN ROEs for SC2
roe62 = compute_roe(sc1, sc2)
roe63 = compute_roe(sc1, sc3)


# Start the simulation here.
timeNow, duration, timestep = 0.0, 86400.0, 30.0  # Seconds
samples = int(duration / timestep)
k = 0  # Sample count

# Matrix to store the data
rtn_states_glm2 = np.zeros((samples, 6))
rtn_states_glm3 = np.zeros((samples, 6))
rtn_states_true = np.zeros((samples, 6))

# Initialize the data
rv_c_eci = np.array([sc1.px, sc1.py, sc1.pz, sc1.vx, sc1.vy, sc1.vz])
rv_d_eci = np.array([sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz])
rv_cd_eci = rv_d_eci - rv_c_eci
r_rtn, v_rtn = rv_eci_to_rtn(rv_c_eci, rv_cd_eci)
rv_rtn = np.concatenate((r_rtn, v_rtn))  # Merge the pos and vel

# Make 2 copies of the initial RTN states (one for HCW, and one for truth)
#rv_rtn_hcw = rv_rtn
rv_rtn_true = rv_rtn

# ACTUAL SIMULATION CODE BELOW. Note: the actual SC2 object isn't used below.
while timeNow < duration:

    # Record states for SC2 copy (using HCW state transitions)
    rtn_states_glm2[k, :] = stm_lgm_propagate_willis(sc1, timeNow, roe62)
    rtn_states_glm3[k, :] = stm_lgm_propagate_willis(sc1, timeNow, roe63)
    print(rtn_states_glm2[k, 1])

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
    rv_rtn_true = np.concatenate((r_rtn_true, v_rtn_true))

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
# plt.figure(1)

#plt.subplot(3, 1, 1)
#plt.title('`HCW` minus `Truth` position error in RTN [km]')
#plt.plot(timeAxis, rtn_states_true[:, 0] - rtn_states_hcw[:, 0])
# plt.grid()
#plt.xlabel('Simulation time [sec]')
#plt.ylabel('R [km]')

#plt.subplot(3, 1, 2)
#plt.plot(timeAxis, rtn_states_true[:, 1] - rtn_states_hcw[:, 1])
# plt.grid()
#plt.xlabel('Simulation time [sec]')
#plt.ylabel('T [km]')

#plt.subplot(3, 1, 3)
#plt.plot(timeAxis, rtn_states_true[:, 2] - rtn_states_hcw[:, 2])
# plt.grid()
#plt.xlabel('Simulation time [sec]')
#plt.ylabel('N [km]')

# plt.show()

###############################################################################
###############################################################################

# Plot velocity errors between HCW and truth
# plt.figure(2)

#plt.subplot(3, 1, 1)
#plt.title('`HCW` minus `Truth` velocity error in RTN [km/s]')
#plt.plot(timeAxis, rtn_states_true[:, 3] - rtn_states_hcw[:, 3])
# plt.grid()
#plt.xlabel('Simulation time [sec]')
#plt.ylabel('R [km/s]')

#plt.subplot(3, 1, 2)
#plt.plot(timeAxis, rtn_states_true[:, 4] - rtn_states_hcw[:, 4])
# plt.grid()
#plt.xlabel('Simulation time [sec]')
#plt.ylabel('T [km/s]')

#plt.subplot(3, 1, 3)
#plt.plot(timeAxis, rtn_states_true[:, 5] - rtn_states_hcw[:, 5])
# plt.grid()
#plt.xlabel('Simulation time [sec]')
#plt.ylabel('N [km/s]')

# plt.show()

###############################################################################
###############################################################################

# Plot RTN of truth and HCW in 3D
fig3 = plt.figure(3).add_subplot(projection='3d')
axisLimit = 1.0  # km

# Plot HCW vs truth
fig3.plot(rtn_states_glm2[:, 1], rtn_states_glm2[:, 2], rtn_states_glm2[:, 0],
          'r-', alpha=0.35)
# fig3.plot(rtn_states_true[:, 1], rtn_states_true[:, 2], rtn_states_true[:, 0],
#          'b:', alpha=0.85)

# Plot a vector triad to represent chief at the origin
# fig3.quiver(0, 0, 0, 1, 0, 0, length=axisLimit / 5, color='g',
#            arrow_length_ratio=0.3)
# fig3.quiver(0, 0, 0, 0, 1, 0, length=axisLimit / 5, color='g',
#            arrow_length_ratio=0.3)
# fig3.quiver(0, 0, 0, 0, 0, 1, length=axisLimit / 5, color='g',
#            arrow_length_ratio=0.3)

# Set plotting parameters
fig3.set_title('Trajectory in RTN of GLM')
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
plt.plot(rtn_states_glm2[:, 1], rtn_states_glm2[:, 0], 'r-')
#plt.plot(rtn_states_true[:, 1], rtn_states_true[:, 0], 'b:')
plt.grid()
plt.xlabel('T component [km]')
plt.ylabel('R component [km]')
plt.axis('equal')
plt.legend(['GLM'])

# NR plane
plt.subplot(1, 3, 2)
plt.title('NR plane plot')
plt.plot(rtn_states_glm2[:, 2], rtn_states_glm2[:, 0], 'r-')
#plt.plot(rtn_states_true[:, 2], rtn_states_true[:, 0], 'b:')
plt.grid()
plt.xlabel('N component [km]')
plt.ylabel('R component [km]')
plt.axis('equal')
plt.legend(['GLM'])

# TN plane
plt.subplot(1, 3, 3)
plt.title('TN plane plot')
plt.plot(rtn_states_glm2[:, 1], rtn_states_glm2[:, 2], 'r-')
#plt.plot(rtn_states_true[:, 1], rtn_states_true[:, 2], 'b:')
plt.grid()
plt.xlabel('T component [km]')
plt.ylabel('N component [km]')
plt.axis('equal')
plt.legend(['GLM'])


plt.show()
