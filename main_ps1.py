# -*- coding: utf-8 -*-

###############################################################################
###############################################################################
##                                                                           ##
##    AA279D Problem Set 1                                                   ##
##                                                                           ##
##    Written by Samuel Y. W. Low, PhD Candidate, Stanford University        ##
##    Created on Mon Apr 10 20:37:53 2023                                    ##
##                                                                           ##
###############################################################################
###############################################################################

# Part (a) and (b)

from source import spacecraft

# Define the spacecraft and its orbital elements.
orb_a = 6928.137 # km
orb_e = 0.001    # degrees
orb_i = 97.5976  # degrees
orb_w = 0.0      # degrees
orb_R = 250.662  # degrees
orb_M = 0.827    # degrees

elemList = [orb_a, orb_e, orb_i, orb_w, orb_R, orb_M]

# Define the spacecraft itself.
sc = spacecraft.Spacecraft(elements = elemList)
sc.name = 'Boaty McBoat-Face'

# Print out the state of the S/C at time = 0
# print(sc.status())

###############################################################################
###############################################################################

# Part (c) and (d): perform numerical propagation with and without J2

import numpy as np

timeNow = 0.0
duration = 86400 * 1
timestep = 30.0
samples = int(duration / timestep)
n = 0; # Sample count

# Prepare three matrices for comparison of states
states_sc1 = np.zeros((samples,3)) # Keplerian
states_sc2 = np.zeros((samples,3)) # RK4 without J2
states_sc3 = np.zeros((samples,3)) # RK4 with J2

# Propagate the different spacecraft and compare.
sc1 = spacecraft.Spacecraft(elements = elemList, name='Kep')
sc2 = spacecraft.Spacecraft(elements = elemList, name='No_J2')
sc3 = spacecraft.Spacecraft(elements = elemList, name='Yes_J2')

# Toggle the force models for each spacecraft.
sc1.forces['Earth Oblate J2'] = False
sc2.forces['Earth Oblate J2'] = False
sc3.forces['Earth Oblate J2'] = True

# # Run a loop and propagate all three spacecraft.
# while timeNow < duration:
    
#     # Record the states.
#     states_sc1[n,0:3] = np.array([sc1.px, sc1.py, sc1.pz])
#     states_sc2[n,0:3] = np.array([sc2.px, sc2.py, sc2.pz])
#     states_sc3[n,0:3] = np.array([sc3.px, sc3.py, sc3.pz])
    
#     # hill = sc1.get_hill_frame()
#     # sc1_eci = np.array([sc1.px, sc1.py, sc1.pz])
#     # sc2_eci = np.array([sc2.px, sc2.py, sc2.pz])
#     # sc2_rtn = hill @ (sc2_eci - sc1_eci)
#     # states_sc2[n,0:3] = sc2_rtn
    
#     # Propagate the spacecraft.
#     sc1.propagate_orbit( timestep )
#     sc2.propagate_perturbed( timestep, timestep )
#     sc3.propagate_perturbed( timestep, timestep )
    
#     # Update time and sample count.
#     timeNow += timestep
#     n += 1

###############################################################################
###############################################################################

# Part (e)

# Plot the results.
import matplotlib.pyplot as plt
plt.close('all')

timeAxis = np.linspace(0,duration,samples)

# # Plot the differences between the RK4 propagators with/without J2
# plt.figure(1)
# plt.plot( timeAxis, states_sc3[:,0] - states_sc2[:,0] )
# plt.plot( timeAxis, states_sc3[:,1] - states_sc2[:,1] )
# plt.plot( timeAxis, states_sc3[:,2] - states_sc2[:,2] )
# plt.legend(['X [km]', 'Y [km]', 'Z [km]'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Propagation Differences [km]')
# plt.title('Differences between the RK4 propagators with/without J2')
# plt.grid()
# plt.show()

# # Plot the differences between numerical and Keplerian propagator
# plt.figure(2)
# plt.plot( timeAxis, states_sc2[:,0] - states_sc1[:,0] )
# plt.plot( timeAxis, states_sc2[:,1] - states_sc1[:,1] )
# plt.plot( timeAxis, states_sc2[:,2] - states_sc1[:,2] )
# plt.legend(['X [km]', 'Y [km]', 'Z [km]'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Propagation Differences [km]')
# plt.title('Differences between analytical and numerical propagators (J2000)')
# plt.grid()
# plt.show()

# Run a loop and propagate only the spacecraft with J2 included.
import numpy as np
from numpy.linalg import norm as norm

array_a = np.zeros((samples,2))    # semi-major axis [km]
array_e = np.zeros((samples,2))    # eccentricity
array_i = np.zeros((samples,2))    # inclination [deg]
array_w = np.zeros((samples,2))    # arg of perigee [deg]
array_R = np.zeros((samples,2))    # right ascension [deg]
array_Rp = np.zeros((samples,1))   # predicted right ascension [deg]
array_M = np.zeros((samples,2))    # mean anomaly [deg]
array_nu = np.zeros((samples,2))   # true anomaly [deg]
array_hVec = np.zeros((samples,6)) # ang momentum vector [deg]
array_eVec = np.zeros((samples,6)) # eccentricity vector [deg]
array_U = np.zeros((samples,2))    # specific mech. energy

while timeNow < duration:
    
    # Record the osculating orbit elements
    array_a[n,0:2] =  [sc2.a, sc3.a]
    array_e[n,0:2] =  [sc2.e, sc3.e]
    array_i[n,0:2] =  [sc2.i, sc3.i]
    array_w[n,0:2] =  [sc2.w, sc3.w]
    array_R[n,0:2] =  [sc2.R, sc3.R]
    array_M[n,0:2] =  [sc2.M, sc3.M]
    array_nu[n,0:2] = [sc2.nu, sc3.nu]
    
    # Record the states for non-perturbed spacecraft (RK4)
    rVec2 = [sc2.px, sc2.py, sc2.pz]
    vVec2 = [sc2.vx, sc2.vy, sc2.vz]
    hVec2 = np.cross(rVec2, vVec2)
    eVec2 = (np.cross(vVec2,hVec2)/sc2.GM) - (rVec2/norm(rVec2))
    U2 = 0.5 * norm(vVec2)**2 - (sc2.GM / norm(rVec2))
    array_hVec[n,0:3] = hVec2
    array_eVec[n,0:3] = eVec2
    array_U[n,0] = U2
    
    # Record the states for J2-perturbed spacecraft (RK4)
    rVec3 = [sc3.px, sc3.py, sc3.pz]
    vVec3 = [sc3.vx, sc3.vy, sc3.vz]
    hVec3 = np.cross(rVec3, vVec3)
    eVec3 = (np.cross(vVec3,hVec3)/sc3.GM) - (rVec3/norm(rVec3))
    U3 = 0.5 * norm(vVec3)**2 - (sc3.GM / norm(rVec3))
    array_hVec[n,3:6] = hVec3
    array_eVec[n,3:6] = eVec3
    array_U[n,1] = U3
    
    # Predict the next RAAN for the perturbed spacecraft.
    dRAAN_dt = -1.5 * sc3.n * 0.0010826 * np.cos(sc.i) * (6378.140/sc.a)**2
    if n != 0:
        array_Rp[n] = array_Rp[n-1] + timestep * dRAAN_dt
    else:
        array_Rp[n] = array_R[n,0]
    
    # Propagate the spacecraft.
    sc2.propagate_perturbed( timestep, timestep )
    sc3.propagate_perturbed( timestep, timestep )
    
    # Update time and sample count.
    timeNow += timestep
    n += 1

# Plot everything out.

# # Semi-major axis
# plt.figure(3)
# plt.plot( timeAxis, array_a[:,0] )
# plt.plot( timeAxis, array_a[:,1] )
# plt.legend(['Unperturbed', 'J2 Perturbed'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Osculating Semi-Major Axis [km]')
# plt.title('Plots of semi-major axis with/without J2 perturbation')
# plt.grid()
# plt.show()

# # Eccentricity
# plt.figure(4)
# plt.plot( timeAxis, array_e[:,0] )
# plt.plot( timeAxis, array_e[:,1] )
# plt.legend(['Unperturbed', 'J2 Perturbed'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Osculating Eccentricity')
# plt.title('Plots of eccentricity with/without J2 perturbation')
# plt.grid()
# plt.show()

# # Inclination
# plt.figure(5)
# plt.plot( timeAxis, array_i[:,0] * 57.3 )
# plt.plot( timeAxis, array_i[:,1] * 57.3 )
# plt.legend(['Unperturbed', 'J2 Perturbed'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Osculating Inclination [deg]')
# plt.title('Plots of Inclination with/without J2 perturbation')
# plt.grid()
# plt.show()

# # Arg. of Periapsis
# plt.figure(6)
# plt.plot( timeAxis, array_w[:,0] * 57.3 )
# plt.plot( timeAxis, array_w[:,1] * 57.3 )
# plt.legend(['Unperturbed', 'J2 Perturbed'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Osculating Arg. of Periapsis [deg]')
# plt.title('Plots of Arg. of Periapsis with/without J2 perturbation')
# plt.grid()
# plt.show()

# # RAAN
# plt.figure(7)
# plt.plot( timeAxis, array_R[:,0] * 57.3 )
# plt.plot( timeAxis, array_R[:,1] * 57.3 )
# plt.legend(['Unperturbed', 'J2 Perturbed'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Right Ascension of Ascending Node [deg]')
# plt.title('Plots of RAAN with/without J2 perturbation')
# plt.grid()
# plt.show()

# # Mean Anomaly
# plt.figure(8)
# plt.plot( timeAxis, array_M[:,0] * 57.3 )
# plt.plot( timeAxis, array_M[:,1] * 57.3 )
# plt.legend(['Unperturbed', 'J2 Perturbed'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Osculating Mean Anomaly [deg]')
# plt.title('Plots of Mean Anomaly with/without J2 perturbation')
# plt.grid()
# plt.show()

# # True Anomaly
# plt.figure(9)
# plt.plot( timeAxis, array_nu[:,0] * 57.3 )
# plt.plot( timeAxis, array_nu[:,1] * 57.3 )
# plt.legend(['Unperturbed', 'J2 Perturbed'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Osculating True Anomaly [deg]')
# plt.title('Plots of True Anomaly with/without J2 perturbation')
# plt.grid()
# plt.show()

# # Specific mechanical energy
# plt.figure(10)
# plt.plot( timeAxis, array_U[:,0])
# plt.plot( timeAxis, array_U[:,1])
# plt.legend(['Unperturbed', 'J2 Perturbed'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Energy [J/kg]')
# plt.title('Plots of specific mech energy with/without J2 perturbation')
# plt.grid()
# plt.show()

# # Eccentricity vector
# plt.figure(11)

# plt.subplot(3, 1, 1)
# plt.title('Plots of eccentricity vector components with/without J2')
# plt.plot( timeAxis, array_eVec[:,0] )
# plt.plot( timeAxis, array_eVec[:,3] )
# plt.grid()
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('X component')
# plt.legend(['X (No J2)', 'X (With J2)'])

# plt.subplot(3, 1, 2)
# plt.plot( timeAxis, array_eVec[:,1] )
# plt.plot( timeAxis, array_eVec[:,4] )
# plt.grid()
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Y component')
# plt.legend(['Y (No J2)', 'Y (With J2)'])

# plt.subplot(3, 1, 3)
# plt.plot( timeAxis, array_eVec[:,2] )
# plt.plot( timeAxis, array_eVec[:,5] )
# plt.grid()
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Z component')
# plt.legend(['Z (No J2)', 'Z (With J2)'])

# plt.show()

# # Angular momentum vector
# plt.figure(12)

# plt.subplot(3, 1, 1)
# plt.title('Plots of ang. momentum vector components with/without J2')
# plt.plot( timeAxis, array_hVec[:,0] )
# plt.plot( timeAxis, array_hVec[:,3] )
# plt.grid()
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('X component')
# plt.legend(['X (No J2)', 'X (With J2)'])

# plt.subplot(3, 1, 2)
# plt.plot( timeAxis, array_hVec[:,1] )
# plt.plot( timeAxis, array_hVec[:,4] )
# plt.grid()
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Y component')
# plt.legend(['Y (No J2)', 'Y (With J2)'])

# plt.subplot(3, 1, 3)
# plt.plot( timeAxis, array_hVec[:,2] )
# plt.plot( timeAxis, array_hVec[:,5] )
# plt.grid()
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Z component')
# plt.legend(['Z (No J2)', 'Z (With J2)'])

# plt.show()


# # True argument of latitude
# plt.figure(13)
# plt.plot( timeAxis, (array_nu[:,0] + array_w[:,0]) * 57.3, '-' )
# plt.plot( timeAxis, (array_nu[:,1] + array_w[:,1]) * 57.3, '--')
# plt.legend(['Unperturbed', 'J2 Perturbed'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('True argument of latitude [deg]')
# plt.title('Plots of true arg. of latitude with/without J2 perturbation')
# plt.grid()
# plt.show()

# ###############################################################################
# ###############################################################################

# # Part (f)

# # RAAN
# plt.figure(14)
# plt.plot( timeAxis, array_R[:,0] * 57.3 )
# plt.plot( timeAxis, array_R[:,1] * 57.3 )
# plt.plot( timeAxis, array_Rp * 57.3 )
# plt.legend(['Unperturbed', 'J2 Perturbed', 'J2 Analytical'])
# plt.xlabel('Simulation time [sec]')
# plt.ylabel('Right Ascension of Ascending Node [deg]')
# plt.title('Plots of RAAN with/without J2 perturbation')
# plt.grid()
# plt.show()