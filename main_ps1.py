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

from source import spacecraft

# Define the spacecraft and its orbital elements.
orb_a = 6928.137 # km
orb_e = 0.0      # degrees
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

# Part (c): perform numerical propagation with and without J2
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

# Run a loop and propagate all three spacecraft.
while timeNow < duration:
    
    # Record the states.
    states_sc1[n,0:3] = np.array([sc1.px, sc1.py, sc1.pz])
    states_sc2[n,0:3] = np.array([sc2.px, sc2.py, sc2.pz])
    states_sc3[n,0:3] = np.array([sc3.px, sc3.py, sc3.pz])
    
    # Propagate the spacecraft.
    sc1.propagate_orbit( timestep )
    sc2.propagate_perturbed( timestep, timestep )
    sc3.propagate_perturbed( timestep, timestep )
    
    # Update time and sample count.
    timeNow += timestep
    n += 1

# Plot the results.
import matplotlib.pyplot as plt
timeAxis = np.linspace(0,duration,samples)

# Plot the differences between the RK4 propagators with/without J2
plt.figure(1)
plt.plot( timeAxis, states_sc3[:,0] - states_sc2[:,0] )
plt.plot( timeAxis, states_sc3[:,1] - states_sc2[:,1] )
plt.plot( timeAxis, states_sc3[:,2] - states_sc2[:,2] )
plt.legend(['X [km]', 'Y [km]', 'Z [km]'])
plt.xlabel('Simulation time [sec]')
plt.ylabel('Propagation Differences [km]')
plt.title('Differences between the RK4 propagators with/without J2')
plt.grid()
plt.show()

# Plot the differences between numerical and Keplerian propagator
plt.figure(2)
plt.plot( timeAxis, states_sc2[:,0] - states_sc1[:,0] )
plt.plot( timeAxis, states_sc2[:,1] - states_sc1[:,1] )
plt.plot( timeAxis, states_sc2[:,2] - states_sc1[:,2] )
plt.legend(['X [km]', 'Y [km]', 'Z [km]'])
plt.xlabel('Simulation time [sec]')
plt.ylabel('Propagation Differences [km]')
plt.title('Differences between analytical and numerical propagators')
plt.grid()
plt.show()