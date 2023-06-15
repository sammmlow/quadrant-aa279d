# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:43:46 2023

Sam Low and Katherine Cao
"""

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from math import sqrt, sin, cos, tan, atan2, pi
from numpy.linalg import norm, pinv

from source import spacecraft

# Initialize both chief and deputy.
sc1_elements = [6918.14, 0.00001, 97.59760, 0.000000, -109.33800,   0.00827]
sc2_elements = [6918.14, 0.00361, 97.5976, 134.88767, -108.92024, -134.76636]
sc1 = spacecraft.Spacecraft( elements = sc1_elements )
sc2 = spacecraft.Spacecraft( elements = sc2_elements )

# Set the chief of the spacecraft. Enable maneuvers for SC2.
sc2.chief = sc1 # ROEs and RTN states computed w.r.t. SC1

# Set the masses
sc1.mass = 10.0
sc2.mass = 10.0

# Toggle forces on each spacecraft
sc2.forces['maneuvers'] = True # Important to switch this on.
sc2.forces['j2'] = True # Enable J2 effects
sc1.forces['j2'] = True # Enable J2 effects
# sc2.forces['drag'] = True # Enable drag effects
# sc1.forces['drag'] = True # Enable drag effects

# Set the highest possible thrust output (absolute value in m/s^2).
u_max = np.array([1E-7, 1E-7, 1E-7]) 

# Set the 3 ROE configurations to track for Deputy 1
dep1_ROE_1 = np.array([0.0, 0.00101, -0.00256, 0.00255, 0.0, 0.00723]) # 50km
dep1_ROE_2 = np.array([0.0, 0.00201, -0.00511, 0.00511, 0.0, 0.01445]) # 100km
dep1_ROE_3 = np.array([0.0, 0.00299, -0.00766, 0.00766, 0.0, 0.02168]) # 150km

# Set the 3 ROE configurations to track for Deputy 2
dep2_ROE_1 = np.array([0.0, 0.00079, 0.00255, 0.00255, 0.00723, 0.0]) # 50km
dep2_ROE_2 = np.array([0.0, 0.00160, 0.00511, 0.00511, 0.01445, 0.0]) # 100km
dep2_ROE_3 = np.array([0.0, 0.00243, 0.00767, 0.00767, 0.02168, 0.0]) # 150km

# Start the simulation here.
timeNow, duration, timestep = 0.0, 3 * 86400.0, 30.0 # Time in seconds
k, samples = 0, int(duration / timestep) # Sample count and total samples

# Matrix to store the data
state_history = np.zeros((samples, 6))
ephem_history = np.zeros((samples, 6))
roe_history = np.zeros((samples, 6))
deltaV_history = np.zeros(samples)
total_deltaV = 0.0

##############################################################################
##############################################################################
###                                                                        ###
###                        CONTINUOUS CONTROL STUFF                        ###
###                                                                        ###
##############################################################################
##############################################################################

# Keplerian plant matrix.
def build_A(sc):
    A = np.zeros((6,6))
    A[1,0] = 1.5 * sc1.n
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

# Reduced dimensional control input matrix (see Eq (7) in Steindorf, 2017)
def build_reduced_B(sc):
    
    ec = sc.e
    ex = ec*cos(sc.w)
    ey = ec*sin(sc.w)
    sfc = sin(sc.nu)
    cfc = cos(sc.nu)
    tic = tan(sc.i)
    swfc = sin(sc.w + sc.nu)
    cwfc = cos(sc.w + sc.nu)
    eta = sqrt(1-sc.e**2)
    B = np.zeros((5,2))
    
    B[0,0] = 2*(1+ec*cfc)/eta
    B[1,0] = eta*((2+ec*cfc)*cwfc+ex)/(1+ec*cfc)
    B[1,1] = eta*ey*swfc/(tic*(1+ec*cfc))
    B[2,0] = eta*((2+ec*cfc)*swfc+ey)/(1+ec*cfc)
    B[2,1] = -eta*ex*swfc/(tic*(1+ec*cfc))
    B[3,1] = eta*cwfc/(1+ec*cfc)
    B[4,1] = eta*swfc/(1+ec*cfc)
   
    return (1/(sc.n * sc.a)) * B

# Gain matrix P for Lyapunov control. Takes as input current ROEs and the
# reference ROEs. Negates the elements pertaining to dL (2nd ROE element).
def build_P(N, K, ROE, rROE, sc):
    dROE = ROE - rROE
    phi_ip = np.arctan2(dROE[3], dROE[2])
    phi_op = np.arctan2(dROE[5], dROE[4])
    phi = sc.M + sc.w
    J = phi - phi_ip
    H = phi - phi_op
    P = np.zeros((5,5))
    P[0,0] = cos(J)**N
    P[1,1] = cos(J)**N
    P[2,2] = cos(J)**N
    P[3,3] = cos(H)**N
    P[4,4] = cos(H)**N
    return K * P

##############################################################################
##############################################################################
###                                                                        ###
###                         DISCRETE CONTROL STUFF                         ###
###                                                                        ###
##############################################################################
##############################################################################

# Function to compute single-impulse out of plane maneuvers.
def compute_dv_op():
    return None

# Function to compute single-impulse in-plane maneuvers.
def compute_dv_ip():
    return None

##############################################################################
##############################################################################
###                                                                        ###
###                        ACTUAL SIMULATION BELOW                         ###
###                                                                        ###
##############################################################################
##############################################################################

flag_reconfig_1 = [False, False, False] # IP/OOP/drift
flag_reconfig_2 = [False, False, False] # IP/OOP/drift
reconfig_timer_1 = None
reconfig_timer_2 = None
dv_lambda = 0.0005 # DV for Keplerian drift compensation.

# Objects to hold maneuvers and location of maneuvers
uIP = None
uOP = None
dv = np.array([0.0,0.0,0.0])

# For reconfiguration, assume 1N thruster. For 30s timestep, 10kg S/C, 1N
# = 0.1m/s^2 => 3.0 m/s DV per time step.

# In the loop, in order for the deputy to properly update its ROEs and RTN, 
# the chief needs to be propagated first...
while timeNow < duration:
    
    # Record states.
    ROE = np.array([sc2.da, sc2.dL, sc2.ex, sc2.ey, sc2.ix, sc2.iy])
    roe_history[k,:] = ROE
    state_history[k,:] = [sc2.pR, sc2.pT, sc2.pN, sc2.vR, sc2.vT, sc2.vN]
    ephem_history[k,:] = [sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz]
    deltaV_history[k] = total_deltaV
    
    ##########################################################################
    ##########################################################################
    ###                                                                    ###
    ###                     RECONFIGURATION TO CONFIG 2                    ###
    ###                                                                    ###
    ##########################################################################
    ##########################################################################

    # On the 10th day, perform the reconfiguration to config 2.
    if (timeNow > (1 * 86400.0)):
        na = sc2.chief.n * sc2.chief.a
        dROE = ROE - dep1_ROE_2
        
        # In-plane maneuver trigger
        if (flag_reconfig_1[0] == False):
            dv = [0,0,0]
            dv[0] = -na * sqrt(norm(dROE[2:4])**2 - dROE[0]**2)
            dv[1] = -0.5 * na * dROE[0]  # Correct for SMA errors
            uIP = atan2( dv[0], 2*dv[1] ) + atan2( dROE[2], dROE[3] )
            uIP = ((uIP + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            u = sc2.M + sc2.w
            u = ((u + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            if (abs(u - uIP) < (2 * pi * timestep / sc1.T)):
                print("t = ", timeNow, '\n')
                print("Before: dROE = ", dROE, '\n')
                rtn2eci = np.transpose( sc2.get_hill_frame() )
                dv_IP = rtn2eci @ np.array([dv[0], dv[1], 0])
                sc2.vx = sc2.vx + dv_IP[0]
                sc2.vy = sc2.vy + dv_IP[1]
                sc2.vz = sc2.vz + dv_IP[2]
                print("In-plane thrust activated! \n")
                total_deltaV += norm( dv_IP )
                flag_reconfig_1[0] = True
                
        # Out-of-plane maneuver trigger
        if (flag_reconfig_1[1] == False):
            dv = [0,0,0]
            dv[1] = -0.5 * na * dROE[0] # Correct for SMA errors
            dv[2] = -na * norm(dROE[4:6])
            uOP = atan2( dROE[5], dROE[4] )
            uOP = ((uOP + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            u = sc2.M + sc2.w
            u = ((u + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            if (abs(u - uOP) < (2 * pi * timestep / sc1.T)):
                print("t = ", timeNow, '\n')
                print("Before: dROE = ", dROE, '\n')
                rtn2eci = np.transpose( sc2.get_hill_frame() )
                dv_OP = rtn2eci @ np.array([0, dv[1], dv[2]])
                sc2.vx = sc2.vx + dv_OP[0]
                sc2.vy = sc2.vy + dv_OP[1]
                sc2.vz = sc2.vz + dv_OP[2]
                print("Out-of-plane thrust activated! \n")
                total_deltaV += norm( dv_OP )
                flag_reconfig_1[1] = True
                
        # Keplerian drift correction
        if (flag_reconfig_1[0] == True) and (flag_reconfig_1[1] == True):
            if (flag_reconfig_1[2] == False):
                if reconfig_timer_1 == None:
                    print("t = ", timeNow, '\n')
                    print("Starting drift correction! \n")
                    if dROE[1] < 0:
                        dv_lambda = dv_lambda * (-1)
                    rtn2eci = np.transpose( sc2.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, dv_lambda, 0])
                    sc2.vx = sc2.vx + dv[0]
                    sc2.vy = sc2.vy + dv[1]
                    sc2.vz = sc2.vz + dv[2]
                    reconfig_timer_1 = sc2.chief.a * dROE[1] / (3 * dv_lambda)
                    total_deltaV += dv_lambda
                elif (reconfig_timer_1 < 0) or (abs(dROE[1]) < 0.001):
                    print("t = ", timeNow, '\n')
                    print("Completed drift correction! \n")
                    print("dROE = ", dROE, '\n')
                    flag_reconfig_1[2] = True
                    rtn2eci = np.transpose( sc2.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, -1*dv_lambda, 0])
                    sc2.vx = sc2.vx + dv[0]
                    sc2.vy = sc2.vy + dv[1]
                    sc2.vz = sc2.vz + dv[2]
                else:
                    reconfig_timer_1 -= timestep
                    
    ##########################################################################
    ##########################################################################
    ###                                                                    ###
    ###                     RECONFIGURATION TO CONFIG 3                    ###
    ###                                                                    ###
    ##########################################################################
    ##########################################################################
    
    # On the 10th day, perform the reconfiguration to config 2.
    if (timeNow > (2 * 86400.0)):
        na = sc2.chief.n * sc2.chief.a
        dROE = ROE - dep1_ROE_3
        
        # In-plane maneuver trigger
        if (flag_reconfig_2[0] == False):
            dv = [0,0,0]
            dv[0] = -na * sqrt(norm(dROE[2:4])**2 - dROE[0]**2)
            dv[1] = -0.5 * na * dROE[0]  # Correct for SMA errors
            uIP = atan2( dv[0], 2*dv[1] ) + atan2( dROE[2], dROE[3] )
            uIP = ((uIP + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            u = sc2.M + sc2.w
            u = ((u + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            if (abs(u - uIP) < (2 * pi * timestep / sc1.T)):
                print("t = ", timeNow, '\n')
                print("Before: dROE = ", dROE, '\n')
                rtn2eci = np.transpose( sc2.get_hill_frame() )
                dv_IP = rtn2eci @ np.array([dv[0], dv[1], 0])
                sc2.vx = sc2.vx + dv_IP[0]
                sc2.vy = sc2.vy + dv_IP[1]
                sc2.vz = sc2.vz + dv_IP[2]
                print("In-plane thrust activated! \n")
                total_deltaV += norm( dv_IP )
                flag_reconfig_2[0] = True
                
        # Out-of-plane maneuver trigger
        if (flag_reconfig_2[1] == False):
            dv = [0,0,0]
            dv[1] = -0.5 * na * dROE[0] # Correct for SMA errors
            dv[2] = -na * norm(dROE[4:6])
            uOP = atan2( dROE[5], dROE[4] )
            uOP = ((uOP + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            u = sc2.M + sc2.w
            u = ((u + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            if (abs(u - uOP) < (2 * pi * timestep / sc1.T)):
                print("t = ", timeNow, '\n')
                print("Before: dROE = ", dROE, '\n')
                rtn2eci = np.transpose( sc2.get_hill_frame() )
                dv_OP = rtn2eci @ np.array([0, dv[1], dv[2]])
                sc2.vx = sc2.vx + dv_OP[0]
                sc2.vy = sc2.vy + dv_OP[1]
                sc2.vz = sc2.vz + dv_OP[2]
                print("Out-of-plane thrust activated! \n")
                total_deltaV += norm( dv_OP )
                flag_reconfig_2[1] = True
                
        # Keplerian drift correction
        if (flag_reconfig_2[0] == True) and (flag_reconfig_2[1] == True):
            if (flag_reconfig_2[2] == False):
                if reconfig_timer_2 == None:
                    print("t = ", timeNow, '\n')
                    print("Starting drift correction! \n")
                    if dROE[1] < 0:
                        dv_lambda = dv_lambda * (-1)
                    rtn2eci = np.transpose( sc2.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, dv_lambda, 0])
                    sc2.vx = sc2.vx + dv[0]
                    sc2.vy = sc2.vy + dv[1]
                    sc2.vz = sc2.vz + dv[2]
                    reconfig_timer_2 = sc2.chief.a * dROE[1] / (3 * dv_lambda)
                    total_deltaV += dv_lambda
                elif (reconfig_timer_2 < 0) or (abs(dROE[1]) < 0.001):
                    print("t = ", timeNow, '\n')
                    print("Completed drift correction! \n")
                    print("dROE = ", dROE, '\n')
                    flag_reconfig_2[2] = True
                    rtn2eci = np.transpose( sc2.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, -1*dv_lambda, 0])
                    sc2.vx = sc2.vx + dv[0]
                    sc2.vy = sc2.vy + dv[1]
                    sc2.vz = sc2.vz + dv[2]
                else:
                    reconfig_timer_2 -= timestep
            
    # Finally, propagate (in absolute motion) the ground truth spacecraft
    sc1.propagate_perturbed(timestep, timestep)
    sc2.propagate_perturbed(timestep, timestep)
    
    # Update the time step and sample count.
    timeNow += timestep
    k += 1
    
# Plot the full trajectory below, with chief as a quiver triad.
plt.close('all')
axisLimit = 1.0 # km
ctime = np.arange(len(roe_history[:,1])) * (timestep / 86400)

fig1 = plt.figure(1).add_subplot(projection='3d')
sc = fig1.scatter(state_history[:,1], state_history[:,2], state_history[:,0], s=4, c = ctime, alpha = 0.25)
fig1.quiver(0,0,0,1,0,0, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig1.quiver(0,0,0,0,1,0, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig1.quiver(0,0,0,0,0,1, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig1.set_title('Trajectory in RTN Frame')
fig1.grid()
fig1.set_xlabel('T [km]')
fig1.set_ylabel('N [km]')
fig1.set_zlabel('R [km]')
plt.colorbar(sc)

# Plot the evolution of ROE plots below.
plt.figure(2)
plt.title('Evolution of Quasi-Nonsingular ROEs')

plt.subplot(1, 3, 1)
plt.scatter(roe_history[:,1], roe_history[:,0], c = ctime, alpha = 0.25)
plt.show()
desiredROE12 = plt.scatter([dep1_ROE_1[1], dep1_ROE_2[1], dep1_ROE_3[1]], 
                           [dep1_ROE_1[0], dep1_ROE_2[0], dep1_ROE_3[0]],
                           c='r', label='References')
plt.xlabel(r'$\delta \lambda$')
plt.ylabel(r'$\delta a$')
plt.grid()
plt.legend(handles=[desiredROE12])

plt.subplot(1, 3, 2)
plt.scatter(roe_history[:,2], roe_history[:,3], c = ctime, alpha = 0.25)
plt.show()
desiredROE34 = plt.scatter([dep1_ROE_1[2], dep1_ROE_2[2], dep1_ROE_3[2]], 
                           [dep1_ROE_1[3], dep1_ROE_2[3], dep1_ROE_3[3]],
                           c='r', label='References')
plt.xlabel(r'$\delta e_x$')
plt.ylabel(r'$\delta e_y$')
plt.grid()
plt.legend(handles=[desiredROE34])
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.scatter(roe_history[:,4], roe_history[:,5], c = ctime, alpha = 0.25)
plt.show()
desiredROE56 = plt.scatter([dep1_ROE_1[4], dep1_ROE_2[4], dep1_ROE_3[4]], 
                           [dep1_ROE_1[5], dep1_ROE_2[5], dep1_ROE_3[5]],
                           c='r', label='References')
plt.xlabel(r'$\delta i_x$')
plt.ylabel(r'$\delta i_y$')
plt.grid()
plt.legend(handles=[desiredROE56])
plt.axis('equal')

# Plot the total DV consumption.
plt.figure(3)
timeAxis = np.linspace(0, duration, samples)
plt.plot(timeAxis, deltaV_history * 1000)
plt.title('Cumulative Delta-V Consumption over Time')
plt.xlabel('Time [seconds]')
plt.ylabel('Delta-V [m/s]')
plt.grid()

# Plot the absolute orbit of the deputy spacecraft
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
ERx = 6371 * np.outer(np.cos(u), np.sin(v))
ERy = 6371 * np.outer(np.sin(u), np.sin(v))
ERz = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(ERx, ERy, ERz, color='k', alpha=0.4)
plt.show()
ax.plot(ephem_history[:,0], ephem_history[:,1], ephem_history[:,2])
plt.show()