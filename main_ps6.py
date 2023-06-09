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
sc2.forces['drag'] = True # Enable drag effects
sc1.forces['drag'] = True # Enable drag effects

# Set the highest possible thrust output (absolute value in m/s^2).
u_max = np.array([1E-7, 1E-7, 1E-7]) 

# Set the reference set of ROEs to track.
rROE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
rROE = np.array([sc2.da, sc2.dL, sc2.ex, sc2.ey, sc2.ix, sc2.iy])

# Start the simulation here.
timeNow, duration, timestep = 0.0, 30.1 * 86400.0, 60.0 # Time in seconds
k, samples = 0, int(duration / timestep) # Sample count and total samples

# Matrix to store the data
state_history = np.zeros((samples, 6))
roe_history = np.zeros((samples, 6))
deltaV_history = np.zeros(samples)
total_deltaV = 0.0

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

# Set a timer for continuous control drift correction
drift_flag = False
drift_timer = 0    
dv_lambda = 0.0001

# In the loop, in order for the deputy to properly update its ROEs and RTN, 
# the chief needs to be propagated first...
while timeNow < duration:
    
    # Record states.
    ROE = np.array([sc2.da, sc2.dL, sc2.ex, sc2.ey, sc2.ix, sc2.iy])
    roe_history[k,:] = ROE
    state_history[k,:] = [sc2.pR, sc2.pT, sc2.pN, sc2.vR, sc2.vT, sc2.vN]
    deltaV_history[k] = total_deltaV
    
    # Compute control input. For now assume desired ROE is zero (rendezvous).
    dROE = ROE - rROE
    
    # Nominal maintenance if not correcting for Keplerian drift
    if drift_flag == False:

        # Compute the plant matrix A (time-varying for non-Keplerian case).
        A = build_A(sc1)
        
        # Compute the control matrix B.
        # B = build_B(sc1)
        B_reduced = build_reduced_B(sc1)
        
        # Build the gain matrix.
        P = build_P(100, 0.001, ROE, rROE, sc1)
        # P = 0.0001 * np.eye(6) # Uncomment out to use the dumb control version
        
        # u = -1 * pinv(B) @ ((A @ dROE) + (P @ dROE))
        dROE_reduced = np.array([dROE[0], dROE[2], dROE[3], dROE[4], dROE[5]])
        u_reduced = -1 * pinv(B_reduced) @ (P @ dROE_reduced)
        u_reduced = np.append([0], u_reduced)
        
        # Apply a maximum threshold constraint on the thrust.
        u_sign = np.sign(u_reduced)
        u_cutoff = np.minimum( np.abs(u_reduced), u_max )
        u_cutoff = u_cutoff * u_sign
        
        # Apply the control maneuver to SC2.
        sc2.set_thruster_acceleration( u_cutoff )
        
    else:
        sc2.set_thruster_acceleration( [0,0,0] )
    
    # Keplerian drift correction
    if (abs(dROE[1]) > 0.002) or (drift_flag == True):
        
        # Flip the sign of the impulse bit if the phase is negative
        if dROE[1] < 0:
            dv_lambda = dv_lambda * (-1)
        
        # Apply the DV if it has not been applied yet
        if drift_flag == False:
            print("t = ", timeNow)
            print("Starting drift correction!")
            print("dROE = ", dROE, "dv = ", dv_lambda, '\n')
            rtn2eci = np.transpose( sc2.get_hill_frame() )
            dv = rtn2eci @ np.array([0, dv_lambda, 0])
            sc2.vx = sc2.vx + dv[0]
            sc2.vy = sc2.vy + dv[1]
            sc2.vz = sc2.vz + dv[2]
            drift_timer = sc2.chief.a * dROE[1] / (3 * dv_lambda)
            total_deltaV += abs(dv_lambda)
            drift_flag = True
        
        # Apply the anti DV once drift has been corrected
        if (drift_timer <= 0) or (abs(dROE[1]) < 0.00075):
            print("t = ", timeNow)
            print("Completed drift correction!")
            print("dROE = ", dROE, '\n')
            rtn2eci = np.transpose( sc2.get_hill_frame() )
            dv = rtn2eci @ np.array([0, -1*dv_lambda, 0])
            sc2.vx = sc2.vx + dv[0]
            sc2.vy = sc2.vy + dv[1]
            sc2.vz = sc2.vz + dv[2]
            drift_timer = 0
            total_deltaV += abs(dv_lambda)
            drift_flag = False
            
        else:
            # print("Drifting... t = ", drift_timer, " flag = ", drift_flag)
            drift_timer -= timestep
    
    # Finally, the chief itself needs to be propagated (in absolute motion)
    sc1.propagate_perturbed(timestep, timestep)
    sc2.propagate_perturbed(timestep, timestep)

    # Update delta-V cost incurred, current time, and sample count.
    total_deltaV += norm( u_reduced ) * timestep
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
desiredROE12 = plt.scatter([rROE[1]], [rROE[0]], c='k', label='Reference')
plt.xlabel(r'$\delta \lambda$')
plt.ylabel(r'$\delta a$')
plt.grid()
plt.legend(handles=[desiredROE12])
# plt.hlines(0, -1.0, 1.0, colors='k')
# plt.vlines(0, -1.0, 1.0, colors='k')

plt.subplot(1, 3, 2)
plt.scatter(roe_history[:,2], roe_history[:,3], c = ctime, alpha = 0.25)
plt.show()
desiredROE34 = plt.scatter([rROE[2]], [rROE[3]], c='k', label='Reference')
plt.xlabel(r'$\delta e_x$')
plt.ylabel(r'$\delta e_y$')
plt.grid()
plt.legend(handles=[desiredROE34])
# plt.hlines(0, -1.0, 1.0, colors='k')
# plt.vlines(0, -1.0, 1.0, colors='k')

plt.subplot(1, 3, 3)
plt.scatter(roe_history[:,4], roe_history[:,5], c = ctime, alpha = 0.25)
plt.show()
desiredROE56 = plt.scatter([rROE[4]], [rROE[5]], c='k', label='Reference')
plt.xlabel(r'$\delta i_x$')
plt.ylabel(r'$\delta i_y$')
plt.grid()
plt.legend(handles=[desiredROE56])
# plt.hlines(0, -1.0, 1.0, colors='k')
# plt.vlines(0, -1.0, 1.0, colors='k')

# Plot the total DV consumption.
timeAxis = np.linspace(0, duration, samples)
plt.figure(3)
plt.plot(timeAxis, deltaV_history * 1000)
plt.title('Cumulative Delta-V Consumption over Time')
plt.xlabel('Time [seconds]')
plt.ylabel('Delta-V [m/s]')
plt.grid()
