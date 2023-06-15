# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:43:46 2023

Sam Low and Katherine Cao
"""

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from math import sqrt, sin, asin, cos, tan, atan2, pi
from numpy.linalg import norm, inv, pinv
from scipy.linalg import expm, sinm, cosm

from source import spacecraft

##############################################################################
##############################################################################
###                                                                        ###
###               SETTING UP OF THE GPS CONSTELLATION BELOW                ###
###                                                                        ###
##############################################################################
##############################################################################

# GPS Almanac Week 204: All available PRN IDs
prns = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
         7, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ]

# GPS Almanac Week 204: Semi-Major Axis [km]
GPS_SMAX = 26599.800

# GPS Almanac Week 204: Eccentricity [unitless]
GPS_ECCN = 0.000001

# GPS Almanac Week 204: Inclination [deg]
GPS_INCL = 55.0
        
# GPS Almanac Week 204: Right Angle of the Ascending Node [deg]
GPS_RAAN = [-143.881352, -149.446614,  -84.778512,  -22.818990,  -87.385619,
            -144.357305,   35.349176,  153.734801,  -26.028092,  -84.931462,
            -141.883192,   99.225855,  -16.924009,   97.083800,  -32.683446,
             100.277002,  158.759866, -143.755546,  161.314101,  -94.320245,
            -149.686511,  -86.467123,   30.236113,   94.562051,   91.458499,
             155.031466,   33.155708,  159.555151,   35.770562,   36.473751,
             -25.294003]

# GPS Almanac Week 204: Argument of Periapsis [deg]
GPS_ARGP = [  53.765545,  -75.631986,   57.743969, -173.043165,   64.681857,
             -46.573319, -127.231250,   13.148360,  112.699728, -139.444742,
            -154.121597,   76.981738,   53.130934, -176.496584,   67.669709,
              44.494200,  -81.162593, -178.499508,  128.866153, -166.197417,
             -45.969586, -177.202756,   50.789967,   59.425199,   26.153212,
              40.480607,   96.978207,  136.728158, -151.319311,   29.645619,
            -128.456075]

# GPS Almanac Week 204: Mean Anomaly [deg]
GPS_ANOM = [ -86.960177, -144.505126, -152.714424,   13.494215,   61.374757,
             -74.233353, -120.546970,   39.999955,   59.759746,  152.572997,
              -2.914681, -167.377052,   14.453008, -163.841686,  -15.388842,
              87.047746,   37.617981,  -80.040336,  160.861859,  -37.675960,
              36.690388, -139.207914,  -70.966229, -173.403954,  142.599792,
              40.680892,  147.604237,   43.801439, -129.846039, -157.112496,
              78.628936]

# Initialize GPS satellite osculating Keplerian elements
gps01oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[ 0], GPS_ARGP[ 0], GPS_ANOM[ 0]]
gps02oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[ 1], GPS_ARGP[ 1], GPS_ANOM[ 1]]
gps03oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[ 2], GPS_ARGP[ 2], GPS_ANOM[ 2]]
gps04oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[ 3], GPS_ARGP[ 3], GPS_ANOM[ 3]]
gps05oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[ 4], GPS_ARGP[ 4], GPS_ANOM[ 4]]
gps06oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[ 5], GPS_ARGP[ 5], GPS_ANOM[ 5]]
gps07oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[ 6], GPS_ARGP[ 6], GPS_ANOM[ 6]]
gps08oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[ 7], GPS_ARGP[ 7], GPS_ANOM[ 7]]
gps09oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[ 8], GPS_ARGP[ 8], GPS_ANOM[ 8]]
gps10oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[ 9], GPS_ARGP[ 9], GPS_ANOM[ 9]]
gps11oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[10], GPS_ARGP[10], GPS_ANOM[10]]
gps12oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[11], GPS_ARGP[11], GPS_ANOM[11]]
gps13oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[12], GPS_ARGP[12], GPS_ANOM[12]]
gps14oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[13], GPS_ARGP[13], GPS_ANOM[13]]
gps15oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[14], GPS_ARGP[14], GPS_ANOM[14]]
gps16oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[15], GPS_ARGP[15], GPS_ANOM[15]]
gps17oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[16], GPS_ARGP[16], GPS_ANOM[16]]
gps18oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[17], GPS_ARGP[17], GPS_ANOM[17]]
gps19oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[18], GPS_ARGP[18], GPS_ANOM[18]]
gps20oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[19], GPS_ARGP[19], GPS_ANOM[19]]
gps21oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[20], GPS_ARGP[20], GPS_ANOM[20]]
gps22oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[21], GPS_ARGP[21], GPS_ANOM[21]]
gps23oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[22], GPS_ARGP[22], GPS_ANOM[22]]
gps24oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[23], GPS_ARGP[23], GPS_ANOM[23]]
gps25oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[24], GPS_ARGP[24], GPS_ANOM[24]]
gps26oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[25], GPS_ARGP[25], GPS_ANOM[25]]
gps27oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[26], GPS_ARGP[26], GPS_ANOM[26]]
gps28oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[27], GPS_ARGP[27], GPS_ANOM[27]]
gps29oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[28], GPS_ARGP[28], GPS_ANOM[28]]
gps30oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[29], GPS_ARGP[29], GPS_ANOM[29]]
gps31oe = [GPS_SMAX, GPS_ECCN, GPS_INCL, GPS_RAAN[30], GPS_ARGP[30], GPS_ANOM[30]]

# Generate all GPS satellites at initial conditions.
gps01 = spacecraft.Spacecraft( elements = gps01oe )
gps02 = spacecraft.Spacecraft( elements = gps02oe )
gps03 = spacecraft.Spacecraft( elements = gps03oe )
gps04 = spacecraft.Spacecraft( elements = gps04oe )
gps05 = spacecraft.Spacecraft( elements = gps05oe )
gps06 = spacecraft.Spacecraft( elements = gps06oe )
gps07 = spacecraft.Spacecraft( elements = gps07oe )
gps08 = spacecraft.Spacecraft( elements = gps08oe )
gps09 = spacecraft.Spacecraft( elements = gps09oe )
gps10 = spacecraft.Spacecraft( elements = gps10oe )
gps11 = spacecraft.Spacecraft( elements = gps11oe )
gps12 = spacecraft.Spacecraft( elements = gps12oe )
gps13 = spacecraft.Spacecraft( elements = gps13oe )
gps14 = spacecraft.Spacecraft( elements = gps14oe )
gps15 = spacecraft.Spacecraft( elements = gps15oe )
gps16 = spacecraft.Spacecraft( elements = gps16oe )
gps17 = spacecraft.Spacecraft( elements = gps17oe )
gps18 = spacecraft.Spacecraft( elements = gps18oe )
gps19 = spacecraft.Spacecraft( elements = gps19oe )
gps20 = spacecraft.Spacecraft( elements = gps20oe )
gps21 = spacecraft.Spacecraft( elements = gps21oe )
gps22 = spacecraft.Spacecraft( elements = gps22oe )
gps23 = spacecraft.Spacecraft( elements = gps23oe )
gps24 = spacecraft.Spacecraft( elements = gps24oe )
gps25 = spacecraft.Spacecraft( elements = gps25oe )
gps26 = spacecraft.Spacecraft( elements = gps26oe )
gps27 = spacecraft.Spacecraft( elements = gps27oe )
gps28 = spacecraft.Spacecraft( elements = gps28oe )
gps29 = spacecraft.Spacecraft( elements = gps29oe )
gps30 = spacecraft.Spacecraft( elements = gps30oe )
gps31 = spacecraft.Spacecraft( elements = gps31oe )

# Group all GPS satellites together into a constellation.
gps_constellation = [ gps01, gps02, gps03, gps04, gps05, gps06, gps07, gps08,
                      gps09, gps10, gps11, gps12, gps13, gps14, gps15, gps16,
                      gps17, gps18, gps19, gps21, gps22, gps23, gps24, gps25,
                      gps25, gps26, gps27, gps28, gps29, gps30, gps31 ]

##############################################################################
##############################################################################
###                                                                        ###
###              AUXILIARY FUNCTIONS FOR GPS CONSTELLATION                 ###
###                                                                        ###
##############################################################################
##############################################################################

# Function to convert ECEF (xyz) to Geodetic (lat-lon-alt) coordinates
def ecef_to_geodetic(pos):
    if len(pos) != 3:
        raise ValueError('ECEF to Geodetic: Position vector must be length 3!')
        return np.array([0,0,0])
    earthRad = 6378136.300       # Radius of Earth (WGS84)
    earthEcc = 0.081819190842622 # Earth eccentricity (WGS84)
    x, y, z, r = pos[0], pos[1], pos[2], norm(pos)
    lon = atan2(y,x)
    lat0 = asin(z/r) 
    lat = lat0 # Initial guess
    n, nMax = 0, 3 # Number of iterations
    error, tol = 1, 1E-12
    rxy = norm([x,y])
    while (error > tol) and (n < nMax):
        lat0 = lat
        N = earthRad / sqrt(1 - (earthEcc * sin(lat0))**2)
        lat = atan2(z + (N * earthEcc * earthEcc) * sin(lat0), rxy)
        error = abs(lat - lat0)
        n += 1
    alt = (rxy - cos(lat)) - N
    geodetic = np.array([lat, lon, alt])
    return geodetic

# Computes the azimuth, elevation, and range from positions 0 to 1
def compute_aer(pos0, pos1):
    lla = ecef_to_geodetic(pos0)
    lat, lon = lla[0], lla[1]
    so = sin(lon)
    co = cos(lon)
    sa = sin(lat)
    ca = cos(lat)
    rotation = np.array(
        [[-so,            co, 0.0],
         [-sa * co, -sa * so, ca ],
         [ ca * co,  ca * so, sa ]])
    enu = rotation @ (pos1 - pos0)
    azim = atan2(enu[0], enu[1]);
    elev = atan2(enu[2], norm(enu[0:2]));
    rnge = norm(enu);
    enu = np.array([azim, elev, rnge]);
    return enu

##############################################################################
##############################################################################
###                                                                        ###
###                 CONTINUOUS CONTROL FOR RELATIVE MOTION                 ###
###                                                                        ###
##############################################################################
##############################################################################

# Keplerian plant matrix for relative motion
def build_A(sc):
    A = np.zeros((6,6))
    A[1,0] = 1.5 * sc.n
    return A

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
###               NAVIGATION: COMPUTE JACOBIANS FOR FILTER                 ###
###                                                                        ###
##############################################################################
##############################################################################

# Computes the state transition matrix for absolute orbital motion
# It requires rough knowledge on the norm of the S/C pose

def build_filter_A(dt, R):
    GM = 398600.4418
    F = -(GM / (R**3)) * dt
    A = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [F, 0, 0, 1, 0, 0],
        [0, F, 0, 0, 1, 0],
        [0, 0, F, 0, 0, 1]])
    return A

# Computes the Jacobian of a single scalar range measurement
def compute_C(rho, pos_rcv, pos_gps):
    dx = pos_rcv[0] - pos_gps[0]
    dy = pos_rcv[1] - pos_gps[1]
    dz = pos_rcv[2] - pos_gps[2]
    C = np.array([ dx/rho, dy/rho, dz/rho, 0, 0, 0 ])
    return C

# Computes the Jacobian of a single scalar Doppler measurement
def compute_Cdot(pos_rcv, pos_gps, vel_rcv, vel_gps):
    dpos = pos_gps - pos_rcv
    dvel = vel_gps - vel_rcv
    rho = norm(pos_gps - pos_rcv)
    prx = -1 * dvel[0] / rho
    pry = -1 * dvel[1] / rho
    prz = -1 * dvel[2] / rho
    pvx = -1 * dpos[0] / rho
    pvy = -1 * dpos[1] / rho
    pvz = -1 * dpos[2] / rho
    Cdot = np.array([ prx, pry, prz, pvx, pvy, pvz ])
    return Cdot

# Computes the Jacobian of a single TDOA measurement
def compute_C_tdoa(pos_rcv2, pos_rcv1, pos_tgt):
    rho2 = pos_rcv2 - pos_tgt
    rho1 = pos_rcv1 - pos_tgt
    dx = rho1[0]/norm(rho1) - rho2[0]/norm(rho2)
    dy = rho1[1]/norm(rho1) - rho2[1]/norm(rho2)
    dz = rho1[2]/norm(rho1) - rho2[2]/norm(rho2)
    C_tdoa = np.array([ dx, dy, dz ])
    return C_tdoa

##############################################################################
##############################################################################
###                                                                        ###
###                SETTING UP OF THE FORMATION FLYING S/C                  ###
###                                                                        ###
##############################################################################
##############################################################################

# Initialize both chief and deputy.
sc1_elements = [6918.14, 0.00001, 97.59760, 0.000000, -109.33800, 0.00827]
sc2_elements = [6918.14, 0.00361, 97.5976, 134.88767, -108.92024, -134.76636]
sc3_elements = [6918.14, 0.00362, 98.01169, 44.88811, -109.338, -44.83482]

sc1 = spacecraft.Spacecraft( elements = sc1_elements )
sc2 = spacecraft.Spacecraft( elements = sc2_elements )
sc3 = spacecraft.Spacecraft( elements = sc3_elements )

# Set the chief of the spacecraft. Enable maneuvers for SC2.
sc2.chief = sc1 # ROEs and RTN states computed w.r.t. SC1
sc3.chief = sc1 # ROEs and RTN states computed w.r.t. SC1

# Set the masses
sc1.mass = 10.0
sc2.mass = 10.0
sc3.mass = 10.0

# Toggle forces on each spacecraft

sc1.forces['j2'] = True # Enable J2 effects
sc1.forces['drag'] = True # Enable drag effects

sc2.forces['j2'] = True # Enable J2 effects
sc2.forces['drag'] = True # Enable drag effects
sc2.forces['maneuvers'] = True # For continuous control

sc3.forces['j2'] = True # Enable J2 effects
sc3.forces['drag'] = True # Enable drag effects
sc3.forces['maneuvers'] = True # For continuous control

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

# Matrix to store the data for deputy 1 and 2
state_history_1 = np.zeros((samples+1, 6))
state_history_2 = np.zeros((samples+1, 6))
roe_history_1 = np.zeros((samples+1, 6))
roe_history_2 = np.zeros((samples+1, 6))
deltaV_history_1 = np.zeros(samples+1)
deltaV_history_2 = np.zeros(samples+1)
total_deltaV_1 = 0.0
total_deltaV_2 = 0.0

# Number of states to track within the filter
N = 21

# Initialize the target coordinates.
tgtX, tgtY, tgtZ = -2243.797, -5732.739, 1640.842
pos_tgt = np.array([ -2243.797, -5732.739, 1640.842 ])

# Setup the initial state and initial (prior) distribution
x = np.array([ sc1.px, sc1.py, sc1.pz, sc1.vx, sc1.vy, sc1.vz,
               sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz,
               sc3.px, sc3.py, sc3.pz, sc3.vx, sc3.vy, sc3.vz,
               tgtX, tgtY, tgtZ ])

COV = 1000 * np.eye(N) # Filter prior distribution covariance

# Corrupt the initial state estimate 0.01% or about 1km
x = x * np.random.normal(1.0, 0.0001, N)

# Setup the process and measurement noise
Qr, Qv, Qt = timestep/100, timestep/100000, timestep/10000

Q = np.diag([Qr,Qr,Qr,Qv,Qv,Qv,
             Qr,Qr,Qr,Qv,Qv,Qv,
             Qr,Qr,Qr,Qv,Qv,Qv,
             Qt,Qt,Qt])

R = 5E-3; # 5 meters
Rdot = 1E-7; # 5 mm/s
Rtgt = sqrt(2) * R # meters

# Matrices to record measured and true state history
xt_history = np.zeros((N, samples+1)); # Truth
x_history = np.zeros((N, samples+1));  # Filter mean
P_history = np.zeros((N, samples+1));  # Filter variance
k = 1;                               # Counter

res_pre_history = np.zeros((2, samples+1));  # Prefit residuals
res_pos_history = np.zeros((2, samples+1));  # Postfit residuals

# Set the elevation angle masking (default 5 degrees)
elevMask = np.deg2rad(5.0)

##############################################################################
##############################################################################
###                                                                        ###
###           INITIALIZE A BUNCH OF FLAGS TO GUIDE CONTROL LOGIC           ###
###                                                                        ###
##############################################################################
##############################################################################

# Flags for discrete control
flag_dep1_reconfig_1 = [False, False, False] # IP/OOP/drift
flag_dep1_reconfig_2 = [False, False, False] # IP/OOP/drift
flag_dep2_reconfig_1 = [False, False, False] # IP/OOP/drift
flag_dep2_reconfig_2 = [False, False, False] # IP/OOP/drift
reconfig_dep1_timer_1 = None
reconfig_dep1_timer_2 = None
reconfig_dep2_timer_1 = None
reconfig_dep2_timer_2 = None
dv_lambda = 0.0005 # DV for Keplerian drift compensation.

# Flags and timers for continuous control for Deputy 2
cont_drift_flag_1 = False
cont_drift_flag_2 = False
cont_drift_timer_1 = 0  
cont_drift_timer_2 = 0    

# Objects to hold maneuvers and location of maneuvers
uIP = None
uOP = None
dv = np.array([0.0,0.0,0.0])

# For reconfiguration, assume 1N thruster. For 30s timestep, 10kg S/C, 1N
# = 0.1m/s^2 => 3.0 m/s DV per time step.

# In the loop, in order for the deputy to properly update its ROEs and RTN, 
# the chief needs to be propagated first...
while timeNow < duration:
    
    # Record ground truth states.
    ROE_1 = np.array([sc2.da, sc2.dL, sc2.ex, sc2.ey, sc2.ix, sc2.iy])
    ROE_2 = np.array([sc3.da, sc3.dL, sc3.ex, sc3.ey, sc3.ix, sc3.iy])
    roe_history_1[k,:] = ROE_1
    roe_history_2[k,:] = ROE_2
    state_history_1[k,:] = [sc2.pR, sc2.pT, sc2.pN, sc2.vR, sc2.vT, sc2.vN]
    state_history_2[k,:] = [sc3.pR, sc3.pT, sc3.pN, sc3.vR, sc3.vT, sc3.vN]
    deltaV_history_1[k] = total_deltaV_1
    deltaV_history_2[k] = total_deltaV_2
    
    # Record filter states.
    x_history[:,k] = x
    P_history[:,k] = np.diagonal(COV)
    xt_history[:,k] = [ sc1.px, sc1.py, sc1.pz, sc1.vx, sc1.vy, sc1.vz,
                        sc2.px, sc2.py, sc2.pz, sc2.vx, sc2.vy, sc2.vz,
                        sc3.px, sc3.py, sc3.pz, sc3.vx, sc3.vy, sc3.vz,
                        tgtX, tgtY, tgtZ ]
    
    ##########################################################################
    ##########################################################################
    ###                                                                    ###
    ###          GROUND TRUTH PROPAGATION OF ALL SPACECRAFT + GPS          ###
    ###                                                                    ###
    ##########################################################################
    ##########################################################################
            
    # Finally, propagate (in absolute motion) the ground truth spacecraft
    sc1.propagate_perturbed(timestep, timestep)
    sc2.propagate_perturbed(timestep, timestep)
    sc3.propagate_perturbed(timestep, timestep)
    for gps in gps_constellation:
        gps.propagate_orbit(timestep)
    
    # Generate a virtual spacecraft using the filtered measurements
    sc1_estimate = spacecraft.Spacecraft( states = x[0:6] )
    sc2_estimate = spacecraft.Spacecraft( states = x[6:12] )
    sc3_estimate = spacecraft.Spacecraft( states = x[12:18] )
    
    # Compute the estimated ROEs (not the ground truth ones)
    sc2_estimate.chief = sc1_estimate
    sc3_estimate.chief = sc1_estimate
    ROE_1 = np.array([ sc2_estimate.da, sc2_estimate.dL, sc2_estimate.ex, 
                       sc2_estimate.ey, sc2_estimate.ix, sc2_estimate.iy])
    ROE_2 = np.array([ sc3_estimate.da, sc3_estimate.dL, sc3_estimate.ex, 
                       sc3_estimate.ey, sc3_estimate.ix, sc3_estimate.iy])
    
    # At this point, ROE_1 has overwritten the original ground truth ROEs and
    # so from here on the controller will be using these estimated quantities.
    
    ##########################################################################
    ##########################################################################
    ###                                                                    ###
    ###                DEPUTY 1: RECONFIGURATION TO CONFIG 2               ###
    ###                                                                    ###
    ##########################################################################
    ##########################################################################

    # On the 10th day, perform the reconfiguration to config 2.
    if (timeNow > (1 * 86400.0)):
        na = sc2.chief.n * sc2.chief.a
        dROE = ROE_1 - dep1_ROE_2
        
        # In-plane maneuver trigger
        if (flag_dep1_reconfig_1[0] == False):
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
                total_deltaV_1 += norm( dv_IP )
                flag_dep1_reconfig_1[0] = True
                
        # Out-of-plane maneuver trigger
        if (flag_dep1_reconfig_1[1] == False):
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
                total_deltaV_1 += norm( dv_OP )
                flag_dep1_reconfig_1[1] = True
                
        # Keplerian drift correction
        if (flag_dep1_reconfig_1[0] == True) and (flag_dep1_reconfig_1[1] == True):
            if (flag_dep1_reconfig_1[2] == False):
                if reconfig_dep1_timer_1 == None:
                    print("t = ", timeNow, '\n')
                    print("Starting drift correction! \n")
                    if dROE[1] < 0:
                        dv_lambda = dv_lambda * (-1)
                    rtn2eci = np.transpose( sc2.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, dv_lambda, 0])
                    sc2.vx = sc2.vx + dv[0]
                    sc2.vy = sc2.vy + dv[1]
                    sc2.vz = sc2.vz + dv[2]
                    reconfig_dep1_timer_1 = sc2.chief.a * dROE[1] / (3 * dv_lambda)
                    total_deltaV_1 += abs(dv_lambda)
                elif (abs(dROE[1]) < 0.002): # or (reconfig_dep1_timer_1 < 0):
                    print("t = ", timeNow, '\n')
                    print("Completed drift correction! \n")
                    print("dROE = ", dROE, '\n')
                    flag_dep1_reconfig_1[2] = True
                    rtn2eci = np.transpose( sc2.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, -1*dv_lambda, 0])
                    sc2.vx = sc2.vx + dv[0]
                    sc2.vy = sc2.vy + dv[1]
                    sc2.vz = sc2.vz + dv[2]
                    total_deltaV_1 += abs(dv_lambda)
                else:
                    reconfig_dep1_timer_1 -= timestep
                    
    ##########################################################################
    ##########################################################################
    ###                                                                    ###
    ###                DEPUTY 1: RECONFIGURATION TO CONFIG 3               ###
    ###                                                                    ###
    ##########################################################################
    ##########################################################################
    
    # On the 10th day, perform the reconfiguration to config 2.
    if (timeNow > (2 * 86400.0)):
        na = sc2.chief.n * sc2.chief.a
        dROE = ROE_1 - dep1_ROE_3
        
        # In-plane maneuver trigger
        if (flag_dep1_reconfig_2[0] == False):
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
                total_deltaV_1 += norm( dv_IP )
                flag_dep1_reconfig_2[0] = True
                
        # Out-of-plane maneuver trigger
        if (flag_dep1_reconfig_2[1] == False):
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
                total_deltaV_1 += norm( dv_OP )
                flag_dep1_reconfig_2[1] = True
                
        # Keplerian drift correction
        if (flag_dep1_reconfig_2[0] == True) and (flag_dep1_reconfig_2[1] == True):
            if (flag_dep1_reconfig_2[2] == False):
                if reconfig_dep1_timer_2 == None:
                    print("t = ", timeNow, '\n')
                    print("Starting drift correction! \n")
                    if dROE[1] < 0:
                        dv_lambda = dv_lambda * (-1)
                    rtn2eci = np.transpose( sc2.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, dv_lambda, 0])
                    sc2.vx = sc2.vx + dv[0]
                    sc2.vy = sc2.vy + dv[1]
                    sc2.vz = sc2.vz + dv[2]
                    reconfig_dep1_timer_2 = sc2.chief.a * dROE[1] / (3 * dv_lambda)
                    total_deltaV_1 += abs(dv_lambda)
                elif (abs(dROE[1]) < 0.001): # or (reconfig_dep1_timer_2 < 0):
                    print("t = ", timeNow, '\n')
                    print("Completed drift correction! \n")
                    print("dROE = ", dROE, '\n')
                    flag_dep1_reconfig_2[2] = True
                    rtn2eci = np.transpose( sc2.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, -1*dv_lambda, 0])
                    sc2.vx = sc2.vx + dv[0]
                    sc2.vy = sc2.vy + dv[1]
                    sc2.vz = sc2.vz + dv[2]
                    total_deltaV_1 += abs(dv_lambda)
                else:
                    reconfig_dep1_timer_2 -= timestep
    
    ##########################################################################
    ##########################################################################
    ###                                                                    ###
    ###                DEPUTY 2: RECONFIGURATION TO CONFIG 2               ###
    ###                                                                    ###
    ##########################################################################
    ##########################################################################

    # On the 10th day, perform the reconfiguration to config 2.
    if (timeNow > (1 * 86400.0)):
        na = sc3.chief.n * sc3.chief.a
        dROE = ROE_2 - dep2_ROE_2
        
        # In-plane maneuver trigger
        if (flag_dep2_reconfig_1[0] == False):
            dv = [0,0,0]
            dv[0] = -na * sqrt(norm(dROE[2:4])**2 - dROE[0]**2)
            dv[1] = -0.5 * na * dROE[0]  # Correct for SMA errors
            uIP = atan2( dv[0], 2*dv[1] ) + atan2( dROE[2], dROE[3] ) + pi
            uIP = ((uIP + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            u = sc3.M + sc3.w
            u = ((u + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            if (abs(u - uIP) < (2 * pi * timestep / sc1.T)):
                print("t = ", timeNow, '\n')
                print("Before: dROE = ", dROE, '\n')
                rtn2eci = np.transpose( sc3.get_hill_frame() )
                dv_IP = rtn2eci @ np.array([dv[0], dv[1], 0])
                sc3.vx = sc3.vx + dv_IP[0]
                sc3.vy = sc3.vy + dv_IP[1]
                sc3.vz = sc3.vz + dv_IP[2]
                print("In-plane thrust activated! \n")
                total_deltaV_2 += norm( dv_IP )
                flag_dep2_reconfig_1[0] = True
                
        # Out-of-plane maneuver trigger
        if (flag_dep2_reconfig_1[1] == False):
            dv = [0,0,0]
            dv[1] = -0.5 * na * dROE[0] # Correct for SMA errors
            dv[2] = -na * norm(dROE[4:6])
            uOP = atan2( dROE[5], dROE[4] )
            uOP = ((uOP + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            u = sc3.M + sc3.w
            u = ((u + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            if (abs(u - uOP) < (2 * pi * timestep / sc1.T)):
                print("t = ", timeNow, '\n')
                print("Before: dROE = ", dROE, '\n')
                rtn2eci = np.transpose( sc3.get_hill_frame() )
                dv_OP = rtn2eci @ np.array([0, dv[1], dv[2]])
                sc3.vx = sc3.vx + dv_OP[0]
                sc3.vy = sc3.vy + dv_OP[1]
                sc3.vz = sc3.vz + dv_OP[2]
                print("Out-of-plane thrust activated! \n")
                total_deltaV_2 += norm( dv_OP )
                flag_dep2_reconfig_1[1] = True
                
        # Keplerian drift correction
        if (flag_dep2_reconfig_1[0] == True) and (flag_dep2_reconfig_1[1] == True):
            if (flag_dep2_reconfig_1[2] == False):
                if reconfig_dep2_timer_1 == None:
                    print("t = ", timeNow, '\n')
                    print("Starting drift correction! \n")
                    if dROE[1] < 0:
                        dv_lambda = dv_lambda * (-1)
                    rtn2eci = np.transpose( sc3.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, dv_lambda, 0])
                    sc3.vx = sc3.vx + dv[0]
                    sc3.vy = sc3.vy + dv[1]
                    sc3.vz = sc3.vz + dv[2]
                    reconfig_dep2_timer_1 = sc3.chief.a * dROE[1] / (3 * dv_lambda)
                    total_deltaV_2 += abs(dv_lambda)
                elif (abs(dROE[1]) < 0.001): # or (reconfig_dep2_timer_1 < 0):
                    print("t = ", timeNow, '\n')
                    print("Completed drift correction! \n")
                    print("dROE = ", dROE, '\n')
                    flag_dep2_reconfig_1[2] = True
                    rtn2eci = np.transpose( sc3.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, -1*dv_lambda, 0])
                    sc3.vx = sc3.vx + dv[0]
                    sc3.vy = sc3.vy + dv[1]
                    sc3.vz = sc3.vz + dv[2]
                    total_deltaV_2 += abs(dv_lambda)
                else:
                    reconfig_dep2_timer_1 -= timestep
                    
    ##########################################################################
    ##########################################################################
    ###                                                                    ###
    ###                DEPUTY 2: RECONFIGURATION TO CONFIG 3               ###
    ###                                                                    ###
    ##########################################################################
    ##########################################################################
    
    # On the 10th day, perform the reconfiguration to config 2.
    if (timeNow > (2 * 86400.0)):
        na = sc3.chief.n * sc3.chief.a
        dROE = ROE_2 - dep2_ROE_3
        
        # In-plane maneuver trigger
        if (flag_dep2_reconfig_2[0] == False):
            dv = [0,0,0]
            dv[0] = -na * sqrt(norm(dROE[2:4])**2 - dROE[0]**2)
            dv[1] = -0.5 * na * dROE[0]  # Correct for SMA errors
            uIP = atan2( dv[0], 2*dv[1] ) + atan2( dROE[2], dROE[3] ) + pi
            uIP = ((uIP + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            u = sc3.M + sc3.w
            u = ((u + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            if (abs(u - uIP) < (2 * pi * timestep / sc1.T)):
                print("t = ", timeNow, '\n')
                print("Before: dROE = ", dROE, '\n')
                rtn2eci = np.transpose( sc3.get_hill_frame() )
                dv_IP = rtn2eci @ np.array([dv[0], dv[1], 0])
                sc3.vx = sc3.vx + dv_IP[0]
                sc3.vy = sc3.vy + dv_IP[1]
                sc3.vz = sc3.vz + dv_IP[2]
                print("In-plane thrust activated! \n")
                total_deltaV_2 += norm( dv_IP )
                flag_dep2_reconfig_2[0] = True
                
        # Out-of-plane maneuver trigger
        if (flag_dep2_reconfig_2[1] == False):
            dv = [0,0,0]
            dv[1] = -0.5 * na * dROE[0] # Correct for SMA errors
            dv[2] = -na * norm(dROE[4:6])
            uOP = atan2( dROE[5], dROE[4] )
            uOP = ((uOP + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            u = sc3.M + sc3.w
            u = ((u + pi) % (2*pi)) - pi # Wrap to [-pi, +pi]
            if (abs(u - uOP) < (2 * pi * timestep / sc1.T)):
                print("t = ", timeNow, '\n')
                print("Before: dROE = ", dROE, '\n')
                rtn2eci = np.transpose( sc3.get_hill_frame() )
                dv_OP = rtn2eci @ np.array([0, dv[1], dv[2]])
                sc3.vx = sc3.vx + dv_OP[0]
                sc3.vy = sc3.vy + dv_OP[1]
                sc3.vz = sc3.vz + dv_OP[2]
                print("Out-of-plane thrust activated! \n")
                total_deltaV_2 += norm( dv_OP )
                flag_dep2_reconfig_2[1] = True
                
        # Keplerian drift correction
        if (flag_dep2_reconfig_2[0] == True) and (flag_dep2_reconfig_2[1] == True):
            if (flag_dep2_reconfig_2[2] == False):
                if reconfig_dep2_timer_2 == None:
                    print("t = ", timeNow, '\n')
                    print("Starting drift correction! \n")
                    if dROE[1] < 0:
                        dv_lambda = dv_lambda * (-1)
                    rtn2eci = np.transpose( sc3.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, dv_lambda, 0])
                    sc3.vx = sc3.vx + dv[0]
                    sc3.vy = sc3.vy + dv[1]
                    sc3.vz = sc3.vz + dv[2]
                    reconfig_dep2_timer_2 = sc3.chief.a * dROE[1] / (3 * dv_lambda)
                    total_deltaV_2 += abs(dv_lambda)
                elif (abs(dROE[1]) < 0.001): # or (reconfig_dep2_timer_2 < 0):
                    print("t = ", timeNow, '\n')
                    print("Completed drift correction! \n")
                    print("dROE = ", dROE, '\n')
                    flag_dep2_reconfig_2[2] = True
                    rtn2eci = np.transpose( sc3.get_hill_frame() )
                    dv = rtn2eci @ np.array([0, -1*dv_lambda, 0])
                    sc3.vx = sc3.vx + dv[0]
                    sc3.vy = sc3.vy + dv[1]
                    sc3.vz = sc3.vz + dv[2]
                    total_deltaV_2 += abs(dv_lambda)
                else:
                    reconfig_dep2_timer_2 -= timestep
                    
    ##########################################################################
    ##########################################################################
    ###                                                                    ###
    ###            DEPUTY 1: CONTINUOUS CONTROL STATION-KEEPING            ###
    ###                                                                    ###
    ##########################################################################
    ##########################################################################
    
    if flag_dep1_reconfig_2 == [True,True,True]:
        rROE = dep1_ROE_3
    elif flag_dep1_reconfig_1 == [True,True,True]:
        rROE = dep1_ROE_2
    else:
        rROE = dep1_ROE_1
    
    dROE = ROE_1 - rROE
        
    # Nominal maintenance if not correcting for Keplerian drift
    if cont_drift_flag_1 == False:

        # Compute the plant matrix A (time-varying for non-Keplerian case).
        A = build_A(sc1)
        
        # Compute the control matrix B.
        B_reduced = build_reduced_B(sc1)
        
        # Build the gain matrix.
        P = build_P(100, 0.001, ROE_1, rROE, sc1)
        
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
    if (abs(dROE[1]) > 0.002) or (cont_drift_flag_1 == True):
        
        # Flip the sign of the impulse bit if the phase is negative
        if dROE[1] < 0:
            dv_lambda = dv_lambda * (-1)
        
        # Apply the DV if it has not been applied yet
        if cont_drift_flag_1 == False:
            print("t = ", timeNow)
            print("Starting drift correction!")
            print("dROE = ", dROE, "dv = ", dv_lambda, '\n')
            rtn2eci = np.transpose( sc2.get_hill_frame() )
            dv = rtn2eci @ np.array([0, dv_lambda, 0])
            sc2.vx = sc2.vx + dv[0]
            sc2.vy = sc2.vy + dv[1]
            sc2.vz = sc2.vz + dv[2]
            cont_drift_timer_1 = sc2.chief.a * dROE[1] / (3 * dv_lambda)
            total_deltaV_1 += abs(dv_lambda)
            cont_drift_flag_1 = True
        
        # Apply the anti DV once drift has been corrected
        if (abs(dROE[1]) < 0.001): #(cont_drift_timer_1 <= 0) or 
            print("t = ", timeNow)
            print("Completed drift correction!")
            print("dROE = ", dROE, '\n')
            rtn2eci = np.transpose( sc2.get_hill_frame() )
            dv = rtn2eci @ np.array([0, -1*dv_lambda, 0])
            sc2.vx = sc2.vx + dv[0]
            sc2.vy = sc2.vy + dv[1]
            sc2.vz = sc2.vz + dv[2]
            cont_drift_timer_1 = 0
            total_deltaV_2 += abs(dv_lambda)
            cont_drift_flag_1 = False
            
        else:
            cont_drift_timer_1 -= timestep
    
    ##########################################################################
    ##########################################################################
    ###                                                                    ###
    ###            DEPUTY 2: CONTINUOUS CONTROL STATION-KEEPING            ###
    ###                                                                    ###
    ##########################################################################
    ##########################################################################
    
    if flag_dep2_reconfig_2 == [True,True,True]:
        rROE = dep2_ROE_3
    elif flag_dep2_reconfig_1 == [True,True,True]:
        rROE = dep2_ROE_2
    else:
        rROE = dep2_ROE_1
    
    dROE = ROE_2 - rROE
        
    # Nominal maintenance if not correcting for Keplerian drift
    if cont_drift_flag_2 == False:

        # Compute the plant matrix A (time-varying for non-Keplerian case).
        A = build_A(sc1)
        
        # Compute the control matrix B.
        # B = build_B(sc1)
        B_reduced = build_reduced_B(sc1)
        
        # Build the gain matrix.
        P = build_P(100, 0.001, ROE_2, rROE, sc1)
        
        # u = -1 * pinv(B) @ ((A @ dROE) + (P @ dROE))
        dROE_reduced = np.array([dROE[0], dROE[2], dROE[3], dROE[4], dROE[5]])
        u_reduced = -1 * pinv(B_reduced) @ (P @ dROE_reduced)
        u_reduced = np.append([0], u_reduced)
        
        # Apply a maximum threshold constraint on the thrust.
        u_sign = np.sign(u_reduced)
        u_cutoff = np.minimum( np.abs(u_reduced), u_max )
        u_cutoff = u_cutoff * u_sign
        
        # Apply the control maneuver to SC3.
        sc3.set_thruster_acceleration( u_cutoff )
        
    else:
        sc3.set_thruster_acceleration( [0,0,0] )
    
    # Keplerian drift correction
    if (abs(dROE[1]) > 0.002) or (cont_drift_flag_2 == True):
        
        # Flip the sign of the impulse bit if the phase is negative
        if dROE[1] < 0:
            dv_lambda = dv_lambda * (-1)
        
        # Apply the DV if it has not been applied yet
        if cont_drift_flag_2 == False:
            print("t = ", timeNow)
            print("Starting drift correction!")
            print("dROE = ", dROE, "dv = ", dv_lambda, '\n')
            rtn2eci = np.transpose( sc3.get_hill_frame() )
            dv = rtn2eci @ np.array([0, dv_lambda, 0])
            sc3.vx = sc3.vx + dv[0]
            sc3.vy = sc3.vy + dv[1]
            sc3.vz = sc3.vz + dv[2]
            cont_drift_timer_2 = sc3.chief.a * dROE[1] / (3 * dv_lambda)
            total_deltaV_2 += abs(dv_lambda)
            cont_drift_flag_2 = True
        
        # Apply the anti DV once drift has been corrected
        if (abs(dROE[1]) < 0.001): #(cont_drift_timer_2 <= 0) or 
            print("t = ", timeNow)
            print("Completed drift correction!")
            print("dROE = ", dROE, '\n')
            rtn2eci = np.transpose( sc3.get_hill_frame() )
            dv = rtn2eci @ np.array([0, -1*dv_lambda, 0])
            sc3.vx = sc3.vx + dv[0]
            sc3.vy = sc3.vy + dv[1]
            sc3.vz = sc3.vz + dv[2]
            cont_drift_timer_2 = 0
            total_deltaV_2 += abs(dv_lambda)
            cont_drift_flag_2 = False
            
        else:
            cont_drift_timer_2 -= timestep
    
    ##########################################################################
    ##########################################################################
    ###                                                                    ###
    ###              NAVIGATION AND LOCALIZATION BLOCK BELOW               ###
    ###                                                                    ###
    ##########################################################################
    ##########################################################################
    
    # Propagate the target's ECI position by performing rotation of the Earth
    ERR = 7.2921150e-5 # EARTH INERTIAL ROTATION RATE (RAD/SEC)
    ROT = ERR * timestep 
    
    # Get the Earth rotation direction cosine matrix, with
    # small angle approximations for simplicity.
    ERRMAT = np.array([[  1.0,   ROT, 0.0 ],
                       [ -1*ROT, 1.0, 0.0 ],
                       [  0.0,   0.0, 1.0 ]])
    
    # Build the block diagonal state transition matrix here
    I3 = np.eye(3)
    Z6 = np.zeros((6,6))
    Z36 = np.zeros((3,6))
    Z63 = np.zeros((6,3))
    A1 = build_filter_A( timestep, norm(x[0:3]) )
    A2 = build_filter_A( timestep, norm(x[6:9]) )
    A3 = build_filter_A( timestep, norm(x[12:15]) )
    A4 = ERRMAT # Earth rotation in ECI
    A_filt = np.block([[A1,  Z6,  Z6,  Z63],
                       [Z6,  A2,  Z6,  Z63],
                       [Z6,  Z6,  A3,  Z63],
                       [Z36, Z36, Z36, I3 ]])
    
    # Propagate the filter with the time update here.
    x = A_filt @ x
    COV = A_filt @ COV @ np.transpose(A_filt) + Q
    
    # Corrupt target state estimate with some noise.
    x[-3:] = x[-3:] + np.random.normal(0,R,3)
    
    # Check which GPS satellites the SC can currently see. If it can
    # detect and receive range, compute a measurement update.
    for prn in prns:
        
        # Fetch GPS ephemeris and corrupt it with some noise
        pn, vn = 0.0015, 0.0000015 # units in km
        gps = gps_constellation[prn-1]
        pos_gps = np.array([gps.px, gps.py, gps.pz]) + np.random.normal(0,pn,3)
        vel_gps = np.array([gps.vx, gps.vy, gps.vz]) + np.random.normal(0,vn,3)
        
        # For SC1:
        pos_sc1 = np.array([sc1.px, sc1.py, sc1.pz])
        aer1 = compute_aer(pos_sc1, pos_gps)
        elev1 = aer1[1]
        if elev1 > elevMask:
            
            # Observed and computed pseudorange measurement
            y = norm(pos_sc1 - pos_gps) + np.random.normal(0, R)
            yc = norm(x[0:3] - pos_gps)
            
            # Compute Jacobian and Kalman gain for position update
            C = compute_C( y, pos_sc1, pos_gps )
            C = np.block([ C, np.zeros((1,N-6)) ])[0]
            CT = np.transpose(C)
            K = COV @ CT / ((C @ COV @ CT) + R)
            x = x + (K * (y-yc))
            COV = COV - (np.outer(K,C) @ COV)
            
            # Observed and computed Doppler measurement
            vel_sc1 = np.array([sc1.vx, sc1.vy, sc1.vz])
            dvel = vel_gps - vel_sc1
            los = (pos_gps - pos_sc1) / norm(pos_gps - pos_sc1)
            losc = (pos_gps - x[0:3]) / norm(pos_gps - x[0:3])
            ydot = np.dot(dvel, los)
            ycdot = np.dot(vel_gps - x[3:6], losc)
            
            # Compute Jacobian and Kalman gain for velocity update
            Cdot = compute_Cdot( x[0:3], pos_gps, x[3:6], vel_gps )
            Cdot = np.block([ Cdot, np.zeros((1,N-6))])[0]
            CdotT = np.transpose(Cdot)
            K = COV @ CdotT / ((Cdot @ COV @ CdotT) + Rdot)
            x = x + (K * (ydot-ycdot))
            COV = COV - (np.outer(K,Cdot) @ COV)
        
        # For SC2:
        pos_sc2 = np.array([sc2.px, sc2.py, sc2.pz])
        aer2 = compute_aer(pos_sc2, pos_gps)
        elev2 = aer2[1]
        if elev2 > elevMask:
            
            # Observed and computed pseudorange measurement
            y = norm(pos_sc2 - pos_gps) + np.random.normal(0, R)
            yc = norm(x[6:9] - pos_gps)
            
            # Compute Jacobian and Kalman gain for position update
            C = compute_C( y, pos_sc2, pos_gps )
            C = np.block([ np.zeros((1,6)), C, np.zeros((1,N-12)) ])[0]
            CT = np.transpose(C)
            K = COV @ CT / ((C @ COV @ CT) + R)
            x = x + (K * (y-yc))
            COV = COV - (np.outer(K,C) @ COV)
            
            # Observed and computed Doppler measurement
            vel_sc2 = np.array([sc2.vx, sc2.vy, sc2.vz])
            dvel = vel_gps - vel_sc2
            los = (pos_gps - pos_sc2) / norm(pos_gps - pos_sc2)
            losc = (pos_gps - x[6:9]) / norm(pos_gps - x[6:9])
            ydot = np.dot(dvel, los)
            ycdot = np.dot(vel_gps - x[9:12], losc)
            
            # Compute Jacobian and Kalman gain for velocity update
            Cdot = compute_Cdot( x[6:9], pos_gps, x[9:12], vel_gps )
            Cdot = np.block([ np.zeros((1,6)), Cdot, np.zeros((1,N-12)) ])[0]
            CdotT = np.transpose(Cdot)
            K = COV @ CdotT / ((Cdot @ COV @ CdotT) + Rdot)
            x = x + (K * (ydot-ycdot))
            COV = COV - (np.outer(K,Cdot) @ COV)
        
        # For SC3:
        pos_sc3 = np.array([sc3.px, sc3.py, sc3.pz])
        aer3 = compute_aer(pos_sc3, pos_gps)
        elev3 = aer3[1]
        if elev3 > elevMask:
            
            # Observed and computed pseudorange measurement
            y = norm(pos_sc3 - pos_gps) + np.random.normal(0, R)
            yc = norm(x[12:15] - pos_gps)
            
            # Compute Jacobian and Kalman gain for position update
            C = compute_C( y, pos_sc3, pos_gps )
            C = np.block([ np.zeros((1,12)), C, np.zeros((1,N-18)) ])[0]
            CT = np.transpose(C)
            K = COV @ CT / ((C @ COV @ CT) + R)
            x = x + (K * (y-yc))
            COV = COV - (np.outer(K,C) @ COV)
            
            # Observed and computed Doppler measurement
            vel_sc3 = np.array([sc3.vx, sc3.vy, sc3.vz])
            dvel = vel_gps - vel_sc3
            los = (pos_gps - pos_sc3) / norm(pos_gps - pos_sc3)
            losc = (pos_gps - x[12:15]) / norm(pos_gps - x[12:15])
            ydot = np.dot(dvel, los)
            ycdot = np.dot(vel_gps - x[15:18], losc)
            
            # Compute Jacobian and Kalman gain for velocity update
            Cdot = compute_Cdot( x[12:15], pos_gps, x[15:18], vel_gps )
            Cdot = np.block([ np.zeros((1,12)), Cdot, np.zeros((1,N-18)) ])[0]
            CdotT = np.transpose(Cdot)
            K = COV @ CdotT / ((Cdot @ COV @ CdotT) + Rdot)
            x = x + (K * (ydot-ycdot))
            COV = COV - (np.outer(K,Cdot) @ COV)
    
    # Check if the target is within the elevation mask
    pos_sc1 = np.array([sc1.px, sc1.py, sc1.pz])
    aer_tgt = compute_aer(pos_tgt, pos_sc1)
    elev_tgt = aer1[1]
    if (elev_tgt > elevMask) and ((timeNow % 86400) < 18000):
    
        # Generate actual observed TDOA measurements.
        y_tdoa_21 = norm(pos_sc2 - pos_tgt) - norm(pos_sc1 - pos_tgt) + np.random.normal(0, R)
        y_tdoa_32 = norm(pos_sc3 - pos_tgt) - norm(pos_sc2 - pos_tgt) + np.random.normal(0, R)
        
        # Get computed TDOA measurements
        state_rcv1 = x[0:3]
        state_rcv2 = x[6:9]
        state_rcv3 = x[12:15]
        state_tgt = x[18:21]
        yc_tdoa_21 = norm(state_rcv2 - state_tgt) - norm(state_rcv1 - state_tgt)
        yc_tdoa_32 = norm(state_rcv3 - state_tgt) - norm(state_rcv2 - state_tgt)
        
        # For the actual target, perform TDOA measurement update between SC1 & 2
        C_tdoa_21 = compute_C_tdoa(state_rcv2, state_rcv1, state_tgt)
        C_tdoa_21 = np.block([ np.zeros((1,18)), C_tdoa_21 ])[0]
        C_tdoa_21T = np.transpose(C_tdoa_21)
        C_tdoa_32 = compute_C_tdoa(state_rcv3, state_rcv2, state_tgt)
        C_tdoa_32 = np.block([ np.zeros((1,18)), C_tdoa_32 ])[0]
        C_tdoa_32T = np.transpose(C_tdoa_32)
        
        # For the actual target, perform TDOA measurement update between SC1 & 2
        K = COV @ C_tdoa_32T / ((C_tdoa_32 @ COV @ C_tdoa_32T) + R)
        x = x + (K * (y_tdoa_32 - yc_tdoa_32))
        COV = COV - (np.outer(K, C_tdoa_32) @ COV)
        
        # For the actual target, perform TDOA measurement update between SC2 & 3
        K = COV @ C_tdoa_32T / ((C_tdoa_32 @ COV @ C_tdoa_32T) + R)
        x = x + (K * (y_tdoa_32 - yc_tdoa_32))
        COV = COV - (np.outer(K, C_tdoa_32) @ COV)
        
        # Save the prefit residuals
        res_pre_history[0,k] = y_tdoa_21 - yc_tdoa_21
        res_pre_history[1,k] = y_tdoa_32 - yc_tdoa_32
        
        # Compute and save postfit residuals
        state_rcv1 = x[0:3]
        state_rcv2 = x[6:9]
        state_rcv3 = x[12:15]
        state_tgt = x[18:21]
        yc_tdoa_21 = norm(state_rcv2 - state_tgt) - norm(state_rcv1 - state_tgt)
        yc_tdoa_32 = norm(state_rcv3 - state_tgt) - norm(state_rcv2 - state_tgt)
        res_pos_history[0,k] = y_tdoa_21 - yc_tdoa_21
        res_pos_history[1,k] = y_tdoa_32 - yc_tdoa_32
        
    else:
        # Save zero as pre and post fit residuals
        res_pre_history[0,k] = 0
        res_pre_history[1,k] = 0
        res_pos_history[0,k] = 0
        res_pos_history[1,k] = 0
    
    ##########################################################################
    ##########################################################################
    ###                                                                    ###
    ###                   UPDATE TIME AND SAMPLE INDEX                     ###
    ###                                                                    ###
    ##########################################################################
    ##########################################################################
    
    # Update the time step and sample count.
    timeNow += timestep
    k += 1
    
##########################################################################
##########################################################################
###                                                                    ###
###                   PLOTTING OF CONTROL RESULTS                      ###
###                                                                    ###
##########################################################################
##########################################################################
    
# Plot the full trajectory below, with chief as a quiver triad.
plt.close('all')

axisLimit = 1.0 # km
ctime = np.arange(len(roe_history_1[:,1])) * (timestep / 86400)

##########################################################################
##########################################################################

# PLOTTING FOR SPACECRAFT 1

fig1 = plt.figure(1).add_subplot(projection='3d')
sc = fig1.scatter(state_history_1[:,1], state_history_1[:,2], state_history_1[:,0], s=4, c = ctime, alpha = 0.25)
fig1.quiver(0,0,0,1,0,0, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig1.quiver(0,0,0,0,1,0, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig1.quiver(0,0,0,0,0,1, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig1.set_title('Deputy 1: Trajectory in RTN Frame')
fig1.grid()
fig1.set_xlabel('T [km]')
fig1.set_ylabel('N [km]')
fig1.set_zlabel('R [km]')
plt.colorbar(sc)

# Plot the evolution of ROE plots below.
plt.figure(2)
plt.title('Deputy 1: Evolution of Quasi-Nonsingular ROEs')

plt.subplot(1, 3, 1)
plt.scatter(roe_history_1[:,1], roe_history_1[:,0], c = ctime, alpha = 0.25)
plt.show()
desiredROE12_1 = plt.scatter([dep1_ROE_1[1], dep1_ROE_2[1], dep1_ROE_3[1]], 
                             [dep1_ROE_1[0], dep1_ROE_2[0], dep1_ROE_3[0]],
                             c='r', label='References')
plt.xlabel(r'$\delta \lambda$')
plt.ylabel(r'$\delta a$')
plt.grid()
plt.legend(handles=[desiredROE12_1])

plt.subplot(1, 3, 2)
plt.scatter(roe_history_1[:,2], roe_history_1[:,3], c = ctime, alpha = 0.25)
plt.show()
desiredROE34_1 = plt.scatter([dep1_ROE_1[2], dep1_ROE_2[2], dep1_ROE_3[2]], 
                             [dep1_ROE_1[3], dep1_ROE_2[3], dep1_ROE_3[3]],
                             c='r', label='References')
plt.xlabel(r'$\delta e_x$')
plt.ylabel(r'$\delta e_y$')
plt.grid()
plt.legend(handles=[desiredROE34_1])
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.scatter(roe_history_1[:,4], roe_history_1[:,5], c = ctime, alpha = 0.25)
plt.show()
desiredROE56_1 = plt.scatter([dep1_ROE_1[4], dep1_ROE_2[4], dep1_ROE_3[4]], 
                             [dep1_ROE_1[5], dep1_ROE_2[5], dep1_ROE_3[5]],
                             c='r', label='References')
plt.xlabel(r'$\delta i_x$')
plt.ylabel(r'$\delta i_y$')
plt.grid()
plt.legend(handles=[desiredROE56_1])
plt.axis('equal')

# Plot the total DV consumption.
plt.figure(3)
timeAxis = np.linspace(0, duration, samples+1)
plt.plot(timeAxis, deltaV_history_1 * 1000)
plt.title('Deputy 1: Cumulative Delta-V Consumption')
plt.xlabel('Time [seconds]')
plt.ylabel('Delta-V [m/s]')
plt.grid()

##########################################################################
##########################################################################

# PLOTTING FOR SPACECRAFT 2

fig4 = plt.figure(4).add_subplot(projection='3d')
sc2 = fig4.scatter(state_history_2[:,1], state_history_2[:,2], state_history_2[:,0], s=4, c = ctime, alpha = 0.25)
fig4.quiver(0,0,0,1,0,0, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig4.quiver(0,0,0,0,1,0, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig4.quiver(0,0,0,0,0,1, length = axisLimit / 5, color = 'g', arrow_length_ratio = 0.3 )
fig4.set_title('Deputy 2: Trajectory in RTN Frame')
fig4.grid()
fig4.set_xlabel('T [km]')
fig4.set_ylabel('N [km]')
fig4.set_zlabel('R [km]')
plt.colorbar(sc2)

# Plot the evolution of ROE plots below.
plt.figure(5)
plt.title('Deputy 2: Evolution of Quasi-Nonsingular ROEs')

plt.subplot(1, 3, 1)
plt.scatter(roe_history_2[:,1], roe_history_2[:,0], c = ctime, alpha = 0.25)
plt.show()
desiredROE12_2 = plt.scatter([dep2_ROE_1[1], dep2_ROE_2[1], dep2_ROE_3[1]], 
                             [dep2_ROE_1[0], dep2_ROE_2[0], dep2_ROE_3[0]],
                             c='r', label='References')
plt.xlabel(r'$\delta \lambda$')
plt.ylabel(r'$\delta a$')
plt.grid()
plt.legend(handles=[desiredROE12_2])

plt.subplot(1, 3, 2)
plt.scatter(roe_history_2[:,2], roe_history_2[:,3], c = ctime, alpha = 0.25)
plt.show()
desiredROE34_2 = plt.scatter([dep2_ROE_1[2], dep2_ROE_2[2], dep2_ROE_3[2]], 
                             [dep2_ROE_1[3], dep2_ROE_2[3], dep2_ROE_3[3]],
                             c='r', label='References')
plt.xlabel(r'$\delta e_x$')
plt.ylabel(r'$\delta e_y$')
plt.grid()
plt.legend(handles=[desiredROE34_2])
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.scatter(roe_history_2[:,4], roe_history_2[:,5], c = ctime, alpha = 0.25)
plt.show()
desiredROE56_2 = plt.scatter([dep2_ROE_1[4], dep2_ROE_2[4], dep2_ROE_3[4]], 
                             [dep2_ROE_1[5], dep2_ROE_2[5], dep2_ROE_3[5]],
                             c='r', label='References')
plt.xlabel(r'$\delta i_x$')
plt.ylabel(r'$\delta i_y$')
plt.grid()
plt.legend(handles=[desiredROE56_2])
plt.axis('equal')

# Plot the total DV consumption.
plt.figure(6)
timeAxis = np.linspace(0, duration, samples+1)
plt.plot(timeAxis, deltaV_history_2 * 1000)
plt.title('Deputy 2: Cumulative Delta-V Consumption')
plt.xlabel('Time [seconds]')
plt.ylabel('Delta-V [m/s]')
plt.grid()

##########################################################################
##########################################################################
###                                                                    ###
###                  PLOTTING OF NAVIGATION RESULTS                    ###
###                                                                    ###
##########################################################################
##########################################################################

# Convert km to m
x_history  = x_history * 1000
xt_history = xt_history * 1000
P_history = P_history * 1000**2

# SC1: Position ECI
stdev = np.sqrt(P_history)

figA, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot( timeAxis, xt_history[0,:] - x_history[0,:])
ax1.fill_between( timeAxis, stdev[0,:], -stdev[0,:], alpha=0.2 )
ax1.set_ylim(-10*1000*R, 10*1000*R)
ax1.grid()
ax1.set_ylabel('X [m]')
ax1.set_title("ECI Position Estimates of SC1 [m]")
ax2.plot( timeAxis, xt_history[1,:] - x_history[1,:])
ax2.fill_between( timeAxis, stdev[1,:], -stdev[1,:], alpha=0.2 )
ax2.set_ylim(-10*1000*R, 10*1000*R)
ax2.grid()
ax2.set_ylabel('Y [m]')
ax3.plot( timeAxis, xt_history[2,:] - x_history[2,:])
ax3.fill_between( timeAxis, stdev[2,:], -stdev[2,:], alpha=0.2 )
ax3.set_ylim(-10*1000*R, 10*1000*R)
ax3.set_xlabel('Time [seconds]')
ax3.grid()
ax3.set_ylabel('Z [m]')

# SC1: Velocity ECI

figB, (ax4, ax5, ax6) = plt.subplots(3)
ax4.plot( timeAxis, xt_history[3,:] - x_history[3,:])
ax4.fill_between( timeAxis, stdev[3,:], -stdev[3,:], alpha=0.2 )
ax4.set_ylim(-100*1000*Rdot, 100*1000*Rdot)
ax4.grid()
ax4.set_ylabel('X [m/s]')
ax4.set_title("ECI Velocity Estimates of SC1 [m/s]")
ax5.plot( timeAxis, xt_history[4,:] - x_history[4,:])
ax5.fill_between( timeAxis, stdev[4,:], -stdev[4,:], alpha=0.2 )
ax5.set_ylim(-100*1000*Rdot, 100*1000*Rdot)
ax5.grid()
ax5.set_ylabel('Y [m/s]')
ax6.plot( timeAxis, xt_history[5,:] - x_history[5,:])
ax6.fill_between( timeAxis, stdev[5,:], -stdev[5,:], alpha=0.2 )
ax6.set_ylim(-100*1000*Rdot, 100*1000*Rdot)
ax6.set_xlabel('Time [seconds]')
ax6.grid()
ax6.set_ylabel('Z [m/s]')

# SC2: Position ECI

figC, (ax7, ax8, ax9) = plt.subplots(3)
ax7.plot( timeAxis, xt_history[6,:] - x_history[6,:])
ax7.fill_between( timeAxis, stdev[6,:], -stdev[6,:], alpha=0.2 )
ax7.set_ylim(-10*1000*R, 10*1000*R)
ax7.grid()
ax7.set_ylabel('X [m]')
ax7.set_title("ECI Position Estimates of SC2 [m]")
ax8.plot( timeAxis, xt_history[7,:] - x_history[7,:])
ax8.fill_between( timeAxis, stdev[7,:], -stdev[7,:], alpha=0.2 )
ax8.set_ylim(-10*1000*R, 10*1000*R)
ax8.grid()
ax8.set_ylabel('Y [m]')
ax9.plot( timeAxis, xt_history[8,:] - x_history[8,:])
ax9.fill_between( timeAxis, stdev[8,:], -stdev[8,:], alpha=0.2 )
ax9.set_ylim(-10*1000*R, 10*1000*R)
ax9.set_xlabel('Time [seconds]')
ax9.grid()
ax9.set_ylabel('Z [m]')

# SC2: Velocity ECI

figD, (ax10, ax11, ax12) = plt.subplots(3)
ax10.plot( timeAxis, xt_history[9,:] - x_history[9,:])
ax10.fill_between( timeAxis, stdev[9,:], -stdev[9,:], alpha=0.2 )
ax10.set_ylim(-100*1000*Rdot, 100*1000*Rdot)
ax10.grid()
ax10.set_ylabel('X [m/s]')
ax10.set_title("ECI Velocity Estimates of SC2 [m/s]")
ax11.plot( timeAxis, xt_history[10,:] - x_history[10,:])
ax11.fill_between( timeAxis, stdev[10,:], -stdev[10,:], alpha=0.2 )
ax11.set_ylim(-100*1000*Rdot, 100*1000*Rdot)
ax11.grid()
ax11.set_ylabel('Y [m/s]')
ax12.plot( timeAxis, xt_history[11,:] - x_history[11,:])
ax12.fill_between( timeAxis, stdev[11,:], -stdev[11,:], alpha=0.2 )
ax12.set_ylim(-100*1000*Rdot, 100*1000*Rdot)
ax12.set_xlabel('Time [seconds]')
ax12.grid()
ax12.set_ylabel('Z [m/s]')

# SC3: Position ECI

figE, (ax13, ax14, ax15) = plt.subplots(3)
ax13.plot( timeAxis, xt_history[12,:] - x_history[12,:])
ax13.fill_between( timeAxis, stdev[12,:], -stdev[12,:], alpha=0.2 )
ax13.set_ylim(-10*1000*R, 10*1000*R)
ax13.grid()
ax13.set_ylabel('X [m]')
ax13.set_title("ECI Position Estimates of SC3 [m]")
ax14.plot( timeAxis, xt_history[13,:] - x_history[13,:])
ax14.fill_between( timeAxis, stdev[13,:], -stdev[13,:], alpha=0.2 )
ax14.set_ylim(-10*1000*R, 10*1000*R)
ax14.grid()
ax14.set_ylabel('Y [m]')
ax15.plot( timeAxis, xt_history[14,:] - x_history[14,:])
ax15.fill_between( timeAxis, stdev[14,:], -stdev[14,:], alpha=0.2 )
ax15.set_ylim(-10*1000*R, 10*1000*R)
ax15.set_xlabel('Time [seconds]')
ax15.grid()
ax15.set_ylabel('Z [m]')

# SC3: Velocity ECI

figF, (ax16, ax17, ax18) = plt.subplots(3)
ax16.plot( timeAxis, xt_history[15,:] - x_history[15,:])
ax16.fill_between( timeAxis, stdev[15,:], -stdev[15,:], alpha=0.2 )
ax16.set_ylim(-100*1000*Rdot, 100*1000*Rdot)
ax16.grid()
ax16.set_ylabel('X [m/s]')
ax16.set_title("ECI Velocity Estimates of SC3 [m/s]")
ax17.plot( timeAxis, xt_history[16,:] - x_history[16,:])
ax17.fill_between( timeAxis, stdev[16,:], -stdev[16,:], alpha=0.2 )
ax17.set_ylim(-100*1000*Rdot, 100*1000*Rdot)
ax17.grid()
ax17.set_ylabel('Y [m/s]')
ax18.plot( timeAxis, xt_history[17,:] - x_history[17,:])
ax18.fill_between( timeAxis, stdev[17,:], -stdev[17,:], alpha=0.2 )
ax18.set_ylim(-100*1000*Rdot, 100*1000*Rdot)
ax18.set_xlabel('Time [seconds]')
ax18.grid()
ax18.set_ylabel('Z [m/s]')

# Target: ECI

figG, (ax19, ax20, ax21) = plt.subplots(3)
ax19.plot( timeAxis, xt_history[18,:] - x_history[18,:])
ax19.fill_between( timeAxis, stdev[18,:], -stdev[18,:], alpha=0.2 )
ax19.set_ylim(-200*1000*R, 200*1000*R)
ax19.grid()
ax19.set_ylabel('X [m]')
ax19.set_title("ECI Position Estimates of Emitter [m]")
ax20.plot( timeAxis, xt_history[19,:] - x_history[19,:])
ax20.fill_between( timeAxis, stdev[19,:], -stdev[19,:], alpha=0.2 )
ax20.set_ylim(-200*1000*R, 200*1000*R)
ax20.grid()
ax20.set_ylabel('Y [m]')
ax21.plot( timeAxis, xt_history[20,:] - x_history[20,:])
ax21.fill_between( timeAxis, stdev[20,:], -stdev[20,:], alpha=0.2 )
ax21.set_ylim(-200*1000*R, 200*1000*R)
ax21.set_xlabel('Time [seconds]')
ax21.grid()
ax21.set_ylabel('Z [m]')

# Pre and post fit residuals of TDOA

figH, (ax22, ax23) = plt.subplots(2)
ax22.scatter( timeAxis, res_pre_history[0,:], 8, alpha=0.35)
ax22.scatter( timeAxis, res_pos_history[0,:], 8, alpha=0.35)
ax22.grid()
ax22.set_ylabel('Localization Residual [m]')
ax22.set_title("Pre/Post-fit residuals for TDOA Between SC1/2")
ax23.scatter( timeAxis, res_pre_history[1,:], 8, alpha=0.35)
ax23.scatter( timeAxis, res_pos_history[1,:], 8, alpha=0.35)
ax23.grid()
ax23.set_ylabel('Localization Residual [m]')
ax23.set_xlabel('Time [seconds]')
ax23.set_title("Pre/Post-fit residuals for TDOA Between SC2/3")