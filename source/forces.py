# -*- coding: utf-8 -*-

###############################################################################
###############################################################################
##                                                                           ##
##      ___  _   _   __   ____  ____   __   _   _ _____                      ##
##     / _ \| | | | /  \ |  _ \| __ \ /  \ | \ | |_   _|                     ##
##    ( |_| ) |_| |/ /\ \| |_| | -/ // /\ \|  \| | | |                       ##
##     \_  /|_____| /--\ |____/|_|\_\ /--\ |_\___| |_|                       ##
##       \/                                               v 0.0              ##
##                                                                           ##
##    Computation of force vector (called in RK4 step)                       ##
##                                                                           ##
##    Written by Samuel Y. W. Low.                                           ##
##    First created 17-Dec-2021 14:36 PM (+8 GMT)                            ##
##    Last modified 10-Apr-2023 20:33 PM (-8 GMT)                            ##
##                                                                           ##
###############################################################################
###############################################################################

import numpy as np
from source import atmosphere

def forces( pos, vel, sc ):
    '''Computation of the total inertial acceleration as a 1x3 vector, from
    Earth's gravity; optionally the J2 perturbation force, and drag force via
    the US Standard Atmosphere 1976.
    
    Parameters
    ----------
    sc : numpy.ndarray
        xxx
        
    Returns
    -------
    acceleration : numpy.ndarray
        Inertial frame acceleration vector (1x3) of the spacecraft (km/s^2)
    
    '''
    
    # Retrieve all parameters from the spacecraft.
    Cd = sc.Cd
    
    # Define all constants
    RE = 6378.140     # Earth equatorial radius (km)
    GM = 398600.4418  # G * Earth Mass (km**3/s**2)
    J2 = 1.0826267e-3 # J2 constant
    
    # Get the radial distance of the satellite.
    R = np.linalg.norm( pos ) # km
    V = np.linalg.norm( vel ) # km/s
    
    # Initialise the acceleration vector.
    acceleration = np.zeros(3)
    
    # Compute the two-body gravitational force by Earth.
    if sc.forces['twobody'] == True:
        acceleration += ( -1 * GM * pos ) / ( R**3 )
    
    # Include the additional J2 acceleration vector if necessary.
    if sc.forces['j2'] == True:
        R_J2 = 1.5 * J2 * GM * ((RE**2)/(R**5))
        zRatio = (pos[2]/R)**2
        oblate_x = R_J2 * pos[0] * (5 * zRatio-1)
        oblate_y = R_J2 * pos[1] * (5 * zRatio-1)
        oblate_z = R_J2 * pos[2] * (5 * zRatio-3)
        acceleration += np.array([oblate_x, oblate_y, oblate_z])
    
    # Include the additional drag acceleration if necessary.
    if sc.forces['drag'] == True:
        areaMassRatio = sc.area / sc.mass # m**2/kg
        dragDensity = atmosphere.density( (R - RE) ) # kg/m**3
        dragAccel = 0.5 * Cd * dragDensity * areaMassRatio * ((V*1000)**2)
        
        # Include uncertainties in the ballistic coefficient up to +/- 10%
        dragAccel = dragAccel * np.random.normal(1.0, (1/10))
        acceleration -= dragAccel * ( vel / V ) / 1000
    
    # Include the addition of continuous thruster force if necessary.
    if sc.forces['maneuvers'] == True:
        # Check if the maneuver vector is expressed in RTN basis.
        if sc.force_frame == 'RTN':
            eci2rtn = sc.get_hill_frame()
            acceleration += np.transpose(eci2rtn) @ sc.thruster_acceleration
        # Check if the maneuver vector is expressed in ECI basis.
        elif sc.force_frame == 'ECI':
            acceleration += sc.thruster_acceleration
        else:
            print("Warning, unknown maneuver frame! No maneuvers applied.")
    
    # Acceleration vector is in km/s**2
    return acceleration
