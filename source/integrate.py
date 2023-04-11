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
##    Vanilla-flavoured Runge-Kutta (Simpson's) 4th Order Integrator         ##
##                                                                           ##
##    Written by Samuel Y. W. Low.                                           ##
##    First created 17-Dec-2021 14:36 PM (+8 GMT)                            ##
##    Last modified 10-Apr-2023 20:33 PM (-8 GMT)                            ##
##                                                                           ##
###############################################################################
###############################################################################

import numpy as np

from source import forces

def RK4( sc, dt ):
    '''Orbit propagator for one step, using Runge-Kutta 4th Order (3/8 Rule)
    
    Parameters
    ----------
    sc : spacecraft
        Spacecraft object (typically a spacecraft calls itself)
    dt : integer
        Time step size (s)
    pos : numpy.ndarray
        Inertial frame position vector (1x3) of the spacecraft (km)
    vel : numpy.ndarray
        Inertial frame velocity vector (1x3) of the spacecraft (km/s)
    Cd : float
        Drag coefficient of the spacecraft
    Ar : float
        Drag area of the spacecraft (m^2)
    Ms : float
        Mass of the spacecraft (kg)
    fJ : bool
        Flag to toggle J2 perturbation (True to toggle on)
    fD : bool
        Flag to toggle atmospheric drag (True to toggle on)

    Returns
    -------
    posf : numpy.ndarray
        Final position vector (1x3) of the spacecraft (km)
    velf : numpy.ndarray
        Final velocity vector (1x3) of the spacecraft (km/s)

    '''
    
    c = 1.0/3.0
    pos, vel = np.array(sc.states[:3]), np.array(sc.states[3:])
    
    # K1
    k1p = vel
    k1v = forces.forces( pos, vel, sc )
    
    # K2
    k2p = vel + dt * (c*k1v)
    k2v = forces.forces( pos + dt*(c*k1p), vel + dt*(c*k1v), sc )
    
    # K3
    k3p = vel + dt * (k2v-c*k1v)
    k3v = forces.forces( pos + dt*(k2p-c*k1p), vel + dt*(k2v-c*k1v), sc )
    
    # K4
    k4p = vel + dt * (k1v-k2v+k3v)
    k4v = forces.forces( pos + dt*(k1p-k2p+k3p), vel + dt*(k1v-k2v+k3v), sc )
    
    # Simpson's Rule variant to RK4 update step
    posf = pos + (dt/8) * (k1p + 3*k2p + 3*k3p + k4p)
    velf = vel + (dt/8) * (k1v + 3*k2v + 3*k3v + k4v)
    
    sc.states = list(posf) + list(velf)
    
    return sc
