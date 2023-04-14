function [posf, velf] = integrate( dt, pos, vel, forceFlag )
    
    % Orbit propagator for one step, using Runge-Kutta 4th Order
    % (modified with Simpson's 3/8 Rule)

    % forceflag: [ 1 0 0 ] => Two-body only
    % forceflag: [ 1 1 0 ] => Two-body + J2
    % forceflag: [ 1 1 1 ] => Two-body + J2 + drag
    
    c = 1/3;
    
    % K1
    k1p = vel;
    k1v = accel( pos, vel, forceFlag );
    
    % K2
    k2p = vel + dt * (c*k1v);
    k2v = accel( pos + dt*(c*k1p), vel + dt*(c*k1v), ...
        forceFlag );
    
    % K3
    k3p = vel + dt * (k2v-c*k1v);
    k3v = accel( pos + dt*(k2p-c*k1p), vel + dt*(k2v-c*k1v), ...
        forceFlag );
    
    % K4
    k4p = vel + dt * (k1v-k2v+k3v);
    k4v = accel( pos + dt*(k1p-k2p+k3p), vel + dt*(k1v-k2v+k3v), ...
        forceFlag );
    
    % Simpson's Rule variant to RK4 update step
    posf = pos + (dt/8) * (k1p + 3*k2p + 3*k3p + k4p); % Returns this
    velf = vel + (dt/8) * (k1v + 3*k2v + 3*k3v + k4v); % Returns this
    
end