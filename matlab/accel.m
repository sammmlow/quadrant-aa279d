function acceleration = accel( pos, vel, forceFlag )

    % forceflag: [ 1 0 ] => Two-body only
    % forceflag: [ 1 1 ] => Two-body + J2

    % Computation of the total inertial acceleration as a 1x3 vector, from
    % Earth's gravity; optionally the J2 perturbation force, and drag 
    % force via the US Standard Atmosphere 1976.
    
    % Initialize all constant parameters from the spacecraft.
    Cd = 2.2;
    RE = 6378.140;     % Earth equatorial radius (km)
    GM = 398600.4418;  % G * Earth Mass (km**3/s**2)
    J2 = 1.0826267e-3; % J2 constant
    
    % Get the radial distance of the satellite.
    R = norm( pos );   % km
    V = norm( vel );   % km/s
    
    % Initialise the acceleration vector.
    acceleration = zeros(1,3);
    
    % Compute the two-body gravitational force by Earth.
    if forceFlag(1)
        acceleration = acceleration + ( -1 * GM * pos ) / ( R^3 );
    end
    
    % Include the additional J2 acceleration vector if necessary.
    if forceFlag(2)
        R_J2 = 1.5 * J2 * GM * ((RE^2)/(R^5));
        zRatio = (pos(3) / R)^2;
        oblate_x = R_J2 * pos(1) * (5 * zRatio-1);
        oblate_y = R_J2 * pos(2) * (5 * zRatio-1);
        oblate_z = R_J2 * pos(3) * (5 * zRatio-3);
        perturbation = [oblate_x oblate_y oblate_z];
        acceleration = acceleration + perturbation;
    end
    
    % Returns acceleration vector in km/s^2
end