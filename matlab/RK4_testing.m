% Test propagate.

% Inertial Position X : -2304.139112 km
% Inertial Position Y : -6525.660192 km
% Inertial Position Z : 99.217674 km
% Inertial Velocity X : -0.910804 km/s
% Inertial Velocity Y : 0.435894 km/s
% Inertial Velocity Z : 7.525236 km/s

pos = [-2304.139112 -6525.660192 99.217674 ];
vel = [-0.910804     0.435894     7.525236 ];
dt = 30;

forceFlag = [1 1]; % Enable both two-body and J2

time = 0.0;
while time < 86400
    [pos, vel] = integrate( dt, pos, vel, forceFlag );
    time = time + dt; % Time step of 30.
end

disp(pos)
disp(vel)