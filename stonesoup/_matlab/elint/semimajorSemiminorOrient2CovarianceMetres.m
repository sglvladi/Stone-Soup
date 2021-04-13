function R = semimajorSemiminorOrient2CovarianceMetres(...
    semimajorSD, semiminorSD, orientDeg)

% R = semimajorSemiminorOrient2CovarianceMetres(...
%    semimajor_sd, semiminor_sd, orientDeg)
%
% orientDeg is orientation of semimajor axis clockwise from 12o'clock (in degrees),
% semi-axis lengths are standard deviation (in metres)
% R matrix is in metres

orientRad = deg2rad(orientDeg);
rot = [cos(orientRad) sin(orientRad); -sin(orientRad) cos(orientRad)];
Rnorm = diag([semiminorSD semimajorSD].^2);
R = rot*Rnorm*rot';
