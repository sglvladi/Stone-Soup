function [R_lonlat, R_metres] = getLonLatR(...
    lonLat, semimajorSD, semiminorSD, orientDeg)

% Get error covariance ellipse in longitude and latitude

npts = size(lonLat,2);
R_lonlat = zeros(2,2,npts);
R_metres = zeros(2,2,npts);
for i=1:npts
    R_metres(:,:,i) = semimajorSemiminorOrient2CovarianceMetres(...
        semimajorSD(i), semiminorSD(i), orientDeg(i));
    % Linear function to convert metres error to lon/lat error
    H = diag(1./degree2metres(lonLat(:,i)));
    % Covariance under linear function
    R_lonlat(:,:,i) = H*R_metres(:,:,i)*H';
end
