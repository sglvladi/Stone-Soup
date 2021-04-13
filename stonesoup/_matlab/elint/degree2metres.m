function d2m = degree2metres(lonLatDeg)

% d2m = degree2metres(pos)
%
% Given a position in lon,lat, calculate the degree to metres conversion
% factor for lon and lat

latDeg = lonLatDeg(2,:);
R_earth = 6371e3; % radius of earth
d2m = pi*R_earth/180*[cosd(latDeg); ones(1,numel(latDeg))];
