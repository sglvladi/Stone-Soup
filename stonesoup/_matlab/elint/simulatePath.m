function [times, lonLat] = simulatePath(xpos0, xvel0, refLonLat, startDate,...
    durationSec, avdt)

% Simulate straight line path (in lon,lat) given local xy position and
% velocity

[lonLat0, dlonLat0] = local2geodeticPosVel(xpos0, xvel0, refLonLat);
nmeas = poissrnd(durationSec/avdt);
ts =  sort(durationSec*rand(1, nmeas));
times = startDate + seconds(ts);
lonLat = lonLat0 + ts.*dlonLat0;

%--------------------------------------------------------------------------

function [lonLat, dlonLat] = local2geodeticPosVel(xpos, xvel, refLonLat)

f = @(t)local2geodetic(xpos + t*xvel, refLonLat);
[dlonLat, lonLat] = getDApprox(f, 0);

function lonLat = local2geodetic(xy, refLonLat)

[lat, lon] = enu2geodetic(xy(1,:), xy(2,:), zeros(1,size(xy,2)),...
    refLonLat(2), refLonLat(1), 0, wgs84Ellipsoid);
lonLat = [lon; lat];
