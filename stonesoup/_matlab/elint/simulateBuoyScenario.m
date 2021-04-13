function [sensorData, colours] = simulateBuoyScenario(buoyOffsetLat)

rng(1, 'twister');

if ~exist('buoyOffsetLat', 'var')
    buoyOffsetLat = 0.05;
end

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

lonLat0 = [0; 55];
t0 = datetime(2020,10,22);
avdt = 1000;
ntimesteps = 30;
tEnd = t0 + seconds(avdt*(ntimesteps-1));%seconds(30e3);%
buoyOffset = [0; buoyOffsetLat];

tBuoyAIS = t0 + (tEnd-t0)/3;%seconds(9667);%
tShipAIS = t0 + 2/3*(tEnd-t0);

shipPos0 = lonLat0;
shipVelMps = [10; 0];
shipVelDeg = shipVelMps./degree2metres(shipPos0);

%buoyPos0 = shipPos0 + seconds(tEnd-t0)/3*shipVelDeg + buoyOffset;
buoyPos0 = shipPos0 + seconds(tBuoyAIS - t0)*shipVelDeg + buoyOffset;
buoyVelDeg = [0; 0];

elint_ship_times = t0:seconds(avdt):tEnd;
ais_ship_times = tShipAIS;
elint_buoy_times = NaT(1,0);
ais_buoy_times = tBuoyAIS;

elint.name = 'ELINT';
elint.sensor.H = [1 0 0 0; 0 0 1 0];
elint.sensor.rates.meas = 1/avdt;
elint.sensor.rates.reveal = 1/days2sec;
elint.sensor.rates.hide = 1/(2*days2sec);
elint.sensor.priorVisProb = 0.5;

ais.name = 'AIS';
ais.sensor.H = [1 0 0 0; 0 0 1 0];
ais.sensor.R = 1e-4*eye(2);
ais.sensor.rates.meas = 1/(60*mins2sec);
ais.sensor.rates.reveal = 1/days2sec;
ais.sensor.rates.hide = 1/(2*days2sec);
ais.sensor.priorVisProb = 0.5;
ais.sensor.logMMSINullLik = log(1e-5);
ais.sensor.logpMMSIMatch = log(1.0);

elint.meas.times = [elint_ship_times(:); elint_buoy_times(:)];
elint.meas.coords = [getCoords(elint_ship_times - t0, shipPos0, shipVelDeg)...
    getCoords(elint_buoy_times - t0, buoyPos0, buoyVelDeg)]';
elint.meas.semimajorSD = repmat(2000, numel(elint.meas.times), 1);
elint.meas.semiminorSD = repmat(2000, numel(elint.meas.times), 1);
elint.meas.orientation = zeros(numel(elint.meas.times), 1);
elint.meas.coords = simulateELINTError(elint.meas.coords',...
    elint.meas.semimajorSD, elint.meas.semiminorSD, elint.meas.orientation)';

ais.meas.times = [ais_ship_times(:); ais_buoy_times(:)];
ais.meas.coords = [getCoords(ais_ship_times - t0, shipPos0, shipVelDeg)...
    getCoords(ais_buoy_times - t0, buoyPos0, buoyVelDeg)]';
ais.meas.coords = (ais.meas.coords' + sqrtm(ais.sensor.R)*...
    randn(2, size(ais.meas.coords,1)))';
ais.meas.mmsi = [repmat({'SHIP'},numel(ais_ship_times),1);...
    repmat({'BUOY'}', numel(ais_buoy_times), 1)];

sensorData = {ais, elint};
for i = 1:numel(sensorData)
    sensorData{i}.meas = sortTimes(sensorData{i}.meas);
end
colours = [];

%--------------------------------------------------------------------------

function coords = getCoords(dt, pos0, vel)

coords = seconds(dt(:)').*vel + pos0;

function data = sortTimes(data)

[~, idx] = sort(data.times);
fn = fieldnames(data);
for i = 1:numel(fn)
    f = fn{i};
    data.(f) = data.(f)(idx,:);
end

%--------------------------------------------------------------------------

function coords = simulateELINTError(coords, smaj_sd, smin_sd, orientDeg)

nmeas = numel(smaj_sd);
for i = 1:nmeas
    R_lonlat = getLonLatR(coords(:,i), smaj_sd(i), smin_sd(i), orientDeg(i));
	coords(:,i) = coords(:,i) + sqrtm(R_lonlat)*randn(2,1);
end
