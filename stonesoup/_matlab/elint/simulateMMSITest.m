function [sensorData, colours] = simulateMMSITest()

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

lonLat0 = [0; 55];
avdt = 1000;
t0 = datetime(2020,10,22);
speed_mps = [10;0];
speed_degPerSec = speed_mps./degree2metres(lonLat0);

elint.name = 'ELINT';
elint.sensor.H = [1 0 0 0; 0 0 1 0];
elint.sensor.rates.meas = 1/avdt;
elint.sensor.rates.reveal = 1/days2sec;
elint.sensor.rates.hide = 1/(2*days2sec);
elint.sensor.priorVisProb = 0.5;

elint.meas.times = t0 + seconds([0; 100]);
elint.meas.coords = lonLat0' + seconds(elint.meas.times-t0).*(speed_degPerSec');
elint.meas.semimajorSD = repmat(1000, numel(elint.meas.times), 1);
elint.meas.semiminorSD = repmat(1000, numel(elint.meas.times), 1);
elint.meas.orientation = zeros(numel(elint.meas.times), 1);

ais.name = 'AIS';
ais.sensor.H = [1 0 0 0; 0 0 1 0];
ais.sensor.R = 1e-4*eye(2);
ais.sensor.rates.meas = 1/(60*mins2sec);
ais.sensor.rates.reveal = 1/days2sec;
ais.sensor.rates.hide = 1/(2*days2sec);
ais.sensor.priorVisProb = 0.5;
ais.sensor.logMMSINullLik = log(1e-5);
ais.sensor.logpMMSIMatch = log(1.0);

ais.meas.times = t0 + seconds([200; 300]);
ais.meas.coords = lonLat0' + seconds(ais.meas.times-t0).*(speed_degPerSec');
ais.meas.mmsi = {'12345','54321'}';

sensorData = {ais, elint};

colours = [];
