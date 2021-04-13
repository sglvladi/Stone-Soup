function [sensorData, colours] = getVisData

% Get data set to test visibility stuff

rng(1,'twister');

% Test both AIS and ELINT tracking

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

d = load('Data\simulatedELINT_AISWithColour1000.mat');

[uqmmsi, tracklen] = uniquewithcounts(d.aisdata.mmsi);
[tracklen,ii] = sort(tracklen, 'descend');
uqmmsi = uqmmsi(ii);
uqmmsidbl = str2double(uqmmsi);
aismmsitouse = uqmmsi([1 2 4]);
elintmmsitouse = uqmmsidbl([1 3 4]);
d.aisdata = selectrows(d.aisdata, ismember(d.aisdata.mmsi, aismmsitouse));
d.elintdata = selectrows(d.elintdata, ismember(d.elintdata.mmsi, elintmmsitouse));
timecutoff = datetime(2017,8,10,12,0,0);
d.aisdata = selectrows(d.aisdata, ~(ismember(d.aisdata.mmsi, uqmmsi(4)) &...
    d.aisdata.times > timecutoff));
d.elintdata = selectrows(d.elintdata, ~(ismember(d.elintdata.mmsi, uqmmsidbl(4)) &...
    d.elintdata.times > timecutoff));

%------

ais.name = 'AIS';
ais.sensor.H = [1 0 0 0; 0 0 1 0];
ais.sensor.R = 1e-4*eye(2);
ais.sensor.rates.meas = 1/(60*mins2sec);
ais.sensor.rates.reveal = 1/days2sec;
ais.sensor.rates.hide = 1/(2*days2sec);
ais.sensor.priorVisProb = 0.5;
ais.sensor.logMMSINullLik = log(1e-5); % prob that a target with unknown MMSI has a particular MMSI
ais.sensor.logpMMSIMatch = log(1.0);   % p(z_mmsi = x_mmsi | x_mmsi)
ais.meas.times = d.aisdata.times;
ais.meas.coords = d.aisdata.coords(:,1:2);
ais.meas.mmsi = d.aisdata.mmsi;

elintcolour = [d.elintdata.colour1 d.elintdata.colour2...
    d.elintdata.colour3 d.elintdata.colour4];

colours(1).name = 'freq3';
colours(1).range = [3 3.1];
colours(1).isSwitch = false;
colours(1).isHarmonic = false;

colours(2).name = 'freq9';
colours(2).range = [9 9.1];
colours(2).isSwitch = false;
colours(2).isHarmonic = false;

colours(3).name = 'scanperiod3';
colours(3).range = [0.1 0.5];
colours(3).isSwitch = false;
colours(3).isHarmonic = true;
colours(3).harmonicLogProbs = log([0.9 0.09 0.01]);

colours(4).name = 'scanperiod9';
colours(4).range = [0.1 0.5];
colours(4).isSwitch = false;
colours(4).isHarmonic = true;
colours(4).harmonicLogProbs = log([0.9 0.09 0.01]);

colours(5).name = 'pri3';
colours(5).range = [1e-3 5e-3];
colours(5).isSwitch = true;
colours(5).priorSwitchProb = 0.1;
colours(5).switchRate0to1 = 1/days2sec;%switchTransProb0to1 = 1e-3; % change to rates
colours(5).switchRate1to0 = 10/days2sec;%switchTransProb1to0 = 1e-3;
colours(5).isHarmonic = false;

colours(6).name = 'pri9';
colours(6).range = [1e-3 5e-3];
colours(6).isSwitch = true;
colours(6).priorSwitchProb = 0.1;
colours(6).switchRate0to1 = 1/days2sec;%switchTransProb0to1 = 1e-3;
colours(6).switchRate1to0 = 10/days2sec;%switchTransProb1to0 = 1e-3;
colours(6).isHarmonic = false;

colours(7).name = 'pulsewidth3';
colours(7).range = [1e-6 5e-6];
colours(7).isSwitch = true;
colours(7).priorSwitchProb = 0.1;
colours(7).switchRate0to1 = 1/days2sec;%switchTransProb0to1 = 1e-3;
colours(7).switchRate1to0 = 10/days2sec;%switchTransProb1to0 = 1e-3;
colours(7).isHarmonic = false;

colours(8).name = 'pulsewidth9';
colours(8).range = [1e-6 5e-6];
colours(8).isSwitch = true;
colours(8).priorSwitchProb = 0.1;
colours(8).switchRate0to1 = 1/days2sec;%switchTransProb0to1 = 1e-3;
colours(8).switchRate1to0 = 10/days2sec;%switchTransProb1to0 = 1e-3;
colours(8).isHarmonic = false;

for i = 1:numel(colours)
    colours(i).q = 0;
    colours(i).measVar = (0.1*diff(colours(i).range)).^2;
    [colours(i).mean, colours(i).var] = fitNormalToUniform(....
        colours(i).range(1), colours(i).range(2));
end

% 1 = 3GHz, 2 = 5GHz
radarType = (rand(size(elintcolour,1),1) > 0.5) + 1;
radarNames = {'ELINT3GHz', 'ELINT9GHz'};
colourIndices = {[1 3 5 7], [2 4 6 8]}; % Which colours for each radar type
nRadarTypes = numel(colourIndices);

% Get colour data from simulated ELINT
elint = cell(1, nRadarTypes);
for i = 1:nRadarTypes
    idx = radarType==i;
    elint{i}.name = radarNames{i};
    elint{i}.sensor.H = [1 0 0 0; 0 0 1 0];
    elint{i}.sensor.rates.meas = 1/(90*mins2sec);
    elint{i}.sensor.rates.reveal = 1/days2sec;
    elint{i}.sensor.rates.hide = 1/(2*days2sec);
    elint{i}.sensor.priorVisProb = 0.5;
    elint{i}.meas.times = d.elintdata.times(idx);
    elint{i}.meas.coords = d.elintdata.coords(idx,[1 2]);
    [elint{i}.meas.semimajorSD, elint{i}.meas.semiminorSD,...
        elint{i}.meas.orientation] = getSmajSminOrientSD(d.elintdata, idx, 0.95);
    for j = 1:numel(colourIndices{i})
        ii = colourIndices{i}(j);
        fn = colours(ii).name;
        thismeas = elintcolour(idx, j);
        x0 = colours(ii).range(1);
        dx = diff(colours(ii).range);
        thismeas = x0 + dx.*thismeas;
        elint{i}.meas.(fn) = thismeas;
    end    
end

sensorData = [{ais} elint];

%--------------------------------------------------------------------------

function [smaj, smin, orient] = getSmajSminOrientSD(elintdata, idx, errorconfidence)

% Convert sso confidence level to standard deviation
measdim = 2;
c = 1/sqrt(chi2inv(errorconfidence, measdim));
smaj = c*elintdata.semi_major(idx);
smin = c*elintdata.semi_minor(idx);
orient = elintdata.orientation(idx);

