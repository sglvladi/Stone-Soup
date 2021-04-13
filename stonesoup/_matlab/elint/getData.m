function [sensorData, colours, truth] = getData

rng(1, 'twister');

% Get ELINT/AIS truth from meas (pos and MMSI)

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

% Get some kind of semi-realistic data set for tracker with appropriate
% ranges

d = load('Data\simulatedELINT_AISWithColour1000.mat');

% ------
maxNumTargets = inf;%10;
if maxNumTargets < inf
    [uqmmsi, cnt] = uniquewithcounts(d.elintdata.mmsi);
	[maxcnt, maxi] = sort(cnt, 'descend');
    ntargets = min(numel(maxcnt), maxNumTargets);
	mmsis = uqmmsi(maxi(1:ntargets));
	elintIdx = find(ismember(d.elintdata.mmsi, mmsis));
	aisIdx = find(ismember(str2double(d.aisdata.mmsi), mmsis));
	d.aisdata = selectrows(d.aisdata, aisIdx);
	d.elintdata = selectrows(d.elintdata, elintIdx);
end

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
ais.truth.mmsi = str2double(d.aisdata.mmsi);
ais.truth.coords = ais.meas.coords;

elintcolour = [d.elintdata.colour1 d.elintdata.colour2...
    d.elintdata.colour3 d.elintdata.colour4];

colours = getColourDefinitions();

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
    elint{i}.truth.mmsi = d.elintdata.mmsi(idx);
    elint{i}.truth.coords = d.elintdata.truth(idx,:);
end

sensorData = [{ais} elint];

t = cat(1, cellfun(@(x)x.meas.times, sensorData, 'UniformOutput', false));
[truth.times, idx] = sort(cat(1, t{:}));
sensorids = ones(numel(t{1}),1);
for i = 2:numel(sensorData)
    sensorids = [sensorids; repmat(i,numel(t{i}),1)]; %#ok<AGROW>
end
truth.sensorids = sensorids(idx);
mmsi = cat(1, cellfun(@(x)x.truth.mmsi, sensorData, 'UniformOutput', false));
truth.mmsi = cat(1, mmsi{:});
truth.mmsi = truth.mmsi(idx,:);
coords = cat(1, cellfun(@(x)x.truth.coords, sensorData, 'UniformOutput', false));
truth.coords = cat(1, coords{:});
truth.coords = truth.coords(idx,:);

%--------------------------------------------------------------------------

function [smaj, smin, orient] = getSmajSminOrientSD(elintdata, idx, errorconfidence)

% Convert sso confidence level to standard deviation
measdim = 2;
c = 1/sqrt(chi2inv(errorconfidence, measdim));
smaj = c*elintdata.semi_major(idx);
smin = c*elintdata.semi_minor(idx);
orient = elintdata.orientation(idx);
