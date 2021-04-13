function [sensorData, colours, truth] = simulateDataHarmonics

% Simulate data for main_testHarmonics

rng(1, 'twister');

days2sec = 24*3600;
refLonLat = [0; 55];
startDate = datetime(2021,01,20);
durationSec = 1*days2sec;
avdt = 1000;
prob3 = 0.5; % probability of measurement being 3GHz
avdt3 = avdt/prob3;
avdt9 = avdt/(1-prob3);
errorSDmetres = 1000;
gatesdThresh = 5;

colours = getColourDefinitions();

xpos0 = [[0; 50000] [0; -50000]];
xvel0 = [[11; -1.1] [11; 1.1]];

nextra = 3;
xpos0 = [xpos0 1000e3*rand(2,nextra) - [0; 500e3]];
xvel0 = [xvel0 zeros(2,nextra)];

ntargets = size(xpos0, 2);

% Simulate truth
for i = 1:ntargets
    [times, lonLat] = simulatePath(xpos0(:,i), xvel0(:,i), refLonLat, startDate,...
        durationSec, avdt);
    truth(i).times = times(:);
    truth(i).coords = lonLat';
    nmeas = numel(truth(i).times);
    truth(i).sensorNum = ((rand(nmeas,1) < prob3) + 1);
    for c = 1:numel(colours)
        thiscol = diff(colours(c).range)*rand + colours(c).range(1);
        % Assume colour values fixed for now
        truth(i).(colours(c).name) = repmat(thiscol, numel(truth(i).times), 1);
    end
end

col3idx = cellfun(@(x)contains(x, '3'), {colours.name});
sensorData{1} = simulateSensor('ELINT3GHz', 1, colours(col3idx), truth,...
    errorSDmetres, gatesdThresh, avdt3);
col9idx = cellfun(@(x)contains(x, '9'), {colours.name});
sensorData{2} = simulateSensor('ELINT9GHz', 1, colours(col9idx), truth,...
    errorSDmetres, gatesdThresh, avdt9);

%--------------------------------------------------------------------------

function sensordata = simulateSensor(sensorname, sensornum, colours, truth,...
    errorSDmetres, gatesdThresh, avdt)

days2sec = 24*3600;

ntargets = numel(truth);
meas = cell(1, ntargets);
for t = 1:ntargets
    thistruth = selectrows(truth(t), truth(t).sensorNum==sensornum);
    nmeas = numel(thistruth.times);
    
    % Simulate measurements for this target
    % Simulate position measurements
    truthLonLat = cat(1, thistruth.coords)';
    measLonLat = simulateLonLatMeasurements(truthLonLat, errorSDmetres, errorSDmetres, 0);
    
    % Simulate colour measurements
    meascolour = struct();
    for c = 1:numel(colours)
        colourname = colours(c).name;
        if isfield(thistruth, colourname)
            thiscoltruth = cat(1, thistruth.(colourname));
            if colours(c).isHarmonic
                thisharmonic = tocolumn(...
                    samplediscrete(exp(colours(c).harmonicLogProbs), nmeas));
                thisharmonic(1) = 1; % ensure initial measurement is not a harmonic
            else
                thisharmonic = ones(size(thiscoltruth));
            end
            meascolour.(colourname) = thiscoltruth.*thisharmonic +...
                sqrt(colours(c).measCov)*randn(size(thiscoltruth));
        end
    end
    
    % Create measurement data
	meas{t}.times = thistruth.times;
	meas{t}.coords = measLonLat';
    meas{t}.semimajorSD = repmat(errorSDmetres, nmeas, 1);
    meas{t}.semiminorSD = repmat(errorSDmetres, nmeas, 1);
    meas{t}.orientation = zeros(nmeas, 1);
    fn = fieldnames(meascolour);
    for c = 1:numel(fn)
        meas{t}.(fn{c}) = meascolour.(fn{c});
    end
end
% Merge data from targets
meas = mergerows(meas);

% Create sensor data
sensordata.name = sensorname;
sensordata.sensor.H = [1 0 0 0; 0 0 1 0];
sensordata.sensor.rates.meas = 1/avdt;
sensordata.sensor.rates.reveal = 1/days2sec;
sensordata.sensor.rates.hide = 1/(2*days2sec);
sensordata.sensor.priorVisProb = 0.5;
sensordata.sensor.gatesdThresh = gatesdThresh;
% Create measurement data
sensordata.meas = meas;
