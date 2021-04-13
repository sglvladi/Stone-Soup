function [sensorData, colours, truth] = simulateDataSwitch

% Under construction

rng(1, 'twister');

hours2sec = 3600;
days2sec = 24*hours2sec;

refLonLat = [0; 55];
startDate = datetime(2021,01,20);
durationSec = 1*days2sec;
avdt = 1000;
errorSDmetres = 1000;
gatesdThresh = 5;

switchDurSec = 6*hours2sec;
switchStart = startDate + seconds(durationSec)/2 - seconds(switchDurSec)/2;
switchEnd = switchStart + seconds(switchDurSec);

colours = getColourDefinitions();
colourToSwitch = colours(5).name;
targetToSwitch = 1;

xpos0 = [[0; 50000] [0; -50000]];
xvel0 = [[11; -1.1] [11; 1.1]];

nextra = 3;
xpos0 = [xpos0 1000e3*rand(2,nextra) - [0; 500e3]];
xvel0 = [xvel0 zeros(2,nextra)];

ntargets = size(xpos0, 2);

% Simulate truth
for t = 1:ntargets
    [times, lonLat] = simulatePath(xpos0(:,t), xvel0(:,t), refLonLat, startDate,...
        durationSec, avdt);
    truth(t).times = times(:);
    truth(t).coords = lonLat';
    nmeas = numel(truth(t).times);
    for c = 1:numel(colours)
        colourname = colours(c).name;
        thiscol = sampleColourInRange(colours(c));        
        if colours(c).isSwitch
            truth(t).(colourname) = [thiscol; nan(nmeas-1,1)];
            if isequal(colours(c).name, colourToSwitch) && targetToSwitch==t            
                idx = (truth(t).times > switchStart) & (truth(t).times < switchEnd);
            else
                idx = false(size(truth(t).times));
            end
            for k = 2:nmeas
                if idx(k)
                    truth(t).(colourname)(k) = sampleColourInRange(colours(c));
                else
                    % Assume non-switching colour values fixed for now
                    truth(t).(colourname)(k) = truth(t).(colourname)(k-1);
                end
            end
            truth(t).([colourname 'switch']) = idx(:);
        else
            % Assume colour values fixed for now
            truth(t).(colourname) = repmat(thiscol, nmeas, 1);
        end
    end
end

% Simulate position measurements
truthLonLat = cat(1, truth.coords)';
measLonLat = simulateLonLatMeasurements(truthLonLat, errorSDmetres, errorSDmetres, 0);
% Sort them into time order
[meastimes, idx] = sort(cat(1, truth.times));
measLonLat = measLonLat(:,idx);

% Simulate colour measurements
meascolour = struct();
for c = 1:numel(colours)
    thisname = colours(c).name;
    if isfield(truth, thisname)
        thismeas = cat(1, truth.(thisname));
        %if colours(c).isHarmonic
        %    thisharmonic = cat(1, truth.([colours(c).name 'harmonic']));
        %else
            thisharmonic = ones(size(thismeas));
        %end
        meascolour.(thisname) = thismeas(idx).*thisharmonic(idx) +...
        	sqrt(colours(c).measCov)*randn(size(thismeas));
    end
end

% Create sensor data
elint.name = 'ELINT';
elint.sensor.H = [1 0 0 0; 0 0 1 0];
elint.sensor.rates.meas = 1/avdt;
elint.sensor.rates.reveal = 1/days2sec;
elint.sensor.rates.hide = 1/(2*days2sec);
elint.sensor.priorVisProb = 0.5;
elint.sensor.gatesdThresh = gatesdThresh;
% Create measurement data
elint.meas.times = meastimes(:);
elint.meas.coords = measLonLat';
elint.meas.semimajorSD = repmat(errorSDmetres, numel(elint.meas.times), 1);
elint.meas.semiminorSD = repmat(errorSDmetres, numel(elint.meas.times), 1);
elint.meas.orientation = zeros(numel(elint.meas.times), 1);
fn = fieldnames(meascolour);
for c = 1:numel(fn)
    elint.meas.(fn{c}) = meascolour.(fn{c});
end

[elint3, elint9] = splitElintSensors(elint);
sensorData = {elint3, elint9};

%--------------------------------------------------------------------------

function col = sampleColourInRange(colour)

col = diff(colour.range)*rand + colour.range(1);
