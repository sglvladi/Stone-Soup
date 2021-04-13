function [sensorData, colours, truth] = simulateData2Targets

rng(1, 'twister');

days2sec = 24*3600;
lonLat0 = [0; 55];

endtime = 1*days2sec;
avdt = 1000;
errorSDmetres = 1000;

colours = getColourDefinitions();
colours = colours(1:2:end); % 3GHz colours

xpos0 = [0 50000; 0 -50000]';
xvel0 = [10 -1; 10 1]';
ntargets = size(xpos0, 2);
%truth_freq3 = [3 3.1];

% Simulate truth
for i = 1:ntargets
    nmeas = poissrnd(endtime/avdt);
    ts =  sort(endtime*rand(nmeas, 1));
    truth(i).times = datetime(2020,10,22) + seconds(ts);
    truth(i).xyCoords = (xpos0(:,i) + (ts').*xvel0(:,i))';
    truth(i).coords = local2geodetic(truth(i).xyCoords', lonLat0)';
    for c = 1:numel(colours)
        thiscol = diff(colours(c).range)*rand + colours(c).range(1);
        % Assume colour values fixed for now
        truth(i).(colours(c).name) = repmat(thiscol, numel(truth(i).times), 1);
    end
    % truth(i).freq3 = repmat(truth_freq3(i), numel(truth(i).times), 1);
end

% Simulate measurements
[meastimes, idx] = sort(cat(1, truth.times));
measxy = cat(1, truth.xyCoords);
measxy = measxy + errorSDmetres*randn(size(measxy));
measlonlat = local2geodetic(measxy(idx,:)', lonLat0)';
meascolour = struct();
for c = 1:numel(colours)
    thisname = colours(c).name;
    if isfield(truth, thisname)
        thismeas = cat(1, truth.(thisname));
        meascolour.(thisname) = thismeas(idx) +...
            sqrt(colours(c).measCov)*randn(size(thismeas));
    end
end
%measfreq3 = cat(1, truth.freq3);
%measfreq3 = measfreq3(idx,:) + sqrt(colours(1).measCov)*randn(size(measfreq3));

elint.name = 'ELINT';
elint.sensor.H = [1 0 0 0; 0 0 1 0];
elint.sensor.rates.meas = 1/avdt;
elint.sensor.rates.reveal = 1/days2sec;
elint.sensor.rates.hide = 1/(2*days2sec);
elint.sensor.priorVisProb = 0.5;

elint.meas.times = meastimes;
elint.meas.coords = measlonlat;
elint.meas.semimajorSD = repmat(errorSDmetres, numel(elint.meas.times), 1);
elint.meas.semiminorSD = repmat(errorSDmetres, numel(elint.meas.times), 1);
elint.meas.orientation = zeros(numel(elint.meas.times), 1);
fn = fieldnames(meascolour);
for c = 1:numel(fn)
    elint.meas.(fn{c}) = meascolour.(fn{c});
end

sensorData = {elint};

doplotfig = false;
if doplotfig
    figure
    hold on
    for i = 1:ntargets
    plot(truth(i).coords(:,1), truth(i).coords(:,2), 'b.-');
    end
    plot(measlonlat(:,1), measlonlat(:,2), 'rx')
end

%--------------------------------------------------------------------------

function lonLat = local2geodetic(xy, lonLat0)

[lat, lon] = enu2geodetic(xy(1,:), xy(2,:), zeros(1,size(xy,2)),...
    lonLat0(2), lonLat0(1), 0, wgs84Ellipsoid);
lonLat = [lon; lat];
