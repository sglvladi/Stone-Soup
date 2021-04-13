function [sensorData, colours, truth, timeIndices] = simulateDenseTargets_LV(ntargets)

[sensorData, colours, truth] = simulateDenseTargets(ntargets, [-84; 34], [9; 62]);

% Get which sensor and which line to use for each measurement (in time
% order)
[timeIndices.sensor, timeIndices.line] = getTimeIndices(...
    cellfun(@(x)x.meas.times, sensorData, 'UniformOutput', false));

for i = 1:numel(sensorData)
    sensorData{i}.sensor.gatesdThresh = 5.0;
end


%% PYTHON COMPAT

% Convert times to datestr
for i=1:length(sensorData)
    times = {};
    for j=1:length(sensorData{i}.meas.times)
        times{end+1} = datestr(sensorData{i}.meas.times(j));
    end
    sensorData{i}.meas.times = times;
end

% Colours to cell array
colours = struct2cellArray(colours);

% Truth mmsi to cell array of strings
truth_mmsi = {};
for i=1:size(truth.mmsi,1)
    truth_mmsi{i} = truth.mmsi(i,:);
end
truth.mmsi = truth_mmsi;

% Truth to cell array
truth = struct2cellArray(truth);
end
