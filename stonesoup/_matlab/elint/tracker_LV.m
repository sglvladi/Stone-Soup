function [trackdata, tracks] = tracker_LV(sensorData, txmodel, prior, colours,...
    params, verbose)

% Modified to have weights which take into account missed detections for
% the sensors

% Things to do
% 4) Estimate the number of unconfirmed targets with a Poisson-based
% approach?
% 5) Test properly and try on more challenging scenarios

doplottracks = false;%true;%
if ~exist('verbose', 'var')
    verbose = true;
end
plottracks_axis = [0 1 55 56] + 0.5*[-1 1 -1 1];

% true to output the most likely Gaussian comp for the state rather than a
% single best-fitting Gaussian
if ~isfield(params, 'outputMostLikelyComp')
    params.outputMostLikelyComp = false;
end

% Get which sensor and which line to use for each measurement (in time
% order)
[timeIndices.sensor, timeIndices.line] = getTimeIndices(...
    cellfun(@(x)x.meas.times, sensorData, 'UniformOutput', false));
numScans = numel(timeIndices.sensor);
initDateTime = sensorData{timeIndices.sensor(1)}.meas.times(timeIndices.line(1));

tracks = cell(0);
nextTrackID = 1;
currenttime = 0;
trackdata = [];

for measNum = 1:numScans
    
    sensorIndex = timeIndices.sensor(measNum);
    lineIndex = timeIndices.line(measNum);
    
    thismeas = getMeasurementData(sensorData{sensorIndex},...
        lineIndex, colours);

    thismeasDateTime = thismeas.time;
    thismeas.time = seconds(thismeasDateTime - initDateTime);
    newtime = thismeas.time;
    dtSeconds = newtime - currenttime;
    
    numTracks = numel(tracks);
    maxTrackID = 0;
    if numTracks > 0
        maxTrackID = tracks{end}.id;
    end
    if ~mod(measNum, 1) && verbose
        fprintf(['Timestep %d / %d (%s @ %f), %d tracks ('...
            'maxtrackid=%d)\n'],...
            measNum, numScans, sensorData{sensorIndex}.name, newtime,...
            numTracks, maxTrackID);
    end
    
    
    tracks = ...
    tracker_single(tracks, thismeas, dtSeconds, txmodel, sensorIndex, ...
                   sensorData, prior, colours, params, measNum); %, verbose, numScans
               
               
    currenttime = newtime;
               
    %------
    if doplottracks
        clf
        plotTrackGaussians(tracks);
%         axis(plottracks_axis);
        plot(thismeas.pos(1), thismeas.pos(2), 'r+')
        if isfield(params, 'truth')
            plot(params.truth.lonLat(:,1), params.truth.lonLat(:,2), 'k+')
        end
        pause(0.1)
        drawnow
    end
    %------
    
end
trackdata = mergeFields(trackdata);
trackdata.initDateTime = initDateTime;

%--------------------------------------------------------------------------

function plotTrackGaussians(tracks)

hold on
for i = 1:numel(tracks)
    [mn, cv] = getTrackStateGaussian(tracks{i});
    gaussellipse(mn([1 3]), cv([1 3],[1 3]), 'b', 'x');
    text(mn(1), mn(3), ['  ' num2str(tracks{i}.id)]);
end
axis equal

function newtrackdata = mergeFields(trackdata)

if isempty(trackdata)
    newtrackdata = [];
    return
end

fn = fieldnames(trackdata);
for i = 1:numel(fn)
    newtrackdata.(fn{i}) = cat(1, trackdata(:).(fn{i}));
end
