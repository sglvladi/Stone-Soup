function [trackdata, tracks] = tracker(sensorData, txmodel, prior, colours,...
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
    
    numTracks = numel(tracks);
    sensorIndex = timeIndices.sensor(measNum);
    lineIndex = timeIndices.line(measNum);
    
    thismeas = getMeasurementData(sensorData{sensorIndex},...
        lineIndex, colours);

    thismeasDateTime = thismeas.time;
    thismeas.time = seconds(thismeasDateTime - initDateTime);
    newtime = thismeas.time;
    dtSeconds = newtime - currenttime;
    
    gatedTrackIndices = doGating(tracks, txmodel, thismeas,...
        sensorData{sensorIndex}.sensor.gatesdThresh, colours);
    numGatedTracks = numel(gatedTrackIndices);
    
    if ~mod(measNum, 1) && verbose
        fprintf(['Timestep %d / %d (%s @ %f), %d tracks (%d gated, '...
            'maxtrackid=%d)\n'],...
            measNum, numScans, sensorData{sensorIndex}.name, newtime,...
            numTracks, numGatedTracks, nextTrackID-1);
    end
    
    % Get prior distribution for measurement
    nullLogLik = getLogPriorMeasurementLikelihood(...
        prior, thismeas, sensorData{sensorIndex}, colours);
    % Get null weight from likelihood (meas rate cancels!)
    nullLogWeight = nullLogLik + log(params.estNumUnconfirmedTargets);
    
    % (PRH 17/12/2020: Add term for not detected by other sensors?)
    % (PRH 26/01/2021: Accounted for in track null weights)
    %measRates = cellfun(@(x)x.sensor.rates.meas, sensorData);
    %measRates(sensorIndex) = [];
    %logNotDetOthers = sum(-dtSeconds*measRates);
    %nullLogWeight = nullLogWeight + logNotDetOthers;
    
    % Predict all the track existence, visibility and detection probabilities
    tracks = predictTrackExistenceAndVisibility(...
        tracks, txmodel, sensorData, dtSeconds);
    
    % Predict the gated track states
    tracks(gatedTrackIndices) = predictTrackStates(...
        tracks(gatedTrackIndices), txmodel, thismeas, colours);%, params);
    currenttime = newtime;
    
    % Get measurement likelihoods for each track
    logTrackWeights = zeros(1, numGatedTracks);
    logAssocWeights = cell(1, numGatedTracks);
    logNullWeights = cell(1, numGatedTracks);
    partCompLogLik = cell(1, numGatedTracks);
    % Predict forward tracks and get likelihoods
    for i = 1:numGatedTracks
        tracki = gatedTrackIndices(i);
        % Get p(z | a_{1:k-1}, a_k=1)
        %     p(z^c | m^c, a_{1:k-1}, a_k=1)
        [measLogLik, partCompLogLik{i}] =...
            getTrackMeasurementLikelihood(tracks{tracki}, thismeas,...
            sensorData{sensorIndex}.sensor, colours);
        % Get logTrackWeights(i) = log assignment weight of ith gated track
        logAssocWeights{i} = getTrackLogAssocWeights(...
            tracks{tracki}, sensorData{sensorIndex}.sensor, dtSeconds,...
            measLogLik, sensorIndex);
        logNullWeights{i} = getTrackLogNullWeights(tracks{tracki});
        logTrackWeights(i) = sumvectorinlogs(logAssocWeights{i} +...
            tracks{tracki}.logweights) -...
            sumvectorinlogs(logNullWeights{i} + tracks{tracki}.logweights);
    end
    
    % Get log probability of assignment to existing tracks and new track
    hypLogWeights = normaliseinlogs([logTrackWeights nullLogWeight]);
    
    for i = 1:numTracks
        [~, gidx] = ismember(i, gatedTrackIndices);
        if gidx > 0 && hypLogWeights(gidx) > -inf
            logAssignProb = hypLogWeights(gidx);
            tracks{i} = updateDetectedTrack(tracks{i}, thismeas,...
                logAssignProb, logAssocWeights{gidx}, logNullWeights{gidx},...
                partCompLogLik{gidx}, colours,...
                sensorIndex, measNum);
        else
            tracks{i} = updateMissedTrack(tracks{i});
        end
    end
    
    % Create a new track if the measurement wasn't assigned any existing tracks
    % (change later?)
    if max(hypLogWeights)==hypLogWeights(end) %numGatedTracks == 0
        [tracks{end+1}, nextTrackID] = createNewTrack(thismeas, prior,...
            colours, sensorData, sensorIndex, measNum, nextTrackID,...
            hypLogWeights(end));      
        numTracks = numTracks + 1;
    end
    
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
    
    trackdata = outputTracks(trackdata, tracks, thismeasDateTime, measNum,...
        sensorIndex, lineIndex,...
        colours, gatedTrackIndices, hypLogWeights, params.trackstooutput,...
        params.outputMostLikelyComp);
        
    % Do mixture reduction on assignments for gated tracks (model reduction
    % done separately)
    tracks(gatedTrackIndices) = mixtureReduceAssigns(...
        tracks(gatedTrackIndices), params);
    
    % Kill tracks that have too low a probability of existence
    killtrack = zeros(1, numTracks);
    for i = 1:numTracks
        % Probability that target exists and is visible to at least one
        % sensor
        pe = tracks{i}.existProbs;
        pv = tracks{i}.visProbsGivenExist;
        p = sum(exp(tracks{i}.logweights).*(pe.*(1 - prod(1 - pv, 1))));
        if p < params.killProbThresh
            killtrack(i) = true;
        end
    end
    tracks = tracks(~killtrack);
    
end
trackdata = mergeFields(trackdata);
trackdata.initDateTime = initDateTime;

%--------------------------------------------------------------------------

function trackdata = outputTracks(trackdata, tracks, time, measNum,...
    sensorNum, lineNum, colours, gatedTrackIndices, hypLogWeights,...
    trackstooutput, mostLikelyComp)

% Get indices of tracks to output
[~,mostlikelyidx] = max(hypLogWeights);
switch lower(trackstooutput)
    case 'mostlikely'
        if mostlikelyidx > numel(gatedTrackIndices)
            outputIndices = numel(tracks);
        else
            outputIndices = gatedTrackIndices(mostlikelyidx);
        end
        outputHypLogProbs = hypLogWeights(mostlikelyidx);
 	case 'gated'
        if mostlikelyidx > numel(gatedTrackIndices)
             outputIndices = [gatedTrackIndices numel(tracks)];
             outputHypLogProbs = hypLogWeights;
        else
            outputIndices = gatedTrackIndices;
            outputHypLogProbs = hypLogWeights(1:end-1); % exclude new track prob
        end
    case 'all'
        outputIndices = 1:numel(tracks);
        outputHypLogProbs = -inf(1, numel(tracks));
        outputHypLogProbs(gatedTrackIndices) = hypLogWeights(1:end-1);
end     
    
% Output tracks
for i = 1:numel(outputIndices)
    newdata = getTrackData(tracks{outputIndices(i)},...
        time, colours, measNum, sensorNum, lineNum,...
        mostLikelyComp);
    newdata.assignLogProb = outputHypLogProbs(i);
    if isempty(trackdata)
        trackdata = newdata;
    else
        trackdata(end+1) = newdata;
    end
end

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
