function [trackdata, tracks] = tracker(sensorData, txmodel, prior, colours, params)

% Things to do
% 4) Estimate the number of unconfirmed targets with a Poisson-based
% approach?
% 5) Test properly and try on more challenging scenarios

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

for measNum = 1:numScans%1:min(numScans, 1000)%
    
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
    
    if ~mod(measNum, 1)
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
    
    % Predict all the track existence, visibility and detection probabilities
    tracks = predictTrackExistenceAndVisibility(...
        tracks, txmodel, sensorData, dtSeconds);
    
    % Predict the gated track states
    tracks(gatedTrackIndices) = predictTrackStates(...
        tracks(gatedTrackIndices), txmodel, thismeas, colours);%, params);
    currenttime = newtime;
    
    % Get measurement likelihoods for each track
    trackLogWeights = zeros(1, numGatedTracks);
    assocLogLik = cell(1, numGatedTracks);
    partCompLogLik = cell(1, numGatedTracks);
    % Predict forward tracks and get likelihoods
    for i = 1:numGatedTracks
        tracki = gatedTrackIndices(i);
        % Get p(z | a_{1:k-1}, a_k=1)
        %     p(z^c | m^c, a_{1:k-1}, a_k=1)
        [assocLogLik{i}, partCompLogLik{i}] =...
            getTrackMeasurementLikelihood(tracks{tracki}, thismeas,...
            sensorData{sensorIndex}.sensor, colours);
        % Get p(z | a_{1:k-1}, a_k=1)p(V | a_{1:k-1})
        pvis = tracks{tracki}.visProbsGivenExist(sensorIndex,:).*...
            tracks{tracki}.existProbs;
        assocLogLik{i} = assocLogLik{i} + log(pvis);
        % Get track association weight
        trackLogWeights(i) = sumvectorinlogs(assocLogLik{i} +...
            tracks{tracki}.logweights);
    end
    
    % Get best assignment (0 = null) and log probability of assigned
    % hypothesis
    hypLogWeights = normaliseinlogs([trackLogWeights nullLogWeight]);
    
    for i = 1:numTracks
        [~, gidx] = ismember(i, gatedTrackIndices);
        if gidx > 0 && hypLogWeights(gidx) > -inf
            logAssignProb = hypLogWeights(gidx);
            tracks{i} = updateDetectedTrack(tracks{i}, thismeas,...
                assocLogLik{gidx}, partCompLogLik{gidx}, colours,...
                logAssignProb, sensorIndex, measNum);
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
    
    trackdata = outputTracks(trackdata, tracks, thismeasDateTime, measNum,...
        colours, gatedTrackIndices, hypLogWeights, params.trackstooutput);
        
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
    colours, gatedTrackIndices, hypLogWeights, trackstooutput)

% Get indices of tracks to output
[~,mostlikelyidx] = max(hypLogWeights);
switch lower(trackstooutput)
    case 'mostlikely'
        if mostlikelyidx > numel(gatedTrackIndices)
            outputIndices = numel(tracks);
        else
            outputIndices = gatedTrackIndices(mostlikelyidx);
        end
 	case 'gated'
        if mostlikelyidx > numel(gatedTrackIndices)
             outputIndices = [gatedTrackIndices numel(tracks)];
        else
            outputIndices = gatedTrackIndices;
        end
    case 'all'
        outputIndices = 1:numel(tracks);
end     
    
% Output tracks
for i = 1:numel(outputIndices)
    newdata = getTrackData(tracks{outputIndices(i)},...
        time, colours, measNum);
    if isempty(trackdata)
        trackdata = newdata;
    else
        trackdata(end+1) = newdata;
    end
end

function plotTrackGaussians(tracks)

figure
hold on
for i = 1:numel(tracks)
    [mn, cv] = getTrackStateGaussian(tracks{i});
    gaussellipse(mn([1 3]), cv([1 3],[1 3]), 'g', 'x');
    text(mn(1), mn(3), ['  ' num2str(tracks{i}.id)]);
end
axis equal

function trackdata = getTrackData(track, time, colours, scanNum)

trackdata.time = time;
trackdata.id = track.id;
[mean, cov] = getTrackStateGaussian(track, false);

[mmsis, mmsiLogProbs] = getMMSILogProbs(track);
trackdata.mmsis = {mmsis};
trackdata.mmsiLogProbs = {mmsiLogProbs};
[mxlogprob, mxind] = max(mmsiLogProbs);
trackdata.mmsiMostProb = mmsis(mxind);
trackdata.mmsiMaxLogProb = mxlogprob;

trackdata.mean = mean(:)';
trackdata.cov = cov(:)';
%trackdata = [time track.id mean(:)' cov(:)'];

ncolours = numel(track.colours);
colourMeans = nan(1, ncolours);
colourCovs = nan(1, 1, ncolours);
%switchProbs = repmat(-1, 1, ncolours);
for c = 1:ncolours
    col = track.colours(c);
    logcolourweights = track.colours(c).logweightsGivenMeas + track.logweights(...
    track.colours(c).measHistIndices);
    [colourMeans(:,c), colourCovs(:,:,c)] = mergegaussians(col.means,...
        col.covs, logcolourweights);
    % Add model probs later
    %if colours(c).isSwitch
    %    switchProbs(c) = exp(sumvectorinlogs(col.logweights(col.modelHist)));
    %end
end
trackdata.colourMeans = colourMeans;
trackdata.colourVars = colourCovs(:)';

existProb = sum(exp(track.logweights).*track.existProbs);
visProbs = sum(exp(track.logweights).*(track.existProbs.*track.visProbsGivenExist), 2);

trackdata.existProb = existProb;
trackdata.visProbs = visProbs(:)';

% Scan number for this update
trackdata.scanNum = scanNum;

function newtrackdata = mergeFields(trackdata)

if isempty(trackdata)
    newtrackdata = [];
    return
end

fn = fieldnames(trackdata);
for i = 1:numel(fn)
    newtrackdata.(fn{i}) = cat(1, trackdata(:).(fn{i}));
end
