function [tracks, nextTrackID] = ...
    tracker_single_python(tracks, thismeas, dtSeconds, txmodel, sensorIndex, ...
                          sensorData, prior, colours, params, measNum, numScans, nextTrackID) %, verbose, numScans
%TRACKER_SINGLE Summary of this function goes here
%   Detailed explanation goes here
    
    numTracks = numel(tracks);
    
    for i=1:numTracks
        tracks{i}.colours = cell2structArray(tracks{i}.colours);
    end
    colours = cell2structArray(colours);
    
%     nextTrackID = 1;
%     if numTracks > 0
%         nextTrackID = tracks{end}.id + 1;
%     end
    
    thismeasDateTime = thismeas.time;
    
    gatedTrackIndices = doGating(tracks, txmodel, thismeas,...
        sensorData{sensorIndex}.sensor.gatesdThresh, colours);
    numGatedTracks = numel(gatedTrackIndices);
    
    if ~mod(measNum, 1)
        fprintf(['Timestep %d / %d (%s @ %f), %d tracks (%d gated, '...
            'maxtrackid=%d)\n'],...
            measNum, numScans, sensorData{sensorIndex}.name, thismeas.time,...
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
    
%     trackdata = outputTracks(trackdata, tracks, thismeasDateTime, measNum,...
%         sensorIndex, lineIndex,...
%         colours, gatedTrackIndices, hypLogWeights, params.trackstooutput,...
%         params.outputMostLikelyComp);
    
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
    
    asd=1;
    for i=1:numel(tracks)
        tracks{i}.colours = struct2cellArray(tracks{i}.colours);
    end



