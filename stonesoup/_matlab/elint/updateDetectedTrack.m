function track = updateDetectedTrack(track, meas,...
    logAssignProb, logAssocWeights, logNullWeights,...
    partCompLogLik, colours, sensorIndex, measID)

logNotAssignProb = log(1 - exp(logAssignProb));

% Add measurement history with undetected/detected components
% (undetected first)
noldassigncomps = size(track.measHist, 2);
track.measHist = [repmat(track.measHist, 1, 2);...
    zeros(1, noldassigncomps) repmat(measID, 1, noldassigncomps)];
logdetweights = logAssignProb +...
    normaliseinlogs(track.logweights + logAssocWeights);
lognotdetweights = logNotAssignProb +...
    normaliseinlogs(track.logweights + logNullWeights);
track.logweights = normaliseinlogs([lognotdetweights logdetweights]);

% Add MMSI for new component
% If no MMSI on this measurement, get MMSI of parent component, otherwise
% get MMSI measurement
if isempty(meas.mmsi)
    track.mmsis = repmat(track.mmsis, 1, 2);
else
    track.mmsis = [track.mmsis repmat({meas.mmsi}, 1, noldassigncomps)];
end
assert(numel(track.mmsis)==numel(track.logweights));

% Probability of not being detected by each sensor given existence and each
% assignment
pNotDetGivenExist_eachSensor = track.pNotVisAndNotDetGivenExist +...
    track.pVisAndNotDetGivenExist;
% Probability of not being detected at all given existence
pNotDetGivenExist_all = prod(pNotDetGivenExist_eachSensor, 1);
pNotDetAndExist_all = track.existProbs.*pNotDetGivenExist_all;
% Probability of not being detected at all (including nonexistence)
pNotDet_all = pNotDetAndExist_all + (1 - track.existProbs);

% Update visibility probabilities
% p(V^j | nD^j, E) = p(V^j, nD^j | E)/p(nD^j | E)
visProbsGivenExistNotDet = track.pVisAndNotDetGivenExist./...
    pNotDetGivenExist_eachSensor;
% If detected, visibility probability of this sensor is 1
visProbsGivenExistDet = visProbsGivenExistNotDet;
visProbsGivenExistDet(sensorIndex,:) = 1;
track.visProbsGivenExist = min(max(...
    [visProbsGivenExistNotDet visProbsGivenExistDet], 0), 1);
track = rmfield(track, {'pNotVisAndNotDetGivenExist', 'pVisAndNotDetGivenExist'});

% Update existence probabilities
% p(E | nD) = p(E, nD)/p(nD)
existProbsGivenNotDet = pNotDetAndExist_all./pNotDet_all;
existProbsGivenDet = ones(size(existProbsGivenNotDet)); 
track.existProbs = min(max([existProbsGivenNotDet existProbsGivenDet], 0), 1);

% Add assignments to state
noldmodelcomps = numel(track.state.logweightsGivenMeas);
% Detected weights are old weights multiplied by likelihood, normalised so
% components for each association sums to 1
logdetweights = normaliseLogWeightsGivenMeas(...
    track.state.logweightsGivenMeas + partCompLogLik{1},...
    track.state.measHistIndices);
lognotdetweights = track.state.logweightsGivenMeas;
track.state.logweightsGivenMeas = [lognotdetweights logdetweights];
track.state.measHistIndices = [track.state.measHistIndices...
    track.state.measHistIndices + noldassigncomps];
track.state.modelHists = repmat(track.state.modelHists, 1, 2);
% Update state means and covariances
track.state.means = repmat(track.state.means, 1, 2);
track.state.covs = repmat(track.state.covs, 1, 1, 2);
for i = noldmodelcomps+(1:noldmodelcomps)
    [track.state.means(:,i), track.state.covs(:,:,i)] = kalmanStep(...
        track.state.means(:,i), track.state.covs(:,:,i), meas.H, meas.R,...
        meas.pos);
end

% Add assignments to colours
for c = 1:numel(colours)
    [~, cidx] = ismember(c, meas.coloursDefined);
    % Add new components
    noldmodelcomps = numel(track.colours(c).logweightsGivenMeas);
    track.colours(c).means = repmat(track.colours(c).means, 1, 2);
    track.colours(c).covs = repmat(track.colours(c).covs, 1, 1, 2);
    if cidx ~= 0
        logdetweights = normaliseLogWeightsGivenMeas(...
            track.colours(c).logweightsGivenMeas + partCompLogLik{cidx+1},...
            track.colours(c).measHistIndices);
        lognotdetweights = track.colours(c).logweightsGivenMeas;
        track.colours(c).logweightsGivenMeas = [lognotdetweights logdetweights];
        % Update means and covariances
        cpidx = noldmodelcomps+(1:noldmodelcomps);
        if colours(c).isHarmonic
            [newmeans, newcovs] = kalmanStepHarmonic1D(...
                track.colours(c).means(:,cpidx),...
                track.colours(c).covs(:,:,cpidx), 1,...
                colours(c).measCov, meas.colour(cidx),...
                colours(c).harmonicLogProbs);
        else
            [newmeans, newcovs] = kalmanStep1D(...
                track.colours(c).means(:,cpidx),...
                track.colours(c).covs(:,:,cpidx),...
                1, colours(c).measCov, meas.colour(cidx));
        end
        track.colours(c).means(:,cpidx) = newmeans;
        track.colours(c).covs(:,cpidx) = newcovs;
    else
        track.colours(c).logweightsGivenMeas = repmat(...
            track.colours(c).logweightsGivenMeas, 1, 2);
    end
    track.colours(c).measHistIndices = [track.colours(c).measHistIndices...
        track.colours(c).measHistIndices + noldassigncomps];
    track.colours(c).modelHists = repmat(track.colours(c).modelHists, 1, 2);
end
