function [track, nextTrackID] = createNewTrack(meas, prior,...
    colours, sensorData, sensorindex, measID, nextTrackID, logExistProb)

% track.id
% track.prevTime
% track.mmsi
% track.state.       (same as colours)
% track.colours.     means, covs, logweightsGivenMeas, measHistIndices, modelHists
% track.logweights
% track.measHist
% track.existProbs
% track.visProbsGivenExist

track.id = nextTrackID;
nextTrackID = nextTrackID + 1;
track.prevTime = meas.time;
track.mmsis = {meas.mmsi}; % MMSI per component

% Set initial kinematic distribution
[priorMean, priorCov] = getInitStatePrior(meas.pos, prior);
[track.state.means, track.state.covs] = kalmanStep(priorMean, priorCov,...
    meas.H, meas.R, meas.pos);
track.state.logweightsGivenMeas = 0;
track.state.measHistIndices = 1;
track.state.modelHists = 1;

% Set colour information
ncolours = numel(colours);
track.colours = repmat(struct('means',[],'covs',[],'logweightsGivenMeas',[],...
    'measHistIndices',[],'modelHists',[]), 1, ncolours);
[~, colourindices] = ismember(1:ncolours, meas.coloursDefined);
for c = 1:ncolours
    colourmean = colours(c).mean;
    colourcov = colours(c).cov;
    if colourindices(c) > 0
        [colourmean, colourcov] = kalmanStep(...
            colourmean, colourcov, 1, colours(c).measCov,...
            meas.colour(colourindices(c)));
    end
    if colours(c).isSwitch
        % Create components for switching and non-switching case
        switchProb = colours(c).priorSwitchProb;
        track.colours(c).means = repmat(colourmean, 1, 2);
        track.colours(c).covs = repmat(colourcov, 1, 1, 2);
        track.colours(c).logweightsGivenMeas = log([1-switchProb switchProb]);
        track.colours(c).measHistIndices = [1 1];
        track.colours(c).modelHists = [1 2];
    else
        track.colours(c).means = colourmean;
        track.colours(c).covs = colourcov;
        track.colours(c).logweightsGivenMeas = 0;
        track.colours(c).measHistIndices = 1;
        track.colours(c).modelHists = 1;
    end
end

% Set existence and visibility probabilities
track.logweights = 0;
track.measHist = measID;
track.existProbs = exp(logExistProb);
track.visProbsGivenExist = tocolumn(cellfun(@(x)x.sensor.priorVisProb, sensorData));
track.visProbsGivenExist(sensorindex) = 1;

%--------------------------------------------------------------------------

function [priorMean, priorCov] = getInitStatePrior(pos, prior)

% Get the prior on the kinematic state using the position to convert the
% velocity from m/s to deg/s

[posMean, posCov] = fitNormalToUniform(prior.posMin, prior.posMax);
velMean = zeros(size(posMean));
initvel_metres = prior.speedMetresSD^2*eye(2);
H = diag(1./degree2metres(pos));
velCov = H*initvel_metres*H';

posdim = size(posMean, 1);
priorMean = zeros(2*posdim, 1);
priorMean(1:2:end) = posMean;
priorMean(2:2:end) = velMean;
priorCov = zeros(2*posdim, 2*posdim);
priorCov(1:2:end, 1:2:end) = posCov;
priorCov(2:2:end, 2:2:end) = velCov;
