function trackdata = getTrackData(...
    track, time, colours, scanNum, sensorNum, lineNum, mostlikelyComp)

% trackdata = getTrackData(track, time, colours, scanNum)
%
% Get a struct of data for a single track

trackdata.time = time;
trackdata.id = track.id;
[mean, cov] = getTrackStateGaussian(track, mostlikelyComp);

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
switchProbs = zeros(1, ncolours);
for c = 1:ncolours
    col = track.colours(c);
    logcolourweights = track.colours(c).logweightsGivenMeas + track.logweights(...
        track.colours(c).measHistIndices);
    [colourMeans(:,c), colourCovs(:,:,c)] = mergegaussians(col.means,...
        col.covs, logcolourweights);
    switchProbs(c) = exp(sumvectorinlogs(logcolourweights(...
        col.modelHists(end,:)==2)));
end
trackdata.colourMeans = colourMeans;
trackdata.colourVars = colourCovs(:)';

existProb = sum(exp(track.logweights).*track.existProbs);
visProbs = sum(exp(track.logweights).*(track.existProbs.*track.visProbsGivenExist), 2);

trackdata.existProb = existProb;
trackdata.visProbs = visProbs(:)';
trackdata.switchProbs = switchProbs;

% Scan number for this update
trackdata.scanNum = scanNum;
trackdata.sensorNum = sensorNum;
trackdata.lineNum = lineNum;
