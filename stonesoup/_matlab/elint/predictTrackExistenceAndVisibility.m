function tracks = predictTrackExistenceAndVisibility(...
    tracks, txmodel, sensorData, dt)

% Predict all the track existence probabilities
numTracks = numel(tracks);
surviveProb = exp(-txmodel.deathRate*dt);

measRates = cellfun(@(x)x.sensor.rates.meas, sensorData);
hideRates = cellfun(@(x)x.sensor.rates.hide, sensorData);
revealRates = cellfun(@(x)x.sensor.rates.reveal, sensorData);

% Get visDetTrans(v0, v1, d1, j) = probability that we transition from old
% visibility state v0-1 to new visibility state v1-1 and detection state
% d1-1 in time interval dt for sensor j
visDetTrans = getVisibilityDetectionTransition(...
	revealRates, hideRates, measRates, dt);

trans_notOldV_newVnotD = tocolumn(visDetTrans(1, 2, 1, :));
trans_oldV_newVnotD = tocolumn(visDetTrans(2, 2, 1, :));

trans_notOldV_notNewVnotD = tocolumn(visDetTrans(1, 1, 1, :));
trans_oldV_notNewVnotD = tocolumn(visDetTrans(2, 1, 1, :));

hideProb = tocolumn(sum(visDetTrans(2, 1, :, :), 3));
revealProb = tocolumn(sum(visDetTrans(1, 2, :, :), 3));

for i = 1:numTracks
	tracks{i}.existProbs = surviveProb.*tracks{i}.existProbs;
    
    poldVgivenE = tracks{i}.visProbsGivenExist;
    % (s,i) entries are probability for sensor s, component i
    tracks{i}.visProbsGivenExist = (1 - hideProb).*poldVgivenE +...
        revealProb.*(1 - poldVgivenE);
    tracks{i}.pNotVisAndNotDetGivenExist =...
        trans_oldV_notNewVnotD.*poldVgivenE +...
        trans_notOldV_notNewVnotD.*(1 - poldVgivenE);
    tracks{i}.pVisAndNotDetGivenExist =...
        trans_oldV_newVnotD.*poldVgivenE +...
        trans_notOldV_newVnotD.*(1 - poldVgivenE);
end
