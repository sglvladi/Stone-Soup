function track = updateMissedTrack(track)

% Multiply associations by null weights (null weights are not calculated
% earlier for non-gated tracks)
logNullWeights = getTrackLogNullWeights(track);
track.logweights = normaliseinlogs(track.logweights + logNullWeights);

% Probability of not being detected by each sensor given existence and each
% assignment
pNotDetGivenExist_eachSensor = track.pNotVisAndNotDetGivenExist +...
    track.pVisAndNotDetGivenExist;
% Probability of not being detected at all given existence
pNotDetGivenExist_all = prod(pNotDetGivenExist_eachSensor, 1);
pNotDetAndExist_all = track.existProbs.*pNotDetGivenExist_all;
% Probability of not being detected at all (including nonexistence)
pNotDet_all = pNotDetAndExist_all + (1 - track.existProbs);

% Get visibility probabilities given existence
% p(V^j | nD^j, E) = p(V^j, nD^j | E)/p(nD^j | E)
track.visProbsGivenExist = min(max(track.pVisAndNotDetGivenExist./...
    pNotDetGivenExist_eachSensor, 0), 1);

% Update probability of existence 
% p(E | nD) = p(E, nD)/p(nD)
track.existProbs = min(max(pNotDetAndExist_all./pNotDet_all, 0), 1);
