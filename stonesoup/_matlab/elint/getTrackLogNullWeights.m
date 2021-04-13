function logNullWeights = getTrackLogNullWeights(track)

% getTrackLogAssocWeights(track, meas, sensor, colours)
%
% Get null log likelihoods for track
%
% Input:
%   track:       Struct specifying the track
%
% Output:
%   logNullweights:  Log weight of assignment to null given assignments
%       (log w_{null}[a_{1:k}] in note)

% Probability of not being detected by each sensor given existence
pNotDetGivenExist = track.pNotVisAndNotDetGivenExist +...
    track.pVisAndNotDetGivenExist;
% Probability of not being detected at all given existence
pAllNotDetGivenExist = prod(pNotDetGivenExist, 1);
pAllNotDetAndExist = track.existProbs.*pAllNotDetGivenExist;
% Probability of not being detected at all (including nonexistence)
logNullWeights = log(pAllNotDetAndExist + (1 - track.existProbs));
