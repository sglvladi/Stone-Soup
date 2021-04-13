function logAssocWeights = getTrackLogAssocWeights(...
    track, sensor, dtSeconds, logLikGivenAssoc, sensorIndex)

% getTrackLogAssocWeights(track, meas, sensor, colours)
%
% Get association log likelihoods for track
%
% Input:
%   track:       Struct specifying the track
%   sensor:      Struct specifying the sensor data
%   dtSeconds:   Number of seconds elapsed since last measurement
%   logLikGivenAssoc(1,a): p(z | a_{1:k-1}, a_k=1) where a is the
%       assignment history index
%   sensorIndex: Integer index of the sensor which detected the measurement
%
% Output:
%   logAssocWeights: Log weight of assignment to track given assignments
%       (log w_{post}[a_{1:k}] in note)
%   logNullweights:  Log weight of assignment to null given assignments
%       (log w_{null}[a_{1:k}] in note)

% Probability of visibility and existence for each sensor and each
% component
pVis = track.visProbsGivenExist(sensorIndex,:).*track.existProbs;

% Probability of not being detected by each sensor given existence
pNotDetGivenExist = track.pNotVisAndNotDetGivenExist +...
    track.pVisAndNotDetGivenExist;
% Probability of not being detected by other sensors
logpNotDetOthers = sum(log(pNotDetGivenExist((1:end)~=sensorIndex,:)),1);
% Weights on components given the track is detected
logAssocWeights = logLikGivenAssoc + log(pVis) -...
    sensor.rates.meas*dtSeconds + logpNotDetOthers;
