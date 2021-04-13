function [F, Q] = getStateTransition(track, dt, txmodel)

% Get weights for each component (my multiplying assoc and model
% probabilities together)
logweights = track.state.logweightsGivenMeas +...
    track.logweights(track.state.measHistIndices);

% Get mean position
meanpos = sum(exp(logweights).*track.state.means([1 3],:), 2);

% Get transition kernel (dependent on the mean track position)
[F, Q] = getTransitionModel(meanpos, dt, txmodel);
