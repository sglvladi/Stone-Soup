function [mean, cov] = getTrackStateGaussian(track, mostlikely)

if ~exist('mostlikely', 'var')
    mostlikely = false;
end

logstateweights = track.state.logweightsGivenMeas + track.logweights(...
    track.state.measHistIndices);
if ~mostlikely
    % Merged component
    [mean, cov] = mergegaussians(track.state.means, track.state.covs,...
        logstateweights);
else
    % Most likely component
    [~,idx] = max(logstateweights);
    mean = track.state.means(:,idx);
    cov = track.state.covs(:,:,idx);
end
