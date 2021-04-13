function tracks = predictTrackStates(tracks, txmodel, meas, colours)%, params)

% Predict the existence and visibility probabilities separately

for i = 1:numel(tracks)
    tracks{i} = predictTrack(tracks{i}, txmodel, meas, colours);%, params);
end

%--------------------------------------------------------------------------

function track = predictTrack(track, txmodel, meas, colours)%, params)

dt = meas.time - track.prevTime;

% Get transition kernel (based on track mean)
[F, Q] = getStateTransition(track, dt, txmodel);

% Predict target kinematics
track.state.means = F*track.state.means;
ncomps = numel(track.state.logweightsGivenMeas);
for c = 1:ncomps
    track.state.covs(:,:,c) = F*track.state.covs(:,:,c)*F' + Q;
end

% Predict colour information
ncolours = numel(track.colours);
for c = 1:ncolours
    track.colours(c) = predictTrackColour(track.colours(c), colours(c), dt);%, params);
end
track.prevTime = meas.time;

%--------------------------------------------------------------------------

