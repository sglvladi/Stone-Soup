function [logLikGivenAssoc, partLogLikGivenModelAndAssoc] =...
    getTrackMeasurementLikelihood(track, meas, sensor, colours)

% getTrackMeasurementLikelihood(track, meas, sensor, colours)
%
% Get log likelihoods for track
%
% Input:
%   track:  Struct specifying the track
%   meas:   Struct specifying the measurement data
%   sensor: Struct specifying the sensor data
%
% Output:
%   logLikGivenAssoc(1, a) = p(z | a_{1:k-1}, a_k=1) where a is the
%       assignment history index
%   partLogLikGivenModelAndAssoc{c}(1, m) =
%       p(z_c | a_{1:k-1}, m_{1:k-1}, a_k=1) where m is the assignment and
%       model history index and c is the part of the state (kinematics
%       corresponding to c=1)

nassocs = numel(track.logweights);
ncoloursdef = numel(meas.coloursDefined);

% p(y^c | m^c_{1:k}, a_{1:k-1}, a_k=1)
partLogLikGivenModelAndAssoc = cell(1, ncoloursdef + 1);
% p(y^c | a_{1:k-1}, a_k=1) =
%     p(y^c | m^c_{1:k}, a_{1:k-1}, a_k=1)
partLogLikGivenAssoc = -inf(ncoloursdef + 1, nassocs);

% Get likelihood for target kinematics
ncomps = numel(track.state.logweightsGivenMeas);    
partLogLikGivenModelAndAssoc{1} = nan(1, ncomps);
zbar = meas.H*track.state.means;
for r = 1:ncomps
    S = meas.H*track.state.covs(:,:,r)*meas.H' + meas.R;
    thisloglik = lognormpdf(meas.pos, zbar(:,r), S);
    partLogLikGivenModelAndAssoc{1}(r) = thisloglik;
    % Add components with same assignment history
    idx = track.state.measHistIndices(r);
    partLogLikGivenAssoc(1, idx) = suminlogs(partLogLikGivenAssoc(1, idx),...
        track.state.logweightsGivenMeas(r) + thisloglik);
end

% Get log weights of colour components which were defined in the
% measurement
for c = 1:ncoloursdef
    cidx = meas.coloursDefined(c);
    thisloglik = getColourLogLikelihood(track.colours(cidx), colours(cidx),...
        meas.colour(c));
    partLogLikGivenModelAndAssoc{c + 1} = thisloglik;
    for r = 1:numel(thisloglik)
        idx = track.colours(cidx).measHistIndices(r);
        partLogLikGivenAssoc(c + 1, idx) = suminlogs(partLogLikGivenAssoc(c + 1, idx),...
            track.colours(cidx).logweightsGivenMeas(r) + thisloglik(r));
    end
end

% % p(y | a_{1:k-1}, a_k=1) = \prod_c p(y^c | a_{1:k-1}, a_k=1)
logLikGivenAssoc = sum(partLogLikGivenAssoc, 1);

% Get likelihood of MMSI (if applicable)
logMmsiLik = getMMSILogLikelihood(track, meas, sensor);
logLikGivenAssoc = logLikGivenAssoc + logMmsiLik;

function logweights = getColourLogLikelihood(trackcolour, colour, meas)

assert(size(trackcolour.means, 1)==1); % Currently assume colour is 1-d
if colour.isHarmonic
    nharmonics = numel(colour.harmonicLogProbs);
    zbars = (1:nharmonics)'.*trackcolour.means;
    vars = (1:nharmonics)'.^2.*(trackcolour.covs(:)') + colour.measCov;
    logweights = sumcolumnsinlogs(colour.harmonicLogProbs(:) +...
        lognormpdf1d(meas, zbars, vars));
else
    zbars = trackcolour.means;
    vars = trackcolour.covs(:)' + colour.measCov;
    logweights = lognormpdf1d(meas, zbars, vars);
end

function logMmsiLik = getMMSILogLikelihood(track, meas, sensor)

ncomp = numel(track.logweights);
logMmsiLik = zeros(1, ncomp);

% Get MMSI likelihood
if ~isempty(meas.mmsi)
    for c = 1:ncomp
        thismmsi = track.mmsis{c};
        if isempty(thismmsi)
            % Track MMSI is unknown
            % Likelihood p(z_mmsi | x_mmsi = null) (uniform distribution over
            % space of MMSIs)
            logMmsiLik(c) = sensor.logMMSINullLik;
        else
            % Track has a MMSI
            % logpMatch = log probability of a measurement MMSI from a ship
            % with known MMSI matching (e.g. very high)
            logpMatch = sensor.logpMMSIMatch;
            % Probability of the MMSI incorrectly sent (e.g. very low)
            logpNoMatch = log(1 - exp(logpMatch));
            if isequal(thismmsi, meas.mmsi)
                logMmsiLik(c) = logpMatch;
            else
                % Likelihood of the ship sending an incorrect measurement
                % multiplied by a uniform MMSI distribution
                logMmsiLik(c) = logpNoMatch + sensor.logMMSINullLik;
            end
        end
    end
end
