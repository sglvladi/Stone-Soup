function trackcolour = predictTrackColour(trackcolour, colour, dt)

trackcolour.covs = trackcolour.covs + dt*colour.q;
    
% Predict switch
if colour.isSwitch
    
    % Get transition matrix
    logtransmatrix = log(getSwitchProbsFromRates(colour.switchRate0to1,...
        colour.switchRate1to0, dt));
    [assign,~,assignidx] = uniquewithcounts(trackcolour.measHistIndices);
    nassign = numel(assignidx);
    
    newlogweights = nan(1, 2*nassign);
    newmeans = nan(1, 2*nassign);
    newcovs = nan(1, 1, 2*nassign);
    newmodelhist = repmat([1 2], 1, nassign);
    newmeasindices = repelem(assign, 1, 2);
    
    for i = 1:nassign
        isswitch = (trackcolour.modelHists(end, assignidx{i}) == 2);
        switchidx = assignidx{i}(isswitch);
        notswitchidx = assignidx{i}(~isswitch);
        
        % Get old probabilities of switching and nonswitching components
        oldlogPSwitch = trackcolour.logweightsGivenMeas(switchidx);
        oldlogPNotSwitch = trackcolour.logweightsGivenMeas(notswitchidx);
        
        % Set nonswitching component
        logw = [oldlogPNotSwitch + logtransmatrix(1,1)...
            oldlogPSwitch + logtransmatrix(2,1)];
        [newmeans(:,2*i-1), newcovs(:,:,2*i-1), newlogweights(2*i-1)] =...
            mergegaussians(...
            trackcolour.means(:,[notswitchidx switchidx]),...
            trackcolour.covs(:,:,[notswitchidx switchidx]), logw);
        % Set switching component
        newmeans(:,2*i) = colour.mean;
        newcovs(:,:,2*i) = colour.cov;
        newlogweights(2*i) = sumvectorinlogs([(oldlogPSwitch +...
            logtransmatrix(2,2)) (oldlogPNotSwitch + logtransmatrix(1,2))]);
        
    end

    % Set track colour parameters
    trackcolour.means = newmeans;
    trackcolour.covs = newcovs;
    trackcolour.logweightsGivenMeas = newlogweights;
    trackcolour.measHistIndices = newmeasindices;
    trackcolour.modelHists = newmodelhist;
end
