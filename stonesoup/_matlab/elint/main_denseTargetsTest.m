function main_denseTargetsTest

% addpath(cd(cd('MatlabFunctions')));

rng(1, 'twister')

estNumUnconfirmedTargetsList = 10.^(-3:5);
nlist = numel(estNumUnconfirmedTargetsList);
nruns = 50;

ntargets = 5;%10; % Number of targets to simulate

ntracksTot = nan(nruns, nlist);
ntracksEnd = nan(nruns, nlist);
score = nan(nruns, nlist);

for i = 1:nruns
    [sensorData, colours, truth] = simulateDenseTargets(ntargets);
    for j = 1:nlist
        fprintf('Run %d / %d, EstNTargets %d / %d\n', i, nruns, j, nlist);
        [ntracksTot(i,j), ntracksEnd(i,j), score(i,j)] = testExpNumTargets(...
            sensorData, colours, truth, estNumUnconfirmedTargetsList(j));
    end
end

save main_denseTargetsTest_out_m3_p5

%figure
%semilogx(ntargetsList, ntracksOut, 'bx-');

figure
semilogx(estNumUnconfirmedTargetsList, mean(ntracksTot,1), 'bx-')
hold on
semilogx(estNumUnconfirmedTargetsList, mean(ntracksEnd,1), 'rx-')
xlabel('Initial expected number of targets')
ylabel('Number of tracks');
legend('Total','At end')

figure
semilogx(estNumUnconfirmedTargetsList, mean(score,1), 'bx-')
hold on
xlabel('Initial expected number of targets')
ylabel('Mean assignment probability');


keyboard

%--------------------------------------------------------------------------

% function [ntracksTot, ntracksEnd, score] = testExpNumTargetsBatch(...
%     estNumUnconfirmedTargetsList)
% 
% [sensorData, colours, truth] = simulateDenseTargets(ntargets);
% nlist = numel(estNumUnconfirmedTargetsList);
% 
% ntracksTot = nan(1, nlist);
% ntracksEnd = nan(1, nlist);
% score = nan(1, nlist);
% for i = 1:nlist
%     [ntracksTot(i), ntracksEnd(i), score(i)] = testExpNumTargets(...
%         sensorData, colours, truth, estNumUnconfirmedTargetsList(i));
% end
% 
% %------

function [ntracksTot, ntracksEnd, score] = testExpNumTargets(...
    sensorData, colours, truth, estNumUnconfirmedTargets)

ntargets = size(truth.lonLat, 1);
verbose = false;

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

for i = 1:numel(sensorData)
    sensorData{i}.sensor.gatesdThresh = 5;
end
sensorData = sensorData(1:3);

%------
% Strip out targets of interest
targetIds = 1:ntargets;%[4 10];%[2 4 8 10];%[7 8];%[2 4 8];%1:ntargets;%[7 8];%
truth = selectrows(truth, targetIds);
for i = 1:numel(sensorData)
    sensorData{i}.meas = selectrows(sensorData{i}.meas,...
        ismember(sensorData{i}.meas.targetNum, targetIds));
end
%------

prior.posMin = [-84 34];
prior.posMax = [  9 62];
prior.speedMetresSD = 10;

% Set up transition model stuff
stationary_speed = (prior.speedMetresSD);
txmodel.isOrnsteinUhlenbeck = true;
txmodel.q_metres = 0.1;
% Stationary velocity s.d. of Ornstein-Uhlenbeck is sqrt(q/(2*K)) so set K
% to make prior velocity the stationary distribution
txmodel.K = txmodel.q_metres/(2*stationary_speed^2);
txmodel.deathRate = 1/(4*days2sec);
% Birth rate?

params.killProbThresh = 0.01;
params.measHistLen = 2;
%params.modelHistLen = 1; (assumed to be 1 currently)
% Delete component if log probability less than this
params.compLogProbThresh = log(1e-3);
params.trackstooutput = 'gated';
params.outputMostLikelyComp = true;
params.estNumUnconfirmedTargets = estNumUnconfirmedTargets;

params.truth = truth;
[trackdata, tracks] = tracker(sensorData, txmodel, prior, colours, params, verbose);
trackdata.targetNum = getTrueTargetNum(trackdata, sensorData);

ntracksTot = max(trackdata.id);
ntracksEnd = numel(tracks);
score = getTracksScore(trackdata, sensorData);

if verbose
    figure
    drawtrackdata(trackdata, [1 3], true);
    plot(truth.lonLat(:,1), truth.lonLat(:,2), 'k+')
    for i = 1:numel(targetIds)
        text(truth.lonLat(i,1), truth.lonLat(i,2), ['  ' num2str(targetIds(i))]);
    end
    plotMeasCoords(sensorData, [1 2], {'r.','k.'})
    xlabel('Lon'); ylabel('Lat');
    
    figure
    drawtrackdata(trackdata, 1, true);
    plotMeasCoords(sensorData, 1, {'r.','k.'})
    xlabel('Time'); ylabel('Lon');
    
    figure
    drawtrackdata(trackdata, 3, true);
    plotMeasCoords(sensorData, 2, {'r.','k.'})
    xlabel('Time'); ylabel('Lat');
end
