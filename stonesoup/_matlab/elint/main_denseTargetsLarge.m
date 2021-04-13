function main_denseTargetsLarge

% addpath(cd(cd('MatlabFunctions')));

% Large target scenario for timings

rng(1, 'twister')

tic

ntargets = 1000;
estNumUnconfirmedTargets = 1000;
[sensorData, colours, truth] = simulateDenseTargets(ntargets, [-84; 34], [9; 62]);

verbose = true;

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

for i = 1:numel(sensorData)
    sensorData{i}.sensor.gatesdThresh = 5;
end

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
params.trackstooutput = 'mostlikely';
params.outputMostLikelyComp = true;
params.estNumUnconfirmedTargets = estNumUnconfirmedTargets;

params.truth = truth;
[trackdata, tracks] = tracker_LV(sensorData, txmodel, prior, colours, params, verbose);
trackdata.targetNum = getTrueTargetNum(trackdata, sensorData);

toc

save large_data
