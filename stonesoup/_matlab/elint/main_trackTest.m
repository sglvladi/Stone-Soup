function main_trackTest

addpath(cd(cd('MatlabFunctions')));

%close all

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

%[sensorData, colours, truth] = simulateDataHarmonics();
[sensorData, colours, truth] = getData();
%[sensorData, colours, truth] = simulateData2Targets();
%[sensorData, colours, truth] = simulateDenseTargets();% sensorData = sensorData([2 3]);
%[sensorData, colours] = simulateMMSITest();
%[sensorData, colours] = simulateBuoyScenario();

for i = 1:numel(sensorData)
    sensorData{i}.sensor.gatesdThresh = 5;
end
%colours = [];

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
params.trackstooutput = 'all';%
params.estNumUnconfirmedTargets = 10000;

tic
[trackdata, tracks] = tracker(sensorData, txmodel, prior, colours, params);
toc

keyboard

sd = sensorData;
for i = 1:numel(sd)
    sd{i}.meas.times = seconds(sd{i}.meas.times - trackdata.initDateTime);
end
%td = [trackdata.time trackdata.id trackdata.mean trackdata.cov];
%drawresults(td, [1 3], 4);
drawtrackdata(trackdata, [1 3], true);
plotMeasCoords(sd, [1 2]);
axis equal

figure; hold on
%drawresults(td, 1, 4)
drawtrackdata(trackdata, 1, true);
plotMeasCoords(sd, 1);

plotMMSIProbs(trackdata, 1);
