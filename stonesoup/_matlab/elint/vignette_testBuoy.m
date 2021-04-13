function vignette_testBuoy

addpath(cd(cd('MatlabFunctions')));

dorun(0)
dorun(0.05)
dorun(0.1)
dorun(0.2)
dorun(0.3)

%--------------------------------------------------------------------------

function dorun(buoyOffsetLat)

outdir = 'TestVignettes/Buoy';

if ~isempty(outdir) && ~exist(outdir, 'dir')
    mkdir(outdir);
end

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

[sensorData, colours] = simulateBuoyScenario(buoyOffsetLat);

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
txmodel.K = 0;%txmodel.q_metres/(2*stationary_speed^2);
txmodel.deathRate = 1/(4*days2sec);
% Birth rate?

params.killProbThresh = 0.01;
params.measHistLen = 1;
params.compLogProbThresh = -inf;%log(1e-2);%
params.trackstooutput = 'all';
params.estNumUnconfirmedTargets = 100;

[trackdata, tracks] = tracker(sensorData, txmodel, prior, colours, params);

h = figure;
sgtitle(['Buoy offset ' num2str(buoyOffsetLat) ' lat'])
subplot(221)
title('Track 1 Lon Lat')
drawtrackdata(trackdata, [1 3], true);
plotMeasCoords(sensorData, [1 2]);
axis equal

subplot(222)
title('Track 1 T Lon')
hold on
drawtrackdata(trackdata, 1, true);
plotMeasCoords(sensorData, 1);

subplot(223)
title('Track 1 MMSI probs')
plotMMSIProbs(trackdata, 1);
% if max(trackdata.id)>1
%     subplot(224)
%     title('Track 2 MMSI probs')
%     plotMMSIProbs(trackdata, 2);
% end

subplot(224)
idx = trackdata.id==1;
title('Track 1 AIS vis prob')
hold on
plot(trackdata.time(idx), trackdata.visProbs(idx,1), 'r-')
plot(trackdata.time(idx), trackdata.existProb(idx,1), 'k--')

if ~isempty(outdir)
    offstr = strrep(num2str(buoyOffsetLat), '.', '_');
    fname = fullfile(outdir, ['buoyoffset' offstr]);
    saveas(h, fname, 'fig');
    saveas(h, fname, 'png');
end
