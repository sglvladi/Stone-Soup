function vignette_testSwitch

% Vignette hopefully showing the utility of modelling switching in tracking

addpath(cd(cd('MatlabFunctions')));

[sensorData, colours, truth] = simulateDataSwitch();

[prior, txmodel, params] = getParams();
txmodel.q_metres = 0.01;

trackdataSw = doTest(sensorData, truth, txmodel, prior, colours, params,...
    true, 'Use Switch');
trackdataNSw = doTest(sensorData, truth, txmodel, prior, colours, params,...
    false, 'Ignore Switch');

%--------------------------------------------------------------------------

function trackdata = doTest(sensorData, truth, txmodel, prior,...
    colours, params, isSwitch, titletext)

outdir = ['TestVignettes\Switch\' strrep(titletext, ' ', '')];

if ~isempty(outdir) && ~exist(outdir, 'dir')
    mkdir(outdir);
end

if ~isSwitch
    for c = 1:numel(colours)
        colours(c).isSwitch = false;
    end
end

[trackdata, tracks] = tracker(sensorData, txmodel, prior, colours, params);

colourname = 'pri3'; % colour name of interest
colourIndex = find(findstringincell(colourname, {colours.name}));

trackcolours = {'r'};

hlonlat = figure; hold on
drawtrackdata(trackdata, [1 3], true, trackcolours);
for t = 1:numel(truth)
    plot(truth(t).coords(:,1), truth(t).coords(:,2), 'k-')
end
plotMeasCoords(sensorData, [1 2], {'b.'});
xlabel('Longitude'); ylabel('Latitude');
title(titletext);
axis equal

htlon = figure; hold on
drawtrackdata(trackdata, 1, true, trackcolours);
for t = 1:numel(truth)
    plot(truth(t).times, truth(t).coords(:,1), 'k-')
end
plotMeasCoords(sensorData, 1, {'b.'});
xlabel('Time'); ylabel('Longitude');
title(titletext);

htlat = figure; hold on
drawtrackdata(trackdata, 3, true, trackcolours);
for t = 1:numel(truth)
    plot(truth(t).times, truth(t).coords(:,2), 'k-')
end
plotMeasCoords(sensorData, 2, {'b.'});
xlabel('Time'); ylabel('Latitude');
title(titletext);

hcolour = figure; hold on
sensorNums = [1 2];
drawtrackdata(trackdata, colourIndex + 4, true, trackcolours, sensorNums);
for t = 1:numel(truth)
    plot(truth(t).times, truth(t).(colourname), 'k-')
end
meas = sensorData{1}.meas;
plot(meas.times, meas.(colourname), 'b.');
xlabel('Time'); ylabel(colourname);
title(titletext);

hswitchprob = figure; hold on
for t = 1:numel(truth)
    plot(truth(t).times, truth(t).([colourname 'switch']), 'k-')
end
drawTrackSwitchProbs(trackdata, colourIndex, trackcolours)
ylim([0 1.1])

xlabel('Time'); ylabel([colourname ' switch probability']);
title(titletext);

if ~isempty(outdir)
    saveas(hlonlat, fullfile(outdir,'lonlat'), 'fig')
    saveas(hlonlat, fullfile(outdir,'lonlat'), 'png')
    saveas(htlon, fullfile(outdir,'tlon'), 'fig')
    saveas(htlon, fullfile(outdir,'tlon'), 'png')
    saveas(htlat, fullfile(outdir,'tlat'), 'fig')
    saveas(htlat, fullfile(outdir,'tlat'), 'png')
    saveas(hcolour, fullfile(outdir,'colour'), 'fig')
    saveas(hcolour, fullfile(outdir,'colour'), 'png')
    saveas(hswitchprob, fullfile(outdir,'switchprob'), 'fig')
    saveas(hswitchprob, fullfile(outdir,'switchprob'), 'png')
end
%--------------------------------------------------------------------------

function [prior, txmodel, params] = getParams()

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

prior.posMin = [-84 34];
prior.posMax = [  9 62];
prior.speedMetresSD = 10;

% Set up transition model stuff
txmodel.isOrnsteinUhlenbeck = true;
txmodel.q_metres = 0.01;
txmodel.K = getOrnsteinUhlenbeckK(txmodel.q_metres, prior.speedMetresSD);
txmodel.deathRate = 1/(4*days2sec);

params.killProbThresh = 0.01;
params.measHistLen = 2;
%params.modelHistLen = 1; (assumed to be 1 currently)
% Delete component if log probability less than this
params.compLogProbThresh = log(1e-3);
params.trackstooutput = 'gated';
params.estNumUnconfirmedTargets = 10;
