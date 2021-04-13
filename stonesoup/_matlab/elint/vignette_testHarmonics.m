function vignette_testHarmonics

% Vignette hopefully showing the utility of modelling harmonics in tracking

addpath(cd(cd('MatlabFunctions')));

[sensorData, colours, truth] = simulateDataHarmonics();

[prior, txmodel, params] = getParams();
txmodel.q_metres = 0.01;

trackdataH = doTest(sensorData, truth, txmodel, prior, colours, params,...
    true, 'Use Harmonics');
trackdataNH = doTest(sensorData, truth, txmodel, prior, colours, params,...
    false, 'Ignore Harmonics');

%--------------------------------------------------------------------------

function trackdata = doTest(sensorData, truth, txmodel, prior,...
    colours, params, isHarmonic, titletext)

outdir = ['TestVignettes\Harmonics\' strrep(titletext, ' ', '')];

if ~isempty(outdir) && ~exist(outdir, 'dir')
    mkdir(outdir);
end

if ~isHarmonic
    for c = 1:numel(colours)
        colours(c).isHarmonic = false;
    end
end

[trackdata, tracks] = tracker(sensorData, txmodel, prior, colours, params);
sensorNums = [1 2];

trackcolours = {'r'};

hlonlat = figure; hold on
drawtrackdata(trackdata, [1 3], true, trackcolours, sensorNums);
for t = 1:numel(truth)
    plot(truth(t).coords(:,1), truth(t).coords(:,2), 'k-')
end
plotMeasCoords(sensorData, [1 2], {'b.'});
xlabel('Longitude'); ylabel('Latitude');
title(titletext);
axis equal

htlon = figure; hold on
drawtrackdata(trackdata, 1, true, trackcolours, sensorNums);
for t = 1:numel(truth)
    plot(truth(t).times, truth(t).coords(:,1), 'k-')
end
plotMeasCoords(sensorData, 1, {'b.'});
xlabel('Time'); ylabel('Longitude');
title(titletext);

htlat = figure; hold on
drawtrackdata(trackdata, 3, true, trackcolours, sensorNums);
for t = 1:numel(truth)
    plot(truth(t).times, truth(t).coords(:,2), 'k-')
end
plotMeasCoords(sensorData, 2, {'b.'});
xlabel('Time'); ylabel('Latitude');
title(titletext);

hcol = figure; hold on
colourname = 'scanperiod3';%colours(1).name;
colouridx = find(findstringincell(colourname, {colours.name}));
drawtrackdata(trackdata, 4 + colouridx, true, trackcolours, sensorNums);
for t = 1:numel(truth)
    plot(truth(t).times, truth(t).(colourname), 'k-')
end
meas = sensorData{1}.meas;
plot(meas.times, meas.(colourname), 'b.');
xlabel('Time'); ylabel(colourname);
title(titletext);

if ~isempty(outdir)
    saveas(hlonlat, fullfile(outdir,'lonlat'), 'fig')
    saveas(hlonlat, fullfile(outdir,'lonlat'), 'png')
    saveas(htlon, fullfile(outdir,'tlon'), 'fig')
    saveas(htlon, fullfile(outdir,'tlon'), 'png')
    saveas(htlat, fullfile(outdir,'tlat'), 'fig')
    saveas(htlat, fullfile(outdir,'tlat'), 'png')
    saveas(hcol, fullfile(outdir,'colour'), 'fig')
    saveas(hcol, fullfile(outdir,'colour'), 'png')
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
