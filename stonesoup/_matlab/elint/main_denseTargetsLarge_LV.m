function main_denseTargetsLarge

% addpath(cd(cd('MatlabFunctions')));

% Large target scenario for timings

verbose = true;

rng(1, 'twister')

tic

ntargets = 5;
estNumUnconfirmedTargets = 5;


% [sensorData, colours, truth] = simulateDenseTargets(ntargets, [-84; 34], [9; 62]);
% 
% for i = 1:numel(sensorData)
%     sensorData{i}.sensor.gatesdThresh = 5;
% end
% 
% % Get which sensor and which line to use for each measurement (in time
% % order)
% [timeIndices.sensor, timeIndices.line] = getTimeIndices(...
%     cellfun(@(x)x.meas.times, sensorData, 'UniformOutput', false));

[sensorData, colours, truth, timeIndices] = simulateDenseTargets_LV(ntargets);
numScans = numel(timeIndices.sensor);
initDateTime = sensorData{timeIndices.sensor(1)}.meas.times(timeIndices.line(1));

prior.posMin = [-84 34];
prior.posMax = [  9 62];
prior.speedMetresSD = 10;

% Set up transition model stuff
mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;
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
% [trackdata, tracks] = tracker_LV(sensorData, txmodel, prior, colours, params, verbose);
doplottracks = false;%true;%
if ~exist('verbose', 'var')
    verbose = true;
end
plottracks_axis = [0 1 55 56] + 0.5*[-1 1 -1 1];

% true to output the most likely Gaussian comp for the state rather than a
% single best-fitting Gaussian
if ~isfield(params, 'outputMostLikelyComp')
    params.outputMostLikelyComp = false;
end

tracks = cell(0);
nextTrackID = 1;
currenttime = 0;
trackdata = [];
sensData = {};
for i=1:length(sensorData)
    sensData{end+1} = rmfield(sensorData{i}, 'meas');
end


for measNum = 1:numScans
    
    sensorIndex = timeIndices.sensor(measNum);
    lineIndex = timeIndices.line(measNum);
    
    thismeas = getMeasurementData_LV(sensorData{sensorIndex},...
        lineIndex, colours);

    thismeasDateTime = datetime(thismeas.time);
    thismeas.time = seconds(thismeasDateTime - initDateTime);
    newtime = thismeas.time;
    dtSeconds = newtime - currenttime;
    
    numTracks = numel(tracks);
    maxTrackID = nextTrackID-1;
%     if numTracks > 0
%         maxTrackID = tracks{end}.id;
%     end
%     if ~mod(measNum, 1) && verbose
%         fprintf(['Timestep %d / %d (%s @ %f), %d tracks ('...
%             'maxtrackid=%d)\n'],...
%             measNum, numScans, sensData{sensorIndex}.name, newtime,...
%             numTracks, maxTrackID);
%     end
    
    
    [tracks, nextTrackID] = ...
    tracker_single_python(tracks, thismeas, dtSeconds, txmodel, sensorIndex, ...
                   sensData, prior, colours, params, measNum, numScans,nextTrackID); %, verbose, numScans
               
               
    currenttime = newtime;
               
    %------
    if doplottracks
        clf
        plotTrackGaussians(tracks);
%         axis(plottracks_axis);
        plot(thismeas.pos(1), thismeas.pos(2), 'r+')
        if isfield(params, 'truth')
            plot(params.truth.lonLat(:,1), params.truth.lonLat(:,2), 'k+')
        end
        pause(0.1)
        drawnow
    end
    %------
    
end
trackdata = mergeFields(trackdata);
trackdata.initDateTime = initDateTime;

trackdata.targetNum = getTrueTargetNum(trackdata, sensorData);

toc

save large_data

function plotTrackGaussians(tracks)

hold on
for i = 1:numel(tracks)
    [mn, cv] = getTrackStateGaussian(tracks{i});
    gaussellipse(mn([1 3]), cv([1 3],[1 3]), 'b', 'x');
    text(mn(1), mn(3), ['  ' num2str(tracks{i}.id)]);
end
axis equal

function newtrackdata = mergeFields(trackdata)

if isempty(trackdata)
    newtrackdata = [];
    return
end

fn = fieldnames(trackdata);
for i = 1:numel(fn)
    newtrackdata.(fn{i}) = cat(1, trackdata(:).(fn{i}));
end
