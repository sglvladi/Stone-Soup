function [meanscore, targetscores] = getTracksScore(trackdata, sensorData)

% For each measurement, for each track, get probability of assignment

[allTargetNum, allSensorNum, allLineNum] = getAllSensorData(sensorData);
uqTargetNum = unique(allTargetNum);
ntargets = numel(uqTargetNum);

uqTrackIds = unique(trackdata.id);
ntracks = numel(uqTrackIds);

% Get assignment probability
nmeas = numel(allTargetNum);
assignProb = zeros(nmeas, ntracks);
for i = 1:nmeas
    idx = find(trackdata.sensorNum==allSensorNum(i) &...
        trackdata.lineNum==allLineNum(i));
    for j = 1:numel(idx)
        jj = idx(j);        
        trackidx = find(trackdata.id(jj)==uqTrackIds);
        thisprob = exp(trackdata.assignLogProb(jj));
        assignProb(i, trackidx) = assignProb(i, trackidx) + thisprob;
    end
end

assignProbTargetTrack = zeros(ntargets, ntracks);
for t = 1:ntargets
    measidx = (allTargetNum==uqTargetNum(t));
    assignProbTargetTrack(t,:) = mean(assignProb(measidx,:),1);
end

valuematrix = [zeros(ntargets,1) assignProbTargetTrack];
assigns = auction(valuematrix);
targetscores = zeros(1,ntargets);
for i = 1:ntargets
    targetscores(i) = valuematrix(i, assigns(i));
end
meanscore = mean(targetscores);

%--------------------------------------------------------------------------

function [allTargetNum, allSensorNum, allLineNum] = getAllSensorData(sensorData)

allTargetNum = cellfun(@(x)x.meas.targetNum, sensorData, 'UniformOutput', false);
allTargetNum = cat(1, allTargetNum{:});

nsensors = numel(sensorData);
allSensorNum = cell(1, nsensors);
allLineNum = cell(1, nsensors);
for i = 1:nsensors
    nmeas = numel(sensorData{i}.meas.times);
    allSensorNum{i} = repmat(i, nmeas, 1);
    allLineNum{i} = (1:nmeas)';
end
allSensorNum = cat(1, allSensorNum{:});
allLineNum = cat(1, allLineNum{:});
