function targetNum = getTrueTargetNum(trackdata, sensorData)

% targetNum = getTrueTargetNum(trackdata, sensorData)
%
% Get true target number corresponding to each trackdata point

npts = numel(trackdata.time);
targetNum = nan(npts, 1);
[uqsens,~,idx] = uniquewithcounts(trackdata.sensorNum);

for s = 1:numel(uqsens)
    if isfield(sensorData{uqsens(s)}.meas, 'targetNum')
        targetNum(idx{s}) = sensorData{uqsens(s)}.meas.targetNum(...
            trackdata.lineNum(idx{s}));
    end
end
