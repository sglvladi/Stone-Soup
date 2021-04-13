function [sensorIndices, lineIndices, measTimes] = getTimeIndices(timecell)

% [sensorIndices, lineIndices, measTimes] = getTimeIndices(timecell)
%
% Given timecell{i} = times of measurements of ith sensor, get
% sensorIndices(i) = sensor of ith measurement,
% lineIndices(i) = measurement number of sensorIndices(i) measurement
%   sorted in increasing time order

sensorLineIndices = cell(1,numel(timecell));
alltimes = cell(1,numel(timecell));
for i=1:numel(timecell)
    nlines = numel(timecell{i});
    sensorLineIndices{i} = [repmat(i,nlines,1) (1:nlines)'];
    alltimes{i} = timecell{i}(:);
end
alltimes = cat(1, alltimes{:});
sensorLineIndices = cat(1, sensorLineIndices{:});
[measTimes, idx] = sort(alltimes);
sensorIndices = sensorLineIndices(idx,1);
lineIndices = sensorLineIndices(idx,2);
