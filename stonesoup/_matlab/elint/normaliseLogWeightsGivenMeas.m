function logweightsGivenMeas = normaliseLogWeightsGivenMeas(...
    logweightsGivenMeas, measHistIndices)

[~,~,idx] = uniquewithcounts(measHistIndices);
for i = 1:numel(idx)
    logweightsGivenMeas(idx{i}) = normaliseinlogs(logweightsGivenMeas(idx{i}));
end
