function [mmsis, logprobs] = getMMSILogProbs(track, compidx)

% compidx = component ids to calculate probs for (default to all)
if ~exist('compidx', 'var')
    compidx = 1:numel(track.logweights);
end

[mmsis, ~, idx] = uniquewithcounts(track.mmsis(compidx));
logprobs = nan(1, numel(idx));
for i = 1:numel(idx)
    logprobs(i) = sumvectorinlogs(track.logweights(compidx(idx{i})));
end
