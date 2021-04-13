function idx = clusterMeasHistory(track, histlen, logprobthresh)

% Cluster components to merge based on common history
%
% Return idx{i} = old component indices to merge for new component i
% Components might be dropped if their probability is sufficiently low

% Prune unlikely hypotheses
tokeep = find(track.logweights > logprobthresh);
if isempty(tokeep)
    [~, tokeep] = max(track.logweights);
end

% Cluster based on measurement history
L = min(histlen, size(track.measHist,1));
% Keep MMSIs separate
% disp(track.mmsis);
[~,~,mmsiid] = unique(track.mmsis(tokeep));
hist = track.measHist(end-L+1:end, tokeep);
[~,~,idx] = uniquecolumnswithcounts([mmsiid(:)'; hist]);
idx = cellfun(@(x)tokeep(x), idx, 'UniformOutput', false);
