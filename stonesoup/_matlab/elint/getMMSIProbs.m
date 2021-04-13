function [uqmmsis, logProbs] = getMMSIProbs(trackdata, lineidx)

% [uqmmsis, logProbs] = getMMSIProbs(trackdata, lineidx)
%
% Get uqmmsis = cell array of MMSIs assigned to <lineidx> lines of
% trackdata (including unassigned '')
%     logProbs(k, i) = log prob of MMSI <uqmmsis{i}> at line <lineidx(k)>
ntimesteps = numel(lineidx);
uqmmsis = trackdata.mmsis(lineidx);
uqmmsis = unique(cat(2, uqmmsis{:}));
nuqmmsis = numel(uqmmsis);

logProbs = -inf(ntimesteps, nuqmmsis);
for k = 1:ntimesteps
    [~,ind] = ismember(trackdata.mmsis{lineidx(k)}, uqmmsis);
    for i = 1:numel(ind)
        logProbs(k, ind(i)) = trackdata.mmsiLogProbs{lineidx(k)}(i);
    end
end
