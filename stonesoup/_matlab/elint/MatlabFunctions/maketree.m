function [parents, roots, acc] = maketree(hyps, trackorder)

% [parents, roots, acc] = maketree(hyps, trackorder)
%
% Make tree for EHM2 algorithm
%
% parents(t) = parent of track t (NaN if root)
% acc{t} = non-null hypotheses which may be used by track t or descendants

ntracks = numel(hyps);
if ~exist('trackorder', 'var')
    trackorder = 1:ntracks;
end
% Remove null hypotheses
hyps = cellfun(@(x)x(~isnan(x)), hyps, 'UniformOutput', false);

parents = nan(1, ntracks);
roots = [];
for t=fliplr(trackorder(:)')
    theseroots = [];
    acc{t} = hyps{t}; %#ok<AGROW>
    for i=roots
        if ~isempty(intersect(hyps{t}, acc{i}))
            parents(i) = t;
            acc{t} = union(acc{t}, acc{i});
        else
            theseroots = [theseroots i]; %#ok<AGROW>
        end
    end
    roots = [t theseroots];
end