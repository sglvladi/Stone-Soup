function wbins = getweightinbins(pts, edges, weight)

% wbins = getweightinbins(pts, edges, weight)
%
% Get wbins(i) = sum of weight(j) where edges(i)<=pts(j)<edges(i+1)

[pts, idx] = sort(pts);
weight = weight(idx);
nbins = numel(edges)-1;
wbins = zeros(1,nbins);

for i=1:nbins
    % Get indices where edges(i)<=pts<edges(i+1)
    idx = lower_bound(pts, edges(i)):lower_bound(pts, edges(i+1))-1;
    wbins(i) = sum(weight(idx));
end
