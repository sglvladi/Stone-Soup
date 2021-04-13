function b = indexwithnans(a, idx)

% b = indexwithnans(a, idx)
%
% Get b(i,j) = a(idx(i,j)) if idx(i,j) is not nan, or nan otherwise

b = nan(size(idx));
b(~isnan(idx)) = a(idx(~isnan(idx)));
