function idx = getinrange(ts, lb, ub)

% idx = getinrange(ts, lb, ub)
%
% Get indices of ts such that lb<=ts(i)<=ub
% (ts assumed sorted)

idx = lower_bound(ts, lb):upper_bound(ts, ub)-1;
