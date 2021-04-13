function idx = upper_bound(ts, val)

% idx = upper_bound(ts, val)
%
% Get idx = first element of ts where ts(idx)>val (idx=len+1 means all
% elements are <= val)
% (ts assumed sorted)

n = numel(ts);
lower = 1;   % idx >= lower
upper = n+1; % idx <= upper

while upper>lower
    c = floor((lower + upper)/2);
    if ts(c)<=val
        lower = c+1;
    else
        upper = c;
    end
end
idx = lower;
