function [logw, logsum] = normaliseinlogs(logw)

% [logw, logsum] = normaliseinlogs(logw)

c = max(logw(:));
if c == -inf
    logw = -inf(size(logw));
    logsum = -inf;
else
    logsum = log(sum(exp(logw(:) - c))) + c;
    logw = logw - logsum;
end