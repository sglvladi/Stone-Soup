function logsum = suminlogs(loga, logb)

% logsum = suminlogs(loga, logb)
%
% Return logsum = log(exp(loga) + exp(logb)) in a numerically safer way

% Make matrices the same size if necessary (bsx)
if ~isequal(size(loga), size(logb))
    loga = loga + zeros(size(logb));
    logb = logb + zeros(size(loga));
end

logsum = loga;
logsum(loga==-Inf) = logb(loga==-Inf);

idx = loga~=-Inf & logb~=-Inf;
c = max(loga(idx), logb(idx));
logsum(idx) = log(exp(loga(idx)-c) + exp(logb(idx)-c)) + c;
