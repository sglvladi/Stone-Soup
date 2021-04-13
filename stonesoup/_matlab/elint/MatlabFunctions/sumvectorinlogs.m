function logsum = sumvectorinlogs(v)

% logsum = sumvectorinlogs(v)

if isempty(v)
    logsum = -inf;
else
    [~, logsum] = normaliseinlogs(v);
end
