function logsum = sumcolumnsinlogs(logA)

% logsum = sumcolumnsinlogs(logA)
%
% Return logsum = log(sum(exp(logA),1)) in hopefully a numerically safe way

maxlog = max(logA, [], 1);
idx = ~(maxlog == -inf);
logsum = -inf(1,size(logA,2));
logsum(idx) = log(sum(exp(logA(:,idx) - maxlog(idx)),1)) + maxlog(idx);
