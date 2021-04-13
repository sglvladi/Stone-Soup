function logsum = sumrowsinlogs(logA)

% logsum = sumrowsinlogs(logA)
%
% Return logsum = log(sum(exp(logA),2)) in hopefully a numerically safe way

maxlog = max(logA, [], 2);
idx = ~(maxlog == -inf);
logsum = -inf(size(logA,1),1);
logsum(idx) = log(sum(exp(logA(idx,:) - maxlog(idx)),2)) + maxlog(idx);
