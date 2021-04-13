function [mu, C, logw] = mergegaussians(means, covs, logweights)

% [mu, C, logw] = mergegaussians(means, covs, logweights)

logw = sumvectorinlogs(logweights);
weights = exp(logweights - logw);
mu = sum(weights.*means, 2);
Ci = mulXXtrans(means - mu) + covs;
C = sum(reshape(weights, [1 1 numel(weights)]).*Ci, 3);

function xx = mulXXtrans(x)

xx = permute(x, [1 3 2]).*permute(x, [3 1 2]);
