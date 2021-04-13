function [mu, cov] = weightedstats(samples, weights)

if ~exist('weights','var')
    weights = ones(1,size(samples,2));
end
weights = weights/sum(weights);
mu = sum(bsxfun(@times, weights, samples),2);
dx = bsxfun(@minus, samples, mu);
cov = bsxfun(@times, weights, dx) * dx';
