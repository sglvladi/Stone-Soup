function [mean, cov] = weightedstatsangle(samples, weights, angleidx)

% [mean, cov] = weightedstatsangle(samples, weights, angleidx)
%
% Fit a Gaussian when some indices might be angles

mean = circularmean(samples, weights, angleidx);
dx = anglewrap(samples - mean, angleidx);
cov = bsxfun(@times, weights, dx) * dx';
