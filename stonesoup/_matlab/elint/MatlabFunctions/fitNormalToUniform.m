function [mu, C] = fitNormalToUniform(minvals, maxvals)

% [mu, C] = fitNormalToUniform(minvals, maxvals)

% Get mean and covariance of best-fitting Gaussian to Uniform distribution
% with specified limits
mu = (maxvals(:)+minvals(:))/2;
C = diag((maxvals(:)-minvals(:)).^2/12);