function [newmeans, newcovs, newlogweights] = mergegaussians1D(...
    means, covs, logweights)

% [newmeans, newcovs, newlogweights] = mergegaussians1D(...
%     means, covs, logweights)
%
% Treat each column for means, covs, logweights as a set of 1-d Gaussians
% to merge

% Normalise columns to sum to 1 (in logs)
newlogweights = sumcolumnsinlogs(logweights);
weights = exp(logweights - newlogweights);

% Get merged means and covariances
newmeans = sum(weights.*means, 1);
dx2 = (means - newmeans).^2;
newcovs = sum(weights.*(dx2 + covs), 1);
