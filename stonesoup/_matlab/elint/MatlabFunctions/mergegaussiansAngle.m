function [newmean, newcov] = mergegaussians(means, covs, logweights, angleidx)

% [newmean, newcov] = mergegaussians(means, covs, logweights, angleidx)

weights = exp(normaliseinlogs(logweights));
newmean = circularmean(means, weights, angleidx);

dx = anglewrap(means - newmean, angleidx);
newcov = zeros(size(covs,1));
for i=1:numel(weights)
    newcov = newcov + weights(i)*(covs(:,:,i) + dx(:,i)*dx(:,i)');
end
newcov = forcesymmetric(newcov);
