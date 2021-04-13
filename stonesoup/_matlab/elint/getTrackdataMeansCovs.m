function [means, covs] = getTrackdataMeansCovs(trackdata, dims, idx)

% [means, covs] = getTrackdataMeansCovs(trackdata, dims, idx)
%
% Get means and covariance matrices of dimensions which can be either
% position or colour (or both)

ndims = numel(dims);
npts = numel(idx);

posidx = 1:size(trackdata.mean, 2);
if isfield(trackdata, 'colourMeans')
    colouridx = numel(posidx) + (1:size(trackdata.colourMeans, 2));
else
    colouridx = [];
end

[isp, pidx] = ismember(dims, posidx);
[isc, cidx] = ismember(dims, colouridx);
assert(all(isp | isc)); % make sure all dimensions are either position or colour

% Get position and colour means
means = nan(ndims, npts);
if any(isp)
    means(isp, :) = trackdata.mean(idx, pidx(isp))';
end
if any(isc)
    means(isc, :) = trackdata.colourMeans(idx, cidx(isc))';
end

% Get position and colour covariances if required
if nargout > 1
    nposdims = numel(posidx);
    nisp = sum(isp);
    cvtab = reshape(1:nposdims^2, nposdims, nposdims);
    cvidx = cvtab(pidx(isp), pidx(isp));
    covs = zeros(ndims, ndims, npts);
    covs(isp, isp, :) = reshape(trackdata.cov(idx, cvidx)', nisp, nisp, npts);
    for i = find(isc)
        covs(i, i, :) = reshape(trackdata.colourVars(idx, cidx(i)), [1 1 npts]);
    end
end
