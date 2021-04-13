function [postmean, postcov, loglikelihood] = kalmanupdate(...
    jointmean, jointcov, meas, angleidx)

% [postmean, postcov, loglikelihood] = kalmanupdate(...
%     jointmean, jointcov, meas, angleidx)

measdim = size(meas,1);
statedim = size(jointmean,1) - measdim;
sidx = 1:statedim;
midx = statedim + (1:measdim);

mux = jointmean(sidx);
muz = jointmean(midx);
Cxx = jointcov(sidx, sidx);
Cxz = jointcov(sidx, midx);
Czz = jointcov(midx, midx);

dz = meas - muz;
if exist('angleidx','var') && ~isempty(angleidx)
    dz = anglewrap(dz, angleidx);
end

K = Cxz*inv(Czz); %#ok<MINV>
postmean = mux + K*dz;
postcov = forcesymmetric(Cxx - K*Cxz');

if nargout>2
    loglikelihood = lognormpdf(dz, 0, Czz);
end
