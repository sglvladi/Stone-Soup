function [A, Q] = randomWalkModel(dt, qs)

% [A, Q] = randomWalkModel(dt, qs)

ndim = numel(qs);
A = eye(ndim);
Q = dt*diag(qs);
