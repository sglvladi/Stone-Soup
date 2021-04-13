function [newmean, newcov] = constantVelocityPredict(oldmean, oldcov, dt, qs)

% [newmean, newcov] = constantVelocityPredict(oldmean, oldcov, dt, qs)

[A, Q] = constantVelocityModel(dt, qs);
newmean = A*oldmean;
newcov = A*oldcov*A' + Q;
