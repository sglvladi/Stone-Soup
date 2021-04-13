function newstate = constantVelocitySample(oldstate, dt, qs)

% [newmean, newcov] = constantVelocityPredict(oldmean, oldcov, dt, qs)

[A, Q] = constantVelocityModel(dt, qs);
newstate = mvnrnd((A*oldstate)', Q)';
