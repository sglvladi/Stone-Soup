function truth = simulateConstantVelocity(priormean, priorcov, thetimes, qs)

% truth = simulateConstantVelocity(priormean, priorcov, thetimes, qs)

dt = diff(thetimes);
statedim = numel(priormean);
ntimesteps = numel(thetimes);

truth = zeros(statedim, ntimesteps);
truth(:,1) = mvnrnd(priormean', priorcov)';
for k=2:ntimesteps
    [A, Q] = constantVelocityModel(dt(k-1), qs);
    truth(:,k) = A*truth(:,k-1) + mvnrnd(zeros(1,statedim), Q)';
end
