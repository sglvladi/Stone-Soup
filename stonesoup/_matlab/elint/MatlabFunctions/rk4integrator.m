function thetap = rk4integrator(theta, p, M, dvfunc, stepsize, nsteps)

% [theta, p] = rk4integrator(theta, p, M, dvfunc, stepsize, nsteps)
%
% Propagate system of equations dtheta/dt = p, dp/dt = -dV/dtheta using the
% RK4 integrator
%
% theta, p: Initial values of variables
% M:        Mass (assume 1 for now?)
% dvfunc:   dV/dtheta as a function of theta
% stepsize: Time forward to propagate
% nsteps:   Number of Leapfrog steps to use

% Deal with mass matrix later
assert(isequal(M,eye(size(M))));

statedim = size(theta,1);
h = stepsize/nsteps;
f = @(thetap)([thetap(statedim+1:end,:); -dvfunc(thetap(1:statedim,:))]);
x = [theta;p];

for j=1:nsteps
    k1 = f(x);
    k2 = f(x + h/2*k1);
    k3 = f(x + h/2*k2);
    k4 = f(x + h*k3);
    x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end

theta = x(1:statedim,:);
p = x(statedim+1:end,:);

thetap = [theta;p];