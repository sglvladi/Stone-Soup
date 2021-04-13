function [x, p] = eulerintegrator(x, p, M, dvfunc, stepsize, nsteps)

smallstepsize = stepsize/nsteps;
for j=1:nsteps
    acc = -dvfunc(x);
    x = x + smallstepsize*(M\p);
	p = p + smallstepsize*acc;
end
