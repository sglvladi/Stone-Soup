function [A, Q] = constantVelocityModel(dt, qs)

AA = [1 dt; 0 1];
QQ = [dt^3/3 dt^2/2; dt^2/2 dt];
A = kron(eye(numel(qs)), AA);
Q = kron(diag(qs), QQ);
